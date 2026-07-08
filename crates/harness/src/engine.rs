//! The agent loop: drive a [`model::ModelBackend`] and a [`ToolRegistry`]
//! through a conversation until the agent calls the `finish` tool with a claim
//! the harness accepts, or a hard iteration cap is hit.
//!
//! ## Claim vs. verify — the load-bearing invariant
//!
//! `finish(done)` is a **claim** the model makes; the harness verifies it
//! mechanically before honoring it. When the run has [`ChecksRunner`] wired in,
//! a `finish(done)` triggers the harness to re-run those exact checks itself
//! — and the disposition is only accepted if they come back green. A red
//! verification is **steering, not termination**: the fed-back tool result is
//! `is_error=true`, the loop continues, and a subsequent turn can react
//! (typically by fixing whatever failed and finishing again). The [`Done`]
//! variant is **constructed by the loop only**, and only alongside the
//! [`Verification`] evidence that justifies it — the two are unified in a
//! single struct so a `Done` value in the outcome always carries proof.
//!
//! `finish(blocked)` and `finish(failed)` are **not** verified — the model
//! declaring defeat needs no proof; those still terminate the loop as the
//! declaration states.
//!
//! ## What lives here
//!
//! - [`RunConfig`] — the shape a caller hands to [`run`]: task, iteration cap,
//!   optional [`ChecksRunner`], per-turn output cap.
//! - [`run`] — the loop itself, generic over any [`model::ModelBackend`].
//! - [`LoopOutcome`] — the four ways the loop can end.
//! - [`FinishTool`] — the tool the model calls to end the run. Its schema is
//!   what the model sees; the loop is what parses the input and (for `done`)
//!   verifies it.
//! - [`LoopOutcome::into_disposition`] — converts a terminal outcome to a
//!   [`crate::run_record::Disposition`] for storage.
//!
//! What does **not** live here yet (tracked separately): budget / token /
//! wall-clock bounds, retry / backoff, loop / no-progress detection,
//! persistence / checkpointing, and context assembly / compaction. The only
//! stopping condition beyond the agent finishing is the hard `max_iterations`
//! cap.
//!
//! ## Loop shape
//!
//! 1. Render the system prompt via [`prompt::render_system_prompt`] and the
//!    task seed via [`prompt::render_task_prompt`]. Both are computed **once**
//!    before the loop and reused verbatim on every iteration — the
//!    prompt-cache correctness invariant.
//! 2. Each iteration: build a [`TurnRequest`] and call
//!    [`model::ModelBackend::turn`].
//! 3. Append the assistant turn to history (via `From<AssistantTurn>`).
//! 4. If the turn made no tool calls, stop ([`LoopOutcome::StoppedWithoutFinish`]).
//! 5. Otherwise execute each call in order, collecting fed-back results into a
//!    single [`Message::User`]. When an executed call is `finish`, the loop
//!    verifies (or accepts) the claim per the invariant above.
//! 6. If the cap is reached before the model reaches an accepted finish, stop
//!    ([`LoopOutcome::MaxIterations`]).
//!
//! A backend error is surfaced immediately as [`LoopOutcome::BackendError`] —
//! the loop does **not** retry; retry / backoff lands with the budget work.
//!
//! [`ChecksRunner`]: crate::exec::ChecksRunner
//! [`TurnRequest`]: crate::model::TurnRequest

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::exec::{CheckReport, ChecksRunner};
use crate::model::{self, Message, SamplingParams, TurnRequest, UserBlock};
use crate::prompt;
use crate::run_record::{
    BudgetConsumed, BudgetLimits, Budgets, Disposition, DurableFacts, Event, FailureMode, Phase,
    ProjectConfig, RunRecord, SCHEMA_VERSION, Task, Verification,
};
use crate::store::{RunStore, StoreError};
use crate::time::format_rfc3339;
use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};

/// The registered name of the finish tool — the loop recognizes termination by
/// matching an executed call's name against this.
pub const FINISH_TOOL_NAME: &str = "finish";

/// Configuration for one call to [`run`].
///
/// Bundles the task text, iteration cap, optional [`ChecksRunner`], and the
/// per-turn `max_tokens` cap into one struct so the [`run`] signature stays
/// tight and adding a knob later doesn't force every caller to change. When
/// `checks` is `Some`, `finish(done)` is verified against the runner before
/// being honored (see the module docs).
///
/// Build via [`RunConfig::new`] (which sets the [`DEFAULT_MAX_TOKENS`] default)
/// and layer optional knobs with [`RunConfig::with_checks`] /
/// [`RunConfig::with_max_tokens`].
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// The seed user message: what the agent is being asked to do.
    pub task: String,
    /// Hard cap on model turns before the loop gives up
    /// ([`LoopOutcome::MaxIterations`]).
    pub max_iterations: u32,
    /// The checks the harness re-runs itself to verify a `finish(done)` claim.
    /// `None` means no automated verification — see
    /// [`crate::run_record::Verification::NoChecksConfigured`].
    pub checks: Option<ChecksRunner>,
    /// Per-turn output cap threaded into [`SamplingParams::max_tokens`].
    pub max_tokens: u32,
}

/// The default per-turn output cap. Budget-aware sizing lands with the budget
/// work; this is a fixed value for the thin slice.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

impl RunConfig {
    /// Build a config with the given `task` and iteration cap. Defaults
    /// `checks` to `None` and `max_tokens` to [`DEFAULT_MAX_TOKENS`].
    #[must_use]
    pub fn new(task: impl Into<String>, max_iterations: u32) -> Self {
        Self {
            task: task.into(),
            max_iterations,
            checks: None,
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    /// Attach a [`ChecksRunner`] — the loop will verify `finish(done)` claims
    /// against it and reject any that come back red.
    #[must_use]
    pub fn with_checks(mut self, checks: ChecksRunner) -> Self {
        self.checks = Some(checks);
        self
    }

    /// Override the per-turn output cap ([`DEFAULT_MAX_TOKENS`] by default).
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

/// Run-specific persistence bundle passed to [`run_persisted`].
///
/// Kept separate from [`RunConfig`] because `Arc<dyn RunStore>` is not
/// `Debug` (the [`RunStore`] trait has no `Debug` bound), so it cannot be
/// placed in a `derive(Debug, Clone)` struct without breaking those derives on
/// [`RunConfig`].
pub struct Persistence {
    /// Store to write events and checkpoints to.
    pub store: Arc<dyn RunStore>,
    /// Task id — used to compute the [`run_id`] and seed the [`RunRecord`].
    pub task_id: String,
    /// Which attempt number this is for the task (used in the run id).
    pub attempt_n: u32,
    /// Human-readable label for the model backend (e.g. `"claude-sonnet-4-6"`).
    /// Carried on every [`Event::ModelCall`]; the model backend deliberately
    /// does not expose its own id at the trait level.
    pub model_label: String,
}

impl std::fmt::Debug for Persistence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Persistence")
            .field("task_id", &self.task_id)
            .field("attempt_n", &self.attempt_n)
            .field("model_label", &self.model_label)
            .field("store", &"<dyn RunStore>")
            .finish()
    }
}

/// Produce a run id from `task_id` and `attempt_n`.
///
/// The id is the plain join `"{task_id}:{attempt_n}"` — NOT a hash. Stable
/// across restarts so the same attempt always addresses the same record
/// (idempotent dispatch).
///
/// # Example
///
/// ```
/// use harness::engine::run_id;
/// assert_eq!(run_id("task-42", 1), "task-42:1");
/// ```
pub fn run_id(task_id: &str, attempt_n: u32) -> String {
    format!("{task_id}:{attempt_n}")
}

/// Private in-loop persistence context — bundles the computed run id with the
/// live [`RunRecord`] being mutated as the loop progresses. Only constructed
/// (and used) when a [`Persistence`] is supplied to [`run_persisted`].
struct RunPersist {
    rid: String,
    record: RunRecord,
}

/// A claim the model made in a `finish` call, parsed from its raw JSON input.
///
/// This is the *pre-verification* view: it captures what the model said, and
/// the loop then decides whether to accept it as a [`FinishDisposition`].
/// Kept internal because callers should only ever see the post-verification
/// [`FinishDisposition`].
enum FinishClaim {
    Done { summary: String },
    Blocked { decision_needed: String },
    Failed { summary: String },
}

impl FinishClaim {
    /// Parse the `finish` tool's raw JSON input into a claim.
    ///
    /// `disposition` selects the variant; `summary` and `decision_needed` are
    /// read as strings (absent → empty). An unrecognized or missing
    /// `disposition` is treated as [`Self::Failed`] — a malformed finish is a
    /// run problem, not a clean stop.
    fn from_input(input: &Value) -> Self {
        let field = |key: &str| {
            input
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        };
        match input.get("disposition").and_then(Value::as_str) {
            Some("done") => Self::Done {
                summary: field("summary"),
            },
            Some("blocked") => Self::Blocked {
                decision_needed: field("decision_needed"),
            },
            _ => Self::Failed {
                summary: field("summary"),
            },
        }
    }
}

/// Mechanical statistics accumulated by the loop, carried alongside every
/// [`LoopOutcome`] on the [`RunResult`] a call to [`run`] returns.
///
/// - `iterations`: how many model turns the loop actually drew — a
///   [`LoopOutcome::BackendError`] on the FIRST turn yields `iterations = 1`
///   (the erroring draw counts). [`LoopOutcome::MaxIterations`] always yields
///   `iterations == config.max_iterations`.
/// - `input_tokens` / `output_tokens`: the sum of
///   [`AssistantTurn.usage.input_tokens`](crate::model::Usage::input_tokens) /
///   [`output_tokens`](crate::model::Usage::output_tokens) across every
///   SUCCESSFUL turn. Turns that returned a
///   [`BackendError`](crate::model::BackendError) contribute nothing. Per-turn
///   `u32` values sum into `u64` so a long run can't overflow.
/// - `wall_clock`: measured across the whole [`run`] call — from just before
///   the loop starts to just after it returns.
///
/// Deliberately NOT `serde`: persistence wiring
/// (into [`crate::run_record`]) is a later milestone; this type is the
/// in-memory shape the loop hands its caller today.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunStats {
    /// Model turns actually drawn — includes an erroring last turn.
    pub iterations: u32,
    /// Sum of `usage.input_tokens` across successful turns.
    pub input_tokens: u64,
    /// Sum of `usage.output_tokens` across successful turns.
    pub output_tokens: u64,
    /// Wall-clock elapsed across the whole [`run`] call.
    pub wall_clock: Duration,
}

/// The full result of one [`run`] call: the terminal [`LoopOutcome`] plus the
/// mechanical [`RunStats`] carried alongside it.
///
/// Not `PartialEq` — inherited from [`LoopOutcome`], whose
/// [`model::BackendError`] variant is a runtime error that doesn't compare.
#[derive(Debug)]
pub struct RunResult {
    /// Why the loop stopped.
    pub outcome: LoopOutcome,
    /// Mechanical stats accumulated over the run.
    pub stats: RunStats,
}

/// Why the agent loop stopped.
///
/// Not `PartialEq` because [`model::BackendError`] is a runtime error type that
/// doesn't compare; tests match on the variant instead.
#[derive(Debug)]
pub enum LoopOutcome {
    /// The agent called the `finish` tool AND the harness accepted the
    /// claim. Carries the post-verification [`Disposition`]; a `Done`
    /// here has evidence by construction (see [`Verification`]).
    Finished(Disposition),
    /// A turn produced no tool calls (the model ended its turn without
    /// finishing). The loop has nothing to feed back, so it stops.
    StoppedWithoutFinish,
    /// The hard `max_iterations` cap was reached before the agent reached an
    /// accepted finish. Repeated `finish(done)` claims that fail verification
    /// bottom out here — loop/rejection-counter detection lands with a
    /// separate item.
    MaxIterations,
    /// The backend returned an error. Surfaced as-is — the loop does not
    /// retry (retry / backoff lands with the budget work).
    BackendError(model::BackendError),
}

impl LoopOutcome {
    /// Convert a terminal [`LoopOutcome`] into its [`Disposition`].
    ///
    /// Maps every loop-exit reason to the appropriate run-record disposition:
    /// - `Finished(d)` → `d` (already a `Disposition`)
    /// - `MaxIterations` → `Failed { mode: BudgetExhausted, .. }`
    /// - `StoppedWithoutFinish` → `Failed { mode: StoppedWithoutFinish, .. }`
    /// - `BackendError(e)` → `Failed { mode: TransientInfra }` if
    ///   `e.is_retryable()`, else `Failed { mode: PersistentToolError }`
    ///
    /// The `into_` prefix (consuming `self` by value) satisfies
    /// `clippy::wrong_self_convention` for a non-`Copy` type.
    pub fn into_disposition(self) -> Disposition {
        match self {
            LoopOutcome::Finished(d) => d,
            LoopOutcome::MaxIterations => Disposition::Failed {
                mode: FailureMode::BudgetExhausted,
                summary: "iteration cap reached before the agent finished".to_string(),
            },
            LoopOutcome::StoppedWithoutFinish => Disposition::Failed {
                mode: FailureMode::StoppedWithoutFinish,
                summary: "agent stopped generating tool calls without calling finish".to_string(),
            },
            LoopOutcome::BackendError(e) => Disposition::Failed {
                mode: if e.is_retryable() {
                    FailureMode::TransientInfra
                } else {
                    FailureMode::PersistentToolError
                },
                summary: format!("backend error: {e:?}"),
            },
        }
    }
}

/// The `finish` tool: the agent calls it to end the run.
///
/// Its input is `{ disposition: "done" | "blocked" | "failed", summary:
/// string, decision_needed?: string }`.
///
/// [`Tool::run`] here just returns an ok acknowledgment; the LOOP is what
/// recognizes the name, parses the input, verifies a `done` claim against
/// the configured [`ChecksRunner`], and (when the claim is accepted) builds
/// the terminal [`LoopOutcome::Finished`]. The tool's `.run` result is only
/// used when no checks are configured or when the disposition is
/// `blocked`/`failed`; a verified-red `done` bypasses it entirely and the
/// loop synthesizes an `is_error=true` [`ToolResult`] the model can react to.
///
/// [`ChecksRunner`]: crate::exec::ChecksRunner
#[derive(Debug, Default, Clone, Copy)]
pub struct FinishTool;

#[async_trait]
impl Tool for FinishTool {
    // The trait fixes the return type as `&str`; a `&'static str` here would
    // diverge from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        FINISH_TOOL_NAME
    }

    fn schema(&self) -> Value {
        json!({
            "name": FINISH_TOOL_NAME,
            "description": "End the run. Call exactly once when the task is complete, \
                            blocked on a decision, or has failed. A `done` claim is \
                            verified by the harness re-running the configured checks; \
                            a failed verification is fed back as a tool-result error \
                            you can react to, not a termination.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "disposition": {
                        "type": "string",
                        "enum": ["done", "blocked", "failed"],
                        "description": "done = task complete (harness will verify via checks); \
                                        blocked = needs a decision before retrying; \
                                        failed = the run is the problem."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A short summary of the outcome."
                    },
                    "decision_needed": {
                        "type": "string",
                        "description": "Required when blocked: the decision a human must make."
                    }
                },
                "required": ["disposition", "summary"]
            }
        })
    }

    async fn run(&self, _input: Value, _ctx: &ToolCtx) -> ToolResult {
        ToolResult::ok("finish acknowledged")
    }
}

/// Render a [`ToolResult`] into the string fed back to the model as a
/// [`UserBlock::ToolResult`]'s content: the always-small `summary`, plus the
/// bounded `detail` when present (already capped at
/// [`crate::tool::DETAIL_CAP`] by the result constructor).
fn render_tool_result(result: &ToolResult) -> String {
    match &result.detail {
        Some(detail) => format!("{}\n{detail}", result.summary),
        None => result.summary.clone(),
    }
}

/// Build the `is_error=true` fed-back content for a REJECTED `finish(done)`:
/// the "rejected" header + the check report's excerpt + a pointer to the
/// full offloaded output when the runner offloaded it.
///
/// The wording ("finish(done) rejected: verification failed") is load-bearing
/// — a test in this module pins the substring "rejected" as the steering
/// signal the model must see.
fn rejection_content(report: &CheckReport) -> String {
    use std::fmt::Write as _;
    let mut content = String::from("finish(done) rejected: verification failed");
    if !report.excerpt.is_empty() {
        content.push('\n');
        content.push_str(&report.excerpt);
    }
    if let Some(path) = &report.offload_path {
        // `write!` into a String is infallible (its `write_str` cannot fail),
        // so `.expect` here is a lint-satisfying no-op, not a real recovery
        // path.
        write!(content, "\n\n[full check output: {}]", path.display())
            .expect("write! into String is infallible");
    }
    content
}

/// One dispatched `finish` call's outcome from the loop's point of view: the
/// fed-back [`UserBlock::ToolResult`] to hand back to the model, and — when
/// the loop should terminate — the accepted [`Disposition`].
///
/// Extracted out of [`run`] so the main loop stays readable; keeps the
/// per-call plumbing (call id, `is_error`, content wording) in one place.
struct FinishOutcome {
    result: UserBlock,
    finish: Option<Disposition>,
}

/// Handle a `finish` call: verify a `done` claim against `config.checks`
/// when configured, or accept it on trust when not. `blocked` and `failed`
/// terminate as declared with no verification.
///
/// Returns the fed-back [`UserBlock::ToolResult`] plus, when the loop should
/// terminate, the accepted [`Disposition`]. A rejected `done` returns
/// `finish = None`, an `is_error=true` result, and the loop continues.
async fn handle_finish_call(
    call_id: &str,
    input: &Value,
    checks: Option<&ChecksRunner>,
    ctx: &ToolCtx,
) -> FinishOutcome {
    match FinishClaim::from_input(input) {
        FinishClaim::Done { summary } => match checks {
            Some(runner) => {
                let report = runner.run(ctx).await;
                if report.passed {
                    FinishOutcome {
                        result: ack(call_id),
                        finish: Some(Disposition::Done {
                            summary,
                            verification: Verification::Checks(report),
                        }),
                    }
                } else {
                    FinishOutcome {
                        result: UserBlock::ToolResult {
                            call_id: call_id.to_string(),
                            content: rejection_content(&report),
                            is_error: true,
                        },
                        finish: None,
                    }
                }
            }
            None => FinishOutcome {
                result: ack(call_id),
                finish: Some(Disposition::Done {
                    summary,
                    verification: Verification::NoChecksConfigured,
                }),
            },
        },
        FinishClaim::Blocked { decision_needed } => FinishOutcome {
            result: ack(call_id),
            finish: Some(Disposition::Blocked { decision_needed }),
        },
        FinishClaim::Failed { summary } => FinishOutcome {
            result: ack(call_id),
            finish: Some(Disposition::Failed {
                mode: FailureMode::Loop,
                summary,
            }),
        },
    }
}

/// The standard `finish acknowledged` fed-back [`UserBlock::ToolResult`] the
/// loop hands the model for an accepted `finish` (any disposition, or a
/// `done` with no checks configured). Kept factored so the wording matches
/// exactly across the four accepted paths.
fn ack(call_id: &str) -> UserBlock {
    UserBlock::ToolResult {
        call_id: call_id.to_string(),
        content: "finish acknowledged".to_string(),
        is_error: false,
    }
}

/// Drive `backend` + `tools` through a conversation until the agent finishes
/// or `config.max_iterations` is hit.
///
/// The system prompt and task seed are rendered ONCE via the [`crate::prompt`]
/// layer before the loop starts and then reused verbatim every iteration —
/// this is the prompt-cache correctness invariant, and the reason the caller
/// hands in a [`RunConfig`] rather than a pre-rendered `&str` system prompt.
///
/// A `finish(done)` claim is verified via [`ChecksRunner::run`] when
/// `config.checks` is `Some`; see the module docs for the full claim-vs-verify
/// contract.
///
/// The returned [`RunResult`] pairs the terminal [`LoopOutcome`] with the
/// mechanical [`RunStats`] accumulated over the run — see [`RunStats`] for
/// how each field is counted (in particular: an erroring last turn still
/// contributes to `iterations` but not to the token totals).
///
/// **No persistence:** this entry point passes `None` for the store, so no
/// events are appended and no checkpoints are written. Use [`run_persisted`]
/// when durability is needed.
///
/// # Panics
///
/// This function is infallible in practice, but internally calls `.expect()` on
/// a `Result` that is structurally `Ok` when no persistence is wired in (no
/// store calls are made). If that invariant were violated, the function would
/// panic with a diagnostic message.
///
/// [`ChecksRunner::run`]: crate::exec::ChecksRunner::run
pub async fn run(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
) -> RunResult {
    // `wall_clock` covers the whole call (prompt rendering included, since
    // that's real work the loop did). `stats` is threaded into the loop by
    // mut-ref so every termination path picks up the same in-progress totals.
    let start = Instant::now();
    let mut stats = RunStats {
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock: Duration::ZERO,
    };
    // No persistence — the Result::Err path is structurally unreachable when
    // persistence is None (no store calls are made), so the expect is a
    // compile-time invariant, not a runtime safety net.
    let outcome = run_loop_impl(backend, tools, ctx, config, None, &mut stats)
        .await
        .expect("no-persistence run cannot produce a StoreError");
    stats.wall_clock = start.elapsed();
    RunResult { outcome, stats }
}

/// Drive `backend` + `tools` with full durability: append events and write
/// checkpoints to `persistence.store` as the loop progresses.
///
/// ## Persistence discipline
///
/// - A [`RunRecord`] is constructed at run start and kept current throughout.
/// - After each successful model turn: [`Event::ModelCall`] then
///   [`Event::BudgetTick`] are appended, followed by a mid-iteration
///   checkpoint (snapshot includes the assistant turn in `messages`).
/// - For each non-`finish` tool call: [`Event::ToolCallStarted`] (before
///   execution) and [`Event::ToolCallResult`] (after execution) are appended.
/// - At end of each loop iteration: a full checkpoint is written.
/// - On every terminal path: [`Event::DispositionSet`] is appended, then a
///   final checkpoint is written with `disposition` set.
///
/// The first [`StoreError`] from any append or checkpoint immediately aborts
/// the loop and is returned as `Err`. The no-store [`run`] path keeps its
/// bare `RunResult` return type unchanged.
pub async fn run_persisted(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    persistence: &Persistence,
) -> Result<RunResult, StoreError> {
    let start = Instant::now();
    let mut stats = RunStats {
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock: Duration::ZERO,
    };
    let outcome = run_loop_impl(backend, tools, ctx, config, Some(persistence), &mut stats).await?;
    stats.wall_clock = start.elapsed();
    Ok(RunResult { outcome, stats })
}

/// The engine loop body, split out of [`run`] / [`run_persisted`] so the outer
/// functions own the `wall_clock` measurement and can finalize [`RunStats`] on
/// every return path.
///
/// `stats` is mutated in place as the loop progresses:
/// - `iterations` is incremented BEFORE each `backend.turn` call — so a
///   backend error on the first draw yields `iterations = 1`, matching the
///   "model turns actually drawn" contract in [`RunStats`].
/// - `input_tokens` / `output_tokens` accumulate the SUCCESSFUL turn's
///   [`crate::model::Usage`] only; errored turns contribute nothing.
///
/// When `persistence` is `Some`, the loop appends events and writes
/// checkpoints per the durability contract documented on [`run_persisted`].
/// When `persistence` is `None`, no store calls are made and the function
/// returns `Ok(outcome)` (the `Err` arm is structurally unreachable).
#[allow(clippy::too_many_lines)]
async fn run_loop_impl(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    persistence: Option<&Persistence>,
    stats: &mut RunStats,
) -> Result<LoopOutcome, StoreError> {
    // Render both prompts ONCE and reuse verbatim every iteration. Both
    // functions are pure and byte-deterministic (see `prompt` module tests),
    // so the borrows below stay stable across the whole loop — the
    // prompt-cache correctness invariant.
    let system = prompt::render_system_prompt(
        &prompt::tool_lines(tools),
        config
            .checks
            .as_ref()
            .map(ChecksRunner::command_display)
            .as_deref(),
    );
    let task_message = prompt::render_task_prompt(&config.task);

    let mut messages: Vec<Message> = vec![Message::User {
        content: vec![UserBlock::Text(task_message)],
    }];
    let tool_schemas = tools.list();
    let params = SamplingParams {
        max_tokens: config.max_tokens,
        temperature: None,
        stop_sequences: Vec::new(),
    };

    // Build the run-record if persistence is configured. This is kept as
    // Option<RunPersist> so the non-persistent path has zero overhead.
    let mut persist: Option<RunPersist> = if let Some(p) = persistence {
        let rid = run_id(&p.task_id, p.attempt_n);
        let wall_clock_start = format_rfc3339(SystemTime::now());

        // run_checks: a single "checks" entry when a ChecksRunner is wired in,
        // empty BTreeMap otherwise (D7 honesty: only what we actually know).
        let run_checks = match &config.checks {
            None => BTreeMap::new(),
            Some(runner) => {
                let mut m = BTreeMap::new();
                m.insert("checks".to_string(), runner.command_display());
                m
            }
        };

        let record = RunRecord {
            run_id: rid.clone(),
            schema_version: SCHEMA_VERSION,
            attempt_n: p.attempt_n,
            task: Task {
                task_id: p.task_id.clone(),
                title: String::new(),
                description: config.task.clone(),
                acceptance_criteria: vec![],
                files_in_scope: vec![],
                scope_out: vec![],
            },
            project_config: ProjectConfig {
                run_checks,
                model_routing_hint: None,
            },
            phase: Phase::InnerLoop,
            durable_facts: DurableFacts::default(),
            budgets: Budgets {
                consumed: BudgetConsumed::default(),
                limits: BudgetLimits {
                    iterations: config.max_iterations,
                    tokens: 0,
                    cost_micros: 0,
                },
                wall_clock_start,
            },
            last_gate_result: None,
            disposition: None,
            messages: messages.clone(),
        };
        Some(RunPersist { rid, record })
    } else {
        None
    };

    for _ in 0..config.max_iterations {
        let req = TurnRequest {
            system: Some(&system),
            messages: &messages,
            tools: &tool_schemas,
            params: &params,
        };

        // Count the draw BEFORE the call so an error on the first turn still
        // shows up as `iterations = 1` in `stats`.
        stats.iterations += 1;
        let turn = match backend.turn(&req).await {
            Ok(turn) => turn,
            Err(err) => {
                // Terminal path: BackendError. Persist the disposition before
                // returning — this path exits early, bypassing the normal
                // end-of-iteration checkpoint.
                if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                    let mode = if err.is_retryable() {
                        FailureMode::TransientInfra
                    } else {
                        FailureMode::PersistentToolError
                    };
                    let disposition = Disposition::Failed {
                        mode,
                        summary: format!("backend error: {err:?}"),
                    };
                    ctx.record.budgets.consumed = BudgetConsumed {
                        iterations: stats.iterations,
                        tokens: stats.input_tokens + stats.output_tokens,
                        cost_micros: 0,
                    };
                    ctx.record.messages.clone_from(&messages);
                    ctx.record.disposition = Some(disposition.clone());
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::DispositionSet {
                                seq: 0,
                                disposition,
                            },
                        )
                        .await?;
                    p.store.checkpoint(&ctx.rid, &ctx.record).await?;
                }
                return Ok(LoopOutcome::BackendError(err));
            }
        };

        // Successful turn — capture per-turn usage BEFORE moving the turn
        // into history (Message::from consumes it).
        let per_turn_input = u64::from(turn.usage.input_tokens);
        let per_turn_output = u64::from(turn.usage.output_tokens);

        // Accumulate into run totals. Per-turn u32 values sum into u64 so a
        // long run can't overflow.
        stats.input_tokens += per_turn_input;
        stats.output_tokens += per_turn_output;

        // Snapshot the calls before moving the turn into history (the `From`
        // impl consumes `turn.content`).
        let calls: Vec<_> = turn.tool_calls().into_iter().cloned().collect();
        messages.push(Message::from(turn));

        // Append ModelCall + BudgetTick events, then write the mid-iteration
        // checkpoint (after assistant turn, BEFORE any tools.invoke). This
        // guarantees that a mid-iteration crash always leaves a snapshot whose
        // messages include the in-flight assistant turn.
        if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
            let consumed = BudgetConsumed {
                iterations: stats.iterations,
                tokens: stats.input_tokens + stats.output_tokens,
                cost_micros: 0,
            };
            p.store
                .append_event(
                    &ctx.rid,
                    Event::ModelCall {
                        seq: 0,
                        model: p.model_label.clone(),
                        prompt_tokens: per_turn_input,
                        completion_tokens: per_turn_output,
                    },
                )
                .await?;
            p.store
                .append_event(&ctx.rid, Event::BudgetTick { seq: 0, consumed })
                .await?;
            // Mid-iteration checkpoint: assistant turn is in messages,
            // tool calls have NOT been invoked yet.
            ctx.record.budgets.consumed = consumed;
            ctx.record.messages.clone_from(&messages);
            p.store.checkpoint(&ctx.rid, &ctx.record).await?;
        }

        if calls.is_empty() {
            // Terminal path: StoppedWithoutFinish.
            if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                let disposition = Disposition::Failed {
                    mode: FailureMode::StoppedWithoutFinish,
                    summary: "agent stopped generating tool calls without calling finish"
                        .to_string(),
                };
                ctx.record.disposition = Some(disposition.clone());
                p.store
                    .append_event(
                        &ctx.rid,
                        Event::DispositionSet {
                            seq: 0,
                            disposition,
                        },
                    )
                    .await?;
                p.store.checkpoint(&ctx.rid, &ctx.record).await?;
            }
            return Ok(LoopOutcome::StoppedWithoutFinish);
        }

        // Execute every requested call in order, collecting fed-back tool
        // results into a single user message. `finish` is special-cased in
        // [`handle_finish_call`]: a `done` claim triggers the harness
        // re-running the configured checks, and only a green verification
        // (or no checks at all) sets the terminal `finish` slot. A red
        // verification is fed back as an `is_error=true` result and the loop
        // CONTINUES with the remaining calls in the same batch.
        let mut results = Vec::with_capacity(calls.len());
        let mut finish: Option<Disposition> = None;
        for call in &calls {
            // Only the FIRST accepted finish in a batch gets to terminate;
            // a later finish (or a finish while one is already accepted)
            // still executes as a normal tool invocation so its
            // acknowledgement lands in the fed-back batch alongside the
            // earlier calls.
            if call.name == FINISH_TOOL_NAME && finish.is_none() {
                let outcome =
                    handle_finish_call(&call.id, &call.input, config.checks.as_ref(), ctx).await;
                results.push(outcome.result);
                finish = outcome.finish;
            } else {
                // Non-finish tool call: append ToolCallStarted before invoke,
                // ToolCallResult after invoke (log-then-snapshot discipline).
                if let (Some(ctx), Some(p)) = (persist.as_ref(), persistence) {
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::ToolCallStarted {
                                seq: 0,
                                name: call.name.clone(),
                                args: call.input.clone(),
                                call_id: call.id.clone(),
                            },
                        )
                        .await?;
                }

                let result = tools.invoke(&call.name, call.input.clone(), ctx).await;

                if let (Some(ctx), Some(p)) = (persist.as_ref(), persistence) {
                    p.store
                        .append_event(
                            &ctx.rid,
                            Event::ToolCallResult {
                                seq: 0,
                                name: call.name.clone(),
                                is_error: result.is_error,
                                summary: result.summary.clone(),
                                // summary only — detail is NOT concatenated (audit record)
                                offload_path: result
                                    .offload_path
                                    .as_ref()
                                    .map(|path| path.display().to_string()),
                            },
                        )
                        .await?;
                }

                results.push(UserBlock::ToolResult {
                    call_id: call.id.clone(),
                    content: render_tool_result(&result),
                    is_error: result.is_error,
                });
            }
        }
        messages.push(Message::User { content: results });

        if let Some(disposition) = finish {
            // Terminal path: Finished. Write DispositionSet + terminal checkpoint.
            if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                ctx.record.messages.clone_from(&messages);
                ctx.record.budgets.consumed = BudgetConsumed {
                    iterations: stats.iterations,
                    tokens: stats.input_tokens + stats.output_tokens,
                    cost_micros: 0,
                };
                ctx.record.disposition = Some(disposition.clone());
                p.store
                    .append_event(
                        &ctx.rid,
                        Event::DispositionSet {
                            seq: 0,
                            disposition: disposition.clone(),
                        },
                    )
                    .await?;
                p.store.checkpoint(&ctx.rid, &ctx.record).await?;
            }
            return Ok(LoopOutcome::Finished(disposition));
        }

        // Non-terminal end of iteration: write the end-of-iteration checkpoint
        // so a crash here loses at most the current iteration's tool results
        // (already in messages).
        if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
            ctx.record.messages.clone_from(&messages);
            ctx.record.budgets.consumed = BudgetConsumed {
                iterations: stats.iterations,
                tokens: stats.input_tokens + stats.output_tokens,
                cost_micros: 0,
            };
            p.store.checkpoint(&ctx.rid, &ctx.record).await?;
        }
    }

    // Terminal path: MaxIterations.
    if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
        let disposition = Disposition::Failed {
            mode: FailureMode::BudgetExhausted,
            summary: "iteration cap reached before the agent finished".to_string(),
        };
        ctx.record.messages.clone_from(&messages);
        ctx.record.budgets.consumed = BudgetConsumed {
            iterations: stats.iterations,
            tokens: stats.input_tokens + stats.output_tokens,
            cost_micros: 0,
        };
        ctx.record.disposition = Some(disposition.clone());
        p.store
            .append_event(
                &ctx.rid,
                Event::DispositionSet {
                    seq: 0,
                    disposition,
                },
            )
            .await?;
        p.store.checkpoint(&ctx.rid, &ctx.record).await?;
    }

    Ok(LoopOutcome::MaxIterations)
}

#[cfg(test)]
mod tests {
    use super::{
        FINISH_TOOL_NAME, FinishTool, LoopOutcome, Persistence, RunConfig, RunResult, RunStats,
        rejection_content, render_tool_result, run, run_id, run_persisted,
    };
    use crate::exec::{CheckCommand, CheckReport, ChecksRunner};
    use crate::model::{
        AssistantTurn, BackendError, ContentBlock, Message, StopReason, TerminalKind,
        ToolCallRequest, Usage, UserBlock,
    };
    use crate::run_record::{
        BudgetConsumed, Disposition, Event, FailureMode, RunRecord, Verification,
    };
    use crate::store::{RunStore, SqliteRunStore, StoreError};
    use crate::test_support::MockBackend;
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use crate::tools::edit_file::EditFileTool;
    use crate::workspace::Workspace;
    use async_trait::async_trait;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::TempDir;

    // ---- test-only store doubles ----------------------------------------

    /// Enumerates the observable store operations. Used by [`RecordingStore`]
    /// to verify log-then-snapshot ordering.
    #[derive(Debug, Clone)]
    enum StoreCall {
        AppendEvent { kind: String },
        Checkpoint,
    }

    /// A [`RunStore`] that delegates to an in-memory `SQLite` store and records
    /// every call in order. Used to assert the log-then-snapshot ordering
    /// discipline (events before checkpoints within each iteration).
    struct RecordingStore {
        inner: SqliteRunStore,
        calls: Mutex<Vec<StoreCall>>,
    }

    impl RecordingStore {
        fn new() -> Self {
            Self {
                inner: SqliteRunStore::open_in_memory().expect("in-memory SQLite"),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn recorded_calls(&self) -> Vec<StoreCall> {
            self.calls.lock().expect("calls lock").clone()
        }
    }

    #[async_trait]
    impl RunStore for RecordingStore {
        async fn load(&self, rid: &str) -> Result<Option<RunRecord>, StoreError> {
            self.inner.load(rid).await
        }

        async fn append_event(&self, rid: &str, event: Event) -> Result<u64, StoreError> {
            let kind = match &event {
                Event::ModelCall { .. } => "ModelCall",
                Event::ToolCallStarted { .. } => "ToolCallStarted",
                Event::ToolCallResult { .. } => "ToolCallResult",
                Event::PhaseTransition { .. } => "PhaseTransition",
                Event::BudgetTick { .. } => "BudgetTick",
                Event::DispositionSet { .. } => "DispositionSet",
            };
            self.calls
                .lock()
                .expect("calls lock")
                .push(StoreCall::AppendEvent {
                    kind: kind.to_string(),
                });
            self.inner.append_event(rid, event).await
        }

        async fn checkpoint(&self, rid: &str, record: &RunRecord) -> Result<(), StoreError> {
            self.calls
                .lock()
                .expect("calls lock")
                .push(StoreCall::Checkpoint);
            self.inner.checkpoint(rid, record).await
        }
    }

    /// A [`RunStore`] that captures every [`RunRecord`] snapshot passed to
    /// [`RunStore::checkpoint`]. Used to inspect intermediate (mid-iteration)
    /// checkpoint states that are overwritten by subsequent checkpoints.
    struct SnapshotStore {
        inner: SqliteRunStore,
        snapshots: Mutex<Vec<RunRecord>>,
    }

    impl SnapshotStore {
        fn new() -> Self {
            Self {
                inner: SqliteRunStore::open_in_memory().expect("in-memory SQLite"),
                snapshots: Mutex::new(Vec::new()),
            }
        }

        fn all_snapshots(&self) -> Vec<RunRecord> {
            self.snapshots.lock().expect("snapshots lock").clone()
        }
    }

    #[async_trait]
    impl RunStore for SnapshotStore {
        async fn load(&self, rid: &str) -> Result<Option<RunRecord>, StoreError> {
            self.inner.load(rid).await
        }

        async fn append_event(&self, rid: &str, event: Event) -> Result<u64, StoreError> {
            self.inner.append_event(rid, event).await
        }

        async fn checkpoint(&self, rid: &str, record: &RunRecord) -> Result<(), StoreError> {
            self.snapshots
                .lock()
                .expect("snapshots lock")
                .push(record.clone());
            self.inner.checkpoint(rid, record).await
        }
    }

    /// A [`RunStore`] whose [`RunStore::append_event`] always returns an error.
    /// Used to assert that the first store error aborts [`run_persisted`].
    struct FailingStore;

    #[async_trait]
    impl RunStore for FailingStore {
        async fn load(&self, _rid: &str) -> Result<Option<RunRecord>, StoreError> {
            Ok(None)
        }

        async fn append_event(&self, _rid: &str, _event: Event) -> Result<u64, StoreError> {
            Err(StoreError::LockPoisoned)
        }

        async fn checkpoint(&self, _rid: &str, _record: &RunRecord) -> Result<(), StoreError> {
            Err(StoreError::LockPoisoned)
        }
    }

    /// Build a [`Persistence`] backed by the given store.
    fn make_persistence(store: Arc<dyn RunStore>) -> Persistence {
        Persistence {
            store,
            task_id: "task-t".to_string(),
            attempt_n: 1,
            model_label: "test-model".to_string(),
        }
    }

    // Expected run_id for the make_persistence fixture above.
    const FIXTURE_RID: &str = "task-t:1";

    fn usage() -> Usage {
        Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    /// Build a [`Usage`] with the given `input_tokens`/`output_tokens` and
    /// every optional field cleared — the smallest thing tests need to script
    /// known per-turn token amounts.
    fn usage_with(input_tokens: u32, output_tokens: u32) -> Usage {
        Usage {
            input_tokens,
            output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn tool_call(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
        ContentBlock::ToolCall(ToolCallRequest {
            id: id.to_string(),
            name: name.to_string(),
            input,
        })
    }

    fn turn_with(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason,
            usage: usage(),
        }
    }

    /// Like [`turn_with`], but with an explicit [`Usage`] — for tests that
    /// pin the accumulated [`RunStats`] token totals.
    fn turn_with_usage(
        content: Vec<ContentBlock>,
        stop_reason: StopReason,
        usage: Usage,
    ) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason,
            usage,
        }
    }

    fn finish_call(id: &str, input: serde_json::Value) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, FINISH_TOOL_NAME, input)],
            StopReason::ToolUse,
        )
    }

    fn registry_with_finish_and_echo() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry.register("echo", Arc::new(EchoTool));
        registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool));
        registry
    }

    /// The last message in history must be the fed-back tool-result user
    /// message whose first block's `call_id` matches `expected_id`.
    fn assert_last_is_tool_result(messages: &[Message], expected_id: &str) {
        let last = messages.last().expect("at least one message");
        match last {
            Message::User { content } => match &content[0] {
                UserBlock::ToolResult { call_id, .. } => {
                    assert_eq!(
                        call_id, expected_id,
                        "fed-back call_id must match request id"
                    );
                }
                UserBlock::Text(_) => panic!("expected a ToolResult block, got Text"),
            },
            Message::Assistant { .. } => panic!("expected a User message, got Assistant"),
        }
    }

    fn passing_runner() -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
    }

    fn failing_runner() -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "echo FAIL_DETAIL; exit 3".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
    }

    #[tokio::test]
    async fn single_finish_done_with_no_checks_terminates_with_no_checks_verification() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "all set" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::NoChecksConfigured,
            }) => {
                assert_eq!(summary, "all set");
            }
            other => panic!("expected Finished(Done{{NoChecksConfigured}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 1, "should finish in a single iteration");
        // The Finished variant carries stats too: one drawn turn, zero tokens
        // (the mock's usage() helper is all zeros).
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
    }

    #[tokio::test]
    async fn finish_done_with_passing_checks_terminates_with_checks_verification() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "shipped" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_checks(passing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::Checks(report),
            }) => {
                assert_eq!(summary, "shipped");
                assert!(report.passed, "checks report must be green");
                assert_eq!(report.exit_code, Some(0));
            }
            other => panic!("expected Finished(Done{{Checks(green)}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 1);
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn finish_done_with_failing_checks_is_rejected_and_loop_continues() {
        // Two scripted turns: (1) finish(done) → rejected; (2) a non-finish
        // tool call so the second draw shows the loop went past the rejection.
        // We stop by drawing (2) then over-drawing the empty script.
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "premature" }),
            ),
            turn_with(
                vec![tool_call("c-echo", "echo", serde_json::json!({ "k": 1 }))],
                StopReason::ToolUse,
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        // With the second turn being a non-finish call and no third turn
        // scripted, the loop hits the over-draw path — the important
        // assertion is that we did NOT terminate at the finish, we drew
        // beyond it.
        assert!(
            matches!(outcome, LoopOutcome::BackendError(_)),
            "loop should continue past the rejected finish; got {outcome:?}"
        );
        assert!(
            backend.calls() >= 2,
            "second turn must have been drawn — loop didn't terminate at rejected finish; got {} calls",
            backend.calls()
        );

        // The rejected finish's fed-back tool result must appear somewhere
        // in the history the loop later sent to the backend: an
        // is_error=true UserBlock::ToolResult whose call_id matches
        // "call-finish" and whose content contains "rejected" + the check's
        // excerpt. Search every user message — the last one holds the
        // subsequent turn's echo result, which is the whole point (the loop
        // did NOT terminate at the rejection).
        let seen = backend.last_messages();
        let rejection = seen.iter().find_map(|m| match m {
            Message::User { content } => content.iter().find_map(|b| match b {
                UserBlock::ToolResult {
                    call_id,
                    content,
                    is_error,
                } if call_id == "call-finish" => Some((content.clone(), *is_error)),
                _ => None,
            }),
            Message::Assistant { .. } => None,
        });
        let (content, is_error) = rejection.expect("fed-back rejection tool-result present");
        assert!(is_error, "rejected finish result is is_error=true");
        assert!(
            content.contains("rejected"),
            "content must announce rejection; got:\n{content}"
        );
        assert!(
            content.contains("FAIL_DETAIL"),
            "content must include the check excerpt; got:\n{content}"
        );
        // The BackendError variant carries stats too. The mock overdraws on
        // the third turn, so `iterations` counts all three drawn turns.
        assert_eq!(stats.iterations, 3);
    }

    #[tokio::test]
    async fn max_iterations_hits_when_model_keeps_claiming_done_with_red_checks() {
        // The model claims done twice in a row against a failing runner; both
        // are rejected; the loop hits max_iterations rather than terminating.
        let backend = MockBackend::from_turns(vec![
            finish_call(
                "c-1",
                serde_json::json!({ "disposition": "done", "summary": "first claim" }),
            ),
            finish_call(
                "c-2",
                serde_json::json!({ "disposition": "done", "summary": "second claim" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 2).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "repeated red claims must hit MaxIterations; got {outcome:?}"
        );
        assert_eq!(
            backend.calls(),
            2,
            "drew exactly max_iterations turns before giving up"
        );
        // MaxIterations carries stats — iterations equals the cap exactly.
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn blocked_terminates_without_verification() {
        // The runner is set to fail — if the loop were verifying blocked,
        // this test would not terminate. It does verify termination is
        // unconditional for `blocked`.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-blocked",
            serde_json::json!({
                "disposition": "blocked",
                "summary": "ambiguous spec",
                "decision_needed": "which API version?"
            }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "which API version?");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn failed_terminates_without_verification() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-failed",
            serde_json::json!({ "disposition": "failed", "summary": "tool kept erroring" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(failing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { summary, .. }) => {
                assert_eq!(summary, "tool kept erroring");
            }
            other => panic!("expected Finished(Failed), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn full_claim_vs_verify_integration_flag_file() {
        // The full loop: checks look for a file `flag` that doesn't exist
        // initially. Scripted turns:
        //   (1) finish(done) → REJECTED (flag missing → checks fail).
        //   (2) edit_file creates `flag` in the workspace.
        //   (3) finish(done) → VERIFIED GREEN → Finished(Done).
        // This is the "harness itself re-ran the checks" assertion — the
        // model never called run_checks, but a red claim was rejected and a
        // subsequent green claim was accepted.
        let root = TempDir::new().expect("workspace tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize root");
        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));

        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "test -f flag".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let mut tools = ToolRegistry::new();
        tools.register("edit_file", Arc::new(EditFileTool));
        tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

        let backend = MockBackend::from_turns(vec![
            // 1: claim done — flag doesn't exist, should be rejected.
            finish_call(
                "c-premature",
                serde_json::json!({ "disposition": "done", "summary": "premature" }),
            ),
            // 2: create the flag file.
            turn_with(
                vec![tool_call(
                    "c-edit",
                    "edit_file",
                    serde_json::json!({
                        "path": "flag",
                        "old_string": "",
                        "new_string": "planted\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            // 3: claim done again — flag exists, should be verified.
            finish_call(
                "c-verified",
                serde_json::json!({ "disposition": "done", "summary": "flag planted" }),
            ),
        ]);

        let config = RunConfig::new("plant the flag", 5).with_checks(runner);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Done {
                summary,
                verification: Verification::Checks(report),
            }) => {
                assert_eq!(summary, "flag planted");
                assert!(
                    report.passed,
                    "second claim's report must be green (flag now exists)"
                );
            }
            other => panic!(
                "expected Finished(Done{{Checks(green)}}) after edit_file plants the flag; got {other:?}"
            ),
        }
        assert_eq!(
            backend.calls(),
            3,
            "loop should have drawn: rejected claim, edit, verified claim"
        );
        assert!(
            root_path.join("flag").exists(),
            "flag file should have been created by edit_file"
        );
        assert_eq!(stats.iterations, 3, "rejected claim, edit, verified claim");
    }

    #[tokio::test]
    async fn system_prompt_rendered_once_and_identical_every_turn() {
        // Three iterations: assert every turn's `system` string is identical
        // (byte-equal) — the prompt-cache correctness invariant. Also assert
        // it contains the check command display so the checks configuration
        // flows through into the prompt.
        let script: Vec<AssistantTurn> = (0..3)
            .map(|i| {
                turn_with(
                    vec![tool_call(
                        &format!("c{i}"),
                        "echo",
                        serde_json::json!({ "i": i }),
                    )],
                    StopReason::ToolUse,
                )
            })
            .collect();
        let backend = MockBackend::from_turns(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 3).with_checks(passing_runner());

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(backend.calls(), 3);

        let systems = backend.systems_seen();
        assert_eq!(systems.len(), 3, "one system entry per turn");
        let first = systems[0].as_ref().expect("system prompt was sent");
        assert!(
            first.contains("/bin/sh -c exit 0"),
            "system prompt must include the check command display; got:\n{first}"
        );
        for (i, entry) in systems.iter().enumerate() {
            let s = entry.as_ref().expect("system prompt was sent");
            assert_eq!(
                s.as_bytes(),
                first.as_bytes(),
                "turn {i} system prompt drifted from turn 0 — prompt cache invariant broken"
            );
        }
        assert_eq!(stats.iterations, 3);
    }

    #[tokio::test]
    async fn echo_then_finish_feeds_result_back_and_finishes() {
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "done after echo" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));
        // Two model turns: echo, then finish.
        assert_eq!(backend.calls(), 2, "echo turn then finish turn");
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn fed_back_message_is_user_tool_result_with_matching_call_id() {
        // Turn 1: an echo call. Turn 2: finish. By turn 2, the messages the
        // loop sent to the backend include the fed-back tool-result user
        // message — assert it is a Message::User carrying a UserBlock::ToolResult
        // whose call_id matches the echo request id.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo",
                    "echo",
                    serde_json::json!({ "k": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));

        // The most recent turn (finish) saw history ending in the fed-back
        // echo tool result.
        let seen = backend.last_messages();
        assert_last_is_tool_result(&seen, "call-echo");

        // And the fed-back content is echo's rendered result (its input as a
        // string), proving echo actually executed.
        match seen.last().expect("history non-empty") {
            Message::User { content } => match &content[0] {
                UserBlock::ToolResult {
                    content, is_error, ..
                } => {
                    assert!(!is_error);
                    assert_eq!(content, &serde_json::json!({ "k": 1 }).to_string());
                }
                UserBlock::Text(_) => panic!("expected ToolResult"),
            },
            Message::Assistant { .. } => panic!("expected User message"),
        }
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn never_finishing_script_hits_max_iterations() {
        // Three non-finishing (plain tool_use that isn't finish) turns; cap at 3
        // means the loop draws exactly three turns and stops at the cap.
        let non_finish = || {
            turn_with(
                vec![tool_call("c", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            )
        };
        let backend = MockBackend::from_turns(vec![non_finish(), non_finish(), non_finish()]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 3);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(backend.calls(), 3, "drew exactly max_iterations turns");
        assert_eq!(stats.iterations, 3);
    }

    #[tokio::test]
    async fn first_turn_error_surfaces_backend_error_without_retry() {
        let backend = MockBackend::new(vec![Err(BackendError::Terminal {
            kind: TerminalKind::Auth,
            message: "no creds".to_string(),
        })]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        // Surfaced on the first call — no retry.
        assert_eq!(backend.calls(), 1);
        // BackendError-on-first-turn: iterations = 1 (the erroring draw counts),
        // both token totals stay zero (errored turns contribute nothing).
        assert_eq!(stats.iterations, 1);
        assert_eq!(stats.input_tokens, 0);
        assert_eq!(stats.output_tokens, 0);
    }

    #[tokio::test]
    async fn plain_text_turn_stops_without_finish() {
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![ContentBlock::Text("I am just talking".to_string())],
            StopReason::EndTurn,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::StoppedWithoutFinish));
        assert_eq!(backend.calls(), 1);
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn overdrawn_mock_surfaces_backend_error() {
        // A non-finishing turn followed by an empty script: the loop draws a
        // second turn that over-draws the mock, surfacing a BackendError.
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![tool_call("c", "echo", serde_json::json!({}))],
            StopReason::ToolUse,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        assert_eq!(backend.calls(), 2, "second draw over-draws the script");
        // Both draws count: the successful echo turn AND the second-draw
        // over-draw error.
        assert_eq!(stats.iterations, 2);
    }

    #[tokio::test]
    async fn finish_blocked_parses_decision_needed() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({
                "disposition": "blocked",
                "summary": "ambiguous spec",
                "decision_needed": "which API version?"
            }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "which API version?");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn finish_failed_parses_summary() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "failed", "summary": "tool kept erroring" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { summary, .. }) => {
                assert_eq!(summary, "tool kept erroring");
            }
            other => panic!("expected Finished(Failed), got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[tokio::test]
    async fn finish_unknown_disposition_defaults_to_failed() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "weird", "summary": "huh" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { summary, .. }) => {
                assert_eq!(summary, "huh");
            }
            other => panic!("expected Finished(Failed) for unknown disposition, got {other:?}"),
        }
        assert_eq!(stats.iterations, 1);
    }

    #[test]
    fn render_tool_result_includes_bounded_detail() {
        let ctx = ToolCtx::stub();
        let with_detail = ToolResult::with_detail("summary line", "the detail body", &ctx);
        let rendered = render_tool_result(&with_detail);
        assert!(rendered.starts_with("summary line"));
        assert!(rendered.contains("the detail body"));

        let plain = ToolResult::ok("just a summary");
        assert_eq!(render_tool_result(&plain), "just a summary");
    }

    #[test]
    fn rejection_content_composes_header_excerpt_and_offload() {
        // Report with excerpt + offload path → all three pieces flow into
        // the content the model sees on rejection.
        let report = CheckReport {
            passed: false,
            exit_code: Some(1),
            timed_out: false,
            excerpt: "FAILURE OUTPUT".to_string(),
            offload_path: Some(PathBuf::from("/tmp/offload-0001.txt")),
            duration: Duration::from_millis(100),
        };
        let content = rejection_content(&report);
        assert!(
            content.starts_with("finish(done) rejected: verification failed"),
            "rejection header first; got:\n{content}"
        );
        assert!(content.contains("FAILURE OUTPUT"));
        assert!(content.contains("/tmp/offload-0001.txt"));

        // With no excerpt and no offload path, the header is the whole content.
        let bare = CheckReport {
            passed: false,
            exit_code: None,
            timed_out: true,
            excerpt: String::new(),
            offload_path: None,
            duration: Duration::from_secs(1),
        };
        let bare_content = rejection_content(&bare);
        assert_eq!(
            bare_content, "finish(done) rejected: verification failed",
            "bare report yields just the header"
        );
    }

    #[tokio::test]
    async fn finish_tool_metadata_and_run() {
        let tool = FinishTool;
        assert_eq!(tool.name(), FINISH_TOOL_NAME);
        let schema = tool.schema();
        assert_eq!(schema["name"], FINISH_TOOL_NAME);
        assert_eq!(schema["input_schema"]["type"], "object");

        let ctx = ToolCtx::stub();
        let result = tool
            .run(
                serde_json::json!({ "disposition": "done", "summary": "s" }),
                &ctx,
            )
            .await;
        assert!(!result.is_error);
        assert!(result.summary.contains("finish"));
    }

    #[tokio::test]
    async fn run_stats_accumulate_exact_per_turn_usage_across_a_scripted_run() {
        // Three scripted turns, each with a KNOWN, distinct `Usage`. After the
        // run finishes we assert `stats` sums those exact per-turn values.
        // - turn 1: echo call, usage(input=100, output=10)
        // - turn 2: echo call, usage(input=200, output=20)
        // - turn 3: finish(done), usage(input=300, output=30)
        // Expected: iterations = 3, input_tokens = 600, output_tokens = 60.
        let script = vec![
            turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                StopReason::ToolUse,
                usage_with(100, 10),
            ),
            turn_with_usage(
                vec![tool_call("c2", "echo", serde_json::json!({ "i": 2 }))],
                StopReason::ToolUse,
                usage_with(200, 20),
            ),
            turn_with_usage(
                vec![tool_call(
                    "c-finish",
                    FINISH_TOOL_NAME,
                    serde_json::json!({ "disposition": "done", "summary": "ok" }),
                )],
                StopReason::ToolUse,
                usage_with(300, 30),
            ),
        ];
        let backend = MockBackend::from_turns(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        ));
        assert_eq!(stats.iterations, 3);
        assert_eq!(stats.input_tokens, 600);
        assert_eq!(stats.output_tokens, 60);
    }

    #[tokio::test]
    async fn run_stats_errored_turn_contributes_no_tokens_but_counts_the_iteration() {
        // Two turns: (1) a successful echo with non-zero usage, (2) a terminal
        // backend error. `iterations` counts both draws (2); `input_tokens` /
        // `output_tokens` include ONLY the first (successful) turn — errored
        // turns contribute nothing.
        let script: Vec<Result<AssistantTurn, BackendError>> = vec![
            Ok(turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                StopReason::ToolUse,
                usage_with(500, 50),
            )),
            Err(BackendError::Terminal {
                kind: TerminalKind::Other,
                message: "second turn boom".to_string(),
            }),
        ];
        let backend = MockBackend::new(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        assert_eq!(
            stats.iterations, 2,
            "both draws count, including the erroring one"
        );
        assert_eq!(
            stats.input_tokens, 500,
            "only the successful turn's usage sums"
        );
        assert_eq!(stats.output_tokens, 50);
    }

    #[tokio::test]
    async fn run_stats_wall_clock_populated_across_the_run() {
        // The `wall_clock` field must always be populated (non-zero after any
        // real call) — pin the invariant that it's finalized on every return
        // path. A single finish turn is the smallest case that still measures
        // real elapsed time.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("t", 5);

        let RunResult { outcome: _, stats } = run(&backend, &tools, &ctx, &config).await;

        // Wall clock is monotonic Instant-based, so any real call produces a
        // duration strictly greater than zero.
        assert!(
            stats.wall_clock > Duration::ZERO,
            "wall_clock must be measured across the run; got {:?}",
            stats.wall_clock,
        );
    }

    #[test]
    fn run_stats_is_debug_clone_and_eq() {
        // RunStats derives Clone / Debug / PartialEq / Eq — a value can be
        // moved into a report, cloned into a log line, and compared exactly.
        // The task spec explicitly requires these traits.
        let a = RunStats {
            iterations: 3,
            input_tokens: 100,
            output_tokens: 10,
            wall_clock: Duration::from_millis(250),
        };
        let printed = format!("{a:?}");
        assert!(printed.contains("RunStats"));
        let b = a.clone();
        assert_eq!(a, b);
        let c = RunStats { iterations: 4, ..b };
        assert_ne!(a, c);
    }

    #[tokio::test]
    async fn finish_failed_disposition_yields_failed_with_loop_mode() {
        // A model that self-declares finish(failed) should yield
        // Disposition::Failed { mode: FailureMode::Loop, .. } — the pinned
        // default for "agent gave up / no productive progress".
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "failed", "summary": "gave up" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { mode, .. }) => {
                assert_eq!(
                    mode,
                    FailureMode::Loop,
                    "model-declared failed must yield FailureMode::Loop"
                );
            }
            other => panic!("expected Finished(Failed{{Loop}}), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn finish_missing_disposition_yields_failed_with_loop_mode() {
        // A malformed finish call (missing `disposition` field) is routed
        // through FinishClaim::Failed — verify it produces Failed{{Loop}}.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "summary": "no disposition field" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult { outcome, .. } = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { mode, .. }) => {
                assert_eq!(
                    mode,
                    FailureMode::Loop,
                    "malformed finish (missing disposition) must yield FailureMode::Loop"
                );
            }
            other => panic!("expected Finished(Failed{{Loop}}), got {other:?}"),
        }
    }

    #[test]
    fn into_disposition_maps_all_arms() {
        use crate::model::{BackendError, TerminalKind, TransientKind};

        // Finished(d) → d (pass-through)
        let d = Disposition::Done {
            summary: "ok".to_string(),
            verification: Verification::NoChecksConfigured,
        };
        let out = LoopOutcome::Finished(d.clone()).into_disposition();
        assert_eq!(out, d);

        // MaxIterations → Failed { mode: BudgetExhausted }
        let out = LoopOutcome::MaxIterations.into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::BudgetExhausted,
                    ..
                }
            ),
            "MaxIterations must map to BudgetExhausted; got {out:?}"
        );

        // StoppedWithoutFinish → Failed { mode: StoppedWithoutFinish }
        let out = LoopOutcome::StoppedWithoutFinish.into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::StoppedWithoutFinish,
                    ..
                }
            ),
            "StoppedWithoutFinish must map to FailureMode::StoppedWithoutFinish; got {out:?}"
        );

        // BackendError(Transient) → Failed { mode: TransientInfra }
        let out = LoopOutcome::BackendError(BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        })
        .into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::TransientInfra,
                    ..
                }
            ),
            "Transient BackendError must map to TransientInfra; got {out:?}"
        );

        // BackendError(Terminal) → Failed { mode: PersistentToolError }
        let out = LoopOutcome::BackendError(BackendError::Terminal {
            kind: TerminalKind::Auth,
            message: "no creds".to_string(),
        })
        .into_disposition();
        assert!(
            matches!(
                out,
                Disposition::Failed {
                    mode: FailureMode::PersistentToolError,
                    ..
                }
            ),
            "Terminal BackendError must map to PersistentToolError; got {out:?}"
        );
    }

    #[test]
    fn run_config_defaults_and_builders_compose() {
        // new() sets checks=None and max_tokens=DEFAULT_MAX_TOKENS.
        let config = RunConfig::new("do a thing", 7);
        assert_eq!(config.task, "do a thing");
        assert_eq!(config.max_iterations, 7);
        assert!(config.checks.is_none());
        assert_eq!(config.max_tokens, super::DEFAULT_MAX_TOKENS);

        // Builders layer on top.
        let with_checks = RunConfig::new("t", 1).with_checks(passing_runner());
        assert!(with_checks.checks.is_some());
        let with_mt = RunConfig::new("t", 1).with_max_tokens(1234);
        assert_eq!(with_mt.max_tokens, 1234);
    }

    // =====================================================================
    // Persistence tests (run identity + checkpoint wiring)
    // =====================================================================

    #[test]
    fn run_id_helper_formats_task_id_colon_attempt_n() {
        // Pinned vector: the join must be exactly "task-42:1", never a hash.
        assert_eq!(run_id("task-42", 1_u32), "task-42:1");
        // Also check the fixture we use throughout the persistence tests.
        assert_eq!(run_id("task-t", 1), FIXTURE_RID);
    }

    #[tokio::test]
    async fn run_persisted_produces_same_outcome_and_stats_as_run() {
        // No-store parity: run() and run_persisted() on the same scripted
        // trajectory should produce an identical LoopOutcome variant and
        // identical RunStats (except wall_clock, which is timing-dependent
        // and is not compared).
        let script = || {
            vec![
                turn_with_usage(
                    vec![tool_call("c1", "echo", serde_json::json!({ "i": 1 }))],
                    StopReason::ToolUse,
                    usage_with(10, 5),
                ),
                finish_call(
                    "cf",
                    serde_json::json!({ "disposition": "done", "summary": "ok" }),
                ),
            ]
        };

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);

        let RunResult {
            outcome: o1,
            stats: s1,
        } = run(&MockBackend::from_turns(script()), &tools, &ctx, &config).await;

        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store);
        let RunResult {
            outcome: o2,
            stats: s2,
        } = run_persisted(
            &MockBackend::from_turns(script()),
            &tools,
            &ctx,
            &config,
            &pers,
        )
        .await
        .expect("run_persisted must succeed");

        // LoopOutcome variant
        assert!(
            matches!(o1, LoopOutcome::Finished(Disposition::Done { .. })),
            "run() outcome: {o1:?}"
        );
        assert!(
            matches!(o2, LoopOutcome::Finished(Disposition::Done { .. })),
            "run_persisted() outcome: {o2:?}"
        );
        // RunStats (not wall_clock)
        assert_eq!(s1.iterations, s2.iterations, "iterations must match");
        assert_eq!(s1.input_tokens, s2.input_tokens, "input_tokens must match");
        assert_eq!(
            s1.output_tokens, s2.output_tokens,
            "output_tokens must match"
        );
    }

    #[tokio::test]
    async fn tool_call_events_recorded_before_and_after_invoke() {
        // A scripted echo turn followed by finish. The event log must contain
        // ToolCallStarted("echo") immediately before ToolCallResult("echo").
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "call-finish",
                serde_json::json!({ "disposition": "done", "summary": "done" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list events");

        // Find the ToolCallStarted and ToolCallResult indices.
        let started_idx = events
            .iter()
            .position(|e| matches!(e, Event::ToolCallStarted { name, .. } if name == "echo"))
            .expect("ToolCallStarted(echo) must be in event log");
        let result_idx = events
            .iter()
            .position(|e| matches!(e, Event::ToolCallResult { name, .. } if name == "echo"))
            .expect("ToolCallResult(echo) must be in event log");

        assert!(
            started_idx < result_idx,
            "ToolCallStarted must precede ToolCallResult; started={started_idx}, result={result_idx}"
        );
    }

    #[tokio::test]
    async fn tool_call_started_event_carries_matching_call_id() {
        // The ToolCallStarted.call_id must equal the ToolCallRequest.id from
        // the model response — this is what resume pairs against.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "my-unique-call-id",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list events");
        let started = events.iter().find_map(|e| {
            if let Event::ToolCallStarted { name, call_id, .. } = e
                && name == "echo"
            {
                return Some(call_id.clone());
            }
            None
        });
        assert_eq!(
            started.as_deref(),
            Some("my-unique-call-id"),
            "ToolCallStarted.call_id must equal the model's ToolCallRequest.id"
        );
    }

    #[tokio::test]
    async fn two_successful_turns_produce_exactly_two_model_call_and_budget_tick_events() {
        // [echo turn, finish turn] = 2 model draws → 2 ModelCall + 2 BudgetTick.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "done" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        let model_calls = events
            .iter()
            .filter(|e| matches!(e, Event::ModelCall { .. }))
            .count();
        let budget_ticks = events
            .iter()
            .filter(|e| matches!(e, Event::BudgetTick { .. }))
            .count();

        assert_eq!(
            model_calls, 2,
            "expected 2 ModelCall events, got {model_calls}"
        );
        assert_eq!(
            budget_ticks, 2,
            "expected 2 BudgetTick events, got {budget_ticks}"
        );
    }

    #[tokio::test]
    async fn loaded_checkpoint_has_correct_run_id_schema_version_phase_and_run_checks() {
        // Verifies RunRecord construction: run_id format, schema_version == 2,
        // phase == InnerLoop, and run_checks populated from the ChecksRunner.
        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let runner = passing_runner();
        let runner_display = runner.command_display();
        let config = RunConfig::new("do the task", 5).with_checks(runner);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = Persistence {
            store: store.clone(),
            task_id: "task-42".to_string(),
            attempt_n: 1,
            model_label: "test-model".to_string(),
        };

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let rid = run_id("task-42", 1);
        let rec = store.load(&rid).await.expect("load").expect("present");

        assert_eq!(rec.run_id, "task-42:1", "run_id must be task_id:attempt_n");
        assert_eq!(
            rec.schema_version,
            crate::run_record::SCHEMA_VERSION,
            "schema_version must be SCHEMA_VERSION (= 2)"
        );
        assert_eq!(
            rec.phase,
            crate::run_record::Phase::InnerLoop,
            "phase must be InnerLoop"
        );
        assert_eq!(
            rec.project_config
                .run_checks
                .get("checks")
                .map(String::as_str),
            Some(runner_display.as_str()),
            "run_checks['checks'] must equal runner.command_display()"
        );
    }

    #[tokio::test]
    async fn finish_done_produces_disposition_done_in_persisted_record() {
        // A finish(done) terminal path must write a checkpoint whose
        // disposition is Some(Disposition::Done{..}) and emit a DispositionSet
        // event.
        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "task complete" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let rec = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert!(
            matches!(rec.disposition, Some(Disposition::Done { .. })),
            "disposition must be Done; got {:?}",
            rec.disposition
        );

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        let has_disposition_set = events
            .iter()
            .any(|e| matches!(e, Event::DispositionSet { .. }));
        assert!(has_disposition_set, "DispositionSet event must be in log");
    }

    #[tokio::test]
    async fn stopped_without_finish_produces_failed_stopped_without_finish_disposition() {
        // A plain-text turn (no tool calls) stops as StoppedWithoutFinish.
        // The persisted disposition must be Failed{mode: StoppedWithoutFinish}.
        let backend = MockBackend::from_turns(vec![turn_with(
            vec![ContentBlock::Text("I am just talking".to_string())],
            StopReason::EndTurn,
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "outcome must be StoppedWithoutFinish; got {outcome:?}"
        );

        let rec = store
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert!(
            matches!(
                rec.disposition,
                Some(Disposition::Failed {
                    mode: FailureMode::StoppedWithoutFinish,
                    ..
                })
            ),
            "persisted disposition must be Failed{{StoppedWithoutFinish}}; got {:?}",
            rec.disposition
        );
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn log_then_snapshot_ordering_and_two_turn_iteration_count() {
        // [echo turn, finish turn] = 2 model draws.
        // Asserts:
        //   1. The terminal DispositionSet event is recorded before the last
        //      checkpoint (log-then-snapshot discipline).
        //   2. No checkpoint is recorded before the first AppendEvent
        //      (the very first call must be an event, not a checkpoint).
        //   3. The loaded record's budgets.consumed.iterations == 2.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "done" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let recording = Arc::new(RecordingStore::new());
        let pers = make_persistence(recording.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let calls = recording.recorded_calls();
        assert!(!calls.is_empty(), "should have recorded some store calls");

        // (2) First call must be AppendEvent.
        assert!(
            matches!(calls[0], StoreCall::AppendEvent { .. }),
            "first call must be AppendEvent, not Checkpoint; got {:?}",
            calls[0]
        );

        // (1) Last DispositionSet must precede last Checkpoint.
        let last_ds = calls
            .iter()
            .rposition(|c| matches!(c, StoreCall::AppendEvent { kind } if kind == "DispositionSet"))
            .expect("DispositionSet event must have been appended");
        let last_ckpt = calls
            .iter()
            .rposition(|c| matches!(c, StoreCall::Checkpoint))
            .expect("at least one Checkpoint must have been written");
        assert!(
            last_ds < last_ckpt,
            "DispositionSet (idx {last_ds}) must precede the terminal Checkpoint (idx {last_ckpt})"
        );

        // (3) Final checkpoint has iterations == 2.
        let rec = recording
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(
            rec.budgets.consumed.iterations, 2,
            "loaded record must have iterations == 2; got {:?}",
            rec.budgets.consumed
        );
    }

    #[tokio::test]
    async fn mid_turn_checkpoint_contains_assistant_message_before_tool_execution() {
        // LEAD ADDITION: the mid-iteration checkpoint (written after the
        // assistant turn is appended to messages but BEFORE any tools.invoke)
        // must have the assistant turn as its last message. Specifically, that
        // assistant message must carry the ToolCallRequest with the started
        // call's id.
        //
        // Script: [echo call], [finish]. The first checkpoint (snapshot 0) is
        // the mid-iteration checkpoint of iteration 1.
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call(
                    "call-echo-123",
                    "echo",
                    serde_json::json!({ "x": 1 }),
                )],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let snapshots = snap_store.all_snapshots();
        assert!(
            !snapshots.is_empty(),
            "at least one checkpoint must have been written"
        );

        // Snapshot 0 is the mid-iteration checkpoint of iteration 1:
        // messages = [task_seed_user, assistant_turn_with_echo_call]
        let first = &snapshots[0];
        let last_msg = first.messages.last().expect("messages must be non-empty");

        match last_msg {
            Message::Assistant { content } => {
                let has_echo_call = content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::ToolCall(ToolCallRequest { id, name, .. })
                        if id == "call-echo-123" && name == "echo"
                    )
                });
                assert!(
                    has_echo_call,
                    "first checkpoint's last message must be the assistant turn \
                     carrying call_id 'call-echo-123'; got content: {content:?}"
                );
            }
            other @ Message::User { .. } => {
                panic!("first checkpoint's last message must be Message::Assistant; got {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn failing_store_causes_run_persisted_to_return_err() {
        // The first store error (on append_event) must abort run_persisted
        // and propagate as Err(StoreError).
        let backend = MockBackend::from_turns(vec![
            turn_with(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let pers = make_persistence(Arc::new(FailingStore));

        let result = run_persisted(&backend, &tools, &ctx, &config, &pers).await;
        assert!(
            result.is_err(),
            "run_persisted with a failing store must return Err"
        );
    }

    #[tokio::test]
    async fn budget_consumed_in_record_reflects_current_iteration() {
        // The BudgetTick event and end-of-iteration checkpoint must carry
        // budgets.consumed matching the loop's current running totals.
        let backend = MockBackend::from_turns(vec![
            turn_with_usage(
                vec![tool_call("c1", "echo", serde_json::json!({}))],
                StopReason::ToolUse,
                usage_with(100, 50),
            ),
            finish_call(
                "cf",
                serde_json::json!({ "disposition": "done", "summary": "ok" }),
            ),
        ]);

        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 10);
        let store = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let pers = make_persistence(store.clone());

        run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        let events = store.list_events(FIXTURE_RID).await.expect("list");
        // Second BudgetTick (iter 2, the finish turn) should have
        // accumulated tokens from BOTH turns.
        let budget_ticks: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let Event::BudgetTick { consumed, .. } = e {
                    Some(consumed)
                } else {
                    None
                }
            })
            .collect();

        // iter 1 BudgetTick: iterations=1, tokens=100+50=150
        let BudgetConsumed {
            iterations: i1,
            tokens: t1,
            ..
        } = budget_ticks[0];
        assert_eq!(*i1, 1, "first BudgetTick iterations must be 1");
        assert_eq!(*t1, 150, "first BudgetTick tokens must be 150 (100+50)");
    }
}
