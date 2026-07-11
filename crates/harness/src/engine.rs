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
//! wall-clock bounds, loop / no-progress detection, persistence /
//! checkpointing, and context assembly / compaction. The only stopping
//! condition beyond the agent finishing is the hard `max_iterations` cap.
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
//! A retryable backend error ([`model::BackendError::Transient`]) is retried
//! up to [`RunConfig::max_retries`] additional times with deterministic
//! exponential backoff before surfacing [`LoopOutcome::BackendError`]. A
//! non-retryable error ([`model::BackendError::Terminal`],
//! [`model::BackendError::Protocol`],
//! [`model::BackendError::ContextLengthExceeded`]) is surfaced on first
//! occurrence.
//!
//! [`ChecksRunner`]: crate::exec::ChecksRunner
//! [`TurnRequest`]: crate::model::TurnRequest

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::time::sleep;

use crate::exec::{CheckReport, ChecksRunner};
use crate::model::{self, Message, SamplingParams, TurnRequest, UserBlock};
use crate::prompt;
use crate::run_record::{
    BudgetConsumed, BudgetLimits, Budgets, Disposition, DurableFacts, Event, FailureMode, Phase,
    ProjectConfig, RecoveryFacts, RunRecord, SCHEMA_VERSION, Task, Verification,
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
    /// Static-tree threshold K for finish-recovery: how many consecutive
    /// non-mutating iterations must accumulate AFTER a green `run_checks`
    /// before the harness considers the run "done-but-unclaimed" and injects
    /// a nudge. A successful `edit_file`/`bash` resets the counter. See
    /// [`DEFAULT_STATIC_TREE_K`].
    pub static_tree_k: u32,
    /// Maximum nudges the harness will inject before taking the recovery
    /// terminal ([`FailureMode::FinishDiscipline`]). `0` DISABLES
    /// finish-recovery entirely — no nudge is ever injected and the recovery
    /// terminal is never taken. See [`DEFAULT_MAX_NUDGES`].
    pub max_nudges: u32,
    /// Number of ADDITIONAL attempts after the first try when a
    /// [`model::BackendError::Transient`] failure is returned. With the
    /// default 3, one logical turn calls `backend.turn` at most `1 + 3 = 4`
    /// times. Set to `0` to disable retries.
    ///
    /// **Panic-safety bound:** the exponential schedule
    /// (`retry_backoff_base * 2^attempt`) is panic-safe only for small values;
    /// the default (max exponent 2) is safe. A caller configuring a very large
    /// `max_retries` owns the `Duration`-multiply overflow — saturating math is
    /// deferred (YAGNI at the pinned default).
    ///
    /// See [`DEFAULT_MAX_RETRIES`].
    pub max_retries: u32,
    /// Base delay for the deterministic exponential backoff schedule. The
    /// delay before retry attempt `i` (0-indexed) is `retry_backoff_base *
    /// 2^i`. Set to [`Duration::ZERO`] in tests to run retries with no sleep.
    ///
    /// See [`DEFAULT_RETRY_BACKOFF_BASE`].
    pub retry_backoff_base: Duration,
}

/// The default per-turn output cap. Budget-aware sizing lands with the budget
/// work; this is a fixed value for the thin slice.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Default retry cap: how many ADDITIONAL attempts are made after the first
/// try on a retryable [`model::BackendError::Transient`] failure. With the
/// default of 3, one logical turn calls `backend.turn` at most 4 times before
/// giving up. The exponential backoff schedule (`base * 2^attempt`) is
/// panic-safe for small values; at the pinned default the max exponent is 2
/// (0.5 s / 1 s / 2 s). A caller configuring a very large `max_retries` owns
/// the `Duration`-multiply overflow (saturating math is deferred — YAGNI at
/// the pinned default).
pub const DEFAULT_MAX_RETRIES: u32 = 3;

/// Default base delay for the exponential backoff schedule. The delay before
/// attempt `i` (0-indexed) is `DEFAULT_RETRY_BACKOFF_BASE * 2^i`:
/// 500 ms, 1 000 ms, 2 000 ms for attempts 0/1/2 (with `max_retries = 3`).
pub const DEFAULT_RETRY_BACKOFF_BASE: Duration = Duration::from_millis(500);

/// Default static-tree threshold K — a starting guess, to be tuned against
/// 0.4.0 run data. After K consecutive non-mutating iterations with a green
/// gate, the harness considers the run probably-done-but-unclaimed.
pub const DEFAULT_STATIC_TREE_K: u32 = 3;

/// Default nudge cap N — a starting guess, to be tuned against 0.4.0 run
/// data. After N nudges the harness force-terminates via
/// [`FailureMode::FinishDiscipline`]. Setting `max_nudges == 0` disables
/// finish-recovery entirely.
pub const DEFAULT_MAX_NUDGES: u32 = 2;

impl RunConfig {
    /// Build a config with the given `task` and iteration cap. Defaults
    /// `checks` to `None`, `max_tokens` to [`DEFAULT_MAX_TOKENS`],
    /// `max_retries` to [`DEFAULT_MAX_RETRIES`], and `retry_backoff_base` to
    /// [`DEFAULT_RETRY_BACKOFF_BASE`].
    #[must_use]
    pub fn new(task: impl Into<String>, max_iterations: u32) -> Self {
        Self {
            task: task.into(),
            max_iterations,
            checks: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            static_tree_k: DEFAULT_STATIC_TREE_K,
            max_nudges: DEFAULT_MAX_NUDGES,
            max_retries: DEFAULT_MAX_RETRIES,
            retry_backoff_base: DEFAULT_RETRY_BACKOFF_BASE,
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

    /// Override the static-tree threshold K ([`DEFAULT_STATIC_TREE_K`] by
    /// default). See [`RunConfig::static_tree_k`].
    #[must_use]
    pub fn with_static_tree_k(mut self, static_tree_k: u32) -> Self {
        self.static_tree_k = static_tree_k;
        self
    }

    /// Override the nudge cap N ([`DEFAULT_MAX_NUDGES`] by default). `0`
    /// disables finish-recovery entirely. See [`RunConfig::max_nudges`].
    #[must_use]
    pub fn with_max_nudges(mut self, max_nudges: u32) -> Self {
        self.max_nudges = max_nudges;
        self
    }

    /// Override the retry cap ([`DEFAULT_MAX_RETRIES`] by default). `0`
    /// disables retries. See [`RunConfig::max_retries`].
    #[must_use]
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    /// Override the backoff base delay ([`DEFAULT_RETRY_BACKOFF_BASE`] by
    /// default). Set to [`Duration::ZERO`] in tests to skip the sleep. See
    /// [`RunConfig::retry_backoff_base`].
    #[must_use]
    pub fn with_retry_backoff_base(mut self, base: Duration) -> Self {
        self.retry_backoff_base = base;
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

/// Mode of resume — how to reconstruct context on re-entry after an
/// interruption.
///
/// This is a **runtime call argument**, never persisted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResumeMode {
    /// **D6 — Crash**: reload `messages` from the last checkpoint and
    /// reconcile any dangling [`Event::ToolCallStarted`] in the log tail.
    /// For each interrupted (or never-started) call, a synthetic
    /// `is_error=true` [`UserBlock::ToolResult`] with content
    /// `"interrupted by host restart"` is fed back — the tool is **never
    /// re-executed** (side effects may have already happened). Continues
    /// checkpointing under the same `run_id`.
    Crash,
    /// **D7**: drop the reloaded `messages` and restart from a freshly-rendered
    /// task seed (byte-identical to what [`run`] would produce). Carries
    /// `phase`, `durable_facts`, and `budgets.consumed` forward. Checkpoints
    /// under a new `run_id` = `"{task_id}:{attempt_n+1}"`, leaving the prior
    /// record intact.
    FreshContext,
}

/// Errors surfaced by [`resume`].
#[derive(Debug)]
pub enum ResumeError {
    /// No checkpoint exists for the requested `run_id`. [`resume`] returns
    /// this immediately — no [`model::ModelBackend::turn`] call is made.
    UnknownRunId(String),
    /// A store load, append, or checkpoint operation failed.
    Store(StoreError),
}

impl std::fmt::Display for ResumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRunId(id) => write!(f, "no checkpoint found for run_id {id:?}"),
            Self::Store(e) => write!(f, "store error: {e}"),
        }
    }
}

impl std::error::Error for ResumeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnknownRunId(_) => None,
            Self::Store(e) => Some(e),
        }
    }
}

impl From<StoreError> for ResumeError {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
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
/// - `iterations`: how many **logical** iterations the loop executed — one per
///   `for`-loop pass. A turn that fails transiently and is retried within the
///   same pass still counts as **one** logical iteration; retry draws within a
///   single pass are NOT counted separately. A
///   [`LoopOutcome::BackendError`] on the FIRST turn yields `iterations = 1`.
///   [`LoopOutcome::MaxIterations`] always yields
///   `iterations == config.max_iterations`. Raw `backend.turn` draw count
///   (including retries within a pass) is observable in tests via
///   [`crate::test_support::MockBackend::calls`].
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
    /// Logical iterations executed — one per `for`-loop pass. Retry draws
    /// within one pass count once here; use `MockBackend::calls()` in tests
    /// to observe the raw `backend.turn` draw count including retries.
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
    /// The backend returned an error that exhausted the retry budget. Carries
    /// the **last** attempt's error. Retryable errors
    /// ([`model::BackendError::Transient`]) are retried up to
    /// [`RunConfig::max_retries`] additional times with deterministic
    /// exponential backoff before reaching this variant. Non-retryable errors
    /// ([`model::BackendError::Terminal`], [`model::BackendError::Protocol`],
    /// [`model::BackendError::ContextLengthExceeded`]) reach this variant on
    /// first occurrence.
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
    let task_message = prompt::render_task_prompt(&config.task);
    let initial_messages = vec![Message::User {
        content: vec![UserBlock::Text(task_message)],
    }];
    // No persistence — the Result::Err path is structurally unreachable when
    // persistence is None (no store calls are made), so the expect is a
    // compile-time invariant, not a runtime safety net.
    let outcome = run_loop_impl(
        backend,
        tools,
        ctx,
        config,
        None,
        &mut stats,
        initial_messages,
        BudgetConsumed::default(),
        None,
    )
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
    let task_message = prompt::render_task_prompt(&config.task);
    let initial_messages = vec![Message::User {
        content: vec![UserBlock::Text(task_message)],
    }];
    let outcome = run_loop_impl(
        backend,
        tools,
        ctx,
        config,
        Some(persistence),
        &mut stats,
        initial_messages,
        BudgetConsumed::default(),
        None,
    )
    .await?;
    stats.wall_clock = start.elapsed();
    Ok(RunResult { outcome, stats })
}

/// The engine loop body, shared by [`run`], [`run_persisted`], and [`resume`].
///
/// `stats` is mutated in place as the loop progresses:
/// - `iterations` is incremented once per `for`-loop iteration, BEFORE the
///   first (possibly-retried) `backend.turn` call of that iteration. A turn
///   that fails transiently and is retried within the same pass still counts
///   as ONE logical iteration. A backend error on the first iteration yields
///   `iterations = 1`.
/// - `input_tokens` / `output_tokens` accumulate the SUCCESSFUL turn's
///   [`crate::model::Usage`] only; errored turns contribute nothing.
///
/// `initial_messages` is the starting conversation history (task seed for
/// fresh runs; reloaded/reconciled messages for crash-resume; fresh task seed
/// for fresh-context resume).
///
/// `initial_consumed` offsets all `budgets.consumed` computations — zero for
/// fresh runs; the loaded record's consumed for resume (budget carry-over,
/// accounting-only in 0.3.0).
///
/// `override_persist` when `Some` bypasses the record-construction step and
/// uses the provided [`RunPersist`] directly — used by [`resume`] to inject
/// a pre-loaded (and possibly reconciled) record.
///
/// When `persistence` is `Some`, the loop appends events and writes
/// checkpoints per the durability contract documented on [`run_persisted`].
/// When `persistence` is `None`, no store calls are made and the function
/// returns `Ok(outcome)` (the `Err` arm is structurally unreachable).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
async fn run_loop_impl(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    persistence: Option<&Persistence>,
    stats: &mut RunStats,
    initial_messages: Vec<Message>,
    initial_consumed: BudgetConsumed,
    override_persist: Option<RunPersist>,
) -> Result<LoopOutcome, StoreError> {
    // Render the system prompt ONCE and reuse verbatim every iteration —
    // the prompt-cache correctness invariant (D9).
    let system = prompt::render_system_prompt(
        &prompt::tool_lines(tools),
        config
            .checks
            .as_ref()
            .map(ChecksRunner::command_display)
            .as_deref(),
    );

    let mut messages = initial_messages;
    let tool_schemas = tools.list();
    let params = SamplingParams {
        max_tokens: config.max_tokens,
        temperature: None,
        stop_sequences: Vec::new(),
    };

    // Build the run-record if persistence is configured. This is kept as
    // Option<RunPersist> so the non-persistent path has zero overhead.
    // When override_persist is Some (resume path), use it directly — the
    // caller (resume) already loaded/constructed the record.
    let mut persist: Option<RunPersist> = if let Some(pre) = override_persist {
        Some(pre)
    } else if let Some(p) = persistence {
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
            recovery_facts: None,
            messages: messages.clone(),
        };
        Some(RunPersist { rid, record })
    } else {
        None
    };

    // ---- finish-recovery detection state (loop-local) ----
    // The done-oracle is `last_gate_green`, driven ONLY by `run_checks`'s
    // `is_error` flag (never a model self-report). A successful mutating tool
    // call (`edit_file`/`bash` with `!is_error`) invalidates the green — this
    // closes the stale-green false-trip window so a nudge's "gates are
    // currently green" is always true at trip time. `iters_since_tree_change`
    // counts consecutive non-mutating iterations; reset wins over increment
    // when a mutation and the per-iteration tick collide. `nudges_fired`
    // bounds how many times the harness will nudge before force-terminating;
    // `max_nudges == 0` disables the feature entirely (no nudge is ever
    // injected and the recovery terminal is never taken). `nudge_awaiting_status`
    // gates telemetry capture of the assistant reply text from the turn that
    // follows a nudge without producing an accepted `finish(done)`.
    let mut last_gate_green: bool = false;
    let mut iters_since_tree_change: u32 = 0;
    let mut tree_dirty: bool = false;
    let mut nudges_fired: u32 = 0;
    let mut nudge_awaiting_status: bool = false;
    let mut nudge_statuses: Vec<String> = Vec::new();

    for _ in 0..config.max_iterations {
        let req = TurnRequest {
            system: Some(&system),
            messages: &messages,
            tools: &tool_schemas,
            params: &params,
        };

        // Count the logical iteration BEFORE the retry loop so an error on
        // the first iteration still shows `iterations = 1`. A transient error
        // retried within the same for-loop pass counts as ONE logical
        // iteration — `stats.iterations` is NOT re-incremented per retry.
        stats.iterations += 1;
        let mut attempt = 0u32;
        let turn_result = loop {
            match backend.turn(&req).await {
                Ok(turn) => break Ok(turn),
                Err(err) if err.is_retryable() && attempt < config.max_retries => {
                    sleep(retry_delay(config.retry_backoff_base, attempt)).await;
                    attempt += 1;
                }
                Err(err) => break Err(err),
            }
        };
        let turn = match turn_result {
            Ok(turn) => turn,
            Err(err) => {
                // Terminal path: BackendError (retries exhausted or
                // non-retryable). Persist the disposition before returning —
                // this path exits early, bypassing the normal end-of-iteration
                // checkpoint. `err` is the LAST attempt's error.
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
                        iterations: initial_consumed.iterations + stats.iterations,
                        tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                        cost_micros: initial_consumed.cost_micros,
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
        // Capture the assistant reply text BEFORE `Message::from(turn)`
        // consumes the turn. `AssistantTurn::text` concatenates every
        // `ContentBlock::Text` in order (skipping Reasoning/ToolCall) — used
        // for nudge-status telemetry if this turn follows a nudge without
        // producing an accepted finish(done).
        let turn_text = turn.text();
        messages.push(Message::from(turn));

        // Append ModelCall + BudgetTick events, then write the mid-iteration
        // checkpoint (after assistant turn, BEFORE any tools.invoke). This
        // guarantees that a mid-iteration crash always leaves a snapshot whose
        // messages include the in-flight assistant turn.
        if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
            let consumed = BudgetConsumed {
                iterations: initial_consumed.iterations + stats.iterations,
                tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                cost_micros: initial_consumed.cost_micros,
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
        // Per-iteration mutation flag — reset before the per-call loop. A
        // successful `edit_file`/`bash` latches it (driving the end-of-iteration
        // tree-counter reset) AND clears `last_gate_green`.
        let mut mutated_this_iter: bool = false;
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

                // Observe the actual tool ToolResult for finish-recovery
                // detection — the done-oracle is `run_checks`'s `is_error`
                // flag (the run_checks tool sets is_error = !report.passed),
                // NEVER a model self-report. A successful mutating tool call
                // (`edit_file`/`bash` with `!is_error`) latches `tree_dirty`,
                // marks `mutated_this_iter`, and CLEARS `last_gate_green` — a
                // mutation after a green check invalidates the green, closing
                // the stale-green false-trip window so the nudge's "gates are
                // currently green" is always true at trip time.
                let is_error = result.is_error;
                if call.name == "run_checks" {
                    last_gate_green = !is_error;
                } else if (call.name == "edit_file" || call.name == "bash") && !is_error {
                    mutated_this_iter = true;
                    tree_dirty = true;
                    last_gate_green = false;
                }
            }
        }
        messages.push(Message::User { content: results });

        // Nudge-status telemetry: this turn followed a nudge iff
        // `nudge_awaiting_status` was set. If the turn produced an accepted
        // `finish(done)` the loop terminates normally below — clear the flag
        // without pushing (a Done-after-nudge stays a clean success; its
        // nudge_statuses are recoverable from the event log/messages if later
        // wanted). Otherwise push the (possibly empty for a tool-calls-only
        // turn) captured text and clear.
        if nudge_awaiting_status {
            let is_done = matches!(finish, Some(Disposition::Done { .. }));
            if !is_done {
                nudge_statuses.push(turn_text);
            }
            nudge_awaiting_status = false;
        }

        if let Some(disposition) = finish {
            // Terminal path: Finished. Write DispositionSet + terminal checkpoint.
            if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                ctx.record.messages.clone_from(&messages);
                ctx.record.budgets.consumed = BudgetConsumed {
                    iterations: initial_consumed.iterations + stats.iterations,
                    tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                    cost_micros: initial_consumed.cost_micros,
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

        // End-of-iteration tree-counter update — reached only when `finish`
        // was None. Reset wins over increment when a mutation and the
        // per-iteration tick collide.
        if mutated_this_iter {
            iters_since_tree_change = 0;
        } else {
            iters_since_tree_change += 1;
        }

        // Detection / high-precision trip — evaluated after the counter
        // update, only when finish-recovery is enabled (`max_nudges > 0`).
        // A RED gate (`last_gate_green == false`) MUST NOT trip — that case
        // falls through unchanged to the existing MaxIterations cap.
        if config.max_nudges > 0
            && last_gate_green
            && iters_since_tree_change >= config.static_tree_k
        {
            if nudges_fired < config.max_nudges {
                // Inject the nudge by APPENDING a `UserBlock::Text` onto the
                // content vec of the EXISTING tool-results `Message::User`
                // (the `results` batch just pushed above) — NOT a new
                // `Message::User`. `anthropic::map_message` maps each
                // `Message` 1:1 with NO same-role merge, so two adjacent
                // `Message::User` reach the wire as two `role:"user"` blocks
                // and 400. Appending keeps the conversation a single user
                // turn from the API's view.
                let nudge_text = prompt::render_nudge_prompt();
                if let Some(Message::User { content }) = messages.last_mut() {
                    content.push(UserBlock::Text(nudge_text));
                }
                nudges_fired += 1;
                nudge_awaiting_status = true;
                // Reset so K static iterations must re-accumulate before the
                // next trip.
                iters_since_tree_change = 0;
            } else {
                // Recovery terminal: gates green but agent did not call
                // `finish` after `max_nudges` nudges. Mirror the MaxIterations
                // block's persistence discipline (messages, budgets.consumed,
                // disposition, DispositionSet, checkpoint) AND write
                // `recovery_facts`. The harness NEVER constructs Done here —
                // claim-vs-verify is preserved; the loop is the sole Done
                // constructor (only after a green Verification::Checks via
                // handle_finish_call).
                let summary = format!(
                    "gates green but agent did not call finish after {} nudges",
                    config.max_nudges
                );
                let disposition = Disposition::Failed {
                    mode: FailureMode::FinishDiscipline,
                    summary,
                };
                if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
                    ctx.record.messages.clone_from(&messages);
                    ctx.record.budgets.consumed = BudgetConsumed {
                        iterations: initial_consumed.iterations + stats.iterations,
                        tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                        cost_micros: initial_consumed.cost_micros,
                    };
                    ctx.record.recovery_facts = Some(RecoveryFacts {
                        gates_green_at_exit: last_gate_green,
                        tree_dirty,
                        nudge_statuses: nudge_statuses.clone(),
                    });
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
                // When persistence is None (the `run` path), the facts are
                // computed but there is no record to write (acceptable — `run`
                // returns only LoopOutcome/RunStats). The returned
                // `Finished(Failed{FinishDiscipline})` still distinguishes the
                // recovery terminal from a model finish(failed) by its `mode`.
                return Ok(LoopOutcome::Finished(disposition));
            }
        }

        // Non-terminal end of iteration: write the end-of-iteration checkpoint
        // so a crash here loses at most the current iteration's tool results
        // (already in messages).
        if let (Some(ctx), Some(p)) = (persist.as_mut(), persistence) {
            ctx.record.messages.clone_from(&messages);
            ctx.record.budgets.consumed = BudgetConsumed {
                iterations: initial_consumed.iterations + stats.iterations,
                tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
                cost_micros: initial_consumed.cost_micros,
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
            iterations: initial_consumed.iterations + stats.iterations,
            tokens: initial_consumed.tokens + stats.input_tokens + stats.output_tokens,
            cost_micros: initial_consumed.cost_micros,
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

// =============================================================================
// Crash-resume helpers
// =============================================================================

/// Reconcile a potential dangling log tail for [`ResumeMode::Crash`].
///
/// Loads all events for `run_id`. If the last event is an
/// [`Event::ToolCallStarted`] (the only reliable "interrupted mid-execution"
/// signal — D5 appends Started then Result serially), this is a dangling tail:
/// the call was started but its Result was never recorded.
///
/// Returns the reconstructed initial message history:
///
/// - **Clean tail** (last event is not `ToolCallStarted`): returns
///   `record.messages` verbatim.
/// - **Dangling tail**: returns `record.messages` plus a synthetic
///   `Message::User` covering EVERY tool call in the last assistant turn.
///   Pairing is by `call_id` (D6): each [`Event::ToolCallStarted`]'s `call_id`
///   is matched against the [`model::ToolCallRequest`] ids in the snapshot's
///   in-flight assistant turn. Calls with a real logged
///   [`Event::ToolCallResult`] keep that real `{is_error, summary}`; the
///   interrupted call (and any that were never started) get a synthetic
///   `is_error=true` result with content `"interrupted by host restart"`.
///   Non-tool events between [`Event::ModelCall`] and the first
///   [`Event::ToolCallStarted`] (e.g. [`Event::BudgetTick`]) are filtered out
///   before the walk. Also appends one synthetic [`Event::ToolCallResult`] to
///   the log for the dangling Started, so the log is clean on the next resume.
async fn reconcile_crash_tail(
    record: &RunRecord,
    store: &Arc<dyn RunStore>,
    run_id: &str,
) -> Result<Vec<Message>, StoreError> {
    // Gate on SNAPSHOT shape, not log-tail shape: reconciliation is needed
    // exactly when the reloaded snapshot ends mid-turn — its last message is
    // an Assistant turn carrying tool calls whose results never made it into
    // `messages`. (After a clean end-of-iteration checkpoint the last message
    // is the User results batch; a stopped-without-finish turn carries no
    // tool calls.) Gating on `events.last()` being a `ToolCallStarted` is
    // WRONG: a crash between a `ToolCallResult` and the next `Started` — or
    // after the final result but before the end-of-iteration checkpoint —
    // leaves the log ending in a Result while the snapshot still dangles,
    // and an un-reconciled transcript ending in tool calls is a malformed
    // request on every backend.
    let last_assistant_calls: Vec<model::ToolCallRequest> = match record.messages.last() {
        Some(Message::Assistant { content }) => content
            .iter()
            .filter_map(|b| match b {
                model::ContentBlock::ToolCall(req) => Some(req.clone()),
                model::ContentBlock::Text(_) | model::ContentBlock::Reasoning { .. } => None,
            })
            .collect(),
        _ => return Ok(record.messages.clone()), // clean: ends with results batch
    };
    if last_assistant_calls.is_empty() {
        return Ok(record.messages.clone()); // assistant turn with no tool calls
    }

    let events = store.list_events(run_id).await?;

    // Isolate events after the last ModelCall (the current iteration's tail),
    // then filter to tool events only — ToolCallStarted and ToolCallResult.
    // This skips BudgetTick and any other non-tool events the engine emits
    // between ModelCall and the first ToolCallStarted (e.g. the BudgetTick
    // appended at engine.rs:895 before tool dispatch begins).
    let post_model_idx = events
        .iter()
        .rposition(|e| matches!(e, Event::ModelCall { .. }))
        .map_or(0, |i| i + 1);
    let tool_tail: Vec<&Event> = events[post_model_idx..]
        .iter()
        .filter(|e| {
            matches!(
                e,
                Event::ToolCallStarted { .. } | Event::ToolCallResult { .. }
            )
        })
        .collect();

    // Build a map: call_id → Option<(is_error, summary)>.
    //   None  = ToolCallStarted seen, no matching Result yet (dangling).
    //   Some  = both Started + Result seen (call completed).
    //
    // ToolCallResult carries no call_id field; pair it positionally with the
    // most-recently-unmatched ToolCallStarted (pending_cid). Malformed-tail
    // safety: an orphaned Result (no unmatched Started) is silently ignored;
    // a duplicate call_id in Started leaves the earlier entry unchanged.
    let mut completion_map: HashMap<String, Option<(bool, String)>> = HashMap::new();
    let mut started_names: Vec<(String, String)> = Vec::new(); // (call_id, name)
    let mut pending_cid: Option<String> = None;
    for event in &tool_tail {
        match event {
            Event::ToolCallStarted { call_id, name, .. } => {
                // If a previous Started is still unmatched, leave it as None
                // in the map. Update the pending slot. Record the name on
                // first sighting only (duplicate call_ids degrade safely).
                pending_cid = Some(call_id.clone());
                if !completion_map.contains_key(call_id.as_str()) {
                    started_names.push((call_id.clone(), name.clone()));
                }
                completion_map.entry(call_id.clone()).or_insert(None);
            }
            Event::ToolCallResult {
                is_error, summary, ..
            } => {
                if let Some(cid) = pending_cid.take() {
                    // Pair this Result to the most-recently-unmatched Started.
                    completion_map.insert(cid, Some((*is_error, summary.clone())));
                }
                // Orphaned Result (no unmatched Started): degrade safely, ignore.
            }
            _ => {} // unreachable after filter; never panic on edge cases
        }
    }

    // Build the reconciled UserBlock list: for each call in the last assistant
    // turn, look up by call_id. Calls with a real logged Result keep it; calls
    // with no matching Result (dangling Started or never started) get the
    // synthetic is_error "interrupted by host restart" block.
    //
    // Known fidelity limit: the event log stores `ToolResult.summary` only
    // (the audit record deliberately omits the full rendered detail), so a
    // reconciled completed call feeds the model a terser — but true — result
    // than an uninterrupted run would have. The model can re-read workspace
    // state if it needs the detail.
    let mut results: Vec<UserBlock> = Vec::with_capacity(last_assistant_calls.len());
    for call in &last_assistant_calls {
        let completed = match completion_map.get(&call.id) {
            Some(Some((is_error, summary))) => Some((*is_error, summary.clone())),
            _ => None, // not started at all, or started but no Result (dangling)
        };
        results.push(match completed {
            Some((is_error, summary)) => UserBlock::ToolResult {
                call_id: call.id.clone(),
                content: summary,
                is_error,
            },
            None => UserBlock::ToolResult {
                call_id: call.id.clone(),
                content: "interrupted by host restart".to_string(),
                is_error: true,
            },
        });
    }

    // Append a synthetic ToolCallResult for every unmatched Started so the
    // log is paired on the next resume (avoids double-reconciliation). Calls
    // that never reached a Started event have nothing to pair in the log;
    // their transcript block above is synthesis enough, and re-running this
    // reconciliation is idempotent.
    for (call_id, name) in &started_names {
        if matches!(completion_map.get(call_id.as_str()), Some(None)) {
            store
                .append_event(
                    run_id,
                    Event::ToolCallResult {
                        seq: 0,
                        name: name.clone(),
                        is_error: true,
                        summary: "interrupted by host restart".to_string(),
                        offload_path: None,
                    },
                )
                .await?;
        }
    }

    // Return the original messages + reconciliation user message.
    let mut messages = record.messages.clone();
    messages.push(Message::User { content: results });
    Ok(messages)
}

/// Resume a previously-interrupted run from its last checkpoint.
///
/// Loads the [`RunRecord`] for `run_id` from `store`. If no checkpoint exists
/// (`store.load` returns `Ok(None)`), returns
/// [`ResumeError::UnknownRunId`] immediately — no [`model::ModelBackend::turn`]
/// call is made.
///
/// See [`ResumeMode`] for the two resumption strategies (D6/D7).
///
/// The system and task prompts are **RE-RENDERED** from `config + tools` (not
/// read from the record — D9 byte-identity invariant).
///
/// **Budget carry-over (0.3.0):** `budgets.consumed` accumulates across
/// resume in the [`RunRecord`], but remaining-budget enforcement is deferred
/// to 0.4.0. No enforcement logic is added here.
pub async fn resume(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
    store: Arc<dyn RunStore>,
    run_id_arg: &str,
    mode: ResumeMode,
) -> Result<RunResult, ResumeError> {
    let start = Instant::now();
    let mut stats = RunStats {
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall_clock: Duration::ZERO,
    };

    // Load the checkpoint. Return UnknownRunId immediately — no backend call —
    // when no checkpoint exists for the requested run_id.
    let Some(record) = store.load(run_id_arg).await.map_err(ResumeError::Store)? else {
        return Err(ResumeError::UnknownRunId(run_id_arg.to_string()));
    };

    // Budget carry-over: accounting-only in 0.3.0.
    let initial_consumed = record.budgets.consumed;

    let (initial_messages, pre_persist) = match mode {
        ResumeMode::Crash => {
            // D6: reconcile log tail, continue under the same run_id.
            let messages = reconcile_crash_tail(&record, &store, run_id_arg)
                .await
                .map_err(ResumeError::Store)?;
            let pre = RunPersist {
                rid: run_id_arg.to_string(),
                record: {
                    let mut r = record.clone();
                    r.messages.clone_from(&messages);
                    r
                },
            };
            (messages, pre)
        }
        ResumeMode::FreshContext => {
            // D7: drop messages, fresh task seed, new run_id = task_id:(attempt_n+1).
            let task_message = prompt::render_task_prompt(&config.task);
            let messages = vec![Message::User {
                content: vec![UserBlock::Text(task_message)],
            }];
            let new_attempt_n = record.attempt_n + 1;
            let new_rid = run_id(&record.task.task_id, new_attempt_n);
            let pre = RunPersist {
                rid: new_rid.clone(),
                record: {
                    let mut r = record.clone();
                    r.run_id = new_rid;
                    r.attempt_n = new_attempt_n;
                    // Clear any prior terminal disposition — this is a new
                    // attempt, even though it carries durable state forward.
                    r.disposition = None;
                    r.messages.clone_from(&messages);
                    r
                },
            };
            (messages, pre)
        }
    };

    // Build a Persistence solely for store access inside run_loop_impl.
    // task_id / attempt_n are only used to compute the run_id when
    // override_persist is None; since we always pass Some(pre_persist),
    // those fields are irrelevant and set to empty/zero.
    let pers = Persistence {
        store,
        task_id: String::new(),
        attempt_n: 0,
        model_label: String::new(),
    };

    let outcome = run_loop_impl(
        backend,
        tools,
        ctx,
        config,
        Some(&pers),
        &mut stats,
        initial_messages,
        initial_consumed,
        Some(pre_persist),
    )
    .await
    .map_err(ResumeError::Store)?;

    stats.wall_clock = start.elapsed();
    Ok(RunResult { outcome, stats })
}

/// Compute the delay before retry attempt `attempt` (0-indexed) using a
/// deterministic exponential backoff schedule:
///
/// ```text
/// delay = base * 2^attempt
/// ```
///
/// Examples with `base = 500 ms`: attempt 0 → 500 ms, attempt 1 → 1 000 ms,
/// attempt 2 → 2 000 ms.
///
/// **No jitter, no RNG, no wall-clock read** — the delay is a pure function
/// of the attempt index (determinism invariant 3). Set `base =
/// Duration::ZERO` in tests to run retries with no sleep.
fn retry_delay(base: Duration, attempt: u32) -> Duration {
    base * 2u32.pow(attempt)
}

#[cfg(test)]
mod tests {
    use super::{
        FINISH_TOOL_NAME, FinishTool, LoopOutcome, Persistence, ResumeError, ResumeMode, RunConfig,
        RunResult, RunStats, rejection_content, render_tool_result, resume, retry_delay, run,
        run_id, run_persisted,
    };
    use crate::exec::{CheckCommand, CheckReport, ChecksRunner};
    use crate::model::{
        AssistantTurn, BackendError, ContentBlock, Message, StopReason, TerminalKind,
        ToolCallRequest, TransientKind, Usage, UserBlock,
    };
    use crate::prompt;
    use crate::run_record::{
        BudgetConsumed, BudgetLimits, Budgets, Disposition, DurableFacts, Event, FailureMode,
        Phase, ProjectConfig, RunRecord, SCHEMA_VERSION, Task, Verification,
    };
    use crate::store::{RunStore, SqliteRunStore, StoreError};
    use crate::test_support::MockBackend;
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use crate::tools::edit_file::EditFileTool;
    use crate::tools::standard_registry;
    use crate::workspace::Workspace;
    use async_trait::async_trait;
    use std::collections::BTreeMap;
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

        async fn list_events(&self, rid: &str) -> Result<Vec<Event>, StoreError> {
            self.inner.list_events(rid).await
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

        async fn list_events(&self, rid: &str) -> Result<Vec<Event>, StoreError> {
            self.inner.list_events(rid).await
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

        async fn list_events(&self, _rid: &str) -> Result<Vec<Event>, StoreError> {
            Ok(vec![])
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

        // (2) For EVERY iteration (segment bounded by Checkpoint entries), the
        // first call must be AppendEvent — not a Checkpoint.  Walk the full
        // sequence rather than checking only calls[0] so a regression in any
        // iteration (e.g. iteration 2 starting with a Checkpoint) would be caught.
        {
            let mut iteration_start_idx = 0usize;
            for (idx, call) in calls.iter().enumerate() {
                if matches!(call, StoreCall::Checkpoint) {
                    let first_in_iter = &calls[iteration_start_idx];
                    assert!(
                        matches!(first_in_iter, StoreCall::AppendEvent { .. }),
                        "iteration starting at call index {iteration_start_idx} must begin \
                         with AppendEvent, not Checkpoint; got {first_in_iter:?}"
                    );
                    iteration_start_idx = idx + 1;
                }
            }
        }

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

    // =====================================================================
    // Resume tests (AC-2 through AC-10)
    // =====================================================================

    /// Build a minimal [`RunRecord`] for resume tests — enough structure to
    /// satisfy the type but no interesting content beyond what each test sets.
    fn make_minimal_record(task_id: &str, attempt_n: u32) -> RunRecord {
        RunRecord {
            run_id: run_id(task_id, attempt_n),
            schema_version: SCHEMA_VERSION,
            attempt_n,
            task: Task {
                task_id: task_id.to_string(),
                title: String::new(),
                description: "test task".to_string(),
                acceptance_criteria: vec![],
                files_in_scope: vec![],
                scope_out: vec![],
            },
            project_config: ProjectConfig {
                run_checks: BTreeMap::new(),
                model_routing_hint: None,
            },
            phase: Phase::InnerLoop,
            durable_facts: DurableFacts::default(),
            budgets: Budgets {
                consumed: BudgetConsumed::default(),
                limits: BudgetLimits {
                    iterations: 10,
                    tokens: 0,
                    cost_micros: 0,
                },
                wall_clock_start: "2026-01-01T00:00:00Z".to_string(),
            },
            last_gate_result: None,
            disposition: None,
            recovery_facts: None,
            messages: vec![],
        }
    }

    // AC-3: UnknownRunId — no backend calls.
    #[tokio::test]
    async fn resume_unknown_run_id_returns_error_without_backend_calls() {
        let backend = MockBackend::from_turns(vec![]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("task", 5);
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let err = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "no-such-run",
            ResumeMode::Crash,
        )
        .await
        .expect_err("unknown run_id must return Err");

        assert!(
            matches!(&err, ResumeError::UnknownRunId(id) if id == "no-such-run"),
            "must be UnknownRunId(\"no-such-run\"); got {err:?}"
        );
        assert_eq!(
            backend.calls(),
            0,
            "no backend turn must be drawn for an unknown run_id"
        );
    }

    // AC-4: Crash mode clean tail — first turn sees exact reloaded messages.
    #[tokio::test]
    async fn crash_resume_clean_tail_first_turn_sees_exact_reloaded_messages() {
        // Seed a record with a known two-message history (user task + assistant
        // + user tool-result). The log's last event is NOT a ToolCallStarted
        // (clean tail). The resumed run finishes in one turn.
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-echo".to_string(),
                name: "echo".to_string(),
                input: serde_json::json!({}),
            })],
        };
        let tool_result = Message::User {
            content: vec![UserBlock::ToolResult {
                call_id: "c-echo".to_string(),
                content: "{}".to_string(),
                is_error: false,
            }],
        };
        let pre_messages = vec![task_seed.clone(), asst.clone(), tool_result.clone()];

        let mut record = make_minimal_record("clean-task", 1);
        record.messages = pre_messages.clone();

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint("clean-task:1", &record)
            .await
            .expect("checkpoint");
        // Append a ModelCall (not ToolCallStarted) so the tail is clean.
        store
            .append_event(
                "clean-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "test".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("append");

        // One-turn script: finish immediately.
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        let result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "clean-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // First turn must see EXACTLY the reloaded messages — no reconciliation
        // message added on a clean tail.
        let seen = backend.messages_seen();
        assert!(!seen.is_empty(), "at least one turn was drawn");
        assert_eq!(
            seen[0], pre_messages,
            "first turn must see exactly the reloaded messages (clean-tail crash resume)"
        );
        assert!(
            matches!(
                result.outcome,
                LoopOutcome::Finished(Disposition::Done { .. })
            ),
            "outcome must be Done; got {:?}",
            result.outcome
        );
    }

    // AC-5: Crash mode dangling tail — two-call reconciliation.
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn crash_resume_dangling_tail_reconciles_two_calls_no_reinvocation() {
        // A record whose last assistant turn has TWO tool calls: "count1" and
        // "count2". The event log has a ToolCallResult for call1 + a dangling
        // ToolCallStarted for call2. Resume must:
        //   (a) feed back a User message with 2 ToolResults (call1 real, call2 synthetic)
        //   (b) not invoke either tool during reconciliation
        //   (c) append a synthetic ToolCallResult for call2 to the log

        // --- build a counting tool so we can assert 0 new invocations ---
        struct CountingTool {
            count: Mutex<u32>,
        }
        impl CountingTool {
            fn invocations(&self) -> u32 {
                *self.count.lock().unwrap()
            }
        }
        #[async_trait]
        impl Tool for CountingTool {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "count"
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "name": "count",
                    "input_schema": { "type": "object", "properties": {}, "required": [] }
                })
            }
            async fn run(&self, _input: serde_json::Value, _ctx: &ToolCtx) -> ToolResult {
                *self.count.lock().unwrap() += 1;
                ToolResult::ok("counted")
            }
        }

        let counter = Arc::new(CountingTool {
            count: Mutex::new(0),
        });
        let mut tools = ToolRegistry::new();
        tools.register("count", counter.clone());
        tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

        // Build the interrupted record: task seed + assistant(call1, call2).
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![
                ContentBlock::ToolCall(ToolCallRequest {
                    id: "id-call1".to_string(),
                    name: "count".to_string(),
                    input: serde_json::json!({}),
                }),
                ContentBlock::ToolCall(ToolCallRequest {
                    id: "id-call2".to_string(),
                    name: "count".to_string(),
                    input: serde_json::json!({}),
                }),
            ],
        };
        let mut record = make_minimal_record("dangling-task", 1);
        record.messages = vec![task_seed.clone(), asst.clone()];

        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint("dangling-task:1", &record)
            .await
            .expect("checkpoint");

        // Log seeds the REAL engine-emitted tail shape:
        //   ModelCall → BudgetTick → Started1(call_id=id-call1) → Result1 → Started2(call_id=id-call2) [dangling]
        // The BudgetTick is what the engine always emits (engine.rs:895) between
        // ModelCall and the first ToolCallStarted. The old test omitted it,
        // which caused the positional walk to never advance and mark EVERY call
        // as interrupted — this rewrite closes that seam.
        store
            .append_event(
                "dangling-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "test".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("mc");
        store
            .append_event(
                "dangling-task:1",
                Event::BudgetTick {
                    seq: 0,
                    consumed: BudgetConsumed {
                        iterations: 1,
                        tokens: 2,
                        cost_micros: 0,
                    },
                },
            )
            .await
            .expect("budget_tick");
        store
            .append_event(
                "dangling-task:1",
                Event::ToolCallStarted {
                    seq: 0,
                    name: "count".to_string(),
                    args: serde_json::json!({}),
                    call_id: "id-call1".to_string(),
                },
            )
            .await
            .expect("started1");
        store
            .append_event(
                "dangling-task:1",
                Event::ToolCallResult {
                    seq: 0,
                    name: "count".to_string(),
                    is_error: false,
                    summary: "counted".to_string(),
                    offload_path: None,
                },
            )
            .await
            .expect("result1");
        store
            .append_event(
                "dangling-task:1",
                Event::ToolCallStarted {
                    seq: 0,
                    name: "count".to_string(),
                    args: serde_json::json!({}),
                    call_id: "id-call2".to_string(),
                },
            )
            .await
            .expect("started2_dangling");

        // Resume script: one turn that calls finish immediately (so the loop
        // doesn't re-invoke count after reconciliation).
        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        let _result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            "dangling-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // (a) First turn must see task seed + assistant + reconciled user msg
        //     (2 ToolResults: call1 real, call2 synthetic is_error).
        let seen = backend.messages_seen();
        assert!(!seen.is_empty());
        let first_turn = &seen[0];
        // messages[0] = task seed, [1] = assistant, [2] = reconciled user
        assert_eq!(
            first_turn.len(),
            3,
            "task_seed + asst + reconciled user; got {} msgs",
            first_turn.len()
        );
        let reconciled_user = &first_turn[2];
        match reconciled_user {
            Message::User { content } => {
                assert_eq!(content.len(), 2, "must have 2 ToolResult blocks");
                // call1: real result (not error, summary = "counted")
                match &content[0] {
                    UserBlock::ToolResult {
                        call_id,
                        content: c,
                        is_error,
                    } => {
                        assert_eq!(call_id, "id-call1");
                        assert!(!is_error, "call1 was completed successfully");
                        assert_eq!(c, "counted");
                    }
                    other @ UserBlock::Text(_) => {
                        panic!("expected ToolResult block 0; got {other:?}")
                    }
                }
                // call2: synthetic is_error=true
                match &content[1] {
                    UserBlock::ToolResult {
                        call_id,
                        content: c,
                        is_error,
                    } => {
                        assert_eq!(call_id, "id-call2");
                        assert!(is_error, "call2 must be is_error=true (interrupted)");
                        assert_eq!(c, "interrupted by host restart");
                    }
                    other @ UserBlock::Text(_) => {
                        panic!("expected ToolResult block 1; got {other:?}")
                    }
                }
            }
            other @ Message::Assistant { .. } => {
                panic!("expected User message for reconciled results; got {other:?}")
            }
        }

        // (b) CountingTool must not have been invoked during reconciliation.
        assert_eq!(
            counter.invocations(),
            0,
            "counting tool must not be invoked during reconciliation"
        );

        // (c) The log must contain a synthetic ToolCallResult(is_error=true, name="count")
        //     appended during reconciliation (the exact position is after the dangling
        //     Started; more events may follow from the resumed run itself).
        let events = store.list_events("dangling-task:1").await.expect("list");
        let synthetic = events.iter().find(|e| {
            matches!(
                e,
                Event::ToolCallResult {
                    is_error: true,
                    summary,
                    name,
                    ..
                }
                if summary == "interrupted by host restart" && name == "count"
            )
        });
        assert!(
            synthetic.is_some(),
            "synthetic ToolCallResult(is_error=true, summary='interrupted by host restart') \
             must be in the event log; events: {events:?}"
        );
    }

    // AC-6: FreshContext drops messages and carries budgets.consumed forward.
    #[tokio::test]
    async fn fresh_context_resume_drops_messages_carries_consumed() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        // Build a record with non-empty messages and non-zero consumed.
        let old_msg = Message::User {
            content: vec![UserBlock::Text("old context".to_string())],
        };
        let pre_consumed = BudgetConsumed {
            iterations: 7,
            tokens: 500,
            cost_micros: 100,
        };
        let mut record = make_minimal_record("fc-task", 1);
        record.messages = vec![old_msg.clone()];
        record.budgets.consumed = pre_consumed;
        record
            .durable_facts
            .findings
            .push("prior finding".to_string());

        store
            .checkpoint("fc-task:1", &record)
            .await
            .expect("checkpoint");

        // One-turn script: finish immediately.
        let snap = Arc::new(SnapshotStore::new());
        // We need a combined store so we can observe snapshots AND have the
        // same data available for load. Use a wrapper.
        // For simplicity, use an SqliteStore for load and a SnapshotStore for
        // observing snapshots. But they need to share data...
        // Instead: manually checkpoint into the SnapshotStore's inner store.
        snap.inner
            .checkpoint("fc-task:1", &record)
            .await
            .expect("snap checkpoint");

        let store2: Arc<dyn RunStore> = snap.clone();

        let backend = MockBackend::from_turns(vec![finish_call(
            "c-fin",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store2,
            "fc-task:1",
            ResumeMode::FreshContext,
        )
        .await
        .expect("resume must succeed");

        // First turn must see ONLY the fresh task seed (old messages absent).
        let seen = backend.messages_seen();
        assert!(!seen.is_empty());
        let first = &seen[0];
        assert_eq!(first.len(), 1, "only task seed; old messages absent");
        match &first[0] {
            Message::User { content } => match &content[0] {
                UserBlock::Text(t) => assert!(
                    !t.contains("old context"),
                    "old context must not appear in fresh seed"
                ),
                other @ UserBlock::ToolResult { .. } => {
                    panic!("expected Text; got {other:?}")
                }
            },
            other @ Message::Assistant { .. } => {
                panic!("expected User message; got {other:?}")
            }
        }

        // The checkpointed record must carry the pre-restart consumed value.
        let snaps = snap.all_snapshots();
        assert!(!snaps.is_empty(), "at least one checkpoint written");
        let final_snap = snaps.last().unwrap();
        assert!(
            final_snap.budgets.consumed.iterations >= pre_consumed.iterations,
            "consumed.iterations must carry forward (>= pre-restart value); got {:?}",
            final_snap.budgets.consumed
        );
        // The durable finding from before the restart must still be there.
        assert!(
            final_snap
                .durable_facts
                .findings
                .contains(&"prior finding".to_string()),
            "durable_facts.findings must carry forward; got {:?}",
            final_snap.durable_facts.findings
        );
    }

    // AC-7: FreshContext run identity — new run_id, old intact.
    #[tokio::test]
    async fn fresh_context_run_identity_new_run_id_old_intact() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let mut record = make_minimal_record("id-task", 1);
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("old".to_string())],
        }];
        record.durable_facts.findings.push("carried".to_string());

        let old_rid = "id-task:1";
        store.checkpoint(old_rid, &record).await.expect("cp");

        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            old_rid,
            ResumeMode::FreshContext,
        )
        .await
        .expect("resume must succeed");

        // old run_id must still hold the original record (unchanged).
        let old_rec = store.load(old_rid).await.expect("load").expect("present");
        assert_eq!(
            old_rec.messages, record.messages,
            "prior run_id messages must be unmodified by FreshContext resume"
        );

        // new run_id = "id-task:2" must hold the continued record.
        let new_rid = "id-task:2";
        let new_rec = store
            .load(new_rid)
            .await
            .expect("load")
            .expect("new run_id must be checkpointed");
        assert_eq!(new_rec.attempt_n, 2, "new record must have attempt_n == 2");
        assert!(
            new_rec
                .durable_facts
                .findings
                .contains(&"carried".to_string()),
            "new record must carry durable_facts forward"
        );
    }

    // AC-7 complement: Crash resume checkpoints under the SAME run_id only.
    #[tokio::test]
    async fn crash_resume_checkpoints_under_same_run_id_only() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let mut record = make_minimal_record("crash-id-task", 1);
        // Minimal messages so reconcile finds no dangling tail.
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        }];
        store
            .checkpoint("crash-id-task:1", &record)
            .await
            .expect("cp");
        // Append a non-ToolCallStarted event so the tail is clean.
        store
            .append_event(
                "crash-id-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5);

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store.clone(),
            "crash-id-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // Only the original run_id should have a checkpoint.
        let orig = store
            .load("crash-id-task:1")
            .await
            .expect("load")
            .expect("original run_id must still be checkpointed");
        assert_eq!(orig.attempt_n, 1, "same attempt_n on crash resume");

        // No new run_id must have been created.
        let new_rid = store.load("crash-id-task:2").await.expect("load");
        assert!(
            new_rid.is_none(),
            "crash resume must not create a new run_id; found: {new_rid:?}"
        );
    }

    // AC-8: Prompt byte-identity on resume — system prompt rendered fresh.
    #[tokio::test]
    async fn resume_system_prompt_byte_identical_to_fresh_run() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let mut record = make_minimal_record("prompt-task", 1);
        record.messages = vec![Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        }];
        store
            .checkpoint("prompt-task:1", &record)
            .await
            .expect("cp");
        store
            .append_event(
                "prompt-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        // Use a non-trivial tools setup and a checks runner so the system
        // prompt includes the check command display.
        let tools = registry_with_finish_and_echo();
        let runner = passing_runner();
        let runner_display = runner.command_display();
        let config = RunConfig::new("do the task", 5).with_checks(runner);

        // Three-turn script so we see multiple system-prompt entries.
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
        let ctx = ToolCtx::stub();

        resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "prompt-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        let systems = backend.systems_seen();
        assert!(!systems.is_empty(), "at least one turn drawn");

        // The expected system prompt: re-rendered from the same tools + checks.
        let expected = prompt::render_system_prompt(
            &prompt::tool_lines(&tools),
            Some(runner_display.as_str()),
        );

        for (i, entry) in systems.iter().enumerate() {
            let s = entry.as_ref().expect("system prompt must be sent");
            assert_eq!(
                s.as_bytes(),
                expected.as_bytes(),
                "turn {i} system prompt must be byte-identical to fresh render"
            );
        }
    }

    // AC-10: Crash resume happy path reaches verified Done.
    #[tokio::test]
    async fn crash_resume_happy_path_reaches_verified_done() {
        // Seed a record with a clean log tail, then resume. The script ends
        // with finish(done) against a passing ChecksRunner, proving the
        // claim-vs-verify loop still fires on resume.
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));

        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let mut record = make_minimal_record("happy-task", 1);
        record.messages = vec![task_seed.clone()];
        store.checkpoint("happy-task:1", &record).await.expect("cp");
        // Clean tail.
        store
            .append_event(
                "happy-task:1",
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            )
            .await
            .expect("mc");

        let backend = MockBackend::from_turns(vec![finish_call(
            "cf",
            serde_json::json!({ "disposition": "done", "summary": "task complete" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5).with_checks(passing_runner());

        let result = resume(
            &backend,
            &tools,
            &ctx,
            &config,
            store,
            "happy-task:1",
            ResumeMode::Crash,
        )
        .await
        .expect("resume must succeed");

        // Must reach a verified Done — the claim-vs-verify loop fired.
        assert!(
            matches!(
                result.outcome,
                LoopOutcome::Finished(Disposition::Done {
                    verification: crate::run_record::Verification::Checks(_),
                    ..
                })
            ),
            "crash resume must reach Finished(Done{{Checks(green)}}); got {:?}",
            result.outcome
        );
    }

    // 0.3.0-4: Kill-and-resume proof — the release-gate deterministic integration test.
    //
    // This test proves the crash-resume capability introduced by items 0.3.0-1/2/3.
    // It runs WITHOUT #[ignore] and WITHOUT an env-var gate — it is always in CI.
    //
    // Trajectory:
    //   Leg 1 (pre-crash):  model calls panicky-tool → tool panics → process "dies"
    //   Leg 2 (post-resume): resume() reconciles the dangling tail, model calls finish
    //
    // All storage is FILE-BACKED (tempfile::TempDir + SqliteRunStore::open).
    // No live model, no network.
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn kill_and_resume_proof_deterministic_integration() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // ---- Test-local panicking tool ------------------------------------
        // Follows the EchoTool test pattern (tool.rs:~388).
        // On its FIRST invocation it increments a shared counter then panics,
        // simulating a host crash mid-tool-execution.
        struct PanickyTool {
            counter: Arc<AtomicU32>,
        }

        #[async_trait]
        impl Tool for PanickyTool {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "panicky-tool"
            }

            fn schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "name": "panicky-tool",
                    "description": "test tool that panics on first invocation (kill-resume proof)",
                    "input_schema": { "type": "object", "properties": {}, "required": [] }
                })
            }

            async fn run(&self, _input: serde_json::Value, _ctx: &ToolCtx) -> ToolResult {
                self.counter.fetch_add(1, Ordering::SeqCst);
                panic!("simulated host kill — crash-resume proof leg 1");
            }
        }

        // ---- Constants ---------------------------------------------------
        // run_id format is {task_id}:{attempt_n} (D4).
        const TASK_ID: &str = "kill-resume-task";
        const ATTEMPT_N: u32 = 1;
        const RUN_ID_LITERAL: &str = "kill-resume-task:1";
        const PANICKY_TOOL_NAME: &str = "panicky-tool";
        const PANICKY_CALL_ID: &str = "c-panicky-1";
        // Per-turn token amounts (pinned so totals are computable constants).
        const INPUT_TOKENS: u32 = 100;
        const OUTPUT_TOKENS: u32 = 10;
        const TOKENS_PER_TURN: u64 = (INPUT_TOKENS + OUTPUT_TOKENS) as u64; // 110

        // ---- Setup: file-backed store + shared side-effect counter -------
        let db_dir = TempDir::new().expect("create temp dir for SQLite DB");
        let db_path = db_dir.path().join("harness.db");
        let side_effect_counter = Arc::new(AtomicU32::new(0));

        // ---- Leg 1: run until crash (AC-2) -------------------------------
        // Drive leg 1 inside tokio::spawn so we can catch the panic.
        // D5 guarantees ToolCallStarted is appended BEFORE tools.invoke, so
        // the log ends with a dangling ToolCallStarted when the tool panics.
        //
        // The backend is created OUTSIDE the spawn and shared via Arc so that
        // after the crash we can call leg1_backend.systems_seen() and compare
        // it to leg 2 — asserting byte-identity against what leg 1 ACTUALLY
        // sent, not against a fresh in-test reference render (D9 fix).
        let leg1_backend = Arc::new(MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call(
                PANICKY_CALL_ID,
                PANICKY_TOOL_NAME,
                serde_json::json!({}),
            )],
            StopReason::ToolUse,
            usage_with(INPUT_TOKENS, OUTPUT_TOKENS),
        )]));
        let leg1_backend_for_spawn = leg1_backend.clone();

        let db_path_for_spawn = db_path.clone();
        let counter_for_spawn = side_effect_counter.clone();

        let leg1_handle = tokio::spawn(async move {
            let mut tools = ToolRegistry::new();
            tools.register(
                PANICKY_TOOL_NAME,
                Arc::new(PanickyTool {
                    counter: counter_for_spawn,
                }),
            );
            tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool));
            let ctx = ToolCtx::stub();
            let config = RunConfig::new("do the task", 10).with_checks(passing_runner());
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open(&db_path_for_spawn).expect("open SQLite for leg 1"));
            let pers = Persistence {
                store,
                task_id: TASK_ID.to_string(),
                attempt_n: ATTEMPT_N,
                model_label: "test-model".to_string(),
            };
            // PanickyTool::run() panics here; the spawned task catches it.
            run_persisted(&*leg1_backend_for_spawn, &tools, &ctx, &config, &pers).await
        });

        // AC-2: Leg 1 must panic.
        let leg1_join = leg1_handle.await;
        assert!(
            leg1_join.is_err(),
            "leg 1 tokio::spawn must return Err (panicked task)"
        );
        assert!(
            leg1_join.unwrap_err().is_panic(),
            "leg 1 JoinError must be a panic, not a cancellation"
        );

        // ---- Verify crash state: log ends with dangling ToolCallStarted --
        // Open a FRESH handle — simulates the harness opening the DB after
        // a process restart.
        let store_post_crash: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite post-crash"));
        let events_pre_resume = store_post_crash
            .list_events(RUN_ID_LITERAL)
            .await
            .expect("list events after crash");
        assert!(
            !events_pre_resume.is_empty(),
            "event log must not be empty after leg 1 crash"
        );
        let last_pre_crash = events_pre_resume.last().expect("non-empty — just asserted");
        // AC-2: the LAST event in list_events at crash time is ToolCallStarted.
        assert!(
            matches!(
                last_pre_crash,
                Event::ToolCallStarted { name, .. } if name == PANICKY_TOOL_NAME
            ),
            "last event after crash must be ToolCallStarted({PANICKY_TOOL_NAME}); \
             got {last_pre_crash:?}"
        );
        let pre_crash_last_seq = match last_pre_crash {
            Event::ToolCallStarted { seq, .. } => *seq,
            _ => unreachable!("just matched above"),
        };

        // ---- Leg 2: crash-resume (AC-3) ----------------------------------
        // Fresh store handle over the same file — durability proof: no in-memory dodge.
        let store_leg2: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite for leg 2"));

        // Scripted backend for leg 2: after reconcile feeds the synthetic
        // ToolResult back, the model calls finish immediately.
        let backend_leg2 = MockBackend::from_turns(vec![turn_with_usage(
            vec![tool_call(
                "c-finish-resume",
                FINISH_TOOL_NAME,
                serde_json::json!({
                    "disposition": "done",
                    "summary": "task complete after crash-resume"
                }),
            )],
            StopReason::ToolUse,
            usage_with(INPUT_TOKENS, OUTPUT_TOKENS),
        )]);

        // Same tool registry as leg 1 — D9 byte-identity requires identical
        // tool_lines. Including panicky-tool here: its schema must match so
        // the system prompt is byte-identical. The scripted backend does NOT
        // call it in leg 2, so the counter stays at 1.
        let mut tools_leg2 = ToolRegistry::new();
        tools_leg2.register(
            PANICKY_TOOL_NAME,
            Arc::new(PanickyTool {
                counter: side_effect_counter.clone(),
            }),
        );
        tools_leg2.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

        let ctx_leg2 = ToolCtx::stub();
        let config_leg2 = RunConfig::new("do the task", 10).with_checks(passing_runner());

        let resume_result = resume(
            &backend_leg2,
            &tools_leg2,
            &ctx_leg2,
            &config_leg2,
            store_leg2,
            RUN_ID_LITERAL,
            ResumeMode::Crash,
        )
        .await
        .expect("crash-resume must succeed without error");

        // AC-3: Terminal disposition must be Done{Checks(green)}.
        // A Blocked or Failed terminal fails the test.
        let is_verified_done = matches!(
            &resume_result.outcome,
            LoopOutcome::Finished(Disposition::Done {
                verification: Verification::Checks(report),
                ..
            }) if report.passed
        );
        assert!(
            is_verified_done,
            "crash-resume must terminate in Done{{Checks(green)}}; got {:?}",
            resume_result.outcome
        );

        // ---- AC-4a: Resume-leg transcript contains synthetic ToolResult --
        // D6: reconcile feeds a model::UserBlock::ToolResult{is_error=true,
        // content contains "interrupted"} for the panicky call.
        // The `content` field (not summary) is the assertion target.
        let leg2_messages_seen = backend_leg2.messages_seen();
        assert!(
            !leg2_messages_seen.is_empty(),
            "leg 2 must draw at least one model turn"
        );
        let first_turn = &leg2_messages_seen[0];
        let synthetic_in_transcript = first_turn.iter().any(|msg| match msg {
            Message::User { content } => content.iter().any(|b| {
                matches!(
                    b,
                    UserBlock::ToolResult {
                        is_error: true,
                        call_id,
                        content: c,
                        ..
                    } if call_id == PANICKY_CALL_ID && c.contains("interrupted")
                )
            }),
            Message::Assistant { .. } => false,
        });
        assert!(
            synthetic_in_transcript,
            "first turn of leg 2 must include \
             UserBlock::ToolResult{{is_error=true, call_id={PANICKY_CALL_ID:?}, \
             content contains 'interrupted'}} (D6); messages: {first_turn:?}"
        );

        // ---- Collect final event log for remaining assertions ------------
        let store_final: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite for final assertions"));
        let all_events = store_final
            .list_events(RUN_ID_LITERAL)
            .await
            .expect("list all events after both legs");

        // ---- AC-4b: Exactly ONE synthetic ToolCallResult (is_error=true, "interrupted") --
        // reconcile_crash_tail appends exactly one synthetic ToolCallResult for
        // the dangling ToolCallStarted.
        let synthetic_results: Vec<_> = all_events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Event::ToolCallResult {
                        is_error: true,
                        name,
                        summary,
                        ..
                    } if name == PANICKY_TOOL_NAME && summary.contains("interrupted")
                )
            })
            .collect();
        assert_eq!(
            synthetic_results.len(),
            1,
            "event log must contain exactly ONE synthetic ToolCallResult \
             (is_error=true, name={PANICKY_TOOL_NAME:?}, summary contains 'interrupted'); \
             found {}: {synthetic_results:?}",
            synthetic_results.len()
        );

        // ---- AC-4c: Side-effect counter == 1 (no blind re-execution) ----
        assert_eq!(
            side_effect_counter.load(Ordering::SeqCst),
            1,
            "panicky-tool side-effect counter must be exactly 1 after both legs; \
             the interrupted call must never be blindly re-executed"
        );

        // ---- AC-4d: Every ToolCallStarted has a matching ToolCallResult --
        // After reconcile, the final log has no unpaired ToolCallStarted events:
        // every Started is followed by a Result before the next ModelCall or
        // end-of-log. Walk the events to verify.
        {
            let mut dangling: Option<&str> = None;
            for event in &all_events {
                match event {
                    Event::ToolCallStarted { name, .. } => {
                        assert!(
                            dangling.is_none(),
                            "ToolCallStarted({name}) found while {dangling:?} is still unpaired"
                        );
                        dangling = Some(name.as_str());
                    }
                    Event::ToolCallResult { name, .. } => {
                        assert!(
                            dangling.is_some(),
                            "ToolCallResult({name}) has no matching ToolCallStarted"
                        );
                        dangling = None;
                    }
                    Event::ModelCall { .. } => {
                        assert!(
                            dangling.is_none(),
                            "ModelCall arrived while ToolCallStarted({dangling:?}) is still unpaired"
                        );
                    }
                    Event::PhaseTransition { .. }
                    | Event::BudgetTick { .. }
                    | Event::DispositionSet { .. } => {}
                }
            }
            assert!(
                dangling.is_none(),
                "event log ends with an unpaired ToolCallStarted: {dangling:?}"
            );
        }

        // ---- AC-5: Budget continuity — pinned integer literals -----------
        // Leg 1: 1 turn × (INPUT + OUTPUT) tokens = TOKENS_PER_TURN, 1 iteration.
        // Leg 2: 1 turn × (INPUT + OUTPUT) tokens = TOKENS_PER_TURN, 1 iteration.
        // Total: iterations = 2, tokens = 2 × TOKENS_PER_TURN = 220, cost_micros = 0.
        // item 0.3.0-3 wires no pricing, so cost_micros == 0.
        let final_record = store_final
            .load(RUN_ID_LITERAL)
            .await
            .expect("load final record")
            .expect("record must exist after both legs");
        assert_eq!(
            final_record.budgets.consumed.iterations, 2,
            "consumed.iterations must be exactly 2 (1 pre-crash + 1 post-resume); \
             got {:?}",
            final_record.budgets.consumed
        );
        assert_eq!(
            final_record.budgets.consumed.tokens,
            2 * TOKENS_PER_TURN,
            "consumed.tokens must be exactly {} ({}×{} per turn × 2 turns); got {:?}",
            2 * TOKENS_PER_TURN,
            INPUT_TOKENS + OUTPUT_TOKENS,
            2,
            final_record.budgets.consumed
        );
        assert_eq!(
            final_record.budgets.consumed.cost_micros, 0,
            "consumed.cost_micros must be 0 (no pricing wired in 0.3.0); \
             got {:?}",
            final_record.budgets.consumed
        );

        // ---- AC-6: Seq monotonicity — no reset at the resume boundary ----
        // Extract the monotonic seq from every event in the final log.
        let seqs: Vec<u64> = all_events
            .iter()
            .map(|e| match e {
                Event::ModelCall { seq, .. }
                | Event::ToolCallStarted { seq, .. }
                | Event::ToolCallResult { seq, .. }
                | Event::PhaseTransition { seq, .. }
                | Event::BudgetTick { seq, .. }
                | Event::DispositionSet { seq, .. } => *seq,
            })
            .collect();

        // Strictly increasing — no duplicates, no resets at the resume boundary.
        for (i, window) in seqs.windows(2).enumerate() {
            assert!(
                window[1] > window[0],
                "seq is not strictly increasing at position {i}: {} -> {}; \
                 full seq list: {seqs:?}",
                window[0],
                window[1]
            );
        }

        // The first event appended by the resumed loop has seq exactly one
        // greater than the last event persisted before the crash.
        // grounded in store.rs:322-323 (COALESCE(MAX(seq),-1)+1 per run).
        // events_pre_resume.len() events were written in leg 1;
        // seqs[events_pre_resume.len()] is the first seq added during resume.
        let first_resumed_seq = seqs[events_pre_resume.len()];
        assert_eq!(
            first_resumed_seq,
            pre_crash_last_seq + 1,
            "first event appended by resumed loop (seq={first_resumed_seq}) must be \
             exactly one greater than the last pre-crash event \
             (seq={pre_crash_last_seq})"
        );

        // ---- AC-7: D9 System prompt byte-identity across the restart -----
        // Leg 2's system prompt must be byte-identical to what leg 1 ACTUALLY
        // sent — not to a fresh in-test reference render.  leg1_backend was
        // captured via Arc before the spawn, so systems_seen() reflects the
        // real bytes that the engine transmitted during leg 1.
        let leg1_systems = leg1_backend.systems_seen();
        assert!(
            !leg1_systems.is_empty(),
            "leg 1 must have sent at least one system prompt"
        );
        let leg1_first_system = leg1_systems[0]
            .as_ref()
            .expect("leg 1 turn 0 must send a non-None system prompt");
        let leg2_systems = backend_leg2.systems_seen();
        assert!(
            !leg2_systems.is_empty(),
            "leg 2 must observe at least one system prompt"
        );
        for (i, entry) in leg2_systems.iter().enumerate() {
            let seen = entry
                .as_ref()
                .unwrap_or_else(|| panic!("leg 2 turn {i} must send a non-None system prompt"));
            assert_eq!(
                seen.as_bytes(),
                leg1_first_system.as_bytes(),
                "leg 2 turn {i} system prompt must be byte-identical to what \
                 leg 1 ACTUALLY sent (D9 prompt-cache invariant)"
            );
        }
    }

    // Two-call interrupted turn: first call completes (real result logged),
    // crash before the second call's result. Post-resume the completed call's
    // REAL result content survives in the transcript; the second call gets
    // exactly one synthetic is_error result. Run completes Done{Checks(green)}.
    //
    // This is the sibling of kill_and_resume_proof_deterministic_integration
    // extended to the two-tool-call scenario that the original test skipped
    // (a single-call turn where the buggy all-synthetic output coincides with
    // the correct output — the bug was invisible).
    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn kill_and_resume_two_call_interrupted_turn() {
        use std::sync::atomic::{AtomicU32, Ordering};

        // ---- Test-local panicky tool (always panics on invocation) ---------
        struct PanickyTool {
            counter: Arc<AtomicU32>,
        }

        #[async_trait]
        impl Tool for PanickyTool {
            #[allow(clippy::unnecessary_literal_bound)]
            fn name(&self) -> &str {
                "panicky-tool"
            }

            fn schema(&self) -> serde_json::Value {
                serde_json::json!({
                    "name": "panicky-tool",
                    "description": "test tool that always panics (two-call kill-resume proof)",
                    "input_schema": { "type": "object", "properties": {}, "required": [] }
                })
            }

            async fn run(&self, _input: serde_json::Value, _ctx: &ToolCtx) -> ToolResult {
                self.counter.fetch_add(1, Ordering::SeqCst);
                panic!("simulated host kill — two-call crash-resume proof");
            }
        }

        // ---- Constants ---------------------------------------------------
        const TASK_ID: &str = "two-call-kill-resume-task";
        const ATTEMPT_N: u32 = 1;
        const RUN_ID_LITERAL: &str = "two-call-kill-resume-task:1";
        const ECHO_CALL_ID: &str = "c-echo-1";
        const PANICKY_CALL_ID: &str = "c-panicky-1";
        const PANICKY_TOOL_NAME: &str = "panicky-tool";
        // EchoTool returns input.to_string() as its summary.
        // Called with json!({}) the summary stored in the event is "{}".
        const ECHO_EXPECTED_SUMMARY: &str = "{}";

        // ---- Setup -------------------------------------------------------
        let db_dir = TempDir::new().expect("create temp dir");
        let db_path = db_dir.path().join("harness.db");
        let panicky_counter = Arc::new(AtomicU32::new(0));

        // ---- Leg 1: run until crash ----------------------------------------
        // Model turn: [echo-call, panicky-call]. EchoTool completes; PanickyTool panics.
        let db_path_for_spawn = db_path.clone();
        let counter_for_spawn = panicky_counter.clone();

        let leg1_handle = tokio::spawn(async move {
            let backend = MockBackend::from_turns(vec![turn_with(
                vec![
                    tool_call(ECHO_CALL_ID, "echo", serde_json::json!({})),
                    tool_call(PANICKY_CALL_ID, PANICKY_TOOL_NAME, serde_json::json!({})),
                ],
                StopReason::ToolUse,
            )]);
            let mut tools = ToolRegistry::new();
            tools.register("echo", Arc::new(EchoTool));
            tools.register(
                PANICKY_TOOL_NAME,
                Arc::new(PanickyTool {
                    counter: counter_for_spawn,
                }),
            );
            tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool));
            let ctx = ToolCtx::stub();
            let config = RunConfig::new("do the task", 10).with_checks(passing_runner());
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open(&db_path_for_spawn).expect("open SQLite for leg 1"));
            let pers = Persistence {
                store,
                task_id: TASK_ID.to_string(),
                attempt_n: ATTEMPT_N,
                model_label: "test-model".to_string(),
            };
            run_persisted(&backend, &tools, &ctx, &config, &pers).await
        });

        // Leg 1 must panic (PanickyTool always panics).
        let leg1_join = leg1_handle.await;
        assert!(leg1_join.is_err(), "leg 1 must return Err (panicked task)");
        assert!(
            leg1_join.unwrap_err().is_panic(),
            "leg 1 JoinError must be a panic"
        );

        // ---- Verify crash state ------------------------------------------
        // Log must end with a dangling ToolCallStarted for panicky-tool.
        let store_post_crash: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open post-crash"));
        let events_pre_resume = store_post_crash
            .list_events(RUN_ID_LITERAL)
            .await
            .expect("list events");
        let last_pre_crash = events_pre_resume.last().expect("non-empty log after crash");
        assert!(
            matches!(
                last_pre_crash,
                Event::ToolCallStarted { name, .. } if name == PANICKY_TOOL_NAME
            ),
            "last event after crash must be ToolCallStarted({PANICKY_TOOL_NAME}); \
             got {last_pre_crash:?}"
        );

        // The echo call must have a ToolCallResult with is_error=false in the log.
        let echo_result_in_log = events_pre_resume.iter().any(|e| {
            matches!(
                e,
                Event::ToolCallResult {
                    name,
                    is_error: false,
                    ..
                } if name == "echo"
            )
        });
        assert!(
            echo_result_in_log,
            "echo ToolCallResult(is_error=false) must be in log before resume; \
             events: {events_pre_resume:?}"
        );

        // ---- Leg 2: crash-resume -----------------------------------------
        let store_leg2: Arc<dyn RunStore> =
            Arc::new(SqliteRunStore::open(&db_path).expect("open SQLite for leg 2"));

        let backend_leg2 = MockBackend::from_turns(vec![turn_with(
            vec![tool_call(
                "c-finish-resume",
                FINISH_TOOL_NAME,
                serde_json::json!({
                    "disposition": "done",
                    "summary": "task complete after two-call crash-resume"
                }),
            )],
            StopReason::ToolUse,
        )]);

        let mut tools_leg2 = ToolRegistry::new();
        tools_leg2.register("echo", Arc::new(EchoTool));
        tools_leg2.register(
            PANICKY_TOOL_NAME,
            Arc::new(PanickyTool {
                counter: panicky_counter.clone(),
            }),
        );
        tools_leg2.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

        let ctx_leg2 = ToolCtx::stub();
        let config_leg2 = RunConfig::new("do the task", 10).with_checks(passing_runner());

        let resume_result = resume(
            &backend_leg2,
            &tools_leg2,
            &ctx_leg2,
            &config_leg2,
            store_leg2,
            RUN_ID_LITERAL,
            ResumeMode::Crash,
        )
        .await
        .expect("crash-resume must succeed");

        // Terminal disposition must be Done{Checks(green)}.
        let is_verified_done = matches!(
            &resume_result.outcome,
            LoopOutcome::Finished(Disposition::Done {
                verification: Verification::Checks(report),
                ..
            }) if report.passed
        );
        assert!(
            is_verified_done,
            "two-call crash-resume must terminate in Done{{Checks(green)}}; \
             got {:?}",
            resume_result.outcome
        );

        // First leg-2 turn must see the reconciled user message with two blocks:
        //   [0] echo-call: REAL result (content = ECHO_EXPECTED_SUMMARY, is_error=false)
        //   [1] panicky-call: synthetic is_error=true, content contains "interrupted"
        let leg2_messages = backend_leg2.messages_seen();
        assert!(
            !leg2_messages.is_empty(),
            "leg 2 must draw at least one turn"
        );
        let first_turn = &leg2_messages[0];

        // Find the reconciled User message (last User before the first Assistant).
        let reconciled_user = first_turn
            .iter()
            .find(|m| {
                matches!(m, Message::User { content }
                    if content.iter().any(|b| matches!(b, UserBlock::ToolResult { .. })))
            })
            .expect("first leg-2 turn must include a User message with ToolResult blocks");

        let Message::User { content: blocks } = reconciled_user else {
            panic!("expected User message");
        };
        assert_eq!(
            blocks.len(),
            2,
            "reconciled User must have exactly 2 ToolResult blocks; got {blocks:?}"
        );

        // Block 0: echo-call, real result.
        match &blocks[0] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, ECHO_CALL_ID, "block[0] call_id must be echo");
                assert!(
                    !is_error,
                    "echo call completed successfully — is_error must be false"
                );
                assert_eq!(
                    content, ECHO_EXPECTED_SUMMARY,
                    "echo call must carry its REAL result content, not synthetic"
                );
            }
            UserBlock::Text(_) => panic!("expected ToolResult block 0; got Text"),
        }

        // Block 1: panicky-call, synthetic is_error.
        match &blocks[1] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, PANICKY_CALL_ID, "block[1] call_id must be panicky");
                assert!(
                    *is_error,
                    "panicky call was interrupted — is_error must be true"
                );
                assert!(
                    content.contains("interrupted"),
                    "panicky call content must contain 'interrupted'; got {content:?}"
                );
            }
            UserBlock::Text(_) => panic!("expected ToolResult block 1; got Text"),
        }

        // PanickyTool must have been invoked exactly once (leg 1).
        // Resume does NOT re-execute it.
        assert_eq!(
            panicky_counter.load(Ordering::SeqCst),
            1,
            "panicky-tool must be invoked exactly once (leg 1 only, never re-executed on resume)"
        );
    }

    /// Builds the mid-turn snapshot + log shape shared by the entry-gate
    /// regression tests: a record whose last message is an assistant turn
    /// with the given tool calls, and a log of
    /// `[ModelCall, BudgetTick, Started(a), Result(a)]` — i.e. call `a`
    /// completed, and the log's LAST event is a `ToolCallResult`.
    async fn seed_result_tail_store(
        task_id: &str,
        calls: &[(&str, &str)], // (call_id, name)
    ) -> (RunRecord, Arc<dyn RunStore>, String) {
        let run_id = format!("{task_id}:1");
        let asst = Message::Assistant {
            content: calls
                .iter()
                .map(|(id, name)| {
                    ContentBlock::ToolCall(ToolCallRequest {
                        id: (*id).to_string(),
                        name: (*name).to_string(),
                        input: serde_json::json!({}),
                    })
                })
                .collect(),
        };
        let mut record = make_minimal_record(task_id, 1);
        record.messages = vec![
            Message::User {
                content: vec![UserBlock::Text("do the task".to_string())],
            },
            asst,
        ];
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        store
            .checkpoint(&run_id, &record)
            .await
            .expect("checkpoint");
        store
            .append_event(
                &run_id,
                Event::ModelCall {
                    seq: 0,
                    model: "t".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                },
            )
            .await
            .expect("mc");
        store
            .append_event(
                &run_id,
                Event::BudgetTick {
                    seq: 0,
                    consumed: BudgetConsumed::default(),
                },
            )
            .await
            .expect("bt");
        store
            .append_event(
                &run_id,
                Event::ToolCallStarted {
                    seq: 0,
                    name: calls[0].1.to_string(),
                    args: serde_json::json!({}),
                    call_id: calls[0].0.to_string(),
                },
            )
            .await
            .expect("started-a");
        store
            .append_event(
                &run_id,
                Event::ToolCallResult {
                    seq: 0,
                    name: calls[0].1.to_string(),
                    is_error: false,
                    summary: "real output a".to_string(),
                    offload_path: None,
                },
            )
            .await
            .expect("result-a");
        (record, store, run_id)
    }

    // Entry-gate regression (review follow-up): a crash can land BETWEEN a
    // ToolCallResult and the next ToolCallStarted. The log then ends in a
    // Result, but the snapshot still dangles (assistant turn, no results
    // batch). Gating on `events.last() == ToolCallStarted` skipped
    // reconciliation here and returned a transcript ending in unanswered
    // tool calls — malformed on every backend. The gate is snapshot-shape.
    #[tokio::test]
    async fn crash_between_result_and_next_started_reconciles() {
        let (record, store, run_id) =
            seed_result_tail_store("gate-task", &[("id-a", "work"), ("id-b", "work")]).await;
        // Log deliberately ends at Result(a): call b never reached Started.

        let msgs = super::reconcile_crash_tail(&record, &store, &run_id)
            .await
            .expect("reconcile");

        assert_eq!(msgs.len(), 3, "seed + asst + reconciled results batch");
        let Message::User { content } = &msgs[2] else {
            panic!("third message must be User");
        };
        assert_eq!(content.len(), 2, "one block per assistant tool call");
        match &content[0] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, "id-a");
                assert_eq!(
                    content, "real output a",
                    "completed call keeps its real result"
                );
                assert!(!is_error);
            }
            UserBlock::Text(_) => panic!("expected ToolResult"),
        }
        match &content[1] {
            UserBlock::ToolResult {
                call_id, is_error, ..
            } => {
                assert_eq!(call_id, "id-b");
                assert!(*is_error, "never-started call gets synthetic is_error");
            }
            UserBlock::Text(_) => panic!("expected ToolResult"),
        }
    }

    // Entry-gate regression, all-completed flavor: crash after the final
    // Result but before the end-of-iteration checkpoint. Every call has a
    // real logged result; reconciliation must rebuild the results batch with
    // no synthetic blocks and append no synthetic log events.
    #[tokio::test]
    async fn crash_after_all_results_before_iteration_checkpoint_reconciles() {
        let (record, store, run_id) =
            seed_result_tail_store("gate-task-2", &[("id-a", "work")]).await;
        let events_before = store.list_events(&run_id).await.expect("events").len();

        let msgs = super::reconcile_crash_tail(&record, &store, &run_id)
            .await
            .expect("reconcile");

        let Message::User { content } = &msgs[2] else {
            panic!("third message must be User");
        };
        assert_eq!(content.len(), 1);
        match &content[0] {
            UserBlock::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                assert_eq!(call_id, "id-a");
                assert_eq!(content, "real output a");
                assert!(!is_error, "completed call must NOT be marked interrupted");
            }
            UserBlock::Text(_) => panic!("expected ToolResult"),
        }
        let events_after = store.list_events(&run_id).await.expect("events").len();
        assert_eq!(
            events_before, events_after,
            "no synthetic event when nothing dangles"
        );
    }

    // Entry-gate clean cases: a snapshot ending in a User results batch (the
    // end-of-iteration checkpoint landed) or an assistant turn with NO tool
    // calls must pass through untouched.
    #[tokio::test]
    async fn clean_snapshots_are_not_reconciled() {
        let store: Arc<dyn RunStore> = Arc::new(SqliteRunStore::open_in_memory().expect("open"));
        let mut record = make_minimal_record("clean-task", 1);
        record.messages = vec![
            Message::User {
                content: vec![UserBlock::Text("do the task".to_string())],
            },
            Message::Assistant {
                content: vec![ContentBlock::Text("thinking out loud".to_string())],
            },
        ];
        let msgs = super::reconcile_crash_tail(&record, &store, "clean-task:1")
            .await
            .expect("reconcile");
        assert_eq!(msgs, record.messages, "no-tool-call turn passes through");
    }

    // Malformed-tail protection: reconcile_crash_tail must not panic or
    // index-out-of-bounds when the log contains:
    //   (a) a ToolCallResult with no preceding ToolCallStarted (orphaned result)
    //   (b) duplicate call_ids in ToolCallStarted events
    //
    // In both cases the reconciler must degrade safely and return a valid
    // (possibly all-synthetic) message list.
    #[allow(clippy::too_many_lines)]
    #[tokio::test]
    async fn reconcile_crash_tail_malformed_tail_no_panic() {
        // Build a record with one assistant tool call.
        let task_seed = Message::User {
            content: vec![UserBlock::Text("do the task".to_string())],
        };
        let asst = Message::Assistant {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "id-a".to_string(),
                name: "work".to_string(),
                input: serde_json::json!({}),
            })],
        };
        let mut record = make_minimal_record("malformed-task", 1);
        record.messages = vec![task_seed, asst];

        // ---- Case (a): orphaned ToolCallResult (no matching Started) -------
        // Log ends with ToolCallStarted (so dangling check fires) but before it
        // there's a ToolCallResult with no Started. The reconciler must not panic.
        {
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open_in_memory().expect("open"));
            store
                .checkpoint("malformed-task:1", &record)
                .await
                .expect("checkpoint");

            // Seed an orphaned ToolCallResult followed by a dangling Started.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ModelCall {
                        seq: 0,
                        model: "t".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                    },
                )
                .await
                .expect("mc");
            // Orphaned result — no Started before it.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallResult {
                        seq: 0,
                        name: "work".to_string(),
                        is_error: false,
                        summary: "orphaned".to_string(),
                        offload_path: None,
                    },
                )
                .await
                .expect("orphaned result");
            // Dangling Started — makes events.last() a ToolCallStarted.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallStarted {
                        seq: 0,
                        name: "work".to_string(),
                        args: serde_json::json!({}),
                        call_id: "id-a".to_string(),
                    },
                )
                .await
                .expect("dangling started");

            // Must not panic.
            let msgs = super::reconcile_crash_tail(&record, &store, "malformed-task:1")
                .await
                .expect("reconcile must not fail on orphaned result");

            // Should return task_seed + asst + reconciled user (all-synthetic for id-a).
            assert_eq!(
                msgs.len(),
                3,
                "must return 3 messages (seed + asst + reconciled)"
            );
            let Message::User { content } = &msgs[2] else {
                panic!("third message must be User");
            };
            assert_eq!(content.len(), 1, "one ToolResult block for id-a");
            match &content[0] {
                UserBlock::ToolResult {
                    call_id, is_error, ..
                } => {
                    assert_eq!(call_id, "id-a");
                    // The orphaned result is ignored; the dangling Started has
                    // call_id=id-a with no matching Result → synthetic is_error.
                    assert!(
                        *is_error,
                        "id-a must be synthetic is_error (dangling Started)"
                    );
                }
                UserBlock::Text(_) => panic!("expected ToolResult block"),
            }
        }

        // ---- Case (b): duplicate call_ids in ToolCallStarted ---------------
        {
            let store: Arc<dyn RunStore> =
                Arc::new(SqliteRunStore::open_in_memory().expect("open"));
            store
                .checkpoint("malformed-task:1", &record)
                .await
                .expect("checkpoint");

            store
                .append_event(
                    "malformed-task:1",
                    Event::ModelCall {
                        seq: 0,
                        model: "t".to_string(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                    },
                )
                .await
                .expect("mc");
            // First Started(id-a).
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallStarted {
                        seq: 0,
                        name: "work".to_string(),
                        args: serde_json::json!({}),
                        call_id: "id-a".to_string(),
                    },
                )
                .await
                .expect("started1");
            // Duplicate Started(id-a) — makes events.last() a ToolCallStarted.
            store
                .append_event(
                    "malformed-task:1",
                    Event::ToolCallStarted {
                        seq: 0,
                        name: "work".to_string(),
                        args: serde_json::json!({}),
                        call_id: "id-a".to_string(),
                    },
                )
                .await
                .expect("started2_duplicate");

            // Must not panic.
            let msgs = super::reconcile_crash_tail(&record, &store, "malformed-task:1")
                .await
                .expect("reconcile must not fail on duplicate call_id");

            assert_eq!(msgs.len(), 3, "must return 3 messages");
            let Message::User { content } = &msgs[2] else {
                panic!("third message must be User");
            };
            assert_eq!(content.len(), 1, "one ToolResult block for id-a");
            match &content[0] {
                UserBlock::ToolResult {
                    call_id, is_error, ..
                } => {
                    assert_eq!(call_id, "id-a");
                    assert!(
                        *is_error,
                        "id-a must be synthetic is_error with duplicate Started"
                    );
                }
                UserBlock::Text(_) => panic!("expected ToolResult block"),
            }
        }
    }

    // =====================================================================
    // Finish-recovery protocol tests (detection, nudge, recovery terminal)
    // =====================================================================

    /// The exact nudge wording the harness injects — pinned by a test so a
    /// wording pass that drifts fails loudly. Must match
    /// `crates/harness/templates/nudge_prompt.md` verbatim.
    const NUDGE_TEXT: &str = "The quality gates are currently green. \
        If the acceptance criteria are met, call `finish(done)` now. \
        If they are not yet met, reply with a one-sentence status: \
        what remains, and why you are still working.";

    /// Count how many `UserBlock::Text` blocks in `messages` carry the nudge
    /// text — the harness injects the nudge by APPENDING onto the existing
    /// tool-results `Message::User`, so this counts nudge injections (not new
    /// messages).
    fn count_nudge_injections(messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| match m {
                Message::User { content } => content
                    .iter()
                    .filter(|b| matches!(b, UserBlock::Text(t) if t == NUDGE_TEXT))
                    .count(),
                Message::Assistant { .. } => 0,
            })
            .sum()
    }

    /// Assert no two adjacent `Message::User` appear in `messages` — the
    /// nudge must be APPENDED onto the existing tool-results `Message::User`,
    /// never pushed as a new `Message::User` (which would hit the Anthropic
    /// wire as two `role:"user"` blocks and 400).
    fn assert_no_adjacent_user_messages(messages: &[Message]) {
        for i in 0..messages.len().saturating_sub(1) {
            let a = matches!(messages[i], Message::User { .. });
            let b = matches!(messages[i + 1], Message::User { .. });
            assert!(
                !(a && b),
                "two adjacent Message::User at indices {i} and {} — nudge \
                 must be appended, not pushed as a new message",
                i + 1
            );
        }
    }

    /// A turn that calls `run_checks` (the done-oracle). Used to drive the
    /// green/red gate signal in detection tests.
    fn run_checks_turn(id: &str) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, "run_checks", serde_json::json!({}))],
            StopReason::ToolUse,
        )
    }

    /// A turn that emits a one-sentence status text and a non-mutating echo
    /// call — the post-nudge "status reply" (no green re-observation).
    fn status_echo_turn(id: &str, status: &str) -> AssistantTurn {
        turn_with(
            vec![
                ContentBlock::Text(status.to_string()),
                tool_call(id, "echo", serde_json::json!({})),
            ],
            StopReason::ToolUse,
        )
    }

    /// A non-mutating echo turn (no text — `turn.text()` is empty).
    fn echo_turn(id: &str) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, "echo", serde_json::json!({}))],
            StopReason::ToolUse,
        )
    }

    // AC-1 detection test: green run_checks then K static non-mutating
    // iterations -> exactly one nudge appended (pinned text appears once,
    // no two adjacent Message::User).
    #[tokio::test]
    async fn green_static_spin_injects_exactly_one_nudge_with_pinned_text() {
        // K = DEFAULT_STATIC_TREE_K = 3. The counter increments every
        // non-mutating iteration; the run_checks iteration counts as one.
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: echo — iters 1->2
        //   iter 3: echo — iters 2->3 -> trip, nudge injected (nudges_fired 0->1)
        //   iter 4: echo — iters 0->1 (post-nudge; nudge_awaiting_status cleared)
        //   max_iterations=4 -> MaxIterations
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 4)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        // The nudge is NOT a terminal — after one nudge the loop continues and
        // hits MaxIterations (no finish, cap reached).
        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "one-nudge spin should hit MaxIterations, not the recovery terminal; got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        let nudge_count = count_nudge_injections(&rec.messages);
        assert_eq!(
            nudge_count, 1,
            "exactly one nudge should be injected; got {nudge_count}"
        );
        assert_no_adjacent_user_messages(&rec.messages);
        // recovery_facts is only written on the FinishDiscipline terminal,
        // which was NOT taken here.
        assert_eq!(rec.recovery_facts, None);
    }

    // AC-2 detection test: finish(done) on the turn after a nudge -> the run
    // terminates as a NORMAL Finished(Done{..}) through the existing
    // handle_finish_call path. The recovery terminal is NOT taken, and
    // recovery_facts stays None (a Done-after-nudge is a clean success).
    #[tokio::test]
    async fn finish_done_after_nudge_terminates_normally_without_recovery() {
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: echo — iters 1->2
        //   iter 3: echo — iters 2->3 -> trip, nudge injected. nudge_awaiting_status=true.
        //   iter 4: finish(done) — accepted (green checks). nudge_awaiting_status
        //          cleared WITHOUT pushing (is_done). Terminal: Finished(Done).
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            finish_call(
                "c4",
                serde_json::json!({ "disposition": "done", "summary": "done after nudge" }),
            ),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        match outcome {
            LoopOutcome::Finished(Disposition::Done { summary, .. }) => {
                assert_eq!(summary, "done after nudge");
            }
            other => panic!("expected Finished(Done), got {other:?}"),
        }

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(rec.recovery_facts, None);
        assert!(
            matches!(rec.disposition, Some(Disposition::Done { .. })),
            "persisted disposition must be Done; got {:?}",
            rec.disposition
        );
        // The nudge was injected (once) before the finish.
        assert_eq!(count_nudge_injections(&rec.messages), 1);
    }

    // AC-3 detection test: N nudges then a further K static-green iterations
    // with no finish -> LoopOutcome::Finished(Disposition::Failed{
    // FinishDiscipline }) and record.recovery_facts == Some(RecoveryFacts{
    // gates_green_at_exit: true, tree_dirty: <reflects edits>, nudge_statuses:
    // <non-empty> }).
    #[tokio::test]
    async fn n_nudges_then_static_green_spin_takes_recovery_terminal() {
        // N = DEFAULT_MAX_NUDGES = 2, K = DEFAULT_STATIC_TREE_K = 3.
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: echo — iters 1->2
        //   iter 3: echo — iters 2->3 -> trip, nudge 1. iters=0. awaiting=true.
        //   iter 4: status+echo — push status. iters 0->1. awaiting=false.
        //   iter 5: echo — iters 1->2.
        //   iter 6: echo — iters 2->3 -> trip, nudge 2. iters=0. awaiting=true.
        //   iter 7: status+echo — push status. iters 0->1. awaiting=false.
        //   iter 8: echo — iters 1->2.
        //   iter 9: echo — iters 2->3 -> trip. nudges_fired(2)==max(2) -> RECOVERY TERMINAL.
        //   max_iterations=10 (never reached — recovery returns at iter 9).
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            status_echo_turn("c4", "still writing the failing-case test"),
            echo_turn("c5"),
            echo_turn("c6"),
            status_echo_turn("c7", "still fixing the off-by-one"),
            echo_turn("c8"),
            echo_turn("c9"),
            // iter 10 would over-draw; recovery returns at iter 9 before this.
            echo_turn("c10"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        match outcome {
            LoopOutcome::Finished(Disposition::Failed { mode, summary }) => {
                assert_eq!(
                    mode,
                    FailureMode::FinishDiscipline,
                    "recovery terminal must be FinishDiscipline; got {mode:?}"
                );
                assert!(
                    summary.contains("gates green but agent did not call finish"),
                    "summary must name the recovery; got {summary}"
                );
                assert!(
                    summary.contains("2 nudges"),
                    "summary must name the nudge count; got {summary}"
                );
            }
            other => panic!("expected Finished(Failed{{FinishDiscipline}}), got {other:?}"),
        }

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        let facts = rec
            .recovery_facts
            .as_ref()
            .expect("recovery_facts must be Some on the FinishDiscipline terminal");
        assert!(
            facts.gates_green_at_exit,
            "gates were green at the recovery terminal"
        );
        assert!(
            !facts.tree_dirty,
            "no edit_file/bash was called in this script, so tree_dirty must be false"
        );
        assert!(
            !facts.nudge_statuses.is_empty(),
            "nudge_statuses must be non-empty (one status per nudge that didn't finish)"
        );
        assert_eq!(
            facts.nudge_statuses.len(),
            2,
            "exactly two status replies (one per nudge); got {:?}",
            facts.nudge_statuses
        );
        assert_eq!(count_nudge_injections(&rec.messages), 2);
        assert_no_adjacent_user_messages(&rec.messages);
    }

    // AC-4 detection test: a successful edit_file between green-static iters
    // resets iters_since_tree_change AND clears last_gate_green, so no nudge
    // fires until a fresh green run_checks is observed post-edit.
    #[tokio::test]
    async fn edit_file_resets_counter_and_clears_green_until_fresh_run_checks() {
        //   iter 1: run_checks (green) — iters 0->1
        //   iter 2: edit_file (success) — mutated=true, tree_dirty=true,
        //           last_gate_green=false. iters reset to 0.
        //   iter 3: echo — iters 0->1. green=false -> no trip.
        //   iter 4: echo — iters 1->2. green=false -> no trip.
        //   iter 5: echo — iters 2->3. green=false -> no trip (stale-green window closed).
        //   iter 6: run_checks (green) — last_gate_green=true. iters 3->4. trip! nudge.
        //   iter 7: echo — iters 0->1 (post-nudge).
        //   max_iterations=7 -> MaxIterations
        let root = TempDir::new().expect("workspace tempdir");
        let root_path = root.path().canonicalize().expect("canonicalize root");
        let workspace = Workspace::new(&root_path, None).expect("workspace");
        let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink));

        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let config = RunConfig::new("do the task", 7)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            turn_with(
                vec![tool_call(
                    "c2",
                    "edit_file",
                    serde_json::json!({
                        "path": "flag",
                        "old_string": "",
                        "new_string": "planted\n",
                    }),
                )],
                StopReason::ToolUse,
            ),
            echo_turn("c3"),
            echo_turn("c4"),
            echo_turn("c5"),
            run_checks_turn("c6"),
            echo_turn("c7"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "expected MaxIterations (only one nudge, no recovery); got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        // Exactly one nudge — fired at iter 6 (the fresh green run_checks),
        // NOT at iter 5 (where iters=3 but green=false after the edit).
        assert_eq!(
            count_nudge_injections(&rec.messages),
            1,
            "exactly one nudge, fired only after the fresh green run_checks"
        );
        assert_no_adjacent_user_messages(&rec.messages);
        assert_eq!(rec.recovery_facts, None);
        assert!(
            root_path.join("flag").exists(),
            "edit_file must have created the flag file"
        );
    }

    // AC-5 detection test: RED run_checks + spin does NOT nudge and
    // terminates MaxIterations.
    #[tokio::test]
    async fn red_run_checks_spin_does_not_nudge_and_terminates_max_iterations() {
        //   iter 1: run_checks (RED) — last_gate_green=false. iters 0->1. no trip.
        //   iter 2-5: echo — iters 1->2->3->4->5. green=false -> never trips.
        //   max_iterations=5 -> MaxIterations.
        let runner = failing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 5)
            .with_checks(runner)
            .with_max_nudges(2);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
            echo_turn("c5"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "red-gate spin must fall through to MaxIterations; got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(
            count_nudge_injections(&rec.messages),
            0,
            "red gate must never inject a nudge"
        );
        assert_eq!(rec.recovery_facts, None);
    }

    // AC-6 detection test: max_nudges == 0 disables the feature entirely —
    // no nudge is ever injected and the recovery terminal is never taken;
    // a green-gates + static-tree spin falls through to MaxIterations.
    #[tokio::test]
    async fn max_nudges_zero_disables_finish_recovery_entirely() {
        //   iter 1: run_checks (green) — iters 0->1. max_nudges=0 -> no trip.
        //   iter 2-4: echo — iters 1->2->3->4. no trip (feature off).
        //   max_iterations=4 -> MaxIterations.
        let runner = passing_runner();
        let tools = standard_registry(Some(runner.clone()));
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 4)
            .with_checks(runner)
            .with_max_nudges(0);

        let backend = MockBackend::from_turns(vec![
            run_checks_turn("c1"),
            echo_turn("c2"),
            echo_turn("c3"),
            echo_turn("c4"),
        ]);

        let snap_store = Arc::new(SnapshotStore::new());
        let pers = make_persistence(snap_store.clone());
        let RunResult { outcome, .. } = run_persisted(&backend, &tools, &ctx, &config, &pers)
            .await
            .expect("no error");

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "max_nudges=0 must fall through to MaxIterations; got {outcome:?}"
        );

        let rec = snap_store
            .inner
            .load(FIXTURE_RID)
            .await
            .expect("load")
            .expect("present");
        assert_eq!(
            count_nudge_injections(&rec.messages),
            0,
            "max_nudges=0 must never inject a nudge"
        );
        assert_eq!(rec.recovery_facts, None);
    }

    // ---- retry / backoff tests ------------------------------------------

    /// Load-bearing unit test: pins the exponential schedule with a NON-zero
    /// base so the shape is actually tested. The async retry tests all use
    /// `base = Duration::ZERO` and cannot discriminate between exponential,
    /// linear, or constant schedules.
    #[test]
    fn retry_delay_schedule_is_exponential() {
        assert_eq!(
            retry_delay(Duration::from_millis(500), 0),
            Duration::from_millis(500)
        );
        assert_eq!(
            retry_delay(Duration::from_millis(500), 1),
            Duration::from_secs(1)
        );
        assert_eq!(
            retry_delay(Duration::from_millis(500), 2),
            Duration::from_secs(2)
        );
    }

    /// A single transient error followed by a success completes the run.
    /// Verifies: `backend.calls() == 2`, `stats.iterations == 1`, and
    /// `outcome == LoopOutcome::Finished(Disposition::Done{..})`.
    #[tokio::test]
    async fn transient_then_success_proceeds() {
        let transient = BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        };
        let done_turn = finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "all good" }),
        );

        let backend = MockBackend::new(vec![Err(transient), Ok(done_turn)]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 2, "one transient + one success = 2 draws");
        assert_eq!(stats.iterations, 1, "single logical iteration");
        match outcome {
            LoopOutcome::Finished(Disposition::Done { verification, .. }) => {
                assert_eq!(
                    verification,
                    Verification::NoChecksConfigured,
                    "no checks configured"
                );
            }
            other => panic!("expected Finished(Done), got {other:?}"),
        }
    }

    /// A retried turn is counted as ONE logical iteration even when
    /// `backend.turn` was called twice.
    #[tokio::test]
    async fn iteration_counted_once_across_retried_turn() {
        let transient = BackendError::Transient {
            kind: TransientKind::RateLimit,
            retry_after: None,
        };
        // Plain turn with no tool calls → StoppedWithoutFinish.
        let no_tools_turn = turn_with(
            vec![ContentBlock::Text("thinking...".to_string())],
            StopReason::EndTurn,
        );

        let backend = MockBackend::new(vec![Err(transient), Ok(no_tools_turn)]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 2, "one transient + one success = 2 draws");
        assert_eq!(
            stats.iterations, 1,
            "retry does NOT double-count the iteration"
        );
        assert!(
            matches!(outcome, LoopOutcome::StoppedWithoutFinish),
            "expected StoppedWithoutFinish, got {outcome:?}"
        );
    }

    /// When all retries are exhausted the loop gives up and returns
    /// `LoopOutcome::BackendError` carrying the last attempt's error, with
    /// `stats.iterations == 1` (all retries stay inside iteration 1).
    #[tokio::test]
    async fn consecutive_transients_exhaust_and_give_up_as_transient_infra() {
        // max_retries = 3 (default) → 4 total attempts.
        let script: Vec<Result<AssistantTurn, BackendError>> = (0..4)
            .map(|_| {
                Err(BackendError::Transient {
                    kind: TransientKind::RateLimit,
                    retry_after: None,
                })
            })
            .collect();

        let backend = MockBackend::new(script);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 4, "1 first try + 3 retries = 4 draws");
        assert_eq!(stats.iterations, 1, "all retries stay inside iteration 1");
        // The outcome must be BackendError carrying a Transient error.
        match &outcome {
            LoopOutcome::BackendError(BackendError::Transient { .. }) => {}
            other => panic!("expected BackendError(Transient), got {other:?}"),
        }
        // into_disposition maps a retryable final error to TransientInfra.
        match outcome.into_disposition() {
            Disposition::Failed {
                mode: FailureMode::TransientInfra,
                ..
            } => {}
            other => panic!("expected Failed(TransientInfra), got {other:?}"),
        }
    }

    /// Terminal errors are not retried — `backend.calls() == 1`.
    #[tokio::test]
    async fn terminal_fails_on_first_occurrence_no_retry() {
        let backend = MockBackend::new(vec![Err(BackendError::Terminal {
            kind: TerminalKind::Other,
            message: "bad request".to_string(),
        })]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 1, "terminal error: no retries");
        assert_eq!(stats.iterations, 1);
        match outcome.into_disposition() {
            Disposition::Failed {
                mode: FailureMode::PersistentToolError,
                ..
            } => {}
            other => panic!("expected Failed(PersistentToolError), got {other:?}"),
        }
    }

    /// `ContextLengthExceeded` is NOT retried (deferred out of 0.4.0) —
    /// it maps to `PersistentToolError` on first occurrence.
    #[tokio::test]
    async fn context_length_exceeded_not_retried() {
        let backend = MockBackend::new(vec![Err(BackendError::ContextLengthExceeded)]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();
        let config = RunConfig::new("do the task", 10).with_retry_backoff_base(Duration::ZERO);

        let RunResult { outcome, stats } = run(&backend, &tools, &ctx, &config).await;

        assert_eq!(backend.calls(), 1, "ContextLengthExceeded: no retries");
        assert_eq!(stats.iterations, 1);
        match outcome.into_disposition() {
            Disposition::Failed {
                mode: FailureMode::PersistentToolError,
                ..
            } => {}
            other => panic!("expected Failed(PersistentToolError), got {other:?}"),
        }
    }
}
