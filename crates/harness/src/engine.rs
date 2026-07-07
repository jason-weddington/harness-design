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
//! - [`FinishDisposition`] / [`Verification`] — the parsed decision, plus the
//!   evidence attached to a verified [`FinishDisposition::Done`].
//!
//! What does **not** live here yet (tracked separately): budget / token /
//! wall-clock bounds, retry / backoff, loop / no-progress detection,
//! persistence / checkpointing, context assembly / compaction, and wiring to
//! [`crate::run_record::Disposition`]. The only stopping condition beyond the
//! agent finishing is the hard `max_iterations` cap.
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

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::exec::{CheckReport, ChecksRunner};
use crate::model::{self, Message, SamplingParams, TurnRequest, UserBlock};
use crate::prompt;
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
    /// `None` means no automated verification — see [`Verification::NoChecksConfigured`].
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

/// The evidence the loop attaches to a [`FinishDisposition::Done`] — the
/// justification for accepting the claim.
///
/// The loop is the ONLY constructor of [`FinishDisposition::Done`] and it
/// only constructs it after a green harness-run verification (or explicitly
/// records [`Self::NoChecksConfigured`] when the run had no checks wired in).
/// This makes the "Done carries evidence" property a type-system invariant,
/// not a documentation convention.
#[derive(Debug, Clone, PartialEq)]
pub enum Verification {
    /// The harness re-ran the configured checks and they passed. The full
    /// [`CheckReport`] is attached as the evidence.
    Checks(CheckReport),
    /// No checks were configured for this run — there was nothing to verify
    /// against, so the loop accepted the `done` claim on trust.
    NoChecksConfigured,
}

/// How the agent declared the run finished, parsed from the `finish` tool's
/// input.
///
/// This is **loop-local** and intentionally distinct from
/// [`crate::run_record::Disposition`] — wiring the two together is a later
/// item. The discriminator mirrors the run-record one ("does running the same
/// thing again have any chance of working?") but carries only what this slice
/// needs.
///
/// [`Self::Done`] deliberately carries a [`Verification`]: the loop is the
/// only constructor, and it only produces a `Done` after verifying the claim
/// (or explicitly recording that there were no checks to verify against). A
/// `Done` value in a [`LoopOutcome::Finished`] therefore carries the evidence
/// that justifies it — you cannot get a `Done` without evidence in this API.
///
/// [`Eq`] is not derived because [`Verification::Checks`] wraps a
/// [`CheckReport`], which is [`PartialEq`] but not [`Eq`].
#[derive(Debug, Clone, PartialEq)]
pub enum FinishDisposition {
    /// The agent reports the task done AND the harness has accepted the
    /// claim: either the configured checks came back green
    /// ([`Verification::Checks`]) or there were no checks to verify against
    /// ([`Verification::NoChecksConfigured`]).
    Done {
        /// The short summary the model produced for the outcome.
        summary: String,
        /// The evidence that justifies accepting the claim.
        verification: Verification,
    },
    /// The agent is blocked: the spec or environment is the problem and a
    /// human decision is needed before retrying. Not verified — a model
    /// declaring it can't proceed needs no proof.
    Blocked { decision_needed: String },
    /// The agent failed: the run is the problem (retrying might work). Not
    /// verified — same rationale as [`Self::Blocked`].
    Failed { summary: String },
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

/// Why the agent loop stopped.
///
/// Not `PartialEq` because [`model::BackendError`] is a runtime error type that
/// doesn't compare; tests match on the variant instead.
#[derive(Debug)]
pub enum LoopOutcome {
    /// The agent called the `finish` tool AND the harness accepted the
    /// claim. Carries the post-verification [`FinishDisposition`]; a `Done`
    /// here has evidence by construction (see [`Verification`]).
    Finished(FinishDisposition),
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
/// the loop should terminate — the accepted [`FinishDisposition`].
///
/// Extracted out of [`run`] so the main loop stays readable; keeps the
/// per-call plumbing (call id, `is_error`, content wording) in one place.
struct FinishOutcome {
    result: UserBlock,
    finish: Option<FinishDisposition>,
}

/// Handle a `finish` call: verify a `done` claim against `config.checks`
/// when configured, or accept it on trust when not. `blocked` and `failed`
/// terminate as declared with no verification.
///
/// Returns the fed-back [`UserBlock::ToolResult`] plus, when the loop should
/// terminate, the accepted [`FinishDisposition`]. A rejected `done` returns
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
                        finish: Some(FinishDisposition::Done {
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
                finish: Some(FinishDisposition::Done {
                    summary,
                    verification: Verification::NoChecksConfigured,
                }),
            },
        },
        FinishClaim::Blocked { decision_needed } => FinishOutcome {
            result: ack(call_id),
            finish: Some(FinishDisposition::Blocked { decision_needed }),
        },
        FinishClaim::Failed { summary } => FinishOutcome {
            result: ack(call_id),
            finish: Some(FinishDisposition::Failed { summary }),
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
/// [`ChecksRunner::run`]: crate::exec::ChecksRunner::run
pub async fn run(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    config: &RunConfig,
) -> LoopOutcome {
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

    for _ in 0..config.max_iterations {
        let req = TurnRequest {
            system: Some(&system),
            messages: &messages,
            tools: &tool_schemas,
            params: &params,
        };

        let turn = match backend.turn(&req).await {
            Ok(turn) => turn,
            Err(err) => return LoopOutcome::BackendError(err),
        };

        // Snapshot the calls before moving the turn into history (the `From`
        // impl consumes `turn.content`).
        let calls: Vec<_> = turn.tool_calls().into_iter().cloned().collect();
        messages.push(Message::from(turn));

        if calls.is_empty() {
            return LoopOutcome::StoppedWithoutFinish;
        }

        // Execute every requested call in order, collecting fed-back tool
        // results into a single user message. `finish` is special-cased in
        // [`handle_finish_call`]: a `done` claim triggers the harness
        // re-running the configured checks, and only a green verification
        // (or no checks at all) sets the terminal `finish` slot. A red
        // verification is fed back as an `is_error=true` result and the loop
        // CONTINUES with the remaining calls in the same batch.
        let mut results = Vec::with_capacity(calls.len());
        let mut finish: Option<FinishDisposition> = None;
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
                let result = tools.invoke(&call.name, call.input.clone(), ctx).await;
                results.push(UserBlock::ToolResult {
                    call_id: call.id.clone(),
                    content: render_tool_result(&result),
                    is_error: result.is_error,
                });
            }
        }
        messages.push(Message::User { content: results });

        if let Some(disposition) = finish {
            return LoopOutcome::Finished(disposition);
        }
    }

    LoopOutcome::MaxIterations
}

#[cfg(test)]
mod tests {
    use super::{
        FINISH_TOOL_NAME, FinishDisposition, FinishTool, LoopOutcome, RunConfig, Verification,
        rejection_content, render_tool_result, run,
    };
    use crate::exec::{CheckCommand, CheckReport, ChecksRunner};
    use crate::model::{
        AssistantTurn, BackendError, ContentBlock, Message, StopReason, TerminalKind,
        ToolCallRequest, Usage, UserBlock,
    };
    use crate::test_support::MockBackend;
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use crate::tools::edit_file::EditFileTool;
    use crate::workspace::Workspace;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::TempDir;

    fn usage() -> Usage {
        Usage {
            input_tokens: 0,
            output_tokens: 0,
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Done {
                summary,
                verification: Verification::NoChecksConfigured,
            }) => {
                assert_eq!(summary, "all set");
            }
            other => panic!("expected Finished(Done{{NoChecksConfigured}}), got {other:?}"),
        }
        assert_eq!(backend.calls(), 1, "should finish in a single iteration");
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Done {
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        assert!(
            matches!(outcome, LoopOutcome::MaxIterations),
            "repeated red claims must hit MaxIterations; got {outcome:?}"
        );
        assert_eq!(
            backend.calls(),
            2,
            "drew exactly max_iterations turns before giving up"
        );
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "which API version?");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Failed { summary }) => {
                assert_eq!(summary, "tool kept erroring");
            }
            other => panic!("expected Finished(Failed), got {other:?}"),
        }
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Done {
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

        let outcome = run(&backend, &tools, &ctx, &config).await;
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(FinishDisposition::Done { .. })
        ));
        // Two model turns: echo, then finish.
        assert_eq!(backend.calls(), 2, "echo turn then finish turn");
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

        let outcome = run(&backend, &tools, &ctx, &config).await;
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(FinishDisposition::Done { .. })
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::MaxIterations));
        assert_eq!(backend.calls(), 3, "drew exactly max_iterations turns");
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        // Surfaced on the first call — no retry.
        assert_eq!(backend.calls(), 1);
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::StoppedWithoutFinish));
        assert_eq!(backend.calls(), 1);
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        assert!(matches!(outcome, LoopOutcome::BackendError(_)));
        assert_eq!(backend.calls(), 2, "second draw over-draws the script");
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Blocked { decision_needed }) => {
                assert_eq!(decision_needed, "which API version?");
            }
            other => panic!("expected Finished(Blocked), got {other:?}"),
        }
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Failed { summary }) => {
                assert_eq!(summary, "tool kept erroring");
            }
            other => panic!("expected Finished(Failed), got {other:?}"),
        }
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

        let outcome = run(&backend, &tools, &ctx, &config).await;

        match outcome {
            LoopOutcome::Finished(FinishDisposition::Failed { summary }) => {
                assert_eq!(summary, "huh");
            }
            other => panic!("expected Finished(Failed) for unknown disposition, got {other:?}"),
        }
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
}
