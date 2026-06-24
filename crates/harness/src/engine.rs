//! The minimal agent loop: drive a [`model::ModelBackend`] and a
//! [`ToolRegistry`] through a conversation until the agent calls the `finish`
//! tool or a hard iteration cap is hit.
//!
//! This is the **thin conversational spine** — deliberately *not* the whole
//! harness. What lives here:
//!
//! - [`run`] — the loop itself, generic over any [`model::ModelBackend`].
//! - [`LoopOutcome`] — the four ways the loop can end.
//! - [`FinishTool`] / [`FinishDisposition`] — the first concrete tool and the
//!   loop-local parse of how the agent declared it was done.
//!
//! What does **not** live here yet (tracked separately): budget / token /
//! wall-clock bounds, retry / backoff, loop / no-progress detection,
//! persistence / checkpointing, context assembly / compaction, and wiring to
//! [`crate::run_record::Disposition`]. The only stopping condition beyond the
//! agent finishing is the hard `max_iterations` cap.
//!
//! ## Loop shape
//!
//! 1. Seed history with the task as a [`Message::User`].
//! 2. Each iteration: build a [`TurnRequest`] (system prompt, history, tool
//!    schemas, sampling params) and call [`model::ModelBackend::turn`].
//! 3. Append the assistant turn to history (via `From<AssistantTurn>`).
//! 4. If the turn made no tool calls, stop ([`LoopOutcome::StoppedWithoutFinish`]).
//! 5. Otherwise execute each call in order, collect the rendered results into a
//!    single [`Message::User`], and append it.
//! 6. If any executed call was the `finish` tool, stop
//!    ([`LoopOutcome::Finished`]) with the parsed [`FinishDisposition`].
//! 7. If the cap is reached first, stop ([`LoopOutcome::MaxIterations`]).
//!
//! A backend error is surfaced immediately as [`LoopOutcome::BackendError`] —
//! the loop does **not** retry; retry / backoff lands with the budget work.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::model::{self, Message, SamplingParams, ToolCallRequest, TurnRequest, UserBlock};
use crate::tool::{Tool, ToolCtx, ToolRegistry, ToolResult};

/// The registered name of the finish tool — the loop recognizes termination by
/// matching an executed call's name against this.
pub const FINISH_TOOL_NAME: &str = "finish";

/// A sensible default per-turn output cap for the loop's [`SamplingParams`].
/// Budget-aware sizing lands with the budget work; this is a fixed value for
/// the thin slice.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// How the agent declared the run finished, parsed from the `finish` tool's
/// input.
///
/// This is **loop-local** and intentionally distinct from
/// [`crate::run_record::Disposition`] — wiring the two together is a later
/// item. The discriminator mirrors the run-record one ("does running the same
/// thing again have any chance of working?") but carries only what the thin
/// slice needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishDisposition {
    /// The agent reports the task done.
    Done,
    /// The agent is blocked: the spec or environment is the problem and a
    /// human decision is needed before retrying.
    Blocked { decision_needed: String },
    /// The agent failed: the run is the problem (retrying might work).
    Failed { summary: String },
}

impl FinishDisposition {
    /// Parse the `finish` tool's raw JSON input into a disposition.
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
            Some("done") => Self::Done,
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
    /// The agent called the `finish` tool. Carries the parsed disposition.
    Finished(FinishDisposition),
    /// A turn produced no tool calls (the model ended its turn without
    /// finishing). The loop has nothing to feed back, so it stops.
    StoppedWithoutFinish,
    /// The hard `max_iterations` cap was reached before the agent finished.
    MaxIterations,
    /// The backend returned an error. Surfaced as-is — the loop does not
    /// retry (retry / backoff lands with the budget work).
    BackendError(model::BackendError),
}

/// The first concrete tool: the agent calls `finish` to end the run.
///
/// Its input is `{ disposition: "done" | "blocked" | "failed", summary:
/// string, decision_needed?: string }`. [`Tool::run`] just acknowledges the
/// call with an ok [`ToolResult`]; the loop is what recognizes the name and
/// parses the input into a [`FinishDisposition`] to terminate.
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
                            blocked on a decision, or has failed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "disposition": {
                        "type": "string",
                        "enum": ["done", "blocked", "failed"],
                        "description": "done = task complete; blocked = needs a decision \
                                        before retrying; failed = the run is the problem."
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

/// Drive `backend` + `tools` through a conversation until the agent finishes or
/// the `max_iterations` cap is hit.
///
/// - `system` is the system prompt (rides on the [`TurnRequest`], not as a
///   message).
/// - `task` seeds conversation history as the first user message.
/// - `ctx` is the per-run [`ToolCtx`] threaded into every tool invocation.
///
/// See the module docs for the full loop shape and the [`LoopOutcome`]
/// termination cases.
pub async fn run(
    backend: &impl model::ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    system: &str,
    task: &str,
    max_iterations: u32,
) -> LoopOutcome {
    let mut messages: Vec<Message> = vec![Message::User {
        content: vec![UserBlock::Text(task.to_string())],
    }];
    let tool_schemas = tools.list();
    let params = SamplingParams {
        max_tokens: DEFAULT_MAX_TOKENS,
        temperature: None,
        stop_sequences: Vec::new(),
    };

    for _ in 0..max_iterations {
        let req = TurnRequest {
            system: Some(system),
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
        let calls: Vec<ToolCallRequest> = turn.tool_calls().into_iter().cloned().collect();
        messages.push(Message::from(turn));

        if calls.is_empty() {
            return LoopOutcome::StoppedWithoutFinish;
        }

        // Execute every requested call in order, collecting the fed-back tool
        // results into a single user message.
        let mut results = Vec::with_capacity(calls.len());
        let mut finish: Option<FinishDisposition> = None;
        for call in &calls {
            let result = tools.invoke(&call.name, call.input.clone(), ctx).await;
            results.push(UserBlock::ToolResult {
                call_id: call.id.clone(),
                content: render_tool_result(&result),
                is_error: result.is_error,
            });
            if call.name == FINISH_TOOL_NAME && finish.is_none() {
                finish = Some(FinishDisposition::from_input(&call.input));
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
        FINISH_TOOL_NAME, FinishDisposition, FinishTool, LoopOutcome, render_tool_result, run,
    };
    use crate::model::{
        AssistantTurn, BackendError, ContentBlock, Message, StopReason, TerminalKind,
        ToolCallRequest, Usage, UserBlock,
    };
    use crate::test_support::MockBackend;
    use crate::tool::{EchoTool, Tool, ToolCtx, ToolRegistry, ToolResult};
    use std::sync::Arc;

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

    #[tokio::test]
    async fn single_finish_done_terminates_in_one_iteration() {
        let backend = MockBackend::from_turns(vec![finish_call(
            "c1",
            serde_json::json!({ "disposition": "done", "summary": "all set" }),
        )]);
        let tools = registry_with_finish_and_echo();
        let ctx = ToolCtx::stub();

        let outcome = run(&backend, &tools, &ctx, "be a harness", "do the task", 10).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(FinishDisposition::Done)
        ));
        assert_eq!(backend.calls(), 1, "should finish in a single iteration");
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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

        assert!(matches!(
            outcome,
            LoopOutcome::Finished(FinishDisposition::Done)
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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;
        assert!(matches!(
            outcome,
            LoopOutcome::Finished(FinishDisposition::Done)
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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 3).await;

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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

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

        let outcome = run(&backend, &tools, &ctx, "sys", "task", 10).await;

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
}
