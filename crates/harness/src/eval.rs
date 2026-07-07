//! Eval harness: the measuring rig wrapped around the agent loop.
//!
//! An [`EvalTask`] is a task prompt plus a predicate over the loop's terminal
//! [`LoopOutcome`] that decides whether a trial passed. [`run_eval`] runs `k`
//! independent trials against a [`model::ModelBackend`] and reports the count
//! of passes and the derived rate as an [`EvalReport`].
//!
//! ## Why this exists
//!
//! The point is to make every future loop / model / prompt change *measurable*:
//! once a task has a predicate and a baseline pass-rate at `k`, any
//! regression shows up as a number, not a feeling.
//!
//! ## Pass^k framing
//!
//! `EvalReport::passes` is the count of trials that satisfied the predicate
//! out of `EvalReport::trials = k`; `EvalReport::pass_rate = passes / k` is
//! the comparable number when `k` changes between runs. The name "pass^k" is
//! field-of-evals shorthand for "we ran k trials and counted passes" — that
//! is exactly the math here.
//!
//! ## System prompt sourcing
//!
//! The `EvalTask` deliberately does NOT carry a `system` field: the canonical
//! system prompt is now rendered by [`crate::prompt`] inside the engine, from
//! the registered tool set (plus the check command, when configured). Every
//! trial and every real run render the same prompt from the same source of
//! truth — a wording drift between eval and prod is impossible.
//!
//! ## What this is NOT (deferred)
//!
//! - **Clean-worktree-per-trial isolation is deliberately deferred.** The
//!   harness has no code-editing tools yet, so trials are stateless — the
//!   only stochastic surface is the backend itself. When code-editing tools
//!   land, each trial will need a fresh worktree before it runs.
//! - Real code-editing eval tasks, statistical rigor beyond pass^k counting,
//!   multi-provider / model-routing eval, and CI wiring of the live eval all
//!   land later.

use crate::engine::{self, FinishDisposition, LoopOutcome, RunConfig};
use crate::model::ModelBackend;
use crate::tool::{ToolCtx, ToolRegistry};

/// Predicate that judges whether a single trial's [`LoopOutcome`] is a pass.
///
/// A boxed `Fn` rather than a generic parameter keeps [`EvalTask`] storable in
/// a `Vec` and constructible by helpers like [`finish_task`]. The `Send +
/// Sync` bounds match what backends already require so the predicate doesn't
/// pin the future to a single thread.
pub type SuccessPredicate = Box<dyn Fn(&LoopOutcome) -> bool + Send + Sync>;

/// One eval task: the seed prompt plus how to judge whether it passed.
///
/// All fields are public — the helper [`finish_task`] is just a constructor,
/// and downstream callers are expected to build their own tasks with struct
/// literals. There is no `Debug` impl because [`SuccessPredicate`] is opaque.
pub struct EvalTask {
    /// Human-readable label; flows through to [`EvalReport::task_name`].
    pub name: String,
    /// The seed user message: the actual task description handed to the agent.
    /// The canonical system prompt is rendered by [`crate::prompt`] inside
    /// the engine — see the module docs.
    pub task: String,
    /// Predicate over the terminal [`LoopOutcome`] that decides pass/fail.
    pub success: SuccessPredicate,
}

/// The result of an eval run. "Pass^k" framing: `passes` out of `trials = k`,
/// plus `pass_rate = passes / trials` for cross-`k` comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalReport {
    /// Copy of [`EvalTask::name`] — so a report can be printed in isolation.
    pub task_name: String,
    /// `k`: the number of independent trials that were run.
    pub trials: u32,
    /// How many trials' outcomes satisfied [`EvalTask::success`].
    pub passes: u32,
    /// `passes / trials` as an f64. `0.0` when `trials == 0` (no division by
    /// zero in the report).
    pub pass_rate: f64,
}

/// Run `task` `k` independent times against `backend` + `tools` and report
/// how many trials the [`EvalTask::success`] predicate accepted.
///
/// Each trial is a fresh call to [`engine::run`] with a [`RunConfig`] carrying
/// the task's seed prompt and the given `max_iterations`. `checks` is left
/// unset — the eval fixtures wired here don't verify against real checks yet;
/// a `finish(done)` in these trials yields
/// [`crate::engine::Verification::NoChecksConfigured`]. The backend is the
/// only stochastic surface in this slice — clean-worktree-per-trial isolation
/// is **deferred**.
///
/// `pass_rate` is `0.0` when `k == 0` so a misuse never produces `NaN`; the
/// trial loop runs zero times, so the backend is also never touched in that
/// case.
pub async fn run_eval(
    task: &EvalTask,
    backend: &impl ModelBackend,
    tools: &ToolRegistry,
    ctx: &ToolCtx,
    k: u32,
    max_iterations: u32,
) -> EvalReport {
    let mut passes: u32 = 0;
    for _ in 0..k {
        let config = RunConfig::new(task.task.clone(), max_iterations);
        let outcome = engine::run(backend, tools, ctx, &config).await;
        if (task.success)(&outcome) {
            passes += 1;
        }
    }
    let pass_rate = if k == 0 {
        0.0
    } else {
        // u32 → f64 is lossless; f64::from is the pedantic-clippy-clean form.
        f64::from(passes) / f64::from(k)
    };
    EvalReport {
        task_name: task.name.clone(),
        trials: k,
        passes,
        pass_rate,
    }
}

/// The simplest possible eval task: prove the agent can invoke the `finish`
/// tool with `disposition="done"` to terminate a run.
///
/// This is the smoke-test rung: any working loop + backend + finish-tool
/// wiring should hit `pass_rate ≈ 1.0` here. The predicate matches
/// **any** [`LoopOutcome::Finished`] carrying [`FinishDisposition::Done`] —
/// including the `NoChecksConfigured` variant, which is what the eval yields
/// today (no [`crate::exec::ChecksRunner`] is wired). Every other terminal
/// outcome (blocked, failed, hit max iterations, stopped without finishing,
/// backend error) is a failure.
#[must_use]
pub fn finish_task() -> EvalTask {
    EvalTask {
        name: "finish".to_string(),
        task: "Acknowledge and finish.".to_string(),
        success: Box::new(|outcome| {
            matches!(
                outcome,
                LoopOutcome::Finished(FinishDisposition::Done { .. })
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{EvalReport, EvalTask, finish_task, run_eval};
    use crate::engine::{FINISH_TOOL_NAME, FinishDisposition, FinishTool, LoopOutcome};
    use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
    use crate::test_support::MockBackend;
    use crate::tool::{EchoTool, ToolCtx, ToolRegistry};
    use serde_json::json;
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

    fn finish_done_turn() -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: json!({ "disposition": "done", "summary": "ok" }),
            })],
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    /// A tool-use turn that calls `echo` (not finish), so the loop keeps
    /// drawing turns until it either finishes or hits `max_iterations`.
    fn non_finish_turn() -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-echo".to_string(),
                name: "echo".to_string(),
                input: json!({}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    fn registry() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register("echo", Arc::new(EchoTool));
        r.register(FINISH_TOOL_NAME, Arc::new(FinishTool));
        r
    }

    #[tokio::test]
    async fn always_finishing_script_yields_pass_rate_one() {
        // k trials × 1 turn each (the finish call terminates immediately).
        let k: u32 = 5;
        let script: Vec<AssistantTurn> = (0..k).map(|_| finish_done_turn()).collect();
        let backend = MockBackend::from_turns(script);
        let tools = registry();
        let ctx = ToolCtx::stub();
        let task = finish_task();

        let report = run_eval(&task, &backend, &tools, &ctx, k, 10).await;

        assert_eq!(report.task_name, task.name);
        assert_eq!(report.trials, k);
        assert_eq!(report.passes, k);
        // f64::from(u32) is lossless; an exact equality is safe here.
        assert!((report.pass_rate - 1.0).abs() < f64::EPSILON);
        // And the backend was called exactly once per trial.
        assert_eq!(backend.calls(), k);
    }

    #[tokio::test]
    async fn never_finishing_script_yields_pass_rate_zero() {
        // k trials × max_iter turns of `echo` (no finish), so every trial
        // terminates as LoopOutcome::MaxIterations — never Finished(Done).
        let k: u32 = 3;
        let max_iter: u32 = 2;
        let script: Vec<AssistantTurn> = (0..(k * max_iter)).map(|_| non_finish_turn()).collect();
        let backend = MockBackend::from_turns(script);
        let tools = registry();
        let ctx = ToolCtx::stub();
        let task = finish_task();

        let report = run_eval(&task, &backend, &tools, &ctx, k, max_iter).await;

        assert_eq!(report.task_name, "finish");
        assert_eq!(report.trials, k);
        assert_eq!(report.passes, 0);
        assert!((report.pass_rate - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn mixed_script_yields_correct_fraction() {
        // 4 trials, alternating success / failure. Failure here is a plain
        // tool_use that isn't finish — each such trial hits max_iterations
        // after `max_iter` draws.
        let max_iter: u32 = 1;
        // Alternating turn 0..3 — comment per element documents what each
        // trial does so the expected pass count is self-evident.
        let script: Vec<AssistantTurn> = vec![
            finish_done_turn(), // trial 1: 1 turn, finishes
            non_finish_turn(),  // trial 2: 1 turn, hits max_iter
            finish_done_turn(), // trial 3: 1 turn, finishes
            non_finish_turn(),  // trial 4: 1 turn, hits max_iter
        ];
        let backend = MockBackend::from_turns(script);
        let tools = registry();
        let ctx = ToolCtx::stub();
        let task = finish_task();

        let k: u32 = 4;
        let report = run_eval(&task, &backend, &tools, &ctx, k, max_iter).await;

        assert_eq!(report.trials, k);
        assert_eq!(report.passes, 2);
        assert!((report.pass_rate - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn k_zero_short_circuits_and_yields_zero_pass_rate() {
        // No trials run → backend is never called and pass_rate is 0.0
        // (not NaN). The empty script proves the loop body never executes.
        let backend = MockBackend::from_turns(vec![]);
        let tools = registry();
        let ctx = ToolCtx::stub();
        let task = finish_task();

        let report = run_eval(&task, &backend, &tools, &ctx, 0, 5).await;

        assert_eq!(report.trials, 0);
        assert_eq!(report.passes, 0);
        assert!((report.pass_rate - 0.0).abs() < f64::EPSILON);
        assert_eq!(backend.calls(), 0, "k=0 must not touch the backend");
    }

    #[test]
    fn finish_task_predicate_accepts_only_finished_done() {
        // The whole point of the finish_task predicate: Done means pass, every
        // other terminal outcome means fail. Cover each non-Done outcome
        // explicitly so the contract is locked in.
        let task = finish_task();

        // NoChecksConfigured is the eval-path Done: verify it passes.
        assert!((task.success)(&LoopOutcome::Finished(
            FinishDisposition::Done {
                summary: "ok".to_string(),
                verification: crate::engine::Verification::NoChecksConfigured,
            }
        )));

        assert!(!(task.success)(&LoopOutcome::Finished(
            FinishDisposition::Blocked {
                decision_needed: "which API?".to_string(),
            }
        )));
        assert!(!(task.success)(&LoopOutcome::Finished(
            FinishDisposition::Failed {
                summary: "tool errored".to_string(),
            }
        )));
        assert!(!(task.success)(&LoopOutcome::StoppedWithoutFinish));
        assert!(!(task.success)(&LoopOutcome::MaxIterations));
        // BackendError is also not a Done outcome.
        assert!(!(task.success)(&LoopOutcome::BackendError(
            crate::model::BackendError::Terminal {
                kind: crate::model::TerminalKind::Auth,
                message: "no creds".to_string(),
            }
        )));
    }

    #[test]
    fn finish_task_carries_non_empty_task_prompt() {
        let task = finish_task();
        assert_eq!(task.name, "finish");
        assert!(!task.task.is_empty());
    }

    #[test]
    fn eval_task_fields_are_publicly_constructible() {
        // A custom predicate via struct-literal construction — proves the
        // public-field shape works for non-built-in tasks.
        let task = EvalTask {
            name: "custom".to_string(),
            task: "do something".to_string(),
            success: Box::new(|outcome| matches!(outcome, LoopOutcome::MaxIterations)),
        };
        assert_eq!(task.name, "custom");
        assert!((task.success)(&LoopOutcome::MaxIterations));
        assert!(!(task.success)(&LoopOutcome::Finished(
            FinishDisposition::Done {
                summary: String::new(),
                verification: crate::engine::Verification::NoChecksConfigured,
            }
        )));
    }

    #[test]
    fn eval_report_is_debug_clone_and_partial_eq() {
        let r = EvalReport {
            task_name: "finish".to_string(),
            trials: 3,
            passes: 1,
            pass_rate: 1.0 / 3.0,
        };
        let printed = format!("{r:?}");
        assert!(printed.contains("EvalReport"));
        assert!(printed.contains("task_name"));
        let cloned = r.clone();
        assert_eq!(r, cloned);
    }
}
