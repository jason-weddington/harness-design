//! Live coding-task eval: the 0.1.0 boundary proof.
//!
//! The harness autonomously FIXES a failing test in a real (tiny) Rust crate
//! (`fixtures/broken-adder`), and "done" is HARNESS-VERIFIED — `cargo test` came
//! back green — not model-claimed. Each trial runs against a FRESH copy of the
//! fixture, so the trials are independent.
//!
//! It talks to the live Anthropic API and shells out to `cargo`, so it is **not**
//! wired into the quality gates (the build host has no API key). The gates only
//! verify that this file *compiles*; run it by hand:
//!
//! ```text
//! ANTHROPIC_API_KEY=sk-... cargo run --example coding_eval
//! ANTHROPIC_API_KEY=sk-... ANTHROPIC_MODEL=claude-sonnet-4-6 CODING_EVAL_K=5 \
//!   cargo run --example coding_eval
//! ```
//!
//! Environment:
//! - `ANTHROPIC_API_KEY` (required) — passed through to the backend.
//! - `ANTHROPIC_MODEL`   (optional) — defaults to `claude-haiku-4-5`.
//! - `CODING_EVAL_K`     (optional) — number of trials; defaults to 3.

use std::env;
use std::path::PathBuf;

use harness::anthropic::AnthropicBackend;
use harness::engine::{FinishDisposition, LoopOutcome, Verification};
use harness::eval::{coding_fix_task, run_eval};

/// Default model id when `ANTHROPIC_MODEL` is not set.
const DEFAULT_MODEL: &str = "claude-haiku-4-5";

/// Default trial count when `CODING_EVAL_K` is not set (env-overridable).
const DEFAULT_K: u32 = 3;

/// Per-trial hard cap on agent-loop iterations. A fix-one-bug task needs a few
/// read/edit/verify rounds, so this is a generous safety margin.
const MAX_ITERATIONS: u32 = 12;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let api_key =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in the environment");
    let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let k = env::var("CODING_EVAL_K")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(DEFAULT_K);

    let backend = AnthropicBackend::new(&model, api_key);

    // The fixture lives at the REPO ROOT (`fixtures/broken-adder`). This
    // example's manifest dir is `crates/harness`, so climb two levels.
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("broken-adder");
    let (task, env_factory) = coding_fix_task(&fixture);

    println!(
        "running eval `{}` x{k} against model `{model}` (max_iterations={MAX_ITERATIONS})\n\
         fixture: {}\n",
        task.name,
        fixture.display(),
    );

    // Stream a one-liner per trial as it completes, then print the final report.
    let report = run_eval(
        &task,
        &backend,
        env_factory,
        k,
        MAX_ITERATIONS,
        |i, outcome| println!("  trial {}: {}", i + 1, outcome_one_liner(outcome)),
    )
    .await;

    println!("\n{report:#?}");
}

/// A terse, one-line description of a trial's terminal outcome for the live log.
fn outcome_one_liner(outcome: &LoopOutcome) -> String {
    match outcome {
        LoopOutcome::Finished(FinishDisposition::Done {
            verification: Verification::Checks(report),
            ..
        }) => format!(
            "Done — checks {} (exit {:?})",
            if report.passed { "GREEN" } else { "RED" },
            report.exit_code,
        ),
        LoopOutcome::Finished(FinishDisposition::Done {
            verification: Verification::NoChecksConfigured,
            ..
        }) => "Done — NO CHECKS (unverified)".to_string(),
        LoopOutcome::Finished(FinishDisposition::Blocked { decision_needed }) => {
            format!("Blocked — {decision_needed}")
        }
        LoopOutcome::Finished(FinishDisposition::Failed { summary }) => {
            format!("Failed — {summary}")
        }
        LoopOutcome::StoppedWithoutFinish => "StoppedWithoutFinish".to_string(),
        LoopOutcome::MaxIterations => "MaxIterations".to_string(),
        LoopOutcome::BackendError(err) => format!("BackendError — {err:?}"),
    }
}
