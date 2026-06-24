//! Live eval example: run [`finish_task`] `k` times against the real
//! Anthropic backend and print the [`EvalReport`].
//!
//! This is the interactive end of the eval harness — it talks to a live API,
//! so it is **not** wired into the quality gates (the build host has no API
//! key). The gates only verify that this file *compiles*; the run itself is
//! something you do by hand:
//!
//! ```text
//! ANTHROPIC_API_KEY=sk-... cargo run --example first_eval
//! ANTHROPIC_API_KEY=sk-... ANTHROPIC_MODEL=claude-sonnet-4-6 cargo run --example first_eval
//! ```
//!
//! Environment:
//! - `ANTHROPIC_API_KEY` (required) — passed through to the backend.
//! - `ANTHROPIC_MODEL`   (optional) — defaults to `claude-haiku-4-5`.

use std::env;
use std::sync::Arc;

use harness::anthropic::AnthropicBackend;
use harness::engine::{FINISH_TOOL_NAME, FinishTool};
use harness::eval::{finish_task, run_eval};
use harness::tool::{EchoTool, ToolCtx, ToolRegistry};

/// Number of independent trials per eval run. Small so a live run is cheap.
const TRIALS: u32 = 5;

/// Per-trial hard cap on agent-loop iterations. The trivial `finish_task`
/// should finish in 1, so this is a generous safety margin, not a target.
const MAX_ITERATIONS: u32 = 5;

/// Default model id when `ANTHROPIC_MODEL` is not set in the environment.
const DEFAULT_MODEL: &str = "claude-haiku-4-5";

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let api_key =
        env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in the environment");
    let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let backend = AnthropicBackend::new(&model, api_key);

    let mut tools = ToolRegistry::new();
    tools.register("echo", Arc::new(EchoTool));
    tools.register(FINISH_TOOL_NAME, Arc::new(FinishTool));

    let ctx = ToolCtx::stub();
    let task = finish_task();

    println!(
        "running eval `{}` x{TRIALS} against model `{model}` (max_iterations={MAX_ITERATIONS})",
        task.name,
    );

    let report = run_eval(&task, &backend, &tools, &ctx, TRIALS, MAX_ITERATIONS).await;
    println!("{report:#?}");
}
