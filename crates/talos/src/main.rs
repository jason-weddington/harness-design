//! `talos run` — accept a [`TaskSpec`] JSON (stdin or `--file`), execute
//! it with full persistence, and exit with a disposition-mapped code.
//!
//! ## Exit code contract (locked)
//!
//! | Code | Meaning |
//! |------|---------|
//! | 0    | Task verified Done |
//! | 10   | Task Blocked |
//! | 20   | Task Failed, `StoppedWithoutFinish`, `MaxIterations`, or `BudgetExhausted` |
//! | 1    | Harness/infra error (bad spec, `BackendError`, store error, clap error) |
//!
//! The code is read from [`harness::engine::LoopOutcome`], **not** from the
//! disposition — because `BackendError`'s `into_disposition` also yields
//! `Failed`, which would collapse engine-broke (must be 1) into task-Failed
//! (20). See [`exit_code`] for the full rationale.
//!
//! ## Environment variables
//!
//! - `TALOS_BACKEND` — `anthropic` (default when unset) | `ollama`
//! - `ANTHROPIC_API_KEY` — required for anthropic
//! - `ANTHROPIC_MODEL` — optional; default `claude-haiku-4-5`
//! - `OLLAMA_MODEL` — required for ollama
//! - `OLLAMA_BASE_URL` — optional; default `http://localhost:11434`
//! - `OLLAMA_API_KEY` — optional bearer token
//! - `OLLAMA_NUM_CTX` — optional `u32`; defaults to 32 768 for localhost
//! - `OLLAMA_THINK` — `off|on|low|medium|high|max`
//! - `TALOS_WALL_CLOCK_SECS` — optional `u64` seconds; `0` or unset = unbounded
//!   wall-clock budget. Overridden by `--wall-clock-secs` when the flag is
//!   present. The harness self-terminates gracefully before the worker's hard
//!   kill when this budget is reached.

use std::io::Read as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use harness::anthropic::AnthropicBackend;
use harness::engine::{LoopOutcome, Persistence, RunConfig, run_id, run_persisted};
use harness::exec::{CheckCommand, ChecksRunner};
use harness::model::{AssistantTurn, BackendError, ModelBackend, TurnRequest};
use harness::ollama::{OllamaBackend, ThinkLevel};
use harness::prompt::render_task_prompt_from_spec;
use harness::run_record::Disposition;
use harness::store::{RunStore, SqliteRunStore};
use harness::task_spec::TaskSpec;
use harness::tool::{OffloadSink, ToolCtx};
use harness::tools::standard_registry;
use harness::workspace::{DiskOffloadSink, Workspace};
use serde::Serialize;

// ============================================================================
// CLI types
// ============================================================================

/// Top-level CLI entry point.
#[derive(clap::Parser)]
#[command(name = "talos", about = "Talos agent runner", version = env!("TALOS_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Talos subcommands.
#[derive(clap::Subcommand)]
enum Command {
    /// Execute a `TaskSpec` JSON and report outcome + exit code.
    Run(RunArgs),
}

/// Arguments for `talos run`.
#[derive(clap::Args)]
struct RunArgs {
    /// Workspace root (confined working directory for the agent).
    #[arg(long)]
    workspace: PathBuf,

    /// Path to `TaskSpec` JSON file (reads stdin when omitted).
    #[arg(long)]
    file: Option<PathBuf>,

    /// `SQLite` store path for the run record.
    /// Defaults to `${XDG_STATE_HOME:-~/.local/state}/talos/<task-id>/run.sqlite`.
    #[arg(long)]
    run_store: Option<PathBuf>,

    /// Offload directory for oversized tool output.
    /// Defaults to `${XDG_STATE_HOME:-~/.local/state}/talos/<task-id>/offload`.
    #[arg(long)]
    offload_dir: Option<PathBuf>,

    /// Task identifier — becomes `Persistence.task_id` and seeds the run id.
    /// `TaskSpec` has no `task_id` field; this must come from the CLI.
    #[arg(long, default_value = "talos-run")]
    task_id: String,

    /// Attempt number — combined with `--task-id` to form the run id.
    #[arg(long, default_value_t = 1u32)]
    attempt: u32,

    /// Hard cap on agent-loop iterations.
    ///
    /// History: 12 → 24 after the first (small) dogfood run — a groomed item's
    /// AC list invites per-criterion re-verification, and haiku exhausted 12
    /// iterations one call short of finish(done). 24 → 500 after a 0.4.0 wave
    /// item (18 ACs, 5 files) exhausted 24 in ~48s without reaching a verify
    /// cycle: 24 is calibrated for single-crate dogfood work, far too low for
    /// multi-file changes. With the gate timeout (and the upcoming wall-clock
    /// budget) as the real bounds, a high iteration cap is a backstop, not the
    /// primary limit. 500 mirrors the GTD `max_turns` convention; the eventual
    /// fix is to plumb the dispatch `max_turns` through to this flag.
    #[arg(long, default_value_t = 500u32)]
    max_iterations: u32,

    /// Timeout for the gate command, in seconds.
    #[arg(long, default_value_t = 300u64)]
    gate_timeout_secs: u64,

    /// Wall-clock budget in seconds. `0` or absent = unbounded.
    ///
    /// When set, the harness self-terminates gracefully with recovery facts
    /// before the worker's hard timeout. Can also be set via the environment
    /// variable `TALOS_WALL_CLOCK_SECS` (flag takes precedence over env).
    ///
    /// Note: the `env` feature is NOT enabled for this project's clap
    /// dependency, so `#[arg(env = ...)]` cannot be used — the env fallback
    /// is resolved manually in `main()` via the `env_accessor` closure.
    #[arg(long)]
    wall_clock_secs: Option<u64>,
}

// ============================================================================
// Backend dispatch
// ============================================================================

/// Runtime backend: one variant per supported model provider.
///
/// No `Debug` derive — the api key must not appear in formatter chains.
enum Backend {
    Anthropic(AnthropicBackend),
    Ollama(OllamaBackend),
}

#[async_trait]
impl ModelBackend for Backend {
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        match self {
            Self::Anthropic(b) => b.turn(req).await,
            Self::Ollama(b) => b.turn(req).await,
        }
    }
}

// ============================================================================
// Pure, unit-testable helper functions
// ============================================================================

/// Map a [`LoopOutcome`] to the locked exit-code contract.
///
/// ## Rationale for reading `LoopOutcome`, not `Disposition`
///
/// `BackendError`'s [`LoopOutcome::into_disposition`] also yields
/// `Disposition::Failed`, which would collapse engine-broke (code 1) into
/// task-Failed (code 20) if we read the disposition instead.
/// `BackendError` = transport/auth/rate-limit; it **must never** collide with
/// a task-originated `Failed` (code 20).
///
/// ## Locked map
///
/// | Outcome | Code |
/// |---------|------|
/// | `Finished(Done{..})` | 0 |
/// | `Finished(Blocked{..})` | 10 |
/// | `Finished(Failed{..})` | 20 |
/// | `StoppedWithoutFinish` | 20 |
/// | `MaxIterations` | 20 |
/// | `BudgetExhausted` | 20 |
/// | `BackendError(_)` | 1 |
fn exit_code(outcome: &LoopOutcome) -> i32 {
    match outcome {
        LoopOutcome::Finished(Disposition::Done { .. }) => 0,
        LoopOutcome::Finished(Disposition::Blocked { .. }) => 10,
        LoopOutcome::Finished(Disposition::Failed { .. })
        | LoopOutcome::StoppedWithoutFinish
        | LoopOutcome::MaxIterations
        | LoopOutcome::BudgetExhausted { .. } => 20,
        LoopOutcome::BackendError(_) => 1,
    }
}

/// Closed [`LoopOutcome`] discriminant string for the stdout summary `outcome` field.
///
/// Uses a hand-written match over all variants. `format!("{:?}")` is
/// explicitly **forbidden** — it would leak the `BackendError` payload into
/// the summary and prevent consumers from reliably identifying infra-broke
/// runs by the `outcome` field alone.
fn outcome_str(outcome: &LoopOutcome) -> &'static str {
    match outcome {
        LoopOutcome::Finished(_) => "Finished",
        LoopOutcome::StoppedWithoutFinish => "StoppedWithoutFinish",
        LoopOutcome::MaxIterations => "MaxIterations",
        LoopOutcome::BudgetExhausted { .. } => "BudgetExhausted",
        LoopOutcome::BackendError(_) => "BackendError",
    }
}

/// Render the seed string passed to [`RunConfig::new`] for a task run.
///
/// Returns the [`render_task_prompt_from_spec`] output byte-for-byte — the
/// CLI never hand-formats task text. The engine re-wraps this under a
/// `# Task` heading via `render_task_prompt`; the renderer therefore must
/// NOT emit its own `# Task` heading.
fn make_run_seed(spec: &TaskSpec) -> String {
    render_task_prompt_from_spec(spec)
}

/// Build a [`ChecksRunner`] from a shell gate command string, or `None` if the
/// command is empty or whitespace-only.
///
/// A whitespace-only `gate_command` is treated as empty — closing the
/// `/bin/sh -c ' '` exits-0 rubber-stamp false-Done vector.
///
/// `/bin/sh -c` is used (NOT direct exec) so the gate string can contain
/// shell operators like `&&` and pipes.
fn build_checks_runner(
    gate_command: &str,
    workspace_root: PathBuf,
    gate_timeout_secs: u64,
) -> Option<ChecksRunner> {
    if gate_command.trim().is_empty() {
        return None;
    }
    Some(ChecksRunner::new(
        CheckCommand {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), gate_command.to_string()],
        },
        workspace_root,
        Duration::from_secs(gate_timeout_secs),
    ))
}

/// Write a one-line `{"error": "<message>"}` JSON object to stderr.
fn stderr_json_error(message: &str) {
    let obj = serde_json::json!({"error": message});
    eprintln!("{obj}");
}

// ============================================================================
// stdout summary
// ============================================================================

/// Machine-readable JSON summary written to stdout after a successful
/// `run_persisted` call. Printed regardless of the task's disposition.
#[derive(Serialize)]
struct RunSummary {
    /// Closed [`LoopOutcome`] discriminant — one of
    /// `"Finished"`, `"StoppedWithoutFinish"`, `"MaxIterations"`,
    /// `"BudgetExhausted"`, `"BackendError"`.
    outcome: &'static str,
    /// The run's terminal [`Disposition`] (derived from the `LoopOutcome`).
    disposition: Disposition,
    /// `"{task_id}:{attempt_n}"` — the stable run identifier.
    run_id: String,
    /// Path to the `SQLite` store that holds the full run record.
    record_path: String,
    /// Number of model turns the loop drew.
    iterations: u32,
}

/// Build the stdout [`RunSummary`] from a completed run.
fn build_run_summary(
    outcome_s: &'static str,
    disposition: Disposition,
    run_id_str: String,
    record_path: String,
    iterations: u32,
) -> RunSummary {
    RunSummary {
        outcome: outcome_s,
        disposition,
        run_id: run_id_str,
        record_path,
        iterations,
    }
}

// ============================================================================
// Backend selection from environment
// ============================================================================

/// Build an [`AnthropicBackend`] from the injected environment accessor.
///
/// Reads `ANTHROPIC_API_KEY` (required) and `ANTHROPIC_MODEL` (default
/// `claude-haiku-4-5`). Returns `Err` on any missing required var — never
/// panics.
fn build_anthropic_backend(
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Backend, String), String> {
    let api_key = env("ANTHROPIC_API_KEY").ok_or_else(|| {
        "ANTHROPIC_API_KEY must be set when using the anthropic backend".to_string()
    })?;
    let model = env("ANTHROPIC_MODEL").unwrap_or_else(|| "claude-haiku-4-5".to_string());
    let model_label = model.clone();
    Ok((
        Backend::Anthropic(AnthropicBackend::new(&model, api_key)),
        model_label,
    ))
}

/// Build an [`OllamaBackend`] from the injected environment accessor.
///
/// Reads `OLLAMA_MODEL` (required), `OLLAMA_BASE_URL` (default
/// `http://localhost:11434`), `OLLAMA_API_KEY` (optional), `OLLAMA_NUM_CTX`
/// (optional `u32`; defaults to 32 768 for localhost URLs), and `OLLAMA_THINK`
/// (`off|on|low|medium|high|max`). Returns `Err` on any missing required var,
/// unrecognised `OLLAMA_THINK` value, or non-`u32` `OLLAMA_NUM_CTX` — never
/// panics.
fn build_ollama_backend(
    env: &impl Fn(&str) -> Option<String>,
) -> Result<(Backend, String), String> {
    let model = env("OLLAMA_MODEL")
        .ok_or_else(|| "OLLAMA_MODEL must be set when TALOS_BACKEND=ollama".to_string())?;
    let base_url = env("OLLAMA_BASE_URL").unwrap_or_else(|| "http://localhost:11434".to_string());
    let is_local = base_url.contains("localhost") || base_url.contains("127.0.0.1");

    let num_ctx: Option<u32> = match env("OLLAMA_NUM_CTX") {
        None => is_local.then_some(32_768),
        Some(v) => {
            let n = v
                .parse::<u32>()
                .map_err(|_| format!("OLLAMA_NUM_CTX must be a valid u32, got \"{v}\""))?;
            Some(n)
        }
    };

    let think: Option<ThinkLevel> = match env("OLLAMA_THINK").as_deref() {
        None => None,
        Some("off") => Some(ThinkLevel::Off),
        Some("on") => Some(ThinkLevel::On),
        Some("low") => Some(ThinkLevel::Low),
        Some("medium") => Some(ThinkLevel::Medium),
        Some("high") => Some(ThinkLevel::High),
        Some("max") => Some(ThinkLevel::Max),
        Some(other) => {
            return Err(format!(
                "OLLAMA_THINK must be off|on|low|medium|high|max, got \"{other}\""
            ));
        }
    };

    let mut ollama = OllamaBackend::new(&model, &base_url);
    if let Some(key) = env("OLLAMA_API_KEY") {
        ollama = ollama.with_api_key(key);
    }
    if let Some(n) = num_ctx {
        ollama = ollama.with_num_ctx(n);
    }
    if let Some(level) = think {
        ollama = ollama.with_think(level);
    }

    let model_label = format!("ollama:{model}");
    Ok((Backend::Ollama(ollama), model_label))
}

/// Select and construct the model backend from the injected environment.
///
/// `TALOS_BACKEND`: `"anthropic"` (default when unset) | `"ollama"`.
///
/// Returns `Err(message)` — never panics — for:
/// - `TALOS_BACKEND` set to anything other than `"anthropic"` / `"ollama"`
/// - Missing required provider vars (`ANTHROPIC_API_KEY`, `OLLAMA_MODEL`)
/// - `OLLAMA_THINK` outside the accepted set
/// - `OLLAMA_NUM_CTX` present but unparsable as `u32`
fn backend_from_env(env: &impl Fn(&str) -> Option<String>) -> Result<(Backend, String), String> {
    match env("TALOS_BACKEND").as_deref() {
        Some("anthropic") | None => build_anthropic_backend(env),
        Some("ollama") => build_ollama_backend(env),
        Some(other) => Err(format!(
            "TALOS_BACKEND must be \"anthropic\" or \"ollama\", got \"{other}\""
        )),
    }
}

// ============================================================================
// Filesystem helpers
// ============================================================================

/// Compute the default per-task state directory:
/// `${XDG_STATE_HOME:-$HOME/.local/state}/talos/<task-id>`.
fn talos_state_dir(task_id: &str) -> PathBuf {
    let state_home = std::env::var("XDG_STATE_HOME").map_or_else(
        |_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".local").join("state")
        },
        PathBuf::from,
    );
    state_home.join("talos").join(task_id)
}

/// Read the raw spec JSON from `--file <path>` or stdin.
fn read_spec_json(args: &RunArgs) -> Result<String, String> {
    if let Some(path) = &args.file {
        std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read spec file `{}`: {e}", path.display()))
    } else {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| format!("failed to read spec from stdin: {e}"))?;
        Ok(buf)
    }
}

// ============================================================================
// main
// ============================================================================

#[allow(clippy::too_many_lines)]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // 1. Parse CLI — clap usage errors → exit 1 + JSON error (NOT clap's default exit 2).
    //    `--help`/`--version` also surface as `Err` from try_parse but are not
    //    usage errors: print them plainly and exit 0.
    let cli = match <Cli as clap::Parser>::try_parse() {
        Ok(c) => c,
        Err(e) => {
            if matches!(
                e.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                print!("{e}");
                std::process::exit(0);
            }
            stderr_json_error(&e.to_string());
            std::process::exit(1);
        }
    };
    let Command::Run(args) = cli.command;

    // 2. Read and parse spec — must happen BEFORE the store is opened, so a
    //    bad spec never touches the filesystem.
    let spec_json = match read_spec_json(&args) {
        Ok(s) => s,
        Err(e) => {
            stderr_json_error(&e);
            std::process::exit(1);
        }
    };
    let spec: TaskSpec = match serde_json::from_str(&spec_json) {
        Ok(s) => s,
        Err(e) => {
            stderr_json_error(&format!("invalid TaskSpec: {e}"));
            std::process::exit(1);
        }
    };

    // 3. Select model backend from environment.
    let env_accessor = |k: &str| std::env::var(k).ok();
    let (backend, model_label) = match backend_from_env(&env_accessor) {
        Ok(pair) => pair,
        Err(e) => {
            stderr_json_error(&e);
            std::process::exit(1);
        }
    };

    // 4. Resolve run-artifact paths (default to XDG state dir outside the workspace).
    let state_dir = talos_state_dir(&args.task_id);
    let run_store_path = args
        .run_store
        .unwrap_or_else(|| state_dir.join("run.sqlite"));
    let offload_dir = args
        .offload_dir
        .unwrap_or_else(|| state_dir.join("offload"));

    // 5. Create run-store parent and offload dir (Workspace::new REQUIRES the
    //    offload dir to already exist).
    let store_parent = run_store_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    if let Err(e) = std::fs::create_dir_all(store_parent) {
        stderr_json_error(&format!(
            "failed to create store parent `{}`: {e}",
            store_parent.display()
        ));
        std::process::exit(1);
    }
    if let Err(e) = std::fs::create_dir_all(&offload_dir) {
        stderr_json_error(&format!(
            "failed to create offload dir `{}`: {e}",
            offload_dir.display()
        ));
        std::process::exit(1);
    }

    // 6. Build Workspace (canonicalizes and validates the roots).
    let workspace = match Workspace::new(args.workspace.clone(), Some(offload_dir.clone())) {
        Ok(w) => w,
        Err(e) => {
            stderr_json_error(&format!("workspace error: {e}"));
            std::process::exit(1);
        }
    };
    let workspace_root = workspace.root().to_path_buf();

    // 7. Build tool context with a disk-offload sink.
    let sink = DiskOffloadSink::new(offload_dir.clone());
    let ctx = ToolCtx::new(Arc::new(workspace), Arc::new(sink) as Arc<dyn OffloadSink>);

    // 8. Open the run store.
    let store = match SqliteRunStore::open(&run_store_path) {
        Ok(s) => s,
        Err(e) => {
            stderr_json_error(&format!("failed to open run store: {e}"));
            std::process::exit(1);
        }
    };
    let store: Arc<dyn RunStore> = Arc::new(store);

    // 9. Build the optional ChecksRunner from spec.gate_command, wiring it to
    //    BOTH the tool registry (so the agent can call `run_checks`) AND the
    //    RunConfig (so finish(done) is harness-verified).
    let checks = build_checks_runner(&spec.gate_command, workspace_root, args.gate_timeout_secs);
    let tools = standard_registry(checks.clone());

    // 10. Render the seed prompt — byte-for-byte from the renderer, never
    //     hand-formatted.
    let seed = make_run_seed(&spec);

    // Resolve wall-clock budget: flag > TALOS_WALL_CLOCK_SECS env > 0 (unbounded).
    // The `env` clap feature is NOT enabled (Cargo.toml features=['derive'] only),
    // so the env fallback is resolved here via the env_accessor closure.
    let wall_clock_secs = args
        .wall_clock_secs
        .or_else(|| env_accessor("TALOS_WALL_CLOCK_SECS").and_then(|v| v.parse::<u64>().ok()))
        .unwrap_or(0);

    let config = if let Some(runner) = checks {
        RunConfig::new(seed, args.max_iterations)
            .with_checks(runner)
            .with_wall_clock_secs(wall_clock_secs)
    } else {
        RunConfig::new(seed, args.max_iterations).with_wall_clock_secs(wall_clock_secs)
    };

    // 11. Assemble persistence bundle and run.
    let rid = run_id(&args.task_id, args.attempt);
    let persistence = Persistence {
        store,
        task_id: args.task_id,
        attempt_n: args.attempt,
        model_label,
    };

    let result = match run_persisted(&backend, &tools, &ctx, &config, &persistence).await {
        Ok(r) => r,
        Err(e) => {
            // StoreError — record may be partially written.
            stderr_json_error(&format!("store error during run: {e}"));
            std::process::exit(1);
        }
    };

    // 12. Print machine-readable summary and exit with the locked code.
    let outcome_s = outcome_str(&result.outcome);
    let exit_c = exit_code(&result.outcome);
    let iterations = result.stats.iterations;
    let disposition = result.outcome.into_disposition();
    let record_path = run_store_path.display().to_string();
    let summary = build_run_summary(outcome_s, disposition, rid, record_path, iterations);
    println!(
        "{}",
        serde_json::to_string(&summary)
            .expect("RunSummary serializes infallibly — all fields are owned serde types")
    );
    std::process::exit(exit_c);
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::{
        Backend, RunSummary, backend_from_env, build_checks_runner, build_run_summary, exit_code,
        make_run_seed, outcome_str,
    };
    use harness::engine::LoopOutcome;
    use harness::model::{BackendError, TerminalKind, TransientKind};
    use harness::prompt::render_task_prompt_from_spec;
    use harness::run_record::{Disposition, FailureMode, Verification};
    use harness::task_spec::{FileToModify, TaskSpec};
    use std::path::PathBuf;

    // ---- exit_code: all 6 arms ----------------------------------------

    #[test]
    fn exit_code_finished_done_is_0() {
        let outcome = LoopOutcome::Finished(Disposition::Done {
            summary: "ok".into(),
            verification: Verification::NoChecksConfigured,
        });
        assert_eq!(exit_code(&outcome), 0);
    }

    #[test]
    fn exit_code_finished_blocked_is_10() {
        let outcome = LoopOutcome::Finished(Disposition::Blocked {
            decision_needed: "needs input".into(),
        });
        assert_eq!(exit_code(&outcome), 10);
    }

    #[test]
    fn exit_code_finished_failed_is_20() {
        let outcome = LoopOutcome::Finished(Disposition::Failed {
            mode: FailureMode::Loop,
            summary: "looped".into(),
        });
        assert_eq!(exit_code(&outcome), 20);
    }

    #[test]
    fn exit_code_stopped_without_finish_is_20() {
        assert_eq!(exit_code(&LoopOutcome::StoppedWithoutFinish), 20);
    }

    #[test]
    fn exit_code_max_iterations_is_20() {
        assert_eq!(exit_code(&LoopOutcome::MaxIterations), 20);
    }

    #[test]
    fn exit_code_budget_exhausted_is_20() {
        assert_eq!(
            exit_code(&LoopOutcome::BudgetExhausted {
                summary: "x".into()
            }),
            20,
            "BudgetExhausted must map to exit code 20"
        );
    }

    #[test]
    fn exit_code_backend_error_is_1() {
        let outcome = LoopOutcome::BackendError(BackendError::Transient {
            kind: TransientKind::Network,
            retry_after: None,
        });
        assert_eq!(exit_code(&outcome), 1, "BackendError must be 1, never 20");
    }

    // ---- outcome_str: all 5 literals -----------------------------------

    #[test]
    fn outcome_str_covers_all_five_literals() {
        assert_eq!(
            outcome_str(&LoopOutcome::Finished(Disposition::Done {
                summary: String::new(),
                verification: Verification::NoChecksConfigured,
            })),
            "Finished"
        );
        assert_eq!(
            outcome_str(&LoopOutcome::StoppedWithoutFinish),
            "StoppedWithoutFinish"
        );
        assert_eq!(outcome_str(&LoopOutcome::MaxIterations), "MaxIterations");
        assert_eq!(
            outcome_str(&LoopOutcome::BudgetExhausted {
                summary: "wall-clock budget exhausted".into()
            }),
            "BudgetExhausted"
        );
        assert_eq!(
            outcome_str(&LoopOutcome::BackendError(BackendError::Terminal {
                kind: TerminalKind::Auth,
                message: "bad key".into(),
            })),
            "BackendError"
        );
    }

    // ---- gate_command → ChecksRunner wiring ----------------------------

    fn dummy_root() -> PathBuf {
        PathBuf::from("/")
    }

    #[test]
    fn non_empty_gate_command_produces_sh_runner() {
        let runner = build_checks_runner("cargo nextest run", dummy_root(), 60)
            .expect("non-empty gate_command must yield Some(runner)");
        assert_eq!(runner.command().program, "/bin/sh");
        assert_eq!(runner.command().args, vec!["-c", "cargo nextest run"]);
    }

    #[test]
    fn non_empty_gate_command_registry_has_run_checks() {
        use harness::tools::standard_registry;
        let runner = build_checks_runner("cargo test", dummy_root(), 60).unwrap();
        let registry = standard_registry(Some(runner));
        assert!(
            registry.get("run_checks").is_some(),
            "registry must contain run_checks when gate_command is non-empty"
        );
    }

    #[test]
    fn empty_gate_command_produces_no_runner() {
        assert!(
            build_checks_runner("", dummy_root(), 60).is_none(),
            "empty gate_command must yield None"
        );
    }

    #[test]
    fn whitespace_only_gate_command_is_treated_as_empty() {
        assert!(
            build_checks_runner("   ", dummy_root(), 60).is_none(),
            "whitespace-only gate_command must be treated as empty (no ChecksRunner)"
        );
    }

    // ---- backend_from_env: error branches + model_label ---------------

    fn env_with<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k: &str| {
            vars.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn backend_from_env_defaults_to_anthropic_when_unset() {
        let env = env_with(&[("ANTHROPIC_API_KEY", "sk-test")]);
        let (backend, label) = backend_from_env(&env).expect("must succeed");
        assert!(matches!(backend, Backend::Anthropic(_)));
        // model_label is the model id verbatim (default)
        assert_eq!(label, "claude-haiku-4-5");
    }

    #[test]
    fn backend_from_env_anthropic_explicit() {
        let env = env_with(&[
            ("TALOS_BACKEND", "anthropic"),
            ("ANTHROPIC_API_KEY", "sk-xyz"),
            ("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        ]);
        let (backend, label) = backend_from_env(&env).expect("must succeed");
        assert!(matches!(backend, Backend::Anthropic(_)));
        assert_eq!(
            label, "claude-sonnet-4-6",
            "model_label must be the model id verbatim"
        );
    }

    #[test]
    fn backend_from_env_missing_anthropic_api_key_is_err() {
        let env = env_with(&[]); // ANTHROPIC_API_KEY absent
        assert!(
            backend_from_env(&env).is_err(),
            "missing ANTHROPIC_API_KEY must be Err"
        );
    }

    #[test]
    fn backend_from_env_unknown_backend_is_err() {
        let env = env_with(&[("TALOS_BACKEND", "gemini")]);
        assert!(
            backend_from_env(&env).is_err(),
            "unknown TALOS_BACKEND must be Err"
        );
    }

    #[test]
    fn backend_from_env_ollama_model_label_prefix() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "qwen3:32b"),
            ("OLLAMA_BASE_URL", "https://ollama.com"),
        ]);
        let (backend, label) = backend_from_env(&env).expect("must succeed");
        assert!(matches!(backend, Backend::Ollama(_)));
        assert_eq!(label, "ollama:qwen3:32b");
    }

    #[test]
    fn backend_from_env_ollama_missing_model_is_err() {
        let env = env_with(&[("TALOS_BACKEND", "ollama")]);
        assert!(
            backend_from_env(&env).is_err(),
            "OLLAMA_MODEL must be required for ollama"
        );
    }

    #[test]
    fn backend_from_env_bad_ollama_think_is_err() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "some-model"),
            ("OLLAMA_THINK", "turbo"),
        ]);
        assert!(
            backend_from_env(&env).is_err(),
            "invalid OLLAMA_THINK must be Err"
        );
    }

    #[test]
    fn backend_from_env_bad_ollama_num_ctx_is_err() {
        let env = env_with(&[
            ("TALOS_BACKEND", "ollama"),
            ("OLLAMA_MODEL", "some-model"),
            ("OLLAMA_NUM_CTX", "not-a-number"),
        ]);
        assert!(
            backend_from_env(&env).is_err(),
            "non-u32 OLLAMA_NUM_CTX must be Err"
        );
    }

    #[test]
    fn backend_from_env_all_ollama_think_values_accepted() {
        for level in &["off", "on", "low", "medium", "high", "max"] {
            let vars = [
                ("TALOS_BACKEND", "ollama"),
                ("OLLAMA_MODEL", "m"),
                ("OLLAMA_BASE_URL", "https://ollama.com"),
                ("OLLAMA_THINK", *level),
            ];
            let env = env_with(&vars);
            assert!(
                backend_from_env(&env).is_ok(),
                "OLLAMA_THINK={level} must be accepted"
            );
        }
    }

    // ---- seed: byte-identical to renderer, never raw description -------

    fn sample_spec() -> TaskSpec {
        TaskSpec {
            title: "Test task".into(),
            description: "Test description.".into(),
            acceptance_criteria: vec!["AC one".into(), "AC two".into()],
            files_to_modify: vec![FileToModify {
                path: "src/lib.rs".into(),
                change: "do something".into(),
            }],
            gate_command: "cargo nextest run".into(),
        }
    }

    #[test]
    fn seed_byte_identical_to_renderer_not_raw_description() {
        let spec = sample_spec();
        let seed = make_run_seed(&spec);
        let expected = render_task_prompt_from_spec(&spec);
        assert_eq!(
            seed.as_bytes(),
            expected.as_bytes(),
            "seed must be byte-identical to render_task_prompt_from_spec output"
        );
        // Guard: rendered output includes title/AC/files, so it cannot equal
        // the raw description field.
        assert_ne!(
            seed, spec.description,
            "seed must NOT be the raw description — the renderer must be used"
        );
    }

    // ---- summary: exact field set and outcome literals -----------------

    #[test]
    fn summary_exact_field_set() {
        let summary = build_run_summary(
            "BackendError",
            Disposition::Failed {
                mode: FailureMode::TransientInfra,
                summary: "conn refused".into(),
            },
            "my-task:1".into(),
            "/tmp/run.sqlite".into(),
            3,
        );
        let json = serde_json::to_value(&summary).expect("summary must serialize");
        let obj = json.as_object().expect("must be object");
        let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec![
                "disposition",
                "iterations",
                "outcome",
                "record_path",
                "run_id"
            ],
            "summary must have exactly the five expected fields"
        );
        assert_eq!(
            obj.get("outcome").and_then(serde_json::Value::as_str),
            Some("BackendError")
        );
        assert_eq!(
            obj.get("run_id").and_then(serde_json::Value::as_str),
            Some("my-task:1")
        );
        assert_eq!(
            obj.get("iterations").and_then(serde_json::Value::as_u64),
            Some(3)
        );
    }

    #[test]
    fn summary_outcome_covers_all_five_literals() {
        // Verify all five LoopOutcome discriminants appear in outcome_str.
        let cases: &[(&'static str, LoopOutcome)] = &[
            (
                "Finished",
                LoopOutcome::Finished(Disposition::Done {
                    summary: String::new(),
                    verification: Verification::NoChecksConfigured,
                }),
            ),
            ("StoppedWithoutFinish", LoopOutcome::StoppedWithoutFinish),
            ("MaxIterations", LoopOutcome::MaxIterations),
            (
                "BudgetExhausted",
                LoopOutcome::BudgetExhausted {
                    summary: "wall-clock budget exhausted".into(),
                },
            ),
            (
                "BackendError",
                LoopOutcome::BackendError(BackendError::Terminal {
                    kind: TerminalKind::Auth,
                    message: "x".into(),
                }),
            ),
        ];
        for (expected, outcome) in cases {
            assert_eq!(outcome_str(outcome), *expected);
        }
    }

    // ---- RunSummary is serializable ------------------------------------

    #[test]
    fn run_summary_serializes_disposition() {
        let summary: RunSummary = build_run_summary(
            "Finished",
            Disposition::Done {
                summary: "all green".into(),
                verification: Verification::NoChecksConfigured,
            },
            "task:1".into(),
            "/state/run.sqlite".into(),
            5,
        );
        let json = serde_json::to_value(&summary).expect("must serialize");
        assert!(
            json.get("disposition").is_some(),
            "disposition field must be present"
        );
    }

    // ---- ChecksRunner.clone() used for both registry and config --------
    #[test]
    fn checks_runner_can_be_cloned_for_dual_wiring() {
        let runner =
            build_checks_runner("cargo test", dummy_root(), 30).expect("non-empty yields Some");
        // clone() is required to wire the same runner to both
        // standard_registry(Some(..)) AND RunConfig::with_checks(..).
        let _clone = runner.clone();
        // If this compiles and runs, ChecksRunner is Clone. ✓
        assert_eq!(runner.command().program, "/bin/sh");
    }

    // ---- wall_clock_secs: flag > env > default 0 -------------------------

    /// Helper that simulates the `wall_clock_secs` resolution logic from `main()`:
    ///   `flag > TALOS_WALL_CLOCK_SECS env > default 0`
    fn resolve_wall_clock_secs(flag: Option<u64>, env: &impl Fn(&str) -> Option<String>) -> u64 {
        flag.or_else(|| env("TALOS_WALL_CLOCK_SECS").and_then(|v| v.parse::<u64>().ok()))
            .unwrap_or(0)
    }

    #[test]
    fn wall_clock_secs_flag_beats_env() {
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "999")]);
        assert_eq!(
            resolve_wall_clock_secs(Some(42), &env),
            42,
            "explicit flag must take precedence over env"
        );
    }

    #[test]
    fn wall_clock_secs_env_beats_default() {
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "300")]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            300,
            "env must beat the default 0"
        );
    }

    #[test]
    fn wall_clock_secs_both_unset_yields_zero() {
        let env = env_with(&[]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            0,
            "both-unset must yield the sentinel 0 (unbounded)"
        );
    }

    #[test]
    fn wall_clock_secs_invalid_env_value_falls_back_to_zero() {
        // A non-u64 env value must not panic — it falls through to default 0.
        let env = env_with(&[("TALOS_WALL_CLOCK_SECS", "not-a-number")]);
        assert_eq!(
            resolve_wall_clock_secs(None, &env),
            0,
            "invalid env value must fall back to 0 (unbounded)"
        );
    }
}
