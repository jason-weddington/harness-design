//! Live coding-task eval: the 0.1.0 boundary proof, generalized across every
//! committed fixture.
//!
//! The harness autonomously FIXES a failing test in each real (tiny) Rust
//! crate under `fixtures/`, and "done" is HARNESS-VERIFIED — `cargo test` came
//! back green — not model-claimed. Each trial runs against a FRESH copy of the
//! fixture, so trials are independent. Fixtures run sequentially in sorted
//! discovery order; per-trial outcome lines stream as they complete and each
//! fixture prints its own [`EvalReport`] before the next fixture starts. A
//! single one-line-per-fixture summary is printed at the end.
//!
//! It talks to the live Anthropic API and shells out to `cargo`, so it is **not**
//! wired into the quality gates (the build host has no API key). The gates only
//! verify that this file *compiles*; run it by hand:
//!
//! ```text
//! # Anthropic (default backend)
//! ANTHROPIC_API_KEY=sk-... cargo run --example coding_eval
//! ANTHROPIC_API_KEY=sk-... ANTHROPIC_MODEL=claude-sonnet-4-6 CODING_EVAL_K=5 \
//!   cargo run --example coding_eval
//! # Ollama cloud (GLM-5.2)
//! EVAL_BACKEND=ollama OLLAMA_BASE_URL=https://ollama.com OLLAMA_MODEL=glm-5.2:cloud \
//!   cargo run --example coding_eval
//! # Ollama localhost (small local models; num_ctx defaults to 32768 locally)
//! EVAL_BACKEND=ollama OLLAMA_MODEL=qwen3.6:35b cargo run --example coding_eval
//! # narrow to a single fixture directory name
//! ANTHROPIC_API_KEY=sk-... CODING_EVAL_FIXTURE=lru-cache cargo run --example coding_eval
//! ```
//!
//! Environment:
//! - `EVAL_BACKEND`          (optional) — `anthropic` (default) or `ollama`.
//! - `ANTHROPIC_API_KEY`     (required for anthropic) — passed to the backend.
//! - `ANTHROPIC_MODEL`       (optional) — defaults to `claude-haiku-4-5`.
//! - `OLLAMA_MODEL`          (required for ollama) — e.g. `glm-5.2:cloud`,
//!   `qwen3.6:35b`, `gpt-oss:20b`. Never hardcoded.
//! - `OLLAMA_BASE_URL`       (optional) — defaults to `http://localhost:11434`;
//!   set `https://ollama.com` for Ollama cloud.
//! - `OLLAMA_API_KEY`        (optional) — attached as a Bearer token when set
//!   (required in practice for Ollama cloud).
//! - `OLLAMA_NUM_CTX`        (optional) — context window. When unset it
//!   defaults to `32768` for localhost (local defaults are VRAM-tiered and
//!   overflow truncates SILENTLY — never rely on them) and stays unset for
//!   remote hosts (cloud models default to their max context).
//! - `OLLAMA_THINK`          (optional) — `off|on|low|medium|high|max`
//!   (gpt-oss ignores plain booleans; GLM-5.2 supports high/max).
//! - `CODING_EVAL_K`         (optional) — number of trials; defaults to 3.
//! - `CODING_EVAL_FIXTURE`   (optional) — narrows the run to the single named
//!   fixture directory under `fixtures/` (e.g. `lru-cache`). When unset, every
//!   directory under `fixtures/` is discovered and run in sorted order.
//! - `CODING_EVAL_MAX_ITERATIONS` (optional) — per-trial agent-loop cap;
//!   defaults to 12. The task-spec-shaped tiers (taskdeck, calc) benefit from
//!   more headroom on small models — 24 matches the talos dispatch default.

use std::env;
use std::path::PathBuf;

use async_trait::async_trait;
use harness::anthropic::AnthropicBackend;
use harness::engine::{LoopOutcome, RunStats};
use harness::eval::{EvalReport, TrialResult, coding_fix_task, discover_fixtures, run_eval};
use harness::model::{AssistantTurn, BackendError, ModelBackend, TurnRequest};
use harness::ollama::{OllamaBackend, ThinkLevel};
use harness::run_record::{Disposition, Verification};

/// Default model id when `ANTHROPIC_MODEL` is not set.
const DEFAULT_MODEL: &str = "claude-haiku-4-5";

/// Default `num_ctx` for LOCALHOST Ollama runs when `OLLAMA_NUM_CTX` is unset.
/// Local defaults are VRAM-tiered (4k on small GPUs) and overflow truncates
/// silently, so an explicit value is mandatory hygiene; remote/cloud hosts get
/// no default (cloud models run at their max context).
const DEFAULT_LOCAL_NUM_CTX: u32 = 32_768;

/// Example-local backend selection: one enum over the concrete backends so the
/// generic `run_eval(&impl ModelBackend, ...)` call site stays monomorphic.
enum Backend {
    Anthropic(AnthropicBackend),
    Ollama(OllamaBackend),
}

#[async_trait]
impl ModelBackend for Backend {
    async fn turn(&self, req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
        match self {
            Backend::Anthropic(b) => b.turn(req).await,
            Backend::Ollama(b) => b.turn(req).await,
        }
    }
}

/// Build the backend from the environment (see the module docs for the
/// variables). Returns the backend plus a human-readable description line for
/// the run header — model, endpoint, and the knobs that affect comparability.
fn backend_from_env() -> (Backend, String) {
    match env::var("EVAL_BACKEND").as_deref() {
        Ok("ollama") => {
            let model = env::var("OLLAMA_MODEL")
                .expect("OLLAMA_MODEL must be set when EVAL_BACKEND=ollama");
            let base_url =
                env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".into());
            let is_local = base_url.contains("localhost") || base_url.contains("127.0.0.1");
            let num_ctx = env::var("OLLAMA_NUM_CTX")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .or(is_local.then_some(DEFAULT_LOCAL_NUM_CTX));
            let think = env::var("OLLAMA_THINK").ok().map(|v| match v.as_str() {
                "off" => ThinkLevel::Off,
                "on" => ThinkLevel::On,
                "low" => ThinkLevel::Low,
                "medium" => ThinkLevel::Medium,
                "high" => ThinkLevel::High,
                "max" => ThinkLevel::Max,
                other => panic!("OLLAMA_THINK must be off|on|low|medium|high|max, got `{other}`"),
            });

            let mut backend = OllamaBackend::new(&model, &base_url);
            if let Ok(key) = env::var("OLLAMA_API_KEY") {
                backend = backend.with_api_key(key);
            }
            if let Some(n) = num_ctx {
                backend = backend.with_num_ctx(n);
            }
            if let Some(level) = think {
                backend = backend.with_think(level);
            }
            let desc = format!(
                "ollama `{model}` @ {base_url} (num_ctx={}, think={})",
                num_ctx.map_or("default".into(), |n| n.to_string()),
                env::var("OLLAMA_THINK").unwrap_or_else(|_| "unset".into()),
            );
            (Backend::Ollama(backend), desc)
        }
        Ok("anthropic") | Err(_) => {
            let api_key = env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY must be set in the environment");
            let model = env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
            let desc = format!("anthropic `{model}`");
            (
                Backend::Anthropic(AnthropicBackend::new(&model, api_key)),
                desc,
            )
        }
        Ok(other) => panic!("EVAL_BACKEND must be `anthropic` or `ollama`, got `{other}`"),
    }
}

/// Default trial count when `CODING_EVAL_K` is not set (env-overridable).
const DEFAULT_K: u32 = 3;

/// Default per-trial hard cap on agent-loop iterations when
/// `CODING_EVAL_MAX_ITERATIONS` is not set (env-overridable). A fix-one-bug
/// task needs a few read/edit/verify rounds; the harder task-spec-shaped
/// fixtures (implement-to-spec, write-your-own-tests) need more headroom.
const DEFAULT_MAX_ITERATIONS: u32 = 12;

/// Read a `u32` from the environment, falling back to `default` when the
/// variable is unset or unparsable.
fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let (backend, backend_desc) = backend_from_env();
    let k = env_u32("CODING_EVAL_K", DEFAULT_K);
    let max_iterations = env_u32("CODING_EVAL_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS);
    // Empty string is treated as "unset" — the shell's `VAR= cmd` idiom clears
    // the narrow-to-one-fixture override.
    let fixture_filter = env::var("CODING_EVAL_FIXTURE")
        .ok()
        .filter(|s| !s.is_empty());

    // The fixtures live at the REPO ROOT under `fixtures/`. This example's
    // manifest dir is `crates/harness`, so climb two levels.
    let fixtures_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures");

    let fixtures: Vec<PathBuf> = if let Some(name) = fixture_filter.as_deref() {
        let path = fixtures_root.join(name);
        assert!(
            path.is_dir(),
            "CODING_EVAL_FIXTURE={name} does not resolve to a directory under {}",
            fixtures_root.display(),
        );
        vec![path]
    } else {
        discover_fixtures(&fixtures_root).expect("discover fixtures under fixtures/")
    };

    assert!(
        !fixtures.is_empty(),
        "no fixtures found under {}",
        fixtures_root.display(),
    );

    println!(
        "running coding_fix eval across {} fixture(s) (k={k}) against {backend_desc} \
         (max_iterations={max_iterations})",
        fixtures.len(),
    );

    // Per-fixture reports paired with the display name (the fixture directory
    // name — more useful in a summary than the constant `task.name`).
    let mut summary: Vec<(String, EvalReport)> = Vec::with_capacity(fixtures.len());

    for fixture in &fixtures {
        let fixture_name = fixture
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unnamed>")
            .to_string();
        let (mut task, env_factory) = coding_fix_task(fixture);
        // Stamp the fixture name onto the task so the report says which
        // fixture ran — otherwise every report would just read `coding_fix`.
        task.name = fixture_name.clone();

        println!(
            "\n=== fixture: {fixture_name} ===\n  path: {}\n",
            fixture.display(),
        );

        let report = run_eval(
            &task,
            &backend,
            env_factory,
            k,
            max_iterations,
            |trial: &TrialResult| {
                println!(
                    "  trial {}: {} | {}",
                    trial.trial + 1,
                    outcome_one_liner(&trial.outcome),
                    stats_one_liner(&trial.stats),
                );
            },
        )
        .await;

        println!("\n{report:#?}");
        summary.push((fixture_name, report));
    }

    // Final one-line-per-fixture summary table. Widths are computed so the
    // fixture-name column exactly fits the longest name (no truncation). The
    // extra `mean_iters`, `total_tokens`, `mean_wall`, `holdout`, and
    // `false_dn` columns surface per-trial detail collapsed into a
    // compare-across-fixtures view.
    let name_col = summary
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("fixture".len());

    print_summary(&summary, name_col);
}

/// Render the final one-line-per-fixture summary table.
fn print_summary(summary: &[(String, EvalReport)], name_col: usize) {
    println!("\n=== SUMMARY ===");
    println!(
        "{:<name_col$}  {:>9}  {:>10}  {:>10}  {:>12}  {:>9}  {:>11}  {:>8}",
        "fixture",
        "passes/k",
        "pass_rate",
        "mean_iters",
        "total_tokens",
        "mean_wall",
        "holdout",
        "false_dn",
    );
    for (name, r) in summary {
        let total_tokens = r.total_input_tokens() + r.total_output_tokens();
        // Mean per-trial wall-clock in seconds (0.0 for an empty report).
        // `usize` → `f64` for the divisor: trial counts can't approach f64's
        // precision limit — same rationale as `EvalReport::mean_iterations`.
        #[allow(clippy::cast_precision_loss)]
        let mean_wall = if r.trial_results.is_empty() {
            0.0
        } else {
            r.trial_results
                .iter()
                .map(|t| t.stats.wall_clock.as_secs_f64())
                .sum::<f64>()
                / r.trial_results.len() as f64
        };
        // Count trials that had a holdout re-gate (holdout_passed.is_some()).
        let holdout_n: u32 = r
            .trial_results
            .iter()
            .filter(|t| t.holdout_passed.is_some())
            .map(|_| 1u32)
            .sum();
        let holdout_col = if holdout_n == 0 {
            "-".to_string()
        } else {
            format!("{}/{}", r.holdout_passes(), holdout_n)
        };
        println!(
            "{name:<name_col$}  {:>9}  {:>10.3}  {:>10.2}  {:>12}  {:>8.1}s  {:>11}  {:>8}",
            format!("{}/{}", r.passes, r.trials),
            r.pass_rate,
            r.mean_iterations(),
            format_tokens_compact(total_tokens),
            mean_wall,
            holdout_col,
            r.false_dones(),
        );
    }
}

/// A terse one-line summary of a trial's [`RunStats`] for the per-trial log
/// line — iterations, in/out tokens, and wall-clock. Wall-clock is rendered
/// in whole seconds (small runs might round to 0s, which is fine).
fn stats_one_liner(stats: &RunStats) -> String {
    format!(
        "{} iters | {} in / {} out | {}s",
        stats.iterations,
        format_tokens_compact(stats.input_tokens),
        format_tokens_compact(stats.output_tokens),
        stats.wall_clock.as_secs(),
    )
}

/// Render a token count compactly: below `1_000` as a bare integer, otherwise as
/// `NN.Nk`. Keeps the per-trial log line short without hiding order of
/// magnitude.
fn format_tokens_compact(n: u64) -> String {
    if n < 1_000 {
        return n.to_string();
    }
    // `u64 → f64` loses precision above 2^53, but token totals for a single
    // eval run are nowhere near that; the pedantic-clippy `as` is fine here.
    #[allow(clippy::cast_precision_loss)]
    let k = n as f64 / 1_000.0;
    format!("{k:.1}k")
}

/// A terse, one-line description of a trial's terminal outcome for the live log.
fn outcome_one_liner(outcome: &LoopOutcome) -> String {
    match outcome {
        LoopOutcome::Finished(Disposition::Done {
            verification: Verification::Checks(report),
            ..
        }) => format!(
            "Done — checks {} (exit {:?})",
            if report.passed { "GREEN" } else { "RED" },
            report.exit_code,
        ),
        LoopOutcome::Finished(Disposition::Done {
            verification: Verification::NoChecksConfigured,
            ..
        }) => "Done — NO CHECKS (unverified)".to_string(),
        LoopOutcome::Finished(Disposition::Blocked { decision_needed }) => {
            format!("Blocked — {decision_needed}")
        }
        LoopOutcome::Finished(Disposition::Failed { summary, .. }) => {
            format!("Failed — {summary}")
        }
        LoopOutcome::StoppedWithoutFinish => "StoppedWithoutFinish".to_string(),
        LoopOutcome::MaxIterations => "MaxIterations".to_string(),
        LoopOutcome::BackendError(err) => format!("BackendError — {err:?}"),
    }
}
