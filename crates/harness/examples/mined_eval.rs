//! Live mined-task eval runner: score an agent on ~SWE-bench-shaped tasks
//! hand-mined from real git history.
//!
//! This is the tier-2 counterpart of `examples/coding_eval.rs`. It reads
//! task dirs from a `talos-evals`-shaped root (see `crate::mined_eval` for
//! the schema) and runs `k` independent trials per task. Each trial:
//!
//! 1. Lays down a fresh `git worktree` for the primary repo at `parent_sha`,
//!    plus one for each sibling repo at its `pin`.
//! 2. Runs `env.setup` under a 10-minute timeout.
//! 3. Runs the agent loop with `checks=None` — a `finish(done)` claim is
//!    accepted on trust. Verdicts come 100% from the sealed re-gate.
//! 4. Overwrites the agent's copies of the `sealed[]` files with the mined
//!    truth and re-runs `gate_command` with `PYTEST_ADDOPTS="-rA"` so pytest
//!    emits `PASSED/FAILED/...` short-summary lines the scorer parses.
//! 5. Scores test ids against `fail_to_pass` + `positive_controls` +
//!    `pass_to_pass_exclusions` (never an exit code).
//!
//! It talks to the live Anthropic / Ollama API and shells out to `git` +
//! `uv`, so it is **not** wired into the quality gates (the build host has
//! no API key and no external repo access). The gates only verify that
//! this file *compiles*; run it by hand:
//!
//! ```text
//! # Anthropic (default backend), one task at S2 with k=3
//! ANTHROPIC_API_KEY=sk-... \
//!   MINED_EVAL_TASKS_DIR=~/git/talos-evals/tasks \
//!   MINED_EVAL_TASK=cleanr-<sha> \
//!   cargo run --example mined_eval
//!
//! # Sample the spec grid: run the same task at S1 instead
//! ANTHROPIC_API_KEY=sk-... \
//!   MINED_EVAL_TASKS_DIR=~/git/talos-evals/tasks \
//!   MINED_EVAL_TASK=cleanr-<sha> \
//!   MINED_EVAL_SPEC_LEVEL=s1 MINED_EVAL_K=5 \
//!   cargo run --example mined_eval
//!
//! # Ollama cloud (GLM-5.2)
//! EVAL_BACKEND=ollama OLLAMA_BASE_URL=https://ollama.com \
//!   OLLAMA_MODEL=glm-5.2:cloud OLLAMA_API_KEY=... \
//!   MINED_EVAL_TASKS_DIR=~/git/talos-evals/tasks \
//!   cargo run --example mined_eval
//! ```
//!
//! Environment:
//! - `MINED_EVAL_TASKS_DIR` (required) — task-family root. NO default; must
//!   point at the talos-evals task dir on the host. This crate never
//!   references it directly (contamination hygiene — talos-evals content
//!   stays out of harness-design).
//! - `MINED_EVAL_TASK` (optional) — narrow to one task id (a subdirectory
//!   name). When unset, every subdirectory of `MINED_EVAL_TASKS_DIR` is
//!   discovered and run in sorted order.
//! - `MINED_EVAL_SPEC_LEVEL` (optional) — `s1|s2|s3`, defaults to `s2`.
//! - `MINED_EVAL_K` (optional) — number of trials; defaults to 3.
//! - `MINED_EVAL_MAX_ITERATIONS` (optional) — per-trial agent-loop cap;
//!   defaults to 24 (matches the talos dispatch default).
//! - `EVAL_BACKEND` / `ANTHROPIC_*` / `OLLAMA_*` — same shape as
//!   `examples/coding_eval.rs`. Kept as a duplicated helper (`backend_from_env`)
//!   rather than extracted to the lib, because pulling it into the lib would
//!   couple the harness library to the eval-runner environment schema (the
//!   env-var names are pilot-scoped and might churn — pinning them in the
//!   example file makes rebasing easier for now).
//!
//! ## Task-dir schema
//!
//! Full schema lives in the [`mined_eval`](harness::mined_eval) module docs.

use std::env;
use std::path::PathBuf;

use async_trait::async_trait;
use harness::anthropic::AnthropicBackend;
use harness::mined_eval::{
    self, CLAIMED_DONE, MinedReport, MinedRunConfig, MinedTrialResult, PytestParser, SpecLevel,
    TrialScore, load_statement, load_task,
};
use harness::model::{AssistantTurn, BackendError, ModelBackend, TurnRequest};
use harness::ollama::{OllamaBackend, ThinkLevel, resolve_context_length};

/// Default model id when `ANTHROPIC_MODEL` is not set.
const DEFAULT_MODEL: &str = "claude-haiku-4-5";

/// Audit floor for the resolved `num_ctx`. This is NOT a default or fallback —
/// it is never assigned to `num_ctx`. When the resolved value is below this
/// threshold the runner emits a warning to stderr; the run proceeds with the
/// verbatim advertised value.
const MIN_EXPECTED_NUM_CTX: u32 = 32_768;

/// Default trial count (`k`) when `MINED_EVAL_K` is not set.
const DEFAULT_K: u32 = 3;

/// Default per-trial iteration cap. Matches talos dispatch's `max_iterations`.
const DEFAULT_MAX_ITERATIONS: u32 = 24;

/// Default spec level when `MINED_EVAL_SPEC_LEVEL` is not set.
const DEFAULT_SPEC_LEVEL: SpecLevel = SpecLevel::S2;

/// Example-local backend selection (same shape as `coding_eval.rs`).
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

/// Build the backend from the environment (see module docs). Returns the
/// backend plus a human-readable description for the run header.
///
/// NOTE (DRY): this is a near-verbatim clone of `coding_eval.rs`'s
/// `backend_from_env`. Extracting it to the lib would couple the harness
/// library to eval-runner env names (which are pilot-scoped and may churn);
/// leaving it duplicated in the two example runners keeps churn localised.
/// `num_ctx` resolution itself now lives in `harness::ollama` and is
/// deliberately not duplicated — both runners call `resolve_context_length`.
async fn backend_from_env() -> (Backend, String) {
    match env::var("EVAL_BACKEND").as_deref() {
        Ok("ollama") => {
            let model = env::var("OLLAMA_MODEL")
                .expect("OLLAMA_MODEL must be set when EVAL_BACKEND=ollama");
            let base_url =
                env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".into());
            let is_local = base_url.contains("localhost") || base_url.contains("127.0.0.1");

            // OLLAMA_API_KEY is hoisted ABOVE num_ctx resolution so its value
            // can be forwarded to the /api/show probe.
            let api_key: Option<String> = env::var("OLLAMA_API_KEY").ok().filter(|s| !s.is_empty());

            // Five-branch OLLAMA_NUM_CTX resolution:
            // (0) set but empty/whitespace after trim → treated as unset (falls through to (3)/(4)).
            //     Matches the file-local convention for MINED_EVAL_SPEC_LEVEL
            //     (empty string treated as 'unset' — the shell's `VAR= cmd` idiom).
            // (1) set, non-empty, parses as u32 → use it, no HTTP call.
            // (2) set, non-empty, does NOT parse as u32 → panic! naming the var and raw value.
            // (3) unset/empty AND is_local → probe /api/show; panic! on error (never fall back).
            // (4) unset/empty AND NOT is_local → None (no probe; unchanged behavior).
            let raw_num_ctx = env::var("OLLAMA_NUM_CTX").ok();
            let (num_ctx, num_ctx_desc) = match raw_num_ctx.as_deref().map(str::trim) {
                Some(trimmed) if !trimmed.is_empty() => {
                    let n = trimmed.parse::<u32>().unwrap_or_else(|_| {
                        panic!(
                            "OLLAMA_NUM_CTX must be a valid u32, got `{}`",
                            raw_num_ctx.as_deref().unwrap_or("")
                        )
                    });
                    (Some(n), format!("num_ctx={n} (explicit OLLAMA_NUM_CTX)"))
                }
                _ if is_local => {
                    let resolved = resolve_context_length(&base_url, &model, api_key.as_deref())
                        .await
                        .unwrap_or_else(|e| panic!("{e}"));
                    let v = resolved.value;
                    let a = resolved.architecture;
                    let k = resolved.key;
                    if v < MIN_EXPECTED_NUM_CTX {
                        eprintln!(
                            "WARNING: resolved num_ctx={v} for `{model}` (arch={a}, key={k}) \
                             is BELOW the {MIN_EXPECTED_NUM_CTX} sanity floor; trials may be \
                             truncation-invalid — set OLLAMA_NUM_CTX to override"
                        );
                        (
                            Some(v),
                            format!("num_ctx={v} (resolved, BELOW-FLOOR: arch={a} key={k})"),
                        )
                    } else {
                        (Some(v), format!("num_ctx={v} (resolved: arch={a} key={k})"))
                    }
                }
                _ => (None, "num_ctx=default".to_string()),
            };

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
            if let Some(key) = api_key {
                backend = backend.with_api_key(key);
            }
            if let Some(n) = num_ctx {
                backend = backend.with_num_ctx(n);
            }
            if let Some(level) = think {
                backend = backend.with_think(level);
            }
            let desc = format!(
                "ollama `{model}` @ {base_url} ({num_ctx_desc}, think={})",
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

/// Read a `u32` from the environment, falling back to `default`.
fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

/// Expand a leading `~/` (or `~`) against `$HOME`. Non-tilde input passes
/// through unchanged.
fn expand_home(raw: &str) -> String {
    let Some(home) = env::var_os("HOME").map(PathBuf::from) else {
        return raw.to_string();
    };
    if raw == "~" {
        return home.to_string_lossy().into_owned();
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return home.join(rest).to_string_lossy().into_owned();
    }
    raw.to_string()
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let (backend, backend_desc) = backend_from_env().await;
    let k = env_u32("MINED_EVAL_K", DEFAULT_K);
    let max_iterations = env_u32("MINED_EVAL_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS);
    let spec_level = env::var("MINED_EVAL_SPEC_LEVEL")
        .ok()
        .filter(|s| !s.is_empty())
        .map_or(DEFAULT_SPEC_LEVEL, |raw| {
            SpecLevel::parse(&raw)
                .unwrap_or_else(|| panic!("MINED_EVAL_SPEC_LEVEL must be s1|s2|s3, got `{raw}`"))
        });

    let tasks_root_raw = env::var("MINED_EVAL_TASKS_DIR")
        .expect("MINED_EVAL_TASKS_DIR must be set (path to the talos-evals task-family root)");
    let tasks_root = PathBuf::from(expand_home(&tasks_root_raw));
    assert!(
        tasks_root.is_dir(),
        "MINED_EVAL_TASKS_DIR does not resolve to a directory: {}",
        tasks_root.display(),
    );

    let filter = env::var("MINED_EVAL_TASK").ok().filter(|s| !s.is_empty());

    let task_dirs: Vec<PathBuf> = if let Some(name) = filter.as_deref() {
        let path = tasks_root.join(name);
        assert!(
            path.is_dir(),
            "MINED_EVAL_TASK={name} does not resolve to a directory under {}",
            tasks_root.display(),
        );
        vec![path]
    } else {
        discover_task_dirs(&tasks_root)
    };
    assert!(
        !task_dirs.is_empty(),
        "no task dirs found under {}",
        tasks_root.display(),
    );

    println!(
        "running mined_eval across {} task(s) (k={k}, spec_level={:?}, max_iterations={max_iterations}) against {backend_desc}",
        task_dirs.len(),
        spec_level,
    );

    let parser = PytestParser;
    let mut summary: Vec<MinedReport> = Vec::with_capacity(task_dirs.len());
    for task_dir in &task_dirs {
        let task = load_task(task_dir).unwrap_or_else(|e| {
            panic!("load task.json at {}: {e}", task_dir.display());
        });
        let statement = load_statement(task_dir, spec_level).unwrap_or_else(|e| {
            panic!("load statement for {}: {e}", task_dir.display());
        });

        println!(
            "\n=== task: {} (rung_guess={}) ===\n  dir: {}\n",
            task.id,
            task.rung_guess,
            task_dir.display(),
        );

        let config = MinedRunConfig {
            task_dir,
            task: &task,
            statement: &statement,
            spec_level,
            backend_desc: backend_desc.clone(),
            k,
            max_iterations,
        };
        let mut on_trial = |trial: &MinedTrialResult| {
            // On a tier-2 run today fr_armed is expected false, green_at_exit
            // false, and nudges 0 on EVERY trial (correct behaviour, not a
            // wiring bug — see the doc-comments on MinedTrialResult::gates_green_at_exit
            // and MinedTrialResult::nudges_fired). The discriminating columns
            // are mut_iters, bash_ok, edits_ok, static_iters, and peak_static.
            println!(
                "  trial {}: {} | claimed={} | {} iters | {}s | fr_armed={} | green_at_exit={} | nudges={} | tree_dirty={} | static_iters={} | peak_static={} | mut_iters={} | bash_ok={} | edits_ok={} | gate_output: {}",
                trial.trial + 1,
                trial_score_one_liner(&trial.score),
                trial.claimed_disposition,
                trial.iterations,
                trial.wall.as_secs(),
                trial.finish_recovery_armed,
                trial.gates_green_at_exit,
                trial.nudges_fired,
                trial.tree_dirty,
                trial.iters_since_tree_change_at_exit,
                trial.peak_iters_since_tree_change,
                trial.mutating_iters,
                trial.bash_calls_ok,
                trial.edit_file_calls_ok,
                trial.gate_output_path.display(),
            );
        };
        let report = mined_eval::run_mined_task(&backend, &parser, &config, &mut on_trial).await;
        println!(
            "\n  task summary: resolved {}/{} valid (invalid: {}), false_dones={}",
            report.resolved_count,
            report.valid_denominator(),
            report.invalid_count,
            report.false_dones(),
        );
        summary.push(report);
    }

    print_summary(&summary, &backend_desc, spec_level, max_iterations, k);
}

/// Discover task dirs under `root`, sorted by path.
fn discover_task_dirs(root: &std::path::Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap_or_else(|e| panic!("read_dir({}): {e}", root.display()))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .collect();
    dirs.sort();
    dirs
}

/// A terse one-line description of a [`TrialScore`] for the per-trial log.
fn trial_score_one_liner(score: &TrialScore) -> String {
    match score {
        TrialScore::Resolved => "Resolved".to_string(),
        TrialScore::Unresolved { reason } => {
            format!(
                "Unresolved (missing_ftp={}, unexcluded_red={})",
                reason.missing_fail_to_pass.len(),
                reason.unexcluded_red.len(),
            )
        }
        TrialScore::Invalid { reason } => format!("Invalid ({reason})"),
    }
}

/// Render the final one-line-per-task summary table.
///
/// The header records backend/model/think + `spec_level` + `max_iterations` + k
/// so a printed summary is self-describing (per kb-02909: never compare
/// across a think/level change).
fn print_summary(
    summary: &[MinedReport],
    backend_desc: &str,
    spec_level: SpecLevel,
    max_iterations: u32,
    k: u32,
) {
    let name_col = summary
        .iter()
        .map(|r| r.task_id.len())
        .max()
        .unwrap_or(0)
        .max("task".len());

    println!(
        "\n=== SUMMARY (backend={backend_desc}, spec_level={spec_level:?}, max_iterations={max_iterations}, k={k}, static_tree_k={}, max_nudges={}) ===",
        harness::engine::DEFAULT_STATIC_TREE_K,
        harness::engine::DEFAULT_MAX_NUDGES,
    );
    println!(
        "{:<name_col$}  {:>12}  {:>9}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "task",
        "resolved/val",
        "res_rate",
        "invalid",
        "false_dn",
        "grn_stop",
        "res_unclm",
        "mut_rate",
        "peak_stat",
        "claimed_D",
        "mean_iter",
    );
    for r in summary {
        let claimed_done: u32 = r
            .trials
            .iter()
            .filter(|t| t.claimed_disposition == CLAIMED_DONE)
            .map(|_| 1u32)
            .sum();
        // `usize → f64` for the divisor: trial counts can't approach f64's
        // precision limit (same rationale as EvalReport::mean_iterations).
        #[allow(clippy::cast_precision_loss)]
        let mean_iter = if r.trials.is_empty() {
            0.0
        } else {
            let sum: u64 = r.trials.iter().map(|t| u64::from(t.iterations)).sum();
            sum as f64 / r.trials.len() as f64
        };
        #[allow(clippy::cast_precision_loss)]
        let mut_rate = {
            let iter_sum: u64 = r.trials.iter().map(|t| u64::from(t.iterations)).sum();
            let mut_sum: u64 = r.trials.iter().map(|t| u64::from(t.mutating_iters)).sum();
            if iter_sum == 0 {
                0.0_f64
            } else {
                mut_sum as f64 / iter_sum as f64
            }
        };
        let peak_stat: u32 = r
            .trials
            .iter()
            .map(|t| t.peak_iters_since_tree_change)
            .max()
            .unwrap_or(0);
        println!(
            "{:<name_col$}  {:>12}  {:>9.3}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9.3}  {:>9}  {:>9}  {:>9.2}",
            r.task_id,
            format!("{}/{}", r.resolved_count, r.valid_denominator()),
            r.resolved_rate(),
            r.invalid_count,
            r.false_dones(),
            r.post_green_stops(),
            r.resolved_unclaimed(),
            mut_rate,
            peak_stat,
            claimed_done,
            mean_iter,
        );
    }
}
