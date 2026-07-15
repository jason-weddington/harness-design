//! Claude Code eval runner: drives `claude` (Claude Code) over the **same
//! fixture substrate** the Talos `coding_eval` uses, scored by the **same
//! sealed-holdout gate**, so talos and claude-code rows are directly
//! comparable. Two endpoint modes:
//!
//! - `CLAUDE_CODE_ENDPOINT=ollama` (default): `claude` talks to GLM-5.2 via
//!   **Ollama Cloud** (`ANTHROPIC_BASE_URL=https://ollama.com`,
//!   `ANTHROPIC_AUTH_TOKEN=<OLLAMA_CLOUD_API_KEY>`). This is the original
//!   talos-glm vs claude-code-glm comparison path — byte-identical to the
//!   pre-endpoint-mode behavior.
//! - `CLAUDE_CODE_ENDPOINT=anthropic`: `claude` talks to the **real Anthropic
//!   API** via the ambient `ANTHROPIC_API_KEY` (no `base_url` override, no auth
//!   token injected). `CLAUDE_CODE_MODEL=claude-sonnet-5` produces the
//!   talos-sonnet vs claude-code-sonnet comparison.
//!
//! ## Product-comparison constraints (2026-07-12)
//!
//! - Talos runs with **all** its real advantages (claim-vs-verify, gates
//!   in-loop, finish-recovery). No ablation variant.
//! - Claude Code runs with **its** default agentic system prompt (`--system-prompt`
//!   is deliberately NOT passed — passing it would replace CC's default prompt
//!   and cripple it).
//! - `false_done` rate is a **first-class benchmark output**: CC's
//!   `passed == true` but `holdout_passed == Some(false)` count measures the
//!   gap that Talos's claim-vs-verify closes (structural 0 on Talos; measured
//!   here on CC).
//!
//! ## Runnable fixtures
//!
//! Fixtures lacking **either** a top-level `task.json` **or** a top-level
//! `holdout/` directory are skipped (one-line note printed per skip). Today's
//! runnable set: `calc`, `csv-ledger`, `taskdeck`, `walrus`. Fixtures
//! `broken-adder`, `interval-merge`, `lru-cache`, `text-preview` have neither
//! and are skipped.
//!
//! ## Scoring
//!
//! The holdout re-gate is identical to the Talos path: after `claude` exits,
//! `holdout/` contents are merged into the CC workspace and `cargo test` is
//! run there. Both paths call [`harness::eval::score_holdout`] — the single
//! shared external oracle (do-not-fork requirement).
//!
//! ## Usage
//!
//! ```text
//! # Ollama Cloud / GLM-5.2 (default endpoint mode)
//! OLLAMA_CLOUD_API_KEY=... cargo run --example claude_code_eval
//! OLLAMA_CLOUD_API_KEY=... CODING_EVAL_K=5 CLAUDE_CODE_MAX_TURNS=32 \
//!   cargo run --example claude_code_eval
//! OLLAMA_CLOUD_API_KEY=... CODING_EVAL_FIXTURE=calc \
//!   cargo run --example claude_code_eval
//!
//! # Real Anthropic / Sonnet
//! ANTHROPIC_API_KEY=sk-... CLAUDE_CODE_ENDPOINT=anthropic \
//!   CLAUDE_CODE_MODEL=claude-sonnet-5 cargo run --example claude_code_eval
//! ```
//!
//! ## Environment variables
//!
//! - `CLAUDE_CODE_ENDPOINT`  (optional) — `ollama` (default) or `anthropic`.
//!   Selects which backend `claude` talks to. Panics on any other value.
//! - `CLAUDE_CODE_MODEL`     (optional) — the `--model` arg passed to the
//!   `claude` binary in BOTH modes. Defaults to `glm-5.2:cloud` (the ollama
//!   default); set to e.g. `claude-sonnet-5` in anthropic mode.
//! - `OLLAMA_CLOUD_API_KEY`  (required in `ollama` mode) — Ollama Cloud API key.
//!   **Distinct** from `OLLAMA_API_KEY` (local key). Passed to the child
//!   process as `ANTHROPIC_AUTH_TOKEN`. NOT required in `anthropic` mode.
//! - `ANTHROPIC_API_KEY`    (required in `anthropic` mode) — the ambient
//!   Anthropic API key `claude` authenticates with. NOT required in `ollama`
//!   mode (the child's `ANTHROPIC_API_KEY` is `env_remove`d there so the auth
//!   token takes effect).
//! - `CLAUDE_CODE_MAX_TURNS` (optional) — `--max-turns` cap per CC run; default
//!   **24** (matches the Talos dispatch default for parity — a smaller budget
//!   would misattribute a budget gap as a capability gap; operators may
//!   override at runtime).
//! - `CODING_EVAL_K`         (optional) — trials per fixture; default 3.
//! - `CODING_EVAL_FIXTURE`   (optional) — narrow the run to a single named
//!   fixture directory under `fixtures/` (e.g. `calc`). Empty string treated
//!   as unset (same as `coding_eval.rs`).
//! - `CLAUDE_BIN`            (optional) — path to the `claude` binary; default
//!   `"claude"` (must be on `PATH`).
//!
//! **Note:** `num_turns` (CC) and Talos `iterations` are not identical
//! semantics but are the closest comparable column; wall-clock and token cost
//! are apples-to-apples. The `raw_in` summary column (`input_tokens` +
//! `cache_read_tokens` + `cache_write_tokens`) is the headline
//! harness-overhead number — identical computation on both runners so a
//! talos-sonnet row and a claude-code-sonnet row are directly comparable.

use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use harness::engine::{LoopOutcome, RunStats};
use harness::eval::{
    CODING_CHECK_TIMEOUT, EvalReport, TrialResult, copy_fixture_into_workspace, discover_fixtures,
    score_holdout,
};
use harness::exec::{CheckCommand, ChecksRunner};
use harness::prompt::render_task_prompt_from_spec;
use harness::run_record::{Disposition, FailureMode, Verification};
use harness::task_spec::TaskSpec;
use harness::tool::ToolCtx;
use harness::workspace::{DiskOffloadSink, Workspace};
use tempfile::TempDir;
use tokio::process::Command;

// ── Claude Code JSON result envelope ────────────────────────────────────────

/// The JSON object that `claude --output-format json` emits on stdout.
///
/// All fields are `Option`/`#[serde(default)]` so a missing or renamed field
/// degrades to `None` / empty and never panics — field additions in future CC
/// versions are invisible to this struct.
///
/// `type_`, `is_error`, and `total_cost_usd` are captured for telemetry /
/// spec completeness but are not used in the pass/fail decision; the
/// `#[allow]` silences the resulting dead-code lint.
#[allow(dead_code)]
#[derive(serde::Deserialize, Default)]
struct CcResult {
    /// Always `"result"` in the current CC schema; present for completeness.
    #[serde(rename = "type", default)]
    type_: String,
    /// One of `"success"` | `"error_max_turns"` | `"error_during_execution"`.
    /// `self_reported_done = subtype.as_deref() == Some("success")`.
    subtype: Option<String>,
    /// Whether CC considers this an error; recorded for telemetry only.
    is_error: Option<bool>,
    /// Turns CC consumed; mapped to `RunStats::iterations`.
    num_turns: Option<u32>,
    /// Wall-clock in milliseconds as reported by CC; mapped to `RunStats::wall_clock`.
    duration_ms: Option<u64>,
    /// CC's cost estimate; recorded for telemetry only.
    total_cost_usd: Option<f64>,
    /// CC's final summary message; used as `Disposition::Done.summary`.
    result: Option<String>,
    /// Token breakdown; mapped to `RunStats::{input,output}_tokens`.
    usage: Option<CcUsage>,
}

/// Token counts from the CC JSON envelope.
///
/// `cache_read_input_tokens` / `cache_creation_input_tokens` mirror the
/// Anthropic API usage shape: with prompt caching on, Anthropic MOVES cached
/// input out of `input_tokens` into these two buckets. Both are `Option` /
/// `#[serde(default)]` so ollama-mode runs (which report 0 / omit them)
/// degrade cleanly to 0 — the CC-side `RunStats.cache_*_tokens` then stays 0
/// and the `raw_in` summary column computes identically to the Talos side.
///
/// The field names mirror the Anthropic JSON keys verbatim (no `#[serde(rename)]`
/// needed) — the shared `_tokens` postfix is intentional, hence the allow.
#[allow(clippy::struct_field_names)]
#[derive(serde::Deserialize, Default)]
struct CcUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    /// Anthropic `cache_read_input_tokens` — cache hits. Absent in ollama mode.
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    /// Anthropic `cache_creation_input_tokens` — cache writes. Absent in
    /// ollama mode.
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

// ── Constants / helpers ──────────────────────────────────────────────────────

/// Default trial count when `CODING_EVAL_K` is not set.
const DEFAULT_K: u32 = 3;

/// Default `--max-turns` cap when `CLAUDE_CODE_MAX_TURNS` is not set.
///
/// Pinned to **24** for Talos-parity: the runnable task-spec fixtures were run
/// against Talos with `--max-iterations 24`. A smaller CC budget would
/// misattribute a budget gap as a capability gap.
const DEFAULT_MAX_TURNS: u32 = 24;

/// Default `--model` arg passed to the `claude` binary in BOTH endpoint modes
/// when `CLAUDE_CODE_MODEL` is unset. Pinned to the ollama-cloud GLM model so
/// the default-endpoint run is byte-identical to the pre-endpoint-mode
/// behavior. Set `CLAUDE_CODE_MODEL=claude-sonnet-5` for an anthropic-mode
/// Sonnet run.
const DEFAULT_CLAUDE_MODEL: &str = "glm-5.2:cloud";

/// Endpoint mode: which backend `claude` talks to.
///
/// `Ollama` is the default and is byte-identical to the pre-endpoint-mode
/// behavior (`ANTHROPIC_BASE_URL=https://ollama.com`, auth via
/// `OLLAMA_CLOUD_API_KEY`). `Anthropic` talks to the real Anthropic API via
/// the ambient `ANTHROPIC_API_KEY` — no `base_url` override, no auth token.
enum EndpointMode {
    Ollama,
    Anthropic,
}

impl EndpointMode {
    /// Read `CLAUDE_CODE_ENDPOINT` from the environment, defaulting to
    /// `Ollama`. Panics with a clear message on any other value.
    fn from_env() -> Self {
        match env::var("CLAUDE_CODE_ENDPOINT").as_deref() {
            Ok("ollama") | Err(_) => Self::Ollama,
            Ok("anthropic") => Self::Anthropic,
            Ok(other) => {
                panic!("CLAUDE_CODE_ENDPOINT must be `ollama` or `anthropic`, got `{other}`")
            }
        }
    }
}

/// Read a `u32` from the environment, falling back to `default` when the
/// variable is unset or unparsable. Same pattern as `coding_eval.rs`.
fn env_u32(name: &str, default: u32) -> u32 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

// ── Main ─────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    // Endpoint mode selects which backend `claude` talks to. Ollama mode
    // (default) is byte-identical to the pre-endpoint-mode behavior; anthropic
    // mode talks to the real Anthropic API via the ambient ANTHROPIC_API_KEY.
    let endpoint = EndpointMode::from_env();

    // OLLAMA_CLOUD_API_KEY is required ONLY in ollama mode (it is the auth
    // token passed to ollama.com). Anthropic-mode runs authenticate via the
    // ambient ANTHROPIC_API_KEY and do not touch ollama.com at all.
    //
    // The gate is here (not at the spawn site) so `cargo nextest run` — which
    // never executes examples — passes without the key, while a live
    // ollama-mode run fails fast with a clear message.
    let ollama_cloud_key = match endpoint {
        EndpointMode::Ollama => Some(
            env::var("OLLAMA_CLOUD_API_KEY")
                .expect("OLLAMA_CLOUD_API_KEY must be set in CLAUDE_CODE_ENDPOINT=ollama mode"),
        ),
        EndpointMode::Anthropic => {
            // Sanity: anthropic mode REQUIRES the ambient ANTHROPIC_API_KEY.
            // Fail fast here rather than letting claude fail opaquely later.
            env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY must be set in CLAUDE_CODE_ENDPOINT=anthropic mode");
            None
        }
    };

    // The `--model` arg passed to `claude` in BOTH modes.
    let claude_model =
        env::var("CLAUDE_CODE_MODEL").unwrap_or_else(|_| DEFAULT_CLAUDE_MODEL.to_string());

    let k = env_u32("CODING_EVAL_K", DEFAULT_K);
    let max_turns = env_u32("CLAUDE_CODE_MAX_TURNS", DEFAULT_MAX_TURNS);
    // Empty string is treated as "unset" — `VAR= cmd` clears the filter.
    let fixture_filter = env::var("CODING_EVAL_FIXTURE")
        .ok()
        .filter(|s| !s.is_empty());
    let claude_bin = env::var("CLAUDE_BIN").unwrap_or_else(|_| "claude".into());

    // Fixtures live at the repo root under `fixtures/`.  This example's
    // manifest dir is `crates/harness`, so climb two levels — identical to
    // `coding_eval.rs`.
    let fixtures_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures");

    let all_fixtures: Vec<PathBuf> = if let Some(name) = fixture_filter.as_deref() {
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

    // Filter: skip any fixture that lacks EITHER a top-level `task.json` OR a
    // top-level `holdout/` dir — both are required for apples-to-apples
    // scoring.  Print a one-line note per skip so it's visible in CI output.
    let fixtures: Vec<PathBuf> = all_fixtures
        .into_iter()
        .filter(|f| {
            let name = f
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("<unnamed>");
            let has_task_json = f.join("task.json").is_file();
            let has_holdout = f.join("holdout").is_dir();
            if !has_task_json || !has_holdout {
                println!(
                    "skip {name}: missing {}",
                    match (has_task_json, has_holdout) {
                        (false, false) => "task.json and holdout/",
                        (false, true) => "task.json",
                        (true, false) => "holdout/",
                        (true, true) => unreachable!(),
                    }
                );
                false
            } else {
                true
            }
        })
        .collect();

    assert!(
        !fixtures.is_empty(),
        "no runnable fixtures found (all lack task.json and/or holdout/) under {}",
        fixtures_root.display(),
    );

    println!(
        "claude_code_eval: {} fixture(s) (k={k}, max_turns={max_turns}, bin={claude_bin}, \
         endpoint={}, model={claude_model})",
        fixtures.len(),
        match endpoint {
            EndpointMode::Ollama => "ollama",
            EndpointMode::Anthropic => "anthropic",
        },
    );

    let mut summary: Vec<(String, EvalReport)> = Vec::with_capacity(fixtures.len());

    for fixture in &fixtures {
        let fixture_name = fixture
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unnamed>")
            .to_string();

        // Read task.json → TaskSpec → rendered task prompt (byte-identical to
        // what Talos receives through coding_fix_task).
        let task_json_path = fixture.join("task.json");
        let task_json = std::fs::read_to_string(&task_json_path).unwrap_or_else(|e| {
            panic!("read {}: {e}", task_json_path.display());
        });
        let spec: TaskSpec = serde_json::from_str(&task_json).unwrap_or_else(|e| {
            panic!("parse {} as TaskSpec: {e}", task_json_path.display());
        });
        let task_prompt = render_task_prompt_from_spec(&spec);
        let holdout_src = fixture.join("holdout");

        println!(
            "\n=== fixture: {fixture_name} ===\n  path: {}\n",
            fixture.display(),
        );

        let mut trial_results: Vec<TrialResult> = Vec::with_capacity(k as usize);
        let mut passes: u32 = 0;

        for i in 0..k {
            // Fresh isolated workspace per trial — CC edits here, holdout is
            // NEVER copied in before claude exits (eval-content ban, kb-02965).
            let ws_tmp = TempDir::new().expect("create CC trial workspace");
            let off_tmp = TempDir::new().expect("create trial offload dir");

            // Copy fixture (excluding top-level task.json, holdout/, target/,
            // Cargo.lock) via the single-sourced pub fn — do NOT reimplement.
            copy_fixture_into_workspace(fixture, ws_tmp.path())
                .expect("copy fixture into CC trial workspace");

            // Build the workspace + ToolCtx for the holdout scorer.  Must use
            // Workspace::new (not stub) so score_holdout's file-copy side-effect
            // lands in a real, observable path.
            let workspace = Workspace::new(ws_tmp.path(), Some(off_tmp.path().to_path_buf()))
                .expect("create trial Workspace");
            let ws_root = workspace.root().to_path_buf(); // canonicalized
            let off_canon = off_tmp.path().canonicalize().expect("canon offload dir");
            let ctx = ToolCtx::new(
                Arc::new(workspace),
                Arc::new(DiskOffloadSink::new(off_canon)),
            );

            // The IDENTICAL cargo-test gate + timeout the Talos path uses.
            let checks = ChecksRunner::new(
                CheckCommand {
                    program: "cargo".to_string(),
                    args: vec!["test".to_string()],
                },
                ws_root.clone(),
                CODING_CHECK_TIMEOUT,
            );

            // Spawn CC. Env wiring depends on endpoint mode:
            //   ollama     → ANTHROPIC_BASE_URL=https://ollama.com,
            //               ANTHROPIC_AUTH_TOKEN=<OLLAMA_CLOUD_API_KEY>,
            //               ANTHROPIC_API_KEY env_removed (so the auth token
            //               takes effect). Byte-identical to the pre-endpoint-
            //               mode behavior.
            //   anthropic  → no base_url override, no auth token, no env_remove:
            //               claude authenticates against the real Anthropic API
            //               via the ambient ANTHROPIC_API_KEY.
            // `--model` comes from CLAUDE_CODE_MODEL in BOTH modes. No
            // --system-prompt: CC runs with its own default agentic system
            // prompt (product-comparison DoR — each harness keeps its real
            // advantages).
            let mut cmd = Command::new(&claude_bin);
            cmd.current_dir(&ws_root).args([
                "--model",
                &claude_model,
                "--dangerously-skip-permissions",
                "--max-turns",
                &max_turns.to_string(),
                "--output-format",
                "json",
                "--print",
                &task_prompt,
            ]);
            match endpoint {
                EndpointMode::Ollama => {
                    cmd.env("ANTHROPIC_BASE_URL", "https://ollama.com")
                        .env(
                            "ANTHROPIC_AUTH_TOKEN",
                            ollama_cloud_key
                                .as_ref()
                                .expect("ollama mode requires OLLAMA_CLOUD_API_KEY"),
                        )
                        .env_remove("ANTHROPIC_API_KEY");
                }
                EndpointMode::Anthropic => {
                    // No base_url override; no auth token; no env_remove. The
                    // ambient ANTHROPIC_API_KEY (validated at startup) is what
                    // claude authenticates with.
                }
            }
            let child_output = cmd.output().await.expect("spawn claude binary");

            // Parse CC's JSON envelope.  On parse failure (e.g. empty stdout,
            // binary crash) fall back to a zero-filled default — the subtype
            // will be None → treated as MaxIterations.
            let cc: CcResult = serde_json::from_slice(&child_output.stdout).unwrap_or_default();

            // self_reported_done is derived from subtype ALONE; is_error /
            // num_turns / cost / duration do NOT participate (false-done
            // semantics 1:1 with Talos rows).
            let self_reported_done = cc.subtype.as_deref() == Some("success");
            if self_reported_done {
                passes += 1;
            }

            // Holdout re-gate: copy holdout/ into the CC workspace (where CC
            // has left its edits) and run cargo test.  This is the EXTERNAL
            // oracle — identical to what run_eval runs for Talos rows.
            // Scoring runs AFTER claude exits.
            let holdout_passed = score_holdout(&holdout_src, &checks, &ctx).await;

            // Map CC subtype to the LoopOutcome shape EvalReport/Talos rows use.
            let result_str = cc.result.unwrap_or_default();
            let outcome = match cc.subtype.as_deref() {
                Some("success") => LoopOutcome::Finished(Disposition::Done {
                    summary: result_str,
                    verification: Verification::NoChecksConfigured,
                }),
                Some("error_during_execution") => LoopOutcome::Finished(Disposition::Failed {
                    mode: FailureMode::PersistentToolError,
                    summary: result_str,
                }),
                // "error_max_turns" and any unknown/missing subtype → MaxIterations.
                _ => LoopOutcome::MaxIterations,
            };

            let input_tokens = cc.usage.as_ref().and_then(|u| u.input_tokens).unwrap_or(0);
            let output_tokens = cc.usage.as_ref().and_then(|u| u.output_tokens).unwrap_or(0);
            // Cache tokens from claude's JSON usage — absent in ollama mode
            // (default 0), populated by the real Anthropic API when caching is
            // on. Aggregated into the CC-side RunStats so the summary's
            // cache_rd / cache_wr / raw_in columns compute identically to the
            // Talos side.
            let cache_read_tokens = cc
                .usage
                .as_ref()
                .and_then(|u| u.cache_read_input_tokens)
                .unwrap_or(0);
            let cache_write_tokens = cc
                .usage
                .as_ref()
                .and_then(|u| u.cache_creation_input_tokens)
                .unwrap_or(0);

            let stats = RunStats {
                iterations: cc.num_turns.unwrap_or(0),
                input_tokens,
                output_tokens,
                wall_clock: Duration::from_millis(cc.duration_ms.unwrap_or(0)),
                // CC has no observable in-loop gate; the external oracle lives
                // solely in the holdout column.
                gates_green_at_exit: false,
                cache_read_tokens,
                cache_write_tokens,
            };

            let trial = TrialResult {
                trial: i,
                passed: self_reported_done,
                holdout_passed: Some(holdout_passed),
                outcome,
                stats,
            };

            println!(
                "  trial {}: {} | {}",
                i + 1,
                outcome_one_liner(&trial.outcome),
                stats_one_liner(&trial.stats),
            );

            trial_results.push(trial);
            // ws_tmp and off_tmp drop here, removing the CC workspace.
        }

        let pass_rate = if k == 0 {
            0.0
        } else {
            // u32 → f64 is lossless; the pedantic-clippy-clean form.
            f64::from(passes) / f64::from(k)
        };
        let report = EvalReport {
            task_name: fixture_name.clone(),
            trials: k,
            passes,
            pass_rate,
            trial_results,
        };

        println!("\n{report:#?}");
        summary.push((fixture_name, report));
    }

    // Final one-line-per-fixture summary table (same columns as coding_eval.rs
    // so talos-glm and claude-code-glm rows line up 1:1).
    let name_col = summary
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("fixture".len());

    print_summary(&summary, name_col);
}

// ── Print helpers (duplicated verbatim from coding_eval.rs) ─────────────────

/// Render the final one-line-per-fixture summary table.
fn print_summary(summary: &[(String, EvalReport)], name_col: usize) {
    println!("\n=== SUMMARY ===");
    println!(
        "{:<name_col$}  {:>9}  {:>10}  {:>10}  {:>12}  {:>9}  {:>11}  {:>8}  {:>9}  {:>9}  {:>9}",
        "fixture",
        "passes/k",
        "pass_rate",
        "mean_iters",
        "total_tokens",
        "mean_wall",
        "holdout",
        "false_dn",
        "cache_rd",
        "cache_wr",
        "raw_in",
    );
    for (name, r) in summary {
        let total_tokens = r.total_input_tokens() + r.total_output_tokens();
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
            "{name:<name_col$}  {:>9}  {:>10.3}  {:>10.2}  {:>12}  {:>8.1}s  {:>11}  {:>8}  {:>9}  {:>9}  {:>9}",
            format!("{}/{}", r.passes, r.trials),
            r.pass_rate,
            r.mean_iterations(),
            format_tokens_compact(total_tokens),
            mean_wall,
            holdout_col,
            r.false_dones(),
            format_tokens_compact(r.total_cache_read_tokens()),
            format_tokens_compact(r.total_cache_write_tokens()),
            format_tokens_compact(r.total_raw_input_tokens()),
        );
    }
}

/// A terse one-line summary of a trial's [`RunStats`] for the per-trial log
/// line — iterations, in/out tokens, and wall-clock.
fn stats_one_liner(stats: &RunStats) -> String {
    format!(
        "{} iters | {} in / {} out | {}s | gate_green_at_exit={}",
        stats.iterations,
        format_tokens_compact(stats.input_tokens),
        format_tokens_compact(stats.output_tokens),
        stats.wall_clock.as_secs(),
        stats.gates_green_at_exit,
    )
}

/// Render a token count compactly: below `1_000` as a bare integer, otherwise
/// as `NN.Nk`.
fn format_tokens_compact(n: u64) -> String {
    if n < 1_000 {
        return n.to_string();
    }
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
        LoopOutcome::BudgetExhausted { summary } => format!("BudgetExhausted — {summary}"),
        LoopOutcome::BackendError(err) => format!("BackendError — {err:?}"),
    }
}
