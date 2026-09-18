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
//! ANTHROPIC_API_KEY=sk-... ANTHROPIC_MODEL=claude-sonnet-5 CODING_EVAL_K=5 \
//!   cargo run --example coding_eval
//! # Ollama cloud (GLM-5.3)
//! EVAL_BACKEND=ollama OLLAMA_BASE_URL=https://ollama.com OLLAMA_MODEL=glm-5.3:cloud \
//!   cargo run --example coding_eval
//! # Ollama localhost (small local models; num_ctx resolved from /api/show)
//! EVAL_BACKEND=ollama OLLAMA_MODEL=qwen3.6:35b cargo run --example coding_eval
//! # narrow to a single fixture directory name
//! ANTHROPIC_API_KEY=sk-... CODING_EVAL_FIXTURE=lru-cache cargo run --example coding_eval
//! ```
//!
//! Environment:
//! - `EVAL_BACKEND`          (optional) — `anthropic` (default) or `ollama`.
//! - `ANTHROPIC_API_KEY`     (required for anthropic) — passed to the backend.
//! - `ANTHROPIC_MODEL`       (optional) — defaults to `claude-haiku-4-5`.
//! - `OLLAMA_MODEL`          (required for ollama) — e.g. `glm-5.3:cloud`,
//!   `qwen3.6:35b`, `gpt-oss:20b`. Never hardcoded.
//! - `OLLAMA_BASE_URL`       (optional) — defaults to `http://localhost:11434`;
//!   set `https://ollama.com` for Ollama cloud.
//! - `OLLAMA_API_KEY`        (optional) — attached as a Bearer token when set
//!   (required in practice for Ollama cloud).
//! - `OLLAMA_NUM_CTX`        (optional) — context window override. When unset
//!   for a localhost URL, the runner probes `POST /api/show` and pins the
//!   model's own advertised `{arch}.context_length`; if the probe fails the
//!   runner panics with the full error (fail-loud — never falls back to a
//!   constant). When set, the value is used verbatim and no probe is made; an
//!   unparsable value panics naming `OLLAMA_NUM_CTX` and the raw string.
//!   Use this as an escape hatch in BOTH directions: to pin a smaller window
//!   when the advertised value exceeds available VRAM, or to set a window for
//!   non-localhost daemons (e.g. `http://jason-desktop:11434`, a LAN address)
//!   that are neither matched by `is_local` nor probed — they inherit Ollama's
//!   own low default (~2048, with silent oldest-message dropping). An empty
//!   value (`OLLAMA_NUM_CTX= cmd`) is treated as unset.
//!   Note: `crates/talos/src/main.rs` still pins 32 768 for localhost (no
//!   probe, sync path). Eval `num_ctx` rows are NOT comparable to talos
//!   dispatch rows until that follow-up lands.
//! - `OLLAMA_THINK`          (optional) — `off|on|low|medium|high|max`
//!   (gpt-oss ignores plain booleans; GLM-5.3 supports low/high/max and defaults to max).
//! - `CODING_EVAL_K`         (optional) — number of trials; defaults to 3.
//! - `CODING_EVAL_FIXTURE`   (optional) — narrows the run to the single named
//!   fixture directory under `fixtures/` (e.g. `lru-cache`). When unset, every
//!   directory under `fixtures/` is discovered and run in sorted order.
//! - `CODING_EVAL_MAX_ITERATIONS` (optional) — per-trial agent-loop cap;
//!   defaults to 12. The task-spec-shaped tiers (taskdeck, calc) benefit from
//!   more headroom on small models — 24 matches the talos dispatch default.
//! - `CODING_EVAL_TRANSCRIPTS` (optional) — `1` = on, opt-in full run
//!   transcript per trial (see `harness::transcript`), written under
//!   `<state-root>/talos/coding-eval/<unix-secs>-<pid>/<fixture>/trial-<i>.jsonl`;
//!   `0`/empty/unset = off (the default). See
//!   `harness::transcript::parse_transcripts_flag`.

use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;

use async_trait::async_trait;
use harness::anthropic::AnthropicBackend;
use harness::engine::{LoopOutcome, RunStats};
use harness::eval::{
    EvalReport, EvalTranscripts, TrialResult, coding_fix_task_with, discover_fixtures,
    run_eval_with_transcripts,
};
use harness::model::{AssistantTurn, BackendError, ModelBackend, TurnRequest};
use harness::ollama::{OllamaBackend, ThinkLevel, resolve_context_length};
use harness::run_record::{Disposition, Verification};

/// Default model id when `ANTHROPIC_MODEL` is not set.
const DEFAULT_MODEL: &str = "claude-haiku-4-5";

/// Audit floor for the resolved `num_ctx`. This is NOT a default or fallback —
/// it is never assigned to `num_ctx`. When the resolved value is below this
/// threshold the runner emits a warning to stderr and the provenance token in
/// `desc` uses the BELOW-FLOOR variant; the run still proceeds with the
/// verbatim advertised value.
const MIN_EXPECTED_NUM_CTX: u32 = 32_768;

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
            //     Matches the file-local convention for CODING_EVAL_FIXTURE
            //     ("Empty string is treated as 'unset' — the shell's `VAR= cmd` idiom").
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

/// Default trial count when `CODING_EVAL_K` is not set (env-overridable).
const DEFAULT_K: u32 = 3;

/// Default per-trial hard cap on agent-loop iterations when
/// `CODING_EVAL_MAX_ITERATIONS` is not set (env-overridable). A fix-one-bug
/// task needs a few read/edit/verify rounds; the harder task-spec-shaped
/// fixtures (implement-to-spec, write-your-own-tests) need more headroom.
const DEFAULT_MAX_ITERATIONS: u32 = 12;

/// Root for `CODING_EVAL_TRANSCRIPTS` output:
/// `<state-root>/talos/coding-eval/<unix-secs>-<pid>`. `state-root` follows
/// the same base precedence as `harness::mined_eval::default_state_root`
/// (`XDG_STATE_HOME`, else `HOME/.local/state`, else the process temp dir),
/// but this copy has no `TALOS_MINED_STATE_ROOT`-equivalent override branch.
/// Deliberately duplicated rather than shared — unifying the three XDG-root
/// copies in this repo is out of scope for this item. Computed ONCE in
/// `main` so every fixture/trial in this process shares the same root.
fn coding_eval_transcripts_root() -> PathBuf {
    let state_root = env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .unwrap_or_else(env::temp_dir);
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    state_root
        .join("talos/coding-eval")
        .join(format!("{secs}-{}", std::process::id()))
}

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
    let (backend, backend_desc) = backend_from_env().await;
    let k = env_u32("CODING_EVAL_K", DEFAULT_K);
    let max_iterations = env_u32("CODING_EVAL_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS);
    // Test-first guidance is ON by default, matching what talos dispatch ships.
    // `CODING_EVAL_TEST_FIRST=0` strips it so the same fixtures can be run A/B.
    // Record this on every eval row — like `think`, it is a prompt-surface knob
    // and results must never be compared across it.
    let include_test_first = env::var("CODING_EVAL_TEST_FIRST")
        .map_or(true, |v| !matches!(v.as_str(), "0" | "off" | "false"));
    let transcripts_on = harness::transcript::parse_transcripts_flag(
        "CODING_EVAL_TRANSCRIPTS",
        env::var("CODING_EVAL_TRANSCRIPTS").ok().as_deref(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let transcripts_root = transcripts_on.then(coding_eval_transcripts_root);
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
         (max_iterations={max_iterations}, test_first={include_test_first}, \
         transcripts={})",
        fixtures.len(),
        if transcripts_on { "on" } else { "off" },
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
        let (mut task, env_factory) = coding_fix_task_with(fixture, include_test_first);
        // Stamp the fixture name onto the task so the report says which
        // fixture ran — otherwise every report would just read `coding_fix`.
        task.name = fixture_name.clone();

        println!(
            "\n=== fixture: {fixture_name} ===\n  path: {}\n",
            fixture.display(),
        );

        let transcripts = transcripts_root.as_ref().map(|root| EvalTranscripts {
            dir: root.join(&fixture_name),
            label: backend_desc.clone(),
        });
        let report = run_eval_with_transcripts(
            &task,
            &backend,
            env_factory,
            k,
            max_iterations,
            transcripts.as_ref(),
            |trial: &TrialResult| {
                let mut line = format!(
                    "  trial {}: {} | {}",
                    trial.trial + 1,
                    outcome_one_liner(&trial.outcome),
                    stats_one_liner(&trial.stats),
                );
                if let Some(p) = &trial.transcript_path {
                    let _ = write!(line, " | transcript: {}", p.display());
                }
                println!("{line}");
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
/// line — iterations, in/out tokens, and wall-clock. Wall-clock is rendered
/// in whole seconds (small runs might round to 0s, which is fine).
fn stats_one_liner(stats: &RunStats) -> String {
    let mut line = format!(
        "{} iters | {} in / {} out | {}s | gate_green_at_exit={} | invalid_finish={}",
        stats.iterations,
        format_tokens_compact(stats.input_tokens),
        format_tokens_compact(stats.output_tokens),
        stats.wall_clock.as_secs(),
        stats.gates_green_at_exit,
        stats.invalid_finish_calls,
    );
    if let Some(raw) = &stats.first_invalid_finish_raw {
        use std::fmt::Write as _;
        let _ = write!(
            line,
            " | invalid_raw={}",
            raw.chars().take(80).collect::<String>()
        );
    }
    line
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
        LoopOutcome::Finished(Disposition::AlreadySatisfied { reason, .. }) => {
            format!("AlreadySatisfied — {reason}")
        }
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
