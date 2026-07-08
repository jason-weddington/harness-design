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
//! ANTHROPIC_API_KEY=sk-... cargo run --example coding_eval
//! ANTHROPIC_API_KEY=sk-... ANTHROPIC_MODEL=claude-sonnet-4-6 CODING_EVAL_K=5 \
//!   cargo run --example coding_eval
//! # narrow to a single fixture directory name
//! ANTHROPIC_API_KEY=sk-... CODING_EVAL_FIXTURE=lru-cache cargo run --example coding_eval
//! ```
//!
//! Environment:
//! - `ANTHROPIC_API_KEY`     (required) — passed through to the backend.
//! - `ANTHROPIC_MODEL`       (optional) — defaults to `claude-haiku-4-5`.
//! - `CODING_EVAL_K`         (optional) — number of trials; defaults to 3.
//! - `CODING_EVAL_FIXTURE`   (optional) — narrows the run to the single named
//!   fixture directory under `fixtures/` (e.g. `lru-cache`). When unset, every
//!   directory under `fixtures/` is discovered and run in sorted order.

use std::env;
use std::path::PathBuf;

use harness::anthropic::AnthropicBackend;
use harness::engine::{FinishDisposition, LoopOutcome, RunStats, Verification};
use harness::eval::{EvalReport, TrialResult, coding_fix_task, discover_fixtures, run_eval};

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
    // Empty string is treated as "unset" — the shell's `VAR= cmd` idiom clears
    // the narrow-to-one-fixture override.
    let fixture_filter = env::var("CODING_EVAL_FIXTURE")
        .ok()
        .filter(|s| !s.is_empty());

    let backend = AnthropicBackend::new(&model, api_key);

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
        "running coding_fix eval across {} fixture(s) (k={k}) against model `{model}` \
         (max_iterations={MAX_ITERATIONS})",
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
            MAX_ITERATIONS,
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
    // extra `mean_iters` and `total_tokens` columns surface the per-trial
    // detail collapsed into a compare-across-fixtures view.
    let name_col = summary
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(0)
        .max("fixture".len());

    println!("\n=== SUMMARY ===");
    println!(
        "{:<name_col$}  {:>9}  {:>10}  {:>10}  {:>12}",
        "fixture", "passes/k", "pass_rate", "mean_iters", "total_tokens",
    );
    for (name, r) in &summary {
        let total_tokens = r.total_input_tokens() + r.total_output_tokens();
        println!(
            "{name:<name_col$}  {:>9}  {:>10.3}  {:>10.2}  {:>12}",
            format!("{}/{}", r.passes, r.trials),
            r.pass_rate,
            r.mean_iterations(),
            format_tokens_compact(total_tokens),
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
