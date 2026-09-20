//! The `somnus` library target: everything the nightly map-maintenance loop
//! is made of, minus the (not-yet-wired) run body.
//!
//! Every library construct lives HERE rather than in `src/main.rs` so that
//! `pub` items are exempt from `dead_code`: the bin's run path is a not-wired
//! stub this cut (`pub` does NOT exempt items from `dead_code` in a binary,
//! so a bin-only shape would fail `cargo clippy --all-targets -- -D warnings`
//! on first build). `main.rs` stays in place so the publisher's
//! `crates/*/src/main.rs` discovery rule and the `[[bin]]` path are unchanged.
//!
//! Modules:
//!
//! - [`gate`] — the map-lint gate and the six-tool registry.
//! - [`observer`] — the leg-3 pointer-count change observer.
//! - [`ops`] — the closed op vocabulary, the op seam, and the disposition
//!   mapping.

pub mod gate;
pub mod observer;
pub mod ops;

use std::path::Path;
use std::sync::Arc;

use harness::engine::RunConfig;
use harness::exec::TreeObservation;

use crate::gate::map_lint_runner;
use crate::observer::{MapPointerObserver, PointerCountSource};

/// The seed task for the nightly loop. Unwired this cut: loop wiring is
/// deferred, so the task text says so rather than promising work the run body
/// cannot perform.
pub const SOMNUS_TASK: &str = "somnus nightly map-maintenance loop (unwired: loop wiring deferred)";

/// Hard cap on model turns per nightly run.
pub const SOMNUS_MAX_ITERATIONS: u32 = 24;

/// The nightly wall-clock budget, in seconds: 4 hours.
///
/// The only budget arming available today — `BudgetLimits.tokens` /
/// `cost_micros` arming is harness item `00b5b825`, out of scope.
pub const NIGHTLY_WALL_CLOCK_SECS: u64 = 14_400;

/// Compaction trigger threshold, in percent. `0` DISABLES compaction.
///
/// Compaction is off because per-unit context is ~11k tokens against a 200k
/// window: the compaction machinery would never fire on real work, and
/// arming it would only add a code path with no exercising traffic.
pub const SOMNUS_COMPACT_THRESHOLD_PCT: u64 = 0;

/// Maximum finish-recovery nudges: `0` disables finish-recovery entirely.
///
/// Why, from the vendored spec's stated reason: with the gate registered as
/// `run_checks`, `last_gate_green` DOES get set (that hardcoded match arm is
/// the sole setter); no registered tool mutates a filesystem, so `tree_dirty`
/// stays permanently false; the staleness predicate
/// (`max_nudges > 0 && (last_gate_green || tree_dirty)`) is therefore always
/// true on the gate leg, and the green-static nudge would arm on the FIRST
/// green gate — nudging a loop that legitimately makes no filesystem change.
/// The underlying `last_gate_green`-with-non-filesystem-tools behaviour is
/// filed as a harness defect, and the `49b4445e` gate declaration (see
/// `crate::gate::build_registry`) is the re-enable point.
pub const SOMNUS_MAX_NUDGES: u32 = 0;

/// Build the engine's [`RunConfig`] for one nightly run of `project` against
/// the KB at `kb_base`.
///
/// Every knob the loop needs is set HERE, from named constants, so the run
/// configuration is reviewable in one place:
///
/// - the map-lint gate over `kb_base` (leg 2),
/// - `SOMNUS_MAX_NUDGES = 0` (finish-recovery disabled — see the constant),
/// - the wall-clock budget (the only arming available today),
/// - compaction disabled,
/// - the full transcript (without this the nightly loop runs blind — every
///   decision-telemetry channel the vendored spec leans on is inert),
/// - the pointer-count change observer (leg 3).
pub fn build_run_config(
    project: &str,
    kb_base: &str,
    transcript_path: &Path,
    count_source: Arc<dyn PointerCountSource>,
) -> RunConfig {
    RunConfig::new(SOMNUS_TASK, SOMNUS_MAX_ITERATIONS)
        .with_checks(map_lint_runner(kb_base))
        .with_max_nudges(SOMNUS_MAX_NUDGES)
        .with_wall_clock_secs(NIGHTLY_WALL_CLOCK_SECS)
        .with_compact_threshold_pct(SOMNUS_COMPACT_THRESHOLD_PCT)
        .with_transcript(transcript_path, "somnus")
        .with_change_observer(Arc::new(MapPointerObserver::new(
            project.to_string(),
            count_source,
        )))
}

/// The run-start guard: refuse to start when the run-start baseline could not
/// be observed.
///
/// The baseline is armed by the observer's FIRST successful observation; a
/// first-call failure yields `Unobservable`, and a loop that cannot prove
/// work happened does not get to claim it did — so it does not get to run.
/// The guard's call site in the loop body is deferred loop wiring (the
/// exit-code mapping rides with it); the guard itself and its mapping land
/// and are tested now.
///
/// # Errors
///
/// Returns `Err` with the exact refusal message when `baseline` is
/// [`TreeObservation::Unobservable`]; `Ok(())` for any `Observed` baseline.
pub fn run_start_refusal(baseline: &TreeObservation) -> Result<(), String> {
    match baseline {
        TreeObservation::Observed { .. } => Ok(()),
        TreeObservation::Unobservable { reason } => Err(format!(
            "somnus: refusing to start: run-start baseline could not be observed ({reason})"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::observer::CountSourceError;

    /// A fake [`PointerCountSource`] that always answers `17`. No clock, no
    /// network, no live KB.
    #[derive(Debug)]
    struct FakeSource;

    #[async_trait]
    impl PointerCountSource for FakeSource {
        async fn pointer_count(&self, _project: &str) -> Result<u64, CountSourceError> {
            Ok(17)
        }
    }

    // --- build_run_config: field-by-field pin ---

    #[test]
    fn build_run_config_pins_every_knob() {
        let config = build_run_config(
            "demo",
            "http://kb.invalid",
            std::path::Path::new("transcript.jsonl"),
            Arc::new(FakeSource),
        );
        assert_eq!(config.task, SOMNUS_TASK);
        assert_eq!(config.max_iterations, 24);
        assert_eq!(config.max_nudges, 0);
        assert_eq!(config.wall_clock_secs, 14_400);
        assert_eq!(config.compact_threshold_pct, 0);
        // The `Option<ChecksRunner>` field is not `PartialEq`; the `command()`
        // accessor is the assertable seam.
        let checks = config.checks.as_ref().expect("gate is armed");
        assert_eq!(checks.command().program, "/bin/sh");
        assert!(checks.command().args[1].contains("http://kb.invalid/api/kb/map-lint"));
        assert!(checks.command().args[1].contains("$(cat"));
        // The `Arc<dyn ChangeObserver>` field is not downcastable; the label
        // override is the assertable seam.
        assert_eq!(config.change_observer.label(), "somnus");
        let transcript = config.transcript.as_ref().expect("transcript is armed");
        assert_eq!(transcript.path, std::path::Path::new("transcript.jsonl"));
        assert_eq!(transcript.label, "somnus");
    }

    // --- the run-start guard ---

    #[test]
    fn run_start_refusal_accepts_any_observed_baseline() {
        let baseline = TreeObservation::Observed {
            porcelain: "17".to_string(),
            head: None,
        };
        assert_eq!(run_start_refusal(&baseline), Ok(()));
    }

    #[test]
    fn run_start_refusal_refuses_an_unobservable_baseline_with_the_exact_message() {
        let baseline = TreeObservation::Unobservable {
            reason: "pointer count timed out after 10s".to_string(),
        };
        assert_eq!(
            run_start_refusal(&baseline),
            Err("somnus: refusing to start: run-start baseline could not be observed (pointer count timed out after 10s)".to_string())
        );
    }

    // --- the constants the vendored spec pins ---

    #[test]
    fn named_constants_match_the_pinned_values() {
        assert_eq!(SOMNUS_MAX_ITERATIONS, 24);
        assert_eq!(NIGHTLY_WALL_CLOCK_SECS, 14_400);
        assert_eq!(SOMNUS_COMPACT_THRESHOLD_PCT, 0);
        assert_eq!(SOMNUS_MAX_NUDGES, 0);
        assert_eq!(
            crate::ops::SOMNUS_DONE_SUMMARY,
            "ops applied, map-lint gate green, pointer count moved"
        );
    }
}
