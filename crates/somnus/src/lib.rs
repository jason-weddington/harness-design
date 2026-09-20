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
//! - [`loop_input`] — the rung-1 payload types, the loop-input transport, and
//!   the pure payload consumers.
//! - [`ledger`] — the cluster/decline ledger seam and the decline filter.
//! - [`rungs`] — the single-shot rung-1 and rung-2 inferences.
//! - [`materialize`] — rung 3: code composes the map bodies from ops.
//! - [`unit`] — the per-project pipeline and its run report.
//!
//! Retired with this cut: `build_run_config`, `SOMNUS_MAX_NUDGES`, and
//! `SOMNUS_COMPACT_THRESHOLD_PCT`. The rung discipline — one single-shot
//! inference per project and per cluster, the read path injected as
//! synthetic tool-call/result events, and the read tools deleted from the
//! output union — cannot be expressed by `engine::run`, which builds its
//! own `initial_messages` from `prompt::render_task_prompt` internally (its
//! `RunConfig` exposes no history seam and `run_loop_impl` is private), so
//! there is no engine loop here and therefore no nudge or compaction
//! machinery to configure. `SOMNUS_MAX_ITERATIONS` survived and was
//! repurposed as the per-unit inference budget instead.

pub mod gate;
pub mod ledger;
pub mod loop_input;
pub mod materialize;
pub mod observer;
pub mod ops;
pub mod rungs;
pub mod unit;

use harness::exec::TreeObservation;

/// The seed task for the nightly loop. Unwired this cut: loop wiring is
/// deferred, so the task text says so rather than promising work the run body
/// cannot perform.
pub const SOMNUS_TASK: &str = "somnus nightly map-maintenance loop (unwired: loop wiring deferred)";

/// Hard cap on `ModelBackend::turn` calls inside ONE `run_unit`
/// ([`crate::unit::run_unit`]): 1 rung-1 inference plus at most 23 rung-2
/// inferences (one per cluster). Enforced by the pipeline BEFORE each rung-2
/// turn, count-based (no clock), so a runaway cluster list aborts the unit
/// with a named reason instead of a silent over-spend.
pub const SOMNUS_MAX_ITERATIONS: u32 = 24;

/// The nightly wall-clock budget, in seconds: 4 hours. Consumed by
/// [`crate::unit::run_unit`]'s `tokio::time::timeout` wrapper, so an expiry
/// is a named abort and never a hang.
///
/// The only budget arming available today — `BudgetLimits.tokens` /
/// `cost_micros` arming is harness item `00b5b825`, out of scope.
pub const NIGHTLY_WALL_CLOCK_SECS: u64 = 14_400;

/// The run-start guard: refuse to start when the run-start baseline could not
/// be observed.
///
/// The baseline is armed by the observer's FIRST successful observation; a
/// first-call failure yields `Unobservable`, and a loop that cannot prove
/// work happened does not get to claim it did — so it does not get to run.
/// The guard's call site is [`crate::unit::run_unit`], which aborts the unit
/// with this exact message when the baseline could not be observed.
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
        assert_eq!(
            crate::ops::SOMNUS_DONE_SUMMARY,
            "ops applied, map-lint gate green, pointer count moved"
        );
    }
}
