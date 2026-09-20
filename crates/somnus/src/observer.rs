//! The leg-3 change observer for the nightly loop: the project's KB
//! map-pointer count, observed through a [`PointerCountSource`] seam, fed
//! into the harness's `ChangeObserver` contract.
//!
//! The production transport for the pointer count is pinned NOWHERE in this
//! repo — that is GTD item `somnus-loop-input`'s contract. This module owns
//! only the decision logic, which is pure and unit-tested without a clock, a
//! network, or a live KB:
//!
//! - [`decide_observation`] — the fallback/fail-closed rules.
//! - [`render_fallback_line`] — the escalation line emitted to stderr.
//! - [`audit_violation`] — a pure tripwire re-checking the fail-closed
//!   invariants at the single emit site (the harness's own terminal-tripwire
//!   precedent).
//! - [`MapPointerObserver`] — the `ChangeObserver` implementation that ties
//!   them together.
//!
//! The workspace has NO logging framework (no `tracing`/`log`/`env_logger` in
//! `[workspace.dependencies]`); the harness and talos precedent is bare
//! `eprintln!`, which is what escalation emission uses.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use harness::exec::{ChangeObserver, TreeObservation};
use tokio::time::timeout;

/// Wall-clock bound on one pointer-count observation. One indexed local count
/// that cannot answer in ten seconds means the KB is unhealthy (the vendored
/// spec's bound; the `ChangeObserver` trait doc requires each implementation
/// to pin its own bound).
pub const OBSERVE_BOUND: Duration = Duration::from_secs(10);

/// The seam between the loop and whatever actually counts the project's map
/// pointers. The production transport (which endpoint, which auth, which
/// response shape) is pinned nowhere in this repo, so the trait is the only
/// thing the decision logic is written against.
#[async_trait]
pub trait PointerCountSource: std::fmt::Debug + Send + Sync {
    /// Count the target project's map pointers.
    async fn pointer_count(&self, project: &str) -> Result<u64, CountSourceError>;
}

/// Why a pointer count could not be produced. `Timeout` is the
/// [`OBSERVE_BOUND`] branch (a healthy KB answers well inside ten seconds);
/// `Source` is a transport-level failure reported by the source itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountSourceError {
    /// The source did not answer within [`OBSERVE_BOUND`].
    Timeout,
    /// The source answered with a failure.
    Source {
        /// Why the source failed.
        reason: String,
    },
}

impl std::fmt::Display for CountSourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Mirrors `OBSERVE_BOUND` (10s) byte-for-byte in prose; the
            // constant and this string are pinned together.
            Self::Timeout => write!(f, "pointer count timed out after 10s"),
            Self::Source { reason } => {
                write!(f, "pointer count source failed: {reason}")
            }
        }
    }
}

impl std::error::Error for CountSourceError {}

/// Whether a fallback decision should escalate, and how. The workspace has no
/// logging framework, so emission is `eprintln!` (see [`render_fallback_line`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Escalation {
    /// First consecutive fallback: a single warning line.
    Warn,
    /// Two or more consecutive fallbacks: the count is in the rendered line.
    Error,
}

/// The outcome of one observation, decided purely by [`decide_observation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObserveDecision {
    /// The observation to return to the engine.
    pub observation: TreeObservation,
    /// Whether the returned observation is the cached run-start baseline
    /// rather than a fresh count.
    pub was_fallback: bool,
    /// Whether the fallback should be logged at warn or error severity.
    pub escalation: Option<Escalation>,
    /// The consecutive-fallback counter AFTER this decision (input + 1 on a
    /// fallback, reset to 0 on a success, unchanged on an unobservable).
    pub consecutive_fallbacks: u64,
}

/// Decide what one pointer-count result means for the harness's leg-3
/// observation.
///
/// Pinned rules:
///
/// - `Ok(n)` → `Observed { porcelain: n.to_string(), head: None }`, no
///   fallback, no escalation, counter reset to `0`.
/// - `Err(e)` with a cached baseline → return the cached observation (fail
///   closed), count the fallback, and escalate (`Warn` at exactly 1
///   consecutive, `Error` at 2 or more).
/// - `Err(e)` with no cache (run start) → `Unobservable { reason:
///   e.to_string() }`, no fallback counted. The start guard
///   ([`crate::run_start_refusal`]) refuses to start.
///
/// Why fail closed rather than fail open: returning `Unobservable` fails
/// open — `classify_change` cannot compare an unobservable pair, so the
/// finish path accepts the claim on trust, which is exactly the property the
/// cached-baseline fallback exists to deny. A sentinel like `OBSERVE_FAILED`
/// is worse: a baseline of `17` against the sentinel DIFFERS, so the
/// comparison manufactures `TreeChanged` and hands a broken loop a leg-3
/// pass. No field can signal "I failed" without breaking the whole-value
/// equality the fail-closed property depends on — so failure is signalled by
/// returning the cached value and counting the fallback here, where it can
/// escalate.
#[must_use]
pub fn decide_observation(
    count_result: Result<u64, CountSourceError>,
    cached: Option<&str>,
    consecutive_fallbacks: u64,
) -> ObserveDecision {
    match count_result {
        Ok(n) => ObserveDecision {
            observation: TreeObservation::Observed {
                porcelain: n.to_string(),
                head: None,
            },
            was_fallback: false,
            escalation: None,
            consecutive_fallbacks: 0,
        },
        Err(err) => {
            let reason = err.to_string();
            match cached {
                Some(cached_baseline) => {
                    let new_count = consecutive_fallbacks + 1;
                    ObserveDecision {
                        // The cached run-start baseline, verbatim: whole-value
                        // equality with the run-start observation is what
                        // makes the finish path reject (fail closed).
                        observation: TreeObservation::Observed {
                            porcelain: cached_baseline.to_string(),
                            head: None,
                        },
                        was_fallback: true,
                        escalation: Some(if new_count == 1 {
                            Escalation::Warn
                        } else {
                            Escalation::Error
                        }),
                        consecutive_fallbacks: new_count,
                    }
                }
                None => ObserveDecision {
                    observation: TreeObservation::Unobservable { reason },
                    was_fallback: false,
                    escalation: None,
                    consecutive_fallbacks,
                },
            }
        }
    }
}

/// Render the stderr escalation line for a fallback. Pure so both shapes are
/// byte-pinned by unit tests.
///
/// - `consecutive == 1` → `somnus: map-observe fallback (project {project}): {reason}`
/// - `consecutive >= 2` → `somnus: map-observe fallback x{consecutive} consecutive (project {project}): {reason}`
#[must_use]
pub fn render_fallback_line(project: &str, reason: &str, consecutive: u64) -> String {
    if consecutive == 1 {
        format!("somnus: map-observe fallback (project {project}): {reason}")
    } else {
        format!(
            "somnus: map-observe fallback x{consecutive} consecutive (project {project}): {reason}"
        )
    }
}

/// Pure choke-point audit of a [`ObserveDecision`] against the cache it was
/// decided with, re-checking the fail-closed invariants at the single emit
/// site (mirroring the harness's own terminal tripwire precedent). Returns
/// `Some(message)` — which [`MapPointerObserver::observe`] `eprintln!`s —
/// when:
///
/// - the decision's observation is `Observed` with `head: Some(..)`, or
/// - the decision is a fallback whose observation is NOT whole-value equal to
///   the cached run-start baseline.
///
/// `None` otherwise. A later edit that inverts the fail-open/fail-closed
/// property cannot land silently: one of these arms fires.
#[must_use]
pub fn audit_violation(decision: &ObserveDecision, cached: Option<&str>) -> Option<String> {
    let head_violation = match &decision.observation {
        TreeObservation::Observed { head: Some(_), .. } => {
            Some("invariant violation: observation carries a head value; the pointer-count observer pins head: None on every observation (classify_change compares porcelain AND head by whole-value equality)".to_string())
        }
        _ => None,
    };
    let cache_violation = if decision.was_fallback {
        let expected_ok = cached.map(|c| TreeObservation::Observed {
            porcelain: c.to_string(),
            head: None,
        });
        match expected_ok {
            Some(expected) if expected == decision.observation => None,
            _ => Some("invariant violation: fallback observation diverges from the cached run-start baseline; fail closed requires byte-equality with the cache".to_string()),
        }
    } else {
        None
    };
    head_violation.or(cache_violation)
}

/// The harness `ChangeObserver` for one project's map-pointer count.
///
/// Constructed with the project ref and a [`PointerCountSource`]; the
/// consecutive-fallback counter is exposed via
/// [`MapPointerObserver::consecutive_fallbacks`] so the loop-wiring item can
/// surface per-run fallback totals into telemetry.
#[derive(Debug)]
pub struct MapPointerObserver {
    project: String,
    source: Arc<dyn PointerCountSource>,
    /// The cached run-start baseline (the `porcelain` string of the first
    /// successful observation), used verbatim on every fallback.
    cached: Mutex<Option<String>>,
    consecutive_fallbacks: Arc<AtomicU64>,
}

impl MapPointerObserver {
    /// Observe `project` through `source`.
    pub fn new(project: String, source: Arc<dyn PointerCountSource>) -> Self {
        Self {
            project,
            source,
            cached: Mutex::new(None),
            consecutive_fallbacks: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The current consecutive-fallback count. `0` after a success (reset),
    /// `1` after the first fallback, `2` after the second, and so on.
    #[must_use]
    pub fn consecutive_fallbacks(&self) -> u64 {
        self.consecutive_fallbacks.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ChangeObserver for MapPointerObserver {
    /// Observe the project's map-pointer count under [`OBSERVE_BOUND`].
    ///
    /// The `root: &Path` argument is IGNORED: the trait passes it so
    /// filesystem observers stay drop-in symmetric, but this observer's
    /// durable state lives in the KB, not on the filesystem at `root`.
    async fn observe(&self, _root: &Path) -> TreeObservation {
        let cached = self
            .cached
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();
        let count_result =
            match timeout(OBSERVE_BOUND, self.source.pointer_count(&self.project)).await {
                Ok(result) => result,
                Err(_elapsed) => Err(CountSourceError::Timeout),
            };
        let error_display = count_result.as_ref().err().map(ToString::to_string);
        let current = self.consecutive_fallbacks();
        let decision = decide_observation(count_result, cached.as_deref(), current);

        if let (Some(_escalation), Some(reason)) = (&decision.escalation, &error_display) {
            eprintln!(
                "{}",
                render_fallback_line(&self.project, reason, decision.consecutive_fallbacks)
            );
        }
        if let Some(message) = audit_violation(&decision, cached.as_deref()) {
            eprintln!("{message}");
        }

        self.consecutive_fallbacks
            .store(decision.consecutive_fallbacks, Ordering::SeqCst);
        if !decision.was_fallback
            && let TreeObservation::Observed { porcelain, .. } = &decision.observation
            && let Ok(mut guard) = self.cached.lock()
        {
            *guard = Some(porcelain.clone());
        }
        decision.observation
    }

    /// The `run_start` telemetry label: `"somnus"`, not the default
    /// `"custom"`.
    fn label(&self) -> &'static str {
        "somnus"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// A source that never resolves (a stalled transport), for the
    /// paused-clock timeout test.
    #[derive(Debug)]
    struct StalledSource;

    #[async_trait]
    impl PointerCountSource for StalledSource {
        async fn pointer_count(&self, _project: &str) -> Result<u64, CountSourceError> {
            std::future::pending().await
        }
    }

    /// A source whose results are popped from a scripted sequence (the last
    /// entry repeats), so one observer can be driven success → failure →
    /// failure with its internal cache intact.
    #[derive(Debug)]
    struct SequencedSource(std::sync::Mutex<Vec<Result<u64, CountSourceError>>>);

    impl SequencedSource {
        fn new(script: Vec<Result<u64, CountSourceError>>) -> Self {
            Self(std::sync::Mutex::new(script))
        }
    }

    #[async_trait]
    impl PointerCountSource for SequencedSource {
        async fn pointer_count(&self, _project: &str) -> Result<u64, CountSourceError> {
            let mut script = self.0.lock().expect("script lock");
            if script.len() > 1 {
                script.remove(0)
            } else {
                script[0].clone()
            }
        }
    }

    /// A source that returns one successful count, then never resolves —
    /// driving the `OBSERVE_BOUND` timeout branch with a cache already armed.
    #[derive(Debug)]
    struct ArmedThenStalledSource(AtomicU64);

    #[async_trait]
    impl PointerCountSource for ArmedThenStalledSource {
        async fn pointer_count(&self, _project: &str) -> Result<u64, CountSourceError> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                Ok(17)
            } else {
                std::future::pending().await
            }
        }
    }

    fn source_error(reason: &str) -> Result<u64, CountSourceError> {
        Err(CountSourceError::Source {
            reason: reason.to_string(),
        })
    }

    // --- decide_observation: the success arm ---

    #[test]
    fn success_with_no_cache_yields_fresh_observation() {
        let decision = decide_observation(Ok(17), None, 5);
        assert_eq!(
            decision.observation,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert!(!decision.was_fallback);
        assert_eq!(decision.escalation, None);
        assert_eq!(decision.consecutive_fallbacks, 0);
    }

    #[test]
    fn success_resets_the_fallback_counter() {
        let decision = decide_observation(Ok(3), Some("17"), 2);
        assert_eq!(decision.consecutive_fallbacks, 0);
        assert!(!decision.was_fallback);
    }

    // --- decide_observation: the fallback arm (fail closed) ---

    #[test]
    fn failure_with_cache_yields_cached_baseline() {
        let decision = decide_observation(source_error("fixture outage"), Some("17"), 0);
        assert_eq!(
            decision.observation,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert!(decision.was_fallback);
        assert_eq!(decision.escalation, Some(Escalation::Warn));
        assert_eq!(decision.consecutive_fallbacks, 1);
    }

    #[test]
    fn second_consecutive_failure_escalates_error_with_counter_two() {
        let decision = decide_observation(source_error("fixture outage"), Some("17"), 1);
        assert_eq!(decision.escalation, Some(Escalation::Error));
        assert_eq!(decision.consecutive_fallbacks, 2);
        assert_eq!(
            decision.observation,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
    }

    #[test]
    fn failure_with_no_cache_yields_unobservable_without_counting_a_fallback() {
        let decision = decide_observation(source_error("fixture outage"), None, 3);
        assert_eq!(
            decision.observation,
            TreeObservation::Unobservable {
                reason: "pointer count source failed: fixture outage".to_string(),
            }
        );
        assert!(!decision.was_fallback);
        assert_eq!(decision.escalation, None);
        // "no fallback counted" — the counter is passed through unchanged.
        assert_eq!(decision.consecutive_fallbacks, 3);
    }

    #[test]
    fn timeout_error_renders_the_pinned_display() {
        assert_eq!(
            CountSourceError::Timeout.to_string(),
            "pointer count timed out after 10s"
        );
    }

    #[test]
    fn source_error_renders_the_pinned_display() {
        assert_eq!(
            source_error("fixture outage").unwrap_err().to_string(),
            "pointer count source failed: fixture outage"
        );
    }

    // --- render_fallback_line: byte-pinned escalation shapes ---

    #[test]
    fn fallback_line_at_consecutive_one() {
        assert_eq!(
            render_fallback_line("demo", "pointer count timed out after 10s", 1),
            "somnus: map-observe fallback (project demo): pointer count timed out after 10s"
        );
    }

    #[test]
    fn fallback_line_at_consecutive_two_carries_the_x2_marker() {
        let line = render_fallback_line("demo", "fixture outage", 2);
        assert_eq!(
            line,
            "somnus: map-observe fallback x2 consecutive (project demo): fixture outage"
        );
        assert!(line.contains("x2"));
    }

    // --- audit_violation: the pure tripwire ---

    #[test]
    fn audit_passes_a_good_decision() {
        let decision = decide_observation(Ok(17), Some("17"), 0);
        assert_eq!(audit_violation(&decision, Some("17")), None);
    }

    #[test]
    fn audit_flags_a_head_carrying_observation() {
        let decision = ObserveDecision {
            observation: TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: Some("x".to_string()),
            },
            was_fallback: false,
            escalation: None,
            consecutive_fallbacks: 0,
        };
        let message = audit_violation(&decision, Some("17")).expect("a violation");
        assert!(message.contains("invariant violation"));
        assert!(message.contains("head"));
    }

    #[test]
    fn audit_flags_a_fallback_diverging_from_the_cache() {
        let decision = ObserveDecision {
            observation: TreeObservation::Observed {
                porcelain: "99".to_string(),
                head: None,
            },
            was_fallback: true,
            escalation: Some(Escalation::Warn),
            consecutive_fallbacks: 1,
        };
        let message = audit_violation(&decision, Some("17")).expect("a violation");
        assert!(message.contains("invariant violation"));
        assert!(message.contains("cached run-start baseline"));
    }

    // --- MapPointerObserver: the wired path, no clock, no network, no KB ---

    #[tokio::test]
    async fn observe_counts_consecutive_fallbacks_through_the_accessor() {
        // success → failure → failure: the accessor reads 0, then 1, then 2,
        // and every fallback observation is fail-closed onto the cached
        // run-start baseline.
        let source = Arc::new(SequencedSource::new(vec![
            Ok(17),
            source_error("fixture outage"),
            source_error("fixture outage"),
        ]));
        let observer = MapPointerObserver::new("demo".to_string(), source);

        let first = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            first,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 0);

        let second = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            second,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 1);

        let third = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            third,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 2);
        assert_eq!(observer.label(), "somnus");
    }

    #[tokio::test]
    async fn success_after_a_fallback_resets_the_escalation_ladder() {
        // success → failure → success → failure: the counter resets on the
        // success, so the NEXT failure escalates at `Warn` again with counter
        // `1` (rather than continuing the ladder to `Error`).
        let source = Arc::new(SequencedSource::new(vec![
            Ok(17),
            source_error("fixture outage"),
            Ok(18),
            source_error("later outage"),
        ]));
        let observer = MapPointerObserver::new("demo".to_string(), source);
        let _ = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(observer.consecutive_fallbacks(), 0);
        let _ = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(observer.consecutive_fallbacks(), 1);
        let _ = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(observer.consecutive_fallbacks(), 0);
        let fourth = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            fourth,
            TreeObservation::Observed {
                porcelain: "18".to_string(),
                head: None,
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_source_takes_the_cached_baseline_fallback_at_the_bound() {
        // A source that answers once (arming the run-start baseline) and then
        // never resolves. The paused runtime auto-advances the clock while the
        // pending future is outstanding, so
        // `timeout(OBSERVE_BOUND, pending)` resolves at exactly 10s of paused
        // time with no wall-clock wait — pinning that the timeout branch is
        // reachable from `observe` and identical to the error branch (the
        // same cached-baseline fallback, counted).
        let observer = MapPointerObserver::new(
            "demo".to_string(),
            Arc::new(ArmedThenStalledSource(AtomicU64::new(0))),
        );
        let baseline = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            baseline,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 0);

        let stalled = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            stalled,
            TreeObservation::Observed {
                porcelain: "17".to_string(),
                head: None,
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_with_no_cached_baseline_yields_unobservable() {
        // A stalled source on a FRESH observer (no cache — run start) yields
        // `Unobservable` at the bound and counts no fallback.
        let observer = MapPointerObserver::new("demo".to_string(), Arc::new(StalledSource));
        let outcome = observer.observe(std::path::Path::new(".")).await;
        assert_eq!(
            outcome,
            TreeObservation::Unobservable {
                reason: "pointer count timed out after 10s".to_string(),
            }
        );
        assert_eq!(observer.consecutive_fallbacks(), 0);
    }
}
