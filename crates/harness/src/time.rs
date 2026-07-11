//! Pure, dependency-free RFC 3339 UTC timestamp helper.
//!
//! Formats a [`std::time::SystemTime`] as `"YYYY-MM-DDTHH:MM:SSZ"` using
//! only the standard library — no `chrono`, `time`, or any other date crate.
//! This is a deliberate constraint: the project's `deny.toml` restricts
//! licenses to MIT/Apache-2.0, and adding a date library just for formatting
//! would widen the dependency surface without meaningful benefit.
//!
//! ## Correctness
//!
//! Civil-date arithmetic is based on Howard Hinnant's algorithm
//! (<https://howardhinnant.github.io/date_algorithms.html>), adapted here for
//! unsigned arithmetic because Unix timestamps are always non-negative.
//! Two pinned-vector unit tests anchor the implementation.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

// =======================================================================
// Clock seam
// =======================================================================

/// Seam for wall-clock reads: everything that reads "now" inside the harness
/// goes through this trait so tests can drive time deterministically.
///
/// The `Debug + Send + Sync` supertrait bounds are REQUIRED so that
/// `std::sync::Arc<dyn Clock>` can live inside `RunConfig`'s
/// `#[derive(Debug, Clone)]`: `Arc<T>` is `Clone` unconditionally, `Debug`
/// requires `T: Debug`, and `Send + Sync` let the `Arc` cross thread
/// boundaries safely.
pub trait Clock: std::fmt::Debug + Send + Sync {
    /// Return the current wall-clock instant as a [`SystemTime`].
    fn now(&self) -> SystemTime;
}

/// Production implementation of [`Clock`]: delegates to [`SystemTime::now`].
#[derive(Debug, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

// =======================================================================
// Test-only advanceable clock
// =======================================================================

/// Test-only advanceable clock. Available only when `cfg(test)`.
///
/// Supports two advance modes:
/// - **Manual**: call [`FakeClock::advance`] to move the clock forward
///   explicitly. The clock stays fixed between `advance` calls (each call to
///   [`Clock::now`] returns the same value until the next `advance`).
/// - **Auto-advance**: construct with [`FakeClock::new_auto_advance`]; every
///   call to [`Clock::now`] automatically advances the clock by `step`, then
///   returns the pre-advance value. This lets loop tests drive a breach without
///   external synchronisation — set `step > wall_clock_secs` to fire on the
///   first iteration.
///
/// Interior mutability via `Mutex<(SystemTime, step)>` so both methods accept
/// `&self`.
#[cfg(test)]
pub struct FakeClock {
    inner: std::sync::Mutex<(SystemTime, Duration)>,
}

#[cfg(test)]
impl FakeClock {
    /// Create a [`FakeClock`] anchored at `start` with no auto-advance (the
    /// clock stays fixed until [`FakeClock::advance`] is called).
    pub fn new(start: SystemTime) -> Self {
        Self {
            inner: std::sync::Mutex::new((start, Duration::ZERO)),
        }
    }

    /// Create a [`FakeClock`] that automatically advances by `step` on every
    /// call to [`Clock::now`]. The pre-advance value is returned, so the first
    /// call returns `start`, the second `start + step`, and so on.
    ///
    /// Useful for wall-clock breach tests where the loop runs in a single
    /// async task and there is no opportunity to call `advance` mid-loop.
    pub fn new_auto_advance(start: SystemTime, step: Duration) -> Self {
        Self {
            inner: std::sync::Mutex::new((start, step)),
        }
    }

    /// Advance the clock forward by `d`.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned (another test thread panicked
    /// while holding the lock).
    pub fn advance(&self, d: Duration) {
        let mut guard = self.inner.lock().expect("FakeClock mutex poisoned");
        guard.0 += d;
    }
}

#[cfg(test)]
impl Clock for FakeClock {
    fn now(&self) -> SystemTime {
        let mut guard = self.inner.lock().expect("FakeClock mutex poisoned");
        let t = guard.0;
        let step = guard.1;
        guard.0 += step; // no-op when step == Duration::ZERO
        t
    }
}

#[cfg(test)]
impl std::fmt::Debug for FakeClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.inner.lock().expect("FakeClock mutex poisoned");
        f.debug_struct("FakeClock")
            .field("current", &guard.0)
            .field("step", &guard.1)
            .finish()
    }
}

/// Format `t` as a UTC RFC 3339 timestamp of the form
/// `"YYYY-MM-DDTHH:MM:SSZ"`.
///
/// If `t` is before the Unix epoch (which should not happen in practice),
/// the function saturates to `"1970-01-01T00:00:00Z"`.
#[must_use]
pub fn format_rfc3339(t: SystemTime) -> String {
    let secs = t
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs();

    let sec = (secs % 60) as u32;
    let min = ((secs / 60) % 60) as u32;
    let hour = ((secs / 3_600) % 24) as u32;
    let days = secs / 86_400;

    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert a count of days since the Unix epoch (1970-01-01) to a Gregorian
/// `(year, month, day)` triple.
///
/// Uses Howard Hinnant's unsigned civil-date algorithm:
/// <https://howardhinnant.github.io/date_algorithms.html>
fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    // Shift the reference point from 1970-01-01 to the civil epoch 0000-03-01
    // so the leap-year boundary falls at year-end, simplifying the arithmetic.
    let z = days + 719_468;
    let era = z / 146_097; // 400-year Gregorian cycle
    let doe = z - era * 146_097; // day of era  [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // year of era [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // month period [0, 11] (March=0 … February=11)
    let d = doy - (153 * mp + 2) / 5 + 1; // day [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // month [1, 12]
    // January and February belong to the *following* civil year.
    let y = if m <= 2 { y + 1 } else { y };
    // All three values are bounded by algorithm construction (year fits in u32
    // for any plausible Unix timestamp; month [1,12]; day [1,31]), so the
    // casts cannot truncate in practice.
    #[allow(clippy::cast_possible_truncation)]
    (y as u32, m as u32, d as u32)
}

// =======================================================================
// Tests
// =======================================================================

#[cfg(test)]
mod tests {
    use super::{Clock, FakeClock, SystemClock, format_rfc3339};
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    #[test]
    fn epoch_formats_as_zero_time() {
        // Pinned vector: the Unix epoch must produce the T-zero timestamp.
        assert_eq!(format_rfc3339(UNIX_EPOCH), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn known_timestamp_formats_correctly() {
        // Pinned vector: 1_700_000_000 seconds after the Unix epoch.
        // Verified with `date -u -d @1700000000`: 2023-11-14 22:13:20 UTC.
        let t = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        assert_eq!(format_rfc3339(t), "2023-11-14T22:13:20Z");
    }

    // ---- Clock / SystemClock -------------------------------------------

    #[test]
    fn system_clock_now_is_after_epoch() {
        let clock = SystemClock;
        // SystemClock::now() must return something after the Unix epoch.
        assert!(
            clock.now() > UNIX_EPOCH,
            "SystemClock::now() should be after the Unix epoch"
        );
    }

    #[test]
    fn system_clock_is_debug_send_sync() {
        fn assert_debug_send_sync<T: std::fmt::Debug + Send + Sync>() {}
        assert_debug_send_sync::<SystemClock>();
    }

    #[test]
    fn system_clock_arc_dyn_clock_is_clone_and_debug() {
        let c: Arc<dyn Clock> = Arc::new(SystemClock);
        // Arc<dyn Clock> must be Clone (Arc::clone).
        let _c2 = c.clone();
        // dyn Clock must be Debug via the supertrait.
        let _ = format!("{c:?}");
    }

    // ---- FakeClock --------------------------------------------------------

    #[test]
    fn fake_clock_starts_at_given_time() {
        let t0 = UNIX_EPOCH + Duration::from_secs(1_000);
        let clock = FakeClock::new(t0);
        assert_eq!(clock.now(), t0);
        // Second call returns the same value (no auto-advance).
        assert_eq!(clock.now(), t0);
    }

    #[test]
    fn fake_clock_advance_moves_time_forward() {
        let t0 = UNIX_EPOCH;
        let clock = FakeClock::new(t0);
        clock.advance(Duration::from_secs(42));
        assert_eq!(
            clock.now(),
            t0 + Duration::from_secs(42),
            "advance must move the clock forward by the given duration"
        );
    }

    #[test]
    fn fake_clock_advance_is_cumulative() {
        let t0 = UNIX_EPOCH;
        let clock = FakeClock::new(t0);
        clock.advance(Duration::from_secs(10));
        clock.advance(Duration::from_secs(5));
        assert_eq!(clock.now(), t0 + Duration::from_secs(15));
    }

    #[test]
    fn fake_clock_implements_clock_trait() {
        let clock = FakeClock::new(UNIX_EPOCH + Duration::from_secs(500));
        let dyn_clock: &dyn Clock = &clock;
        assert_eq!(dyn_clock.now(), UNIX_EPOCH + Duration::from_secs(500));
    }

    #[test]
    fn fake_clock_auto_advance_per_now_call() {
        let t0 = UNIX_EPOCH;
        let step = Duration::from_secs(10);
        let clock = FakeClock::new_auto_advance(t0, step);
        // First call returns t0 (pre-advance value), then advances to t0+10s.
        assert_eq!(clock.now(), t0);
        // Second call returns t0+10s, then advances to t0+20s.
        assert_eq!(clock.now(), t0 + Duration::from_secs(10));
        // Third call returns t0+20s.
        assert_eq!(clock.now(), t0 + Duration::from_secs(20));
    }
}
