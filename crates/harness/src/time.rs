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
    use super::format_rfc3339;
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
}
