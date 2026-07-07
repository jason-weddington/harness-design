//! Interval merging — a coding-task eval fixture.
//!
//! This crate is committed in a FAILING state on purpose: the tests below
//! encode the intended contract precisely, and one deviation from that contract
//! keeps the suite red until the harness finds it and repairs it. The workspace
//! `exclude`s `fixtures/*`, so the project's own gates never build or lint this
//! file.

/// Merge overlapping or touching CLOSED intervals `[start, end]`.
///
/// The input is a `Vec<(start, end)>` of closed intervals — both endpoints are
/// inclusive. Two intervals must be merged when they overlap OR when they
/// merely touch: `[1, 3]` and `[3, 5]` share the single endpoint `3` and merge
/// into `[1, 5]`.
///
/// The returned `Vec` is:
///   * sorted by `start` ascending,
///   * non-overlapping,
///   * non-touching — for any two consecutive output intervals `(a, b)` then
///     `(c, d)`, `c > b` holds strictly.
///
/// The order of the input intervals does not matter; the function sorts
/// internally. An empty input yields an empty output.
pub fn merge(mut intervals: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
    if intervals.is_empty() {
        return Vec::new();
    }
    intervals.sort_by_key(|(start, _)| *start);
    let mut out: Vec<(i64, i64)> = Vec::with_capacity(intervals.len());
    out.push(intervals[0]);
    for &next in intervals.iter().skip(1) {
        let current = out.last_mut().expect("just pushed above");
        if next.0 < current.1 {
            current.1 = current.1.max(next.1);
        } else {
            out.push(next);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::merge;

    #[test]
    fn empty_input_yields_empty_output() {
        let out: Vec<(i64, i64)> = merge(Vec::new());
        assert_eq!(out, Vec::<(i64, i64)>::new());
    }

    #[test]
    fn single_interval_passes_through() {
        assert_eq!(merge(vec![(1, 5)]), vec![(1, 5)]);
    }

    #[test]
    fn disjoint_intervals_are_unchanged() {
        // Gap between 2 and 5 — no overlap, no touch.
        assert_eq!(merge(vec![(1, 2), (5, 7)]), vec![(1, 2), (5, 7)]);
    }

    #[test]
    fn properly_overlapping_intervals_merge() {
        // [1,5] and [3,7] overlap on [3,5] → merge to [1,7].
        assert_eq!(merge(vec![(1, 5), (3, 7)]), vec![(1, 7)]);
    }

    #[test]
    fn nested_interval_is_absorbed() {
        // [3,5] is fully contained in [1,10] → single [1,10] out.
        assert_eq!(merge(vec![(1, 10), (3, 5)]), vec![(1, 10)]);
    }

    #[test]
    fn adjacent_touching_intervals_merge() {
        // [1,3] and [3,5] share endpoint 3 → contract says merge to [1,5].
        assert_eq!(merge(vec![(1, 3), (3, 5)]), vec![(1, 5)]);
    }

    #[test]
    fn chain_of_touching_intervals_collapses() {
        // [1,2], [2,3], [3,4] each touch the next → one interval [1,4] out.
        assert_eq!(merge(vec![(1, 2), (2, 3), (3, 4)]), vec![(1, 4)]);
    }
}
