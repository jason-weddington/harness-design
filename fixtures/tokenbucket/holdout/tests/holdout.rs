//! Sealed holdout tests for `TokenBucket::try_acquire`.
//!
//! These tests drive refill EXCLUSIVELY through `try_acquire` (since
//! `available()` performs no refill) on a monotonic non-decreasing logical
//! clock. They live under `holdout/` in the committed source and are copied
//! into the crate's `tests/` directory only when verifying solvability /
//! running the eval:
//!
//!   cp -r fixtures/tokenbucket/holdout/. fixtures/tokenbucket/
//!   cd fixtures/tokenbucket && cargo test
//!
//! All tests are green only under a correct `try_acquire` implementation
//! that satisfies the refill contract: start-full, try_acquire-only refill,
//! `floor(refill_per_sec * elapsed / 1000)` clamped to capacity, carry the
//! sub-token remainder except jump-to-now on clamp, refill committed
//! regardless of acquire outcome, atomic all-or-nothing acquire.

use tokenbucket::TokenBucket;

/// Refill is clamped: a long idle period cannot make the bucket overshoot
/// capacity.
#[test]
fn refill_clamped_at_capacity() {
    // capacity=4, refill_per_sec=2 → 1 token / 500ms.
    let mut b = TokenBucket::new(4, 2);
    assert!(b.try_acquire(4, 0), "drain the full bucket at t=0");
    assert_eq!(b.available(), 0);
    // Idle 5000ms → would gain 10 tokens, but clamped at capacity 4.
    assert!(b.try_acquire(1, 5000));
    assert_eq!(
        b.available(),
        3,
        "long idle must clamp at capacity, then deduct the 1 acquired"
    );
}

/// Fractional refill accumulates across sub-threshold calls: the carried
/// sub-token remainder eventually yields a whole token. A drop-remainder
/// implementation (anchor jumps to `now_ms` every call) stays holdout-red
/// here.
#[test]
fn fractional_remainder_carries() {
    // capacity=100 (no clamp), refill_per_sec=5 → 1 token / 200ms.
    let mut b = TokenBucket::new(100, 5);
    assert!(b.try_acquire(100, 0), "drain");
    // Three sub-threshold calls — each gains 0, but elapsed from anchor 0
    // accumulates (60, 120, 180 ms) because the anchor does not advance
    // when no whole token is gained.
    assert!(!b.try_acquire(1, 60));
    assert!(!b.try_acquire(1, 120));
    assert!(!b.try_acquire(1, 180));
    assert_eq!(b.available(), 0);
    // Accumulated elapsed crosses 200ms → 1 whole token finally refills.
    assert!(
        b.try_acquire(1, 210),
        "carried remainder must yield a token at 210ms; a drop-remainder impl returns false here"
    );
}

/// A FAILED `try_acquire` deducts nothing, yet its committed refill is
/// visible via a subsequent `available()`.
#[test]
fn failed_acquire_commits_refill() {
    // capacity=4, refill_per_sec=2 → 1 token / 500ms.
    let mut b = TokenBucket::new(4, 2);
    assert!(b.try_acquire(4, 0), "drain");
    assert_eq!(b.available(), 0);
    // 500ms passes → 1 token refills; ask for 3 (>available, ≤cap) → fails.
    assert!(!b.try_acquire(3, 500), "1 < 3: acquire must fail");
    // The refill COMMITTED even though the acquire failed:
    assert_eq!(b.available(), 1, "failed acquire must still commit its refill");
}

/// `n > capacity` always returns false and deducts nothing — even when the
/// bucket is full or refilled to capacity.
#[test]
fn acquire_exceeding_capacity_always_false() {
    let mut b = TokenBucket::new(4, 2);
    assert!(!b.try_acquire(5, 0), "n>capacity false even when full");
    assert_eq!(b.available(), 4, "no deduction on n>capacity");
    assert!(b.try_acquire(4, 0), "drain");
    // Long idle refills to capacity (clamped); n>capacity still false.
    assert!(!b.try_acquire(5, 5000), "n>capacity false even with full refill");
    assert_eq!(b.available(), 4, "refill committed but no deduction");
}

/// `now_ms == anchor` refills nothing.
#[test]
fn no_refill_when_now_equals_anchor() {
    let mut b = TokenBucket::new(4, 2);
    assert!(b.try_acquire(4, 0), "drain; anchor stays 0 (elapsed 0)");
    // now_ms == anchor (0 == 0): no refill, insufficient tokens → false.
    assert!(!b.try_acquire(1, 0), "now==anchor: no refill, must fail");
    assert_eq!(b.available(), 0);
}

/// One interleaved acquire/refill sequence yields the exact specified token
/// trajectory. capacity=4, refill_per_sec=2 (1 token / 500ms).
#[test]
fn interleaved_trajectory() {
    let mut b = TokenBucket::new(4, 2);
    assert_eq!(b.available(), 4);
    // t=0: drain (no refill, elapsed 0).
    assert!(b.try_acquire(4, 0));
    assert_eq!(b.available(), 0);
    // t=250: sub-threshold, gained 0, fail.
    assert!(!b.try_acquire(1, 250));
    assert_eq!(b.available(), 0);
    // t=500: gained 1, anchor→500, consume it → 0.
    assert!(b.try_acquire(1, 500));
    assert_eq!(b.available(), 0);
    // t=1500: elapsed 1000 from anchor 500 → gained 2, consume 1 → 1.
    assert!(b.try_acquire(1, 1500));
    assert_eq!(b.available(), 1);
    // t=2000: elapsed 500 from anchor 1500 → gained 1, tokens 2; ask 3 → fail.
    assert!(!b.try_acquire(3, 2000));
    assert_eq!(b.available(), 2, "failed acquire committed its refill");
    // t=2000 again: now==anchor (2000), no refill, ask 1 → succeed on 2.
    assert!(b.try_acquire(1, 2000));
    assert_eq!(b.available(), 1);
    // t=5000: elapsed 3000 from anchor 2000 → gained 6, clamp at 4, consume 1 → 3.
    assert!(b.try_acquire(1, 5000));
    assert_eq!(b.available(), 3, "clamp caps at capacity");
}