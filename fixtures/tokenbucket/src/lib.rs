// NOTES: Tier 6 / hard — withheld-failing-test

//! A token-bucket rate limiter — a coding-eval fixture.
//!
//! This crate is committed in a DELIBERATELY HALF-BUILT state:
//! `try_acquire` is a `todo!()` stub whose contract is described in its doc
//! comment and in `task.json`. There is NO visible failing test in the
//! committed tree — the visible tests below exercise only the already-
//! implemented `new` / `capacity` / `available` surface. The sealed holdout
//! suite (copied into `tests/` only at scoring time) is the sole grader.
//!
//! The refill contract is FULLY DETERMINED by the doc comment on `try_acquire`
//! and by `task.json`: for any monotonic-non-decreasing `now_ms` sequence,
//! every `available()` and `try_acquire` outcome is uniquely determined.

/// A token-bucket rate limiter.
///
/// The bucket starts FULL at `new()` (`tokens == capacity`). Tokens are
/// refilled ONLY by `try_acquire` (see its doc comment for the precise refill
/// rule); [`TokenBucket::available`] performs no refill and simply returns the
/// stored count.
pub struct TokenBucket {
    capacity: u64,
    /// Stored token count. Read by `available()` and mutated only by
    /// `try_acquire` (refill + acquire).
    tokens: u64,
    /// Refill rate in tokens-per-second. Read only inside `try_acquire`.
    #[allow(dead_code)]
    refill_per_sec: u64,
    /// Logical-ms anchor of the last committed refill. Read only inside
    /// `try_acquire`.
    #[allow(dead_code)]
    anchor: u64,
}

impl TokenBucket {
    /// Create a full bucket holding `capacity` tokens, refilling at
    /// `refill_per_sec` tokens per second. The internal refill anchor starts
    /// at 0 logical ms.
    pub fn new(capacity: u64, refill_per_sec: u64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_per_sec,
            anchor: 0,
        }
    }

    /// The bucket's maximum capacity.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The current stored token count. Performs NO refill — it reflects only
    /// the last `try_acquire`'s committed refill.
    pub fn available(&self) -> u64 {
        self.tokens
    }

    /// Attempt to acquire `n` tokens at logical time `now_ms`.
    ///
    /// # Refill rule (committed BEFORE the acquire decision, regardless of
    /// outcome)
    ///
    /// - `elapsed = now_ms - anchor` (logical ms since the last committed
    ///   refill).
    /// - `gained = floor(refill_per_sec * elapsed / 1000)` whole tokens.
    /// - `tokens = min(tokens + gained, capacity)` — refill is clamped so
    ///   tokens never exceed capacity.
    /// - The anchor advances by `gained * 1000 / refill_per_sec` ms (the
    ///   whole-token portion of elapsed); the sub-token fractional remainder
    ///   is CARRIED, not discarded — EXCEPT when the clamp caps the bucket at
    ///   `capacity`, in which case the anchor jumps to `now_ms` (idle-past-full
    ///   time is forfeit).
    /// - `now_ms == anchor` refills nothing (elapsed 0, gained 0, anchor
    ///   unchanged).
    ///
    /// # Acquire (atomic, all-or-nothing, AFTER the refill commits)
    ///
    /// - If `tokens >= n`: deduct `n`, return `true`.
    /// - Else: deduct nothing, return `false`.
    /// - `n > capacity` always returns `false` (tokens can never exceed
    ///   capacity).
    #[allow(unused_variables)]
    pub fn try_acquire(&mut self, n: u64, now_ms: u64) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_full() {
        let b = TokenBucket::new(10, 5);
        assert_eq!(b.available(), 10);
    }

    #[test]
    fn capacity_returns_configured_capacity() {
        let b = TokenBucket::new(7, 3);
        assert_eq!(b.capacity(), 7);
    }

    #[test]
    fn available_at_construction_equals_capacity() {
        let b = TokenBucket::new(16, 4);
        assert_eq!(b.available(), 16);
        assert_eq!(b.capacity(), 16);
    }
}
