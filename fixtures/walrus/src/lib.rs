//! An append-only key–value store — a coding-task eval fixture.
//!
//! This crate is committed in a **FAILING** state on purpose: `compact()` is
//! left as `todo!()`, and exactly one test that calls it panics.  A correct
//! implementation of `compact()` makes all tests — including the sealed
//! holdout suite — green.  The workspace `exclude`s `fixtures/*`, so the
//! project's own gates never build or lint this file.

use std::collections::HashMap;

/// An append-only in-memory key–value store with `String` keys and `i64`
/// values.
///
/// Every `put` and `delete` appends a new record to the internal log; no
/// record is ever removed mid-log.  The log is the source of truth for both
/// value lookups and record ordering.  An index (`HashMap`) shadows the log so
/// `get` is O(1) without scanning.
pub struct WalrusStore {
    /// Append-only record log.  Each entry is `(key, Some(value))` for a put
    /// or `(key, None)` for a tombstone (delete).
    log: Vec<(String, Option<i64>)>,
    /// Index from key to the **position of its latest record** in `log`.
    /// Kept consistent with every `put` and `delete` append.
    index: HashMap<String, usize>,
}

impl WalrusStore {
    /// Create a new, empty store.
    pub fn new() -> Self {
        Self {
            log: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Store `key → value`.
    ///
    /// Always appends a new `Some(value)` record; the index is updated to
    /// point at the new tail position.
    pub fn put(&mut self, key: String, value: i64) {
        let pos = self.log.len();
        self.log.push((key.clone(), Some(value)));
        self.index.insert(key, pos);
    }

    /// Return the current value for `key`, or `None` if absent or deleted.
    pub fn get(&self, key: &str) -> Option<i64> {
        let &pos = self.index.get(key)?;
        self.log[pos].1
    }

    /// Mark `key` as deleted by appending a tombstone (`None`) record.
    ///
    /// Subsequent `get` calls return `None`; the key is completely dropped by
    /// the next `compact()`.
    pub fn delete(&mut self, key: String) {
        let pos = self.log.len();
        self.log.push((key.clone(), None));
        self.index.insert(key, pos);
    }

    /// Return the full record log in append order.
    ///
    /// Each entry is `(key, Some(value))` for a put or `(key, None)` for a
    /// tombstone.  This is the primary observability surface for tests.
    /// Do NOT change this signature when implementing `compact()`.
    pub fn records(&self) -> Vec<(String, Option<i64>)> {
        self.log.clone()
    }

    /// Compact the log: rewrite it so it contains only the **live latest
    /// record per key**, ordered by the position of each key's latest record
    /// in the pre-compaction log.
    ///
    /// Semantics:
    /// - A key is *live* if its latest record is a `put` (`Some(v)`).
    /// - A key whose latest record is a tombstone (`None`) is dropped entirely.
    /// - Surviving records are ordered by the **position of each key's latest
    ///   record** in the pre-compaction log (smallest position first).
    /// - After compaction every `get` returns the same value as before.
    /// - `records()` reflects the new compacted log.
    pub fn compact(&mut self) {
        todo!()
    }
}

impl Default for WalrusStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a CRC-32 digest of a byte slice using the reversed polynomial
/// 0xEDB8_8320.  Useful for integrity checks — for example, you can verify
/// that a `compact()` call leaves all live values unchanged by comparing
/// `crc32(before_bytes)` to `crc32(after_bytes)`.
///
/// This function is **not** called internally by `compact()`; its presence
/// documents a checkpoint pattern for callers who need change-detection
/// between compactions.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::WalrusStore;

    // ── put / get ────────────────────────────────────────────────────────────

    #[test]
    fn put_then_get_round_trips() {
        let mut s = WalrusStore::new();
        s.put("hello".into(), 42);
        assert_eq!(s.get("hello"), Some(42));
    }

    #[test]
    fn put_overwrites_value_for_existing_key() {
        let mut s = WalrusStore::new();
        s.put("k".into(), 1);
        s.put("k".into(), 2);
        assert_eq!(s.get("k"), Some(2));
    }

    #[test]
    fn get_absent_key_returns_none() {
        let s = WalrusStore::new();
        assert_eq!(s.get("nope"), None);
    }

    // ── delete ───────────────────────────────────────────────────────────────

    #[test]
    fn delete_makes_key_absent() {
        let mut s = WalrusStore::new();
        s.put("x".into(), 99);
        s.delete("x".into());
        assert_eq!(s.get("x"), None);
    }

    #[test]
    fn delete_appends_tombstone_to_log() {
        let mut s = WalrusStore::new();
        s.put("x".into(), 7);
        s.delete("x".into());
        assert_eq!(
            s.records(),
            vec![("x".to_string(), Some(7)), ("x".to_string(), None),],
        );
    }

    // ── compact ──────────────────────────────────────────────────────────────
    // This test PANICS because compact() is todo!(). It is the SOLE failing
    // test in this crate. All put/get/delete tests above pass.

    #[test]
    fn compact_keeps_live_latest() {
        let mut s = WalrusStore::new();
        s.put("a".into(), 1);
        s.put("b".into(), 2);
        s.put("a".into(), 3);
        s.delete("b".into());
        s.compact();
        // One surviving key — order is unambiguous.
        assert_eq!(s.get("a"), Some(3));
        assert_eq!(s.get("b"), None);
        assert_eq!(s.records(), vec![("a".to_string(), Some(3))]);
    }
}
