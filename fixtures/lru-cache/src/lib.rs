//! A tiny vec-backed LRU cache — a coding-task eval fixture.
//!
//! This crate is committed in a FAILING state on purpose: the tests below
//! encode the intended contract precisely, and one deviation from that contract
//! keeps the suite red until the harness finds it and repairs it. The workspace
//! `exclude`s `fixtures/*`, so the project's own gates never build or lint this
//! file.

/// A capacity-bounded LRU cache with `String` keys and `i64` values.
///
/// Entries are stored in a `Vec<(String, i64)>`. Position `0` is the
/// LEAST-recently-used slot; the tail (position `len - 1`) is the
/// most-recently-used slot. Both an `insert` that hits an existing key AND a
/// successful `get` REFRESH the entry's recency by moving it to the tail. When
/// the cache is at capacity and a NEW key is inserted, the head (the
/// least-recently-used entry) is evicted to make room.
pub struct Cache {
    /// The stored entries, oldest-first. Public so tests can inspect the
    /// internal ordering directly.
    pub entries: Vec<(String, i64)>,
    /// The maximum number of entries the cache may hold.
    pub capacity: usize,
}

impl Cache {
    /// Create an empty cache with the given `capacity`.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
        }
    }

    /// Insert `key -> value`.
    ///
    /// If `key` already exists, its stored value is replaced with `value` and
    /// the entry is refreshed to the most-recently-used position. If `key` is
    /// new and the cache is at capacity, the least-recently-used entry is
    /// evicted first.
    pub fn insert(&mut self, key: String, value: i64) {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k == &key) {
            let (k, _) = self.entries.remove(pos);
            self.entries.push((k, value));
            return;
        }
        if self.entries.len() >= self.capacity {
            if self.entries.is_empty() {
                return;
            }
            self.entries.remove(0);
        }
        self.entries.push((key, value));
    }

    /// Look up `key`.
    ///
    /// Returns `Some(value)` when the key is present and refreshes the entry
    /// to the most-recently-used position; returns `None` when the key is
    /// absent (and the cache is left unchanged).
    pub fn get(&mut self, key: &str) -> Option<i64> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| *v)
    }
}

#[cfg(test)]
mod tests {
    use super::Cache;

    #[test]
    fn insert_then_get_round_trips() {
        let mut c = Cache::new(3);
        c.insert("a".to_string(), 1);
        assert_eq!(c.get("a"), Some(1));
    }

    #[test]
    fn re_inserting_an_existing_key_updates_the_value() {
        let mut c = Cache::new(3);
        c.insert("a".to_string(), 1);
        c.insert("a".to_string(), 2);
        assert_eq!(c.get("a"), Some(2));
    }

    #[test]
    fn eviction_removes_the_oldest_when_nothing_is_touched() {
        // Fill to capacity, then insert one more — the very first key inserted
        // is the least-recently-used and must be evicted.
        let mut c = Cache::new(2);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        c.insert("c".to_string(), 3);
        assert_eq!(c.get("a"), None, "a was oldest and should be evicted");
        assert_eq!(c.get("b"), Some(2));
        assert_eq!(c.get("c"), Some(3));
    }

    #[test]
    fn get_refreshes_recency_so_touched_key_survives_eviction() {
        // The recency contract: `get` must move the accessed entry to
        // most-recently-used. Sequence: insert a, insert b, get(a), insert c.
        // The correct eviction victim is `b` — `a` was just accessed. If `get`
        // fails to refresh recency, `a` is evicted instead and this test fails.
        let mut c = Cache::new(2);
        c.insert("a".to_string(), 1);
        c.insert("b".to_string(), 2);
        let _ = c.get("a");
        c.insert("c".to_string(), 3);
        assert_eq!(c.get("a"), Some(1), "a was just accessed and must survive");
        assert_eq!(c.get("b"), None, "b was least-recently-used and must be evicted");
        assert_eq!(c.get("c"), Some(3));
    }

    #[test]
    fn get_of_a_missing_key_is_none() {
        let mut c = Cache::new(2);
        c.insert("a".to_string(), 1);
        assert_eq!(c.get("missing"), None);
    }
}
