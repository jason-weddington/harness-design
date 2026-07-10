//! Sealed holdout tests for `WalrusStore::compact()`.
//!
//! These tests cover deeper-semantics behaviors not exercised by the single
//! visible test in `src/lib.rs`.  They live under `holdout/` in the committed
//! source and are copied into the crate's `tests/` directory only when
//! verifying solvability / running the eval:
//!
//!   cp -r fixtures/walrus/holdout/. fixtures/walrus/
//!   cd fixtures/walrus && cargo test
//!
//! All four tests are green only under a correct `compact()` implementation.

use walrus::WalrusStore;

/// (1) Delete-then-reinsert: the key's position in the compacted log is
/// determined by its LATEST put, not its original one.
///
/// Sequence: apple(1) at pos 0, zebra(2) at pos 1, delete apple at pos 2,
/// apple(3) at pos 3.  After compact, "zebra" has its latest record at pos 1
/// and "apple" at pos 3, so the compacted log must be [zebra, apple] —
/// non-alphabetical order.  A shallow impl that sorts keys or iterates the
/// HashMap without position awareness will fail this.
#[test]
fn delete_then_reinsert_positions_at_latest_put() {
    let mut s = WalrusStore::new();
    s.put("apple".into(), 1); // pos 0
    s.put("zebra".into(), 2); // pos 1
    s.delete("apple".into()); // pos 2 — tombstone for apple
    s.put("apple".into(), 3); // pos 3 — apple reinserted; its latest is now here
    s.compact();
    // "zebra" latest at pos 1 < "apple" latest at pos 3 → zebra comes first.
    assert_eq!(
        s.records(),
        vec![
            ("zebra".to_string(), Some(2)),
            ("apple".to_string(), Some(3)),
        ],
    );
    assert_eq!(s.get("apple"), Some(3));
    assert_eq!(s.get("zebra"), Some(2));
}

/// (2) Idempotence: calling compact() twice produces the same records() as
/// calling it once.
#[test]
fn compact_is_idempotent() {
    let mut s = WalrusStore::new();
    s.put("p".into(), 10);
    s.put("q".into(), 20);
    s.put("p".into(), 30);
    s.delete("q".into());
    s.put("r".into(), 40);
    s.compact();
    let after_one = s.records();
    s.compact();
    let after_two = s.records();
    assert_eq!(after_one, after_two, "compact() must be idempotent");
}

/// (3) Trailing tombstone: a key whose final record is a delete is completely
/// absent from the compacted log — no Some entry, no None entry.
#[test]
fn trailing_tombstone_is_dropped() {
    let mut s = WalrusStore::new();
    s.put("keep".into(), 1);
    s.put("drop".into(), 2);
    s.delete("drop".into()); // "drop"'s latest is a tombstone
    s.compact();
    assert_eq!(s.records(), vec![("keep".to_string(), Some(1))]);
    assert_eq!(s.get("drop"), None);
    assert_eq!(s.get("keep"), Some(1));
}

/// (4) Empty store: compact() on a store that has never been written to must
/// not panic, and records() must remain empty.
#[test]
fn compact_on_empty_store_is_no_op() {
    let mut s = WalrusStore::new();
    s.compact();
    assert_eq!(s.records(), vec![]);
}
