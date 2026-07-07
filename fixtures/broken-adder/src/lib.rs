//! A deliberately-broken adder — the coding-task eval fixture.
//!
//! This crate is committed in a FAILING state on purpose: `add` subtracts
//! instead of adds (the planted bug), so `cargo test` here is red. It is DATA,
//! not code under test — the harness's job in the eval is to find the bug, fix
//! it, and make the tests pass. The workspace `exclude`s `fixtures/*`, so the
//! project's own gates never build or lint this file.

/// Add two integers.
///
/// The planted bug: this subtracts instead of adding, so the tests below fail
/// until the harness fixes it (`a - b` → `a + b`).
pub fn add(a: i64, b: i64) -> i64 {
    a - b
}

#[cfg(test)]
mod tests {
    use super::add;

    #[test]
    fn adds_two_and_two() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    fn adds_a_negative_and_a_positive() {
        assert_eq!(add(-3, 10), 7);
    }
}
