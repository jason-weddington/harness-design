//! Byte-safe text truncation — a coding-task eval fixture.
//!
//! This crate is committed in a FAILING state on purpose: the tests below
//! encode the intended contract precisely, and one deviation from that contract
//! keeps the suite red until the harness finds it and repairs it. The workspace
//! `exclude`s `fixtures/*`, so the project's own gates never build or lint this
//! file.

/// Return the longest prefix of `s` that fits within `max_bytes` bytes and
/// ends on a valid UTF-8 `char` boundary.
///
/// The returned slice is always a valid `&str` — a multi-byte character is
/// NEVER split. When `s` already fits within `max_bytes`, all of `s` is
/// returned.
///
/// Concretely: if `max_bytes` lands mid-character, the returned prefix is
/// shortened to the previous `char` boundary. `max_bytes == 0` returns the
/// empty string.
pub fn preview(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    &s[..max_bytes]
}

#[cfg(test)]
mod tests {
    use super::preview;

    #[test]
    fn string_shorter_than_cap_is_returned_whole() {
        assert_eq!(preview("hi", 10), "hi");
    }

    #[test]
    fn ascii_truncates_cleanly_at_the_cap() {
        // Every byte is its own char boundary — the naive slice matches the
        // boundary-aware slice.
        assert_eq!(preview("abcdefghij", 5), "abcde");
    }

    #[test]
    fn multibyte_truncation_walks_back_to_a_char_boundary() {
        // "café au lait": c a f é(2 bytes: 0xC3 0xA9) ' ' a u ' ' l a i t
        // Bytes 0..3 = "caf"; byte index 3 is the START of 'é'; byte index 4
        // is INSIDE 'é'. `max_bytes = 4` must fall back to the char boundary
        // at index 3, returning "caf". A naive `&s[..4]` panics.
        assert_eq!(preview("café au lait", 4), "caf");
    }

    #[test]
    fn cap_of_zero_returns_empty_string() {
        assert_eq!(preview("hello", 0), "");
    }
}
