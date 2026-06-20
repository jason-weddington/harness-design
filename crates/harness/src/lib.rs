//! Core agent-harness library.
//!
//! This is the loop that turns an LLM into an autonomous agent: prompt
//! assembly, tool dispatch, model I/O, and state management. Right now it is a
//! placeholder so the quality-gate harness has something to enforce against —
//! the real harness loop lands next. See the project `CLAUDE.md` for goals.

/// The project's name. Placeholder until the real harness API exists.
#[must_use]
pub fn name() -> &'static str {
    "harness-design"
}

#[cfg(test)]
mod tests {
    use super::name;

    #[test]
    fn name_is_stable() {
        assert_eq!(name(), "harness-design");
    }
}
