//! Core agent-harness library.
//!
//! This is the loop that turns an LLM into an autonomous agent: prompt
//! assembly, tool dispatch, model I/O, and state management. Right now it is a
//! placeholder so the quality-gate harness has something to enforce against —
//! the real harness loop lands next. See the project `CLAUDE.md` for goals.

pub mod anthropic;
pub mod engine;
pub mod eval;
pub mod exec;
pub mod model;
pub mod ollama;
pub mod prompt;
pub mod run_record;
pub mod store;
pub mod time;
pub mod tool;
pub mod tools;
pub mod workspace;

/// Crate-wide, test-only support (scripted backends, etc.). Compiled only
/// under `#[cfg(test)]` so it never ships in a release build.
#[cfg(test)]
mod test_support;

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
