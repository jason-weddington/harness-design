//! Concrete tools the agent can invoke.
//!
//! Each submodule ships one [`Tool`](crate::tool::Tool) implementation. The
//! trait, [`ToolResult`](crate::tool::ToolResult), and
//! [`ToolRegistry`](crate::tool::ToolRegistry) themselves live in
//! [`crate::tool`]; this module just gathers the concrete tools and offers
//! [`standard_registry`] as the one-stop constructor of "the v1 toolset",
//! plus [`answer_registry`] — the READ-ONLY variant answer mode runs on.

use std::sync::Arc;

use crate::engine::{FINISH_TOOL_NAME, FinishTool};
use crate::exec::ChecksRunner;
use crate::tool::ToolRegistry;
use crate::tools::bash::BashTool;
use crate::tools::edit_file::EditFileTool;
use crate::tools::list_files::ListFilesTool;
use crate::tools::read_file::{READ_FILE_TOOL_NAME, ReadFileTool};
use crate::tools::run_checks::RunChecksTool;

pub mod bash;
pub mod edit_file;
pub mod list_files;
pub mod read_file;
pub mod run_checks;

/// Build the standard v1 [`ToolRegistry`]: the file-editing suite
/// (`read_file`, `list_files`, `edit_file`), the shell workhorse
/// (`bash`), the loop's [`FinishTool`], and — when a [`ChecksRunner`]
/// is supplied — the `run_checks` tool.
///
/// Order-of-registration does not matter for the model-facing schema list:
/// [`ToolRegistry`] is a [`BTreeMap`](std::collections::BTreeMap) and
/// [`ToolRegistry::list`](crate::tool::ToolRegistry::list) returns schemas in
/// deterministic name order — same registry input, byte-identical prompt.
///
/// [`FinishTool`]: crate::engine::FinishTool
#[must_use]
pub fn standard_registry(checks: Option<ChecksRunner>) -> ToolRegistry {
    build_registry(checks, false)
}

/// Build the READ-ONLY [`ToolRegistry`] answer mode runs on: everything
/// [`standard_registry`] registers EXCEPT `edit_file`, with a [`FinishTool`]
/// whose `answer_mode` is `true` so the `answer` disposition and its `result`
/// property are advertised.
///
/// The missing `edit_file` is a convenience, not the enforcement: an answer
/// run's read-only guarantee is the engine's inverted tree precondition (an
/// accepted `finish(answer)` requires the working tree to be UNCHANGED since
/// the run started), which holds no matter which tool mutated the tree. What
/// dropping `edit_file` buys is the COMMON path — many answer agents sharing
/// one checkout should not trip over each other's edits — and a prompt that
/// does not advertise a capability the run would then reject.
///
/// `checks` is accepted for symmetry with [`standard_registry`]; the CLI
/// (`talos run --mode answer`) passes `None` this cut, so `run_checks` is
/// absent from a CLI-driven answer run.
///
/// [`FinishTool`]: crate::engine::FinishTool
#[must_use]
pub fn answer_registry(checks: Option<ChecksRunner>) -> ToolRegistry {
    build_registry(checks, true)
}

/// The shared builder behind [`standard_registry`] and [`answer_registry`].
///
/// `answer_mode` does exactly two things: it drops `edit_file`, and it flips
/// [`FinishTool::answer_mode`]. With `answer_mode == false` the output is
/// byte-identical to what `standard_registry` has always produced —
/// [`FinishTool::default()`] IS `FinishTool { answer_mode: false }`.
fn build_registry(checks: Option<ChecksRunner>, answer_mode: bool) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(READ_FILE_TOOL_NAME, Arc::new(ReadFileTool));
    registry.register("list_files", Arc::new(ListFilesTool));
    if !answer_mode {
        registry.register("edit_file", Arc::new(EditFileTool));
    }
    registry.register("bash", Arc::new(BashTool));
    registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool { answer_mode }));
    if let Some(runner) = checks {
        registry.register("run_checks", Arc::new(RunChecksTool::new(runner)));
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::{answer_registry, standard_registry};
    use crate::engine::FINISH_TOOL_NAME;
    use crate::exec::{CheckCommand, ChecksRunner};
    use std::path::PathBuf;
    use std::time::Duration;

    fn runner() -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
    }

    fn names(registry: &crate::tool::ToolRegistry) -> Vec<String> {
        registry
            .list()
            .into_iter()
            .filter_map(|s| {
                s.get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            })
            .collect()
    }

    #[test]
    fn registers_all_v1_tools_without_checks() {
        let registry = standard_registry(None);
        let got = names(&registry);
        // BTreeMap iteration order: alphabetical.
        assert_eq!(
            got,
            vec![
                "bash".to_string(),
                "edit_file".to_string(),
                FINISH_TOOL_NAME.to_string(),
                "list_files".to_string(),
                "read_file".to_string(),
            ],
            "no-checks registry excludes run_checks"
        );
    }

    #[test]
    fn registers_run_checks_when_checks_are_supplied() {
        let registry = standard_registry(Some(runner()));
        let got = names(&registry);
        assert_eq!(
            got,
            vec![
                "bash".to_string(),
                "edit_file".to_string(),
                FINISH_TOOL_NAME.to_string(),
                "list_files".to_string(),
                "read_file".to_string(),
                "run_checks".to_string(),
            ],
            "with-checks registry includes run_checks"
        );
    }

    #[test]
    fn each_registered_tool_is_get_able() {
        let registry = standard_registry(Some(runner()));
        for name in [
            "read_file",
            "list_files",
            "edit_file",
            "bash",
            FINISH_TOOL_NAME,
            "run_checks",
        ] {
            assert!(
                registry.get(name).is_some(),
                "standard_registry must register `{name}`"
            );
        }
    }

    #[test]
    fn answer_registry_is_read_only_without_checks() {
        let registry = answer_registry(None);
        let got = names(&registry);
        // BTreeMap iteration order: alphabetical. `edit_file` is absent.
        assert_eq!(
            got,
            vec![
                "bash".to_string(),
                FINISH_TOOL_NAME.to_string(),
                "list_files".to_string(),
                "read_file".to_string(),
            ],
            "the answer registry excludes edit_file and run_checks"
        );
        assert!(
            registry.get("edit_file").is_none(),
            "an answer run must not be handed edit_file"
        );
    }

    #[test]
    fn answer_registry_adds_run_checks_but_still_no_edit_file() {
        let registry = answer_registry(Some(runner()));
        let got = names(&registry);
        assert_eq!(
            got,
            vec![
                "bash".to_string(),
                FINISH_TOOL_NAME.to_string(),
                "list_files".to_string(),
                "read_file".to_string(),
                "run_checks".to_string(),
            ],
            "with-checks answer registry includes run_checks, never edit_file"
        );
        assert!(
            registry.get("edit_file").is_none(),
            "an answer run must not be handed edit_file even with checks wired"
        );
    }

    /// The two registries must advertise DIFFERENT finish schemas: the answer
    /// registry's carries `answer`, the standard one's does not.
    ///
    /// The exact answer-mode enum is item 1's (`FinishTool { answer_mode:
    /// true }`, pinned by
    /// `answer_mode_finish_schema_advertises_answer_last_without_requiring_result`):
    /// `answer` is appended LAST so the pre-existing members keep their
    /// order. `done` / `already_satisfied` stay advertised there and are
    /// rejected by the ENGINE with mode-specific steering — the registry is
    /// not where that contract lives.
    #[test]
    fn answer_registry_advertises_the_answer_mode_finish_schema() {
        let answer = answer_registry(None);
        let finish = answer.get(FINISH_TOOL_NAME).expect("finish is registered");
        assert_eq!(
            finish.schema()["input_schema"]["properties"]["disposition"]["enum"],
            serde_json::json!(["done", "blocked", "failed", "already_satisfied", "answer"]),
            "the answer registry advertises the answer-mode finish schema"
        );

        let standard = standard_registry(None);
        let finish = standard
            .get(FINISH_TOOL_NAME)
            .expect("finish is registered");
        assert_eq!(
            finish.schema()["input_schema"]["properties"]["disposition"]["enum"],
            serde_json::json!(["done", "blocked", "failed", "already_satisfied"]),
            "standard_registry's finish schema is unchanged — no `answer`"
        );
    }
}
