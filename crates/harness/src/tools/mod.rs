//! Concrete tools the agent can invoke.
//!
//! Each submodule ships one [`Tool`](crate::tool::Tool) implementation. The
//! trait, [`ToolResult`](crate::tool::ToolResult), and
//! [`ToolRegistry`](crate::tool::ToolRegistry) themselves live in
//! [`crate::tool`]; this module just gathers the concrete tools and offers
//! [`standard_registry`] as the one-stop constructor of "the v1 toolset".

use std::sync::Arc;

use crate::engine::{FINISH_TOOL_NAME, FinishTool};
use crate::exec::ChecksRunner;
use crate::tool::ToolRegistry;
use crate::tools::edit_file::EditFileTool;
use crate::tools::list_files::ListFilesTool;
use crate::tools::read_file::{READ_FILE_TOOL_NAME, ReadFileTool};
use crate::tools::run_checks::RunChecksTool;
use crate::tools::run_command::RunCommandTool;

pub mod edit_file;
pub mod list_files;
pub mod read_file;
pub mod run_checks;
pub mod run_command;

/// Build the standard v1 [`ToolRegistry`]: the file-editing suite
/// (`read_file`, `list_files`, `edit_file`), the shell workhorse
/// (`run_command`), the loop's [`FinishTool`], and — when a [`ChecksRunner`]
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
    let mut registry = ToolRegistry::new();
    registry.register(READ_FILE_TOOL_NAME, Arc::new(ReadFileTool));
    registry.register("list_files", Arc::new(ListFilesTool));
    registry.register("edit_file", Arc::new(EditFileTool));
    registry.register("run_command", Arc::new(RunCommandTool));
    registry.register(FINISH_TOOL_NAME, Arc::new(FinishTool));
    if let Some(runner) = checks {
        registry.register("run_checks", Arc::new(RunChecksTool::new(runner)));
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::standard_registry;
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
                "edit_file".to_string(),
                FINISH_TOOL_NAME.to_string(),
                "list_files".to_string(),
                "read_file".to_string(),
                "run_command".to_string(),
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
                "edit_file".to_string(),
                FINISH_TOOL_NAME.to_string(),
                "list_files".to_string(),
                "read_file".to_string(),
                "run_checks".to_string(),
                "run_command".to_string(),
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
            "run_command",
            FINISH_TOOL_NAME,
            "run_checks",
        ] {
            assert!(
                registry.get(name).is_some(),
                "standard_registry must register `{name}`"
            );
        }
    }
}
