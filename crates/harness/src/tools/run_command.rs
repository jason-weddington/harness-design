//! The `run_command` tool: the bounded shell workhorse.
//!
//! Runs a single program (NOT shell-interpreted) in the workspace root with a
//! clamped timeout, and reports the exit code plus a bounded stdout/stderr
//! extract. A non-zero exit or a timeout is surfaced as `is_error` — a failing
//! command is *steering* the agent, not a harness failure.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::exec::{ExecSpec, format_duration, run};
use crate::tool::{Tool, ToolCtx, ToolResult};

/// Default timeout when the model omits `timeout_secs`.
const DEFAULT_TIMEOUT_SECS: u64 = 300;
/// Hard upper bound; larger requests are clamped down to this.
const MAX_TIMEOUT_SECS: u64 = 600;

/// The `run_command` tool. See the module docs for the contract.
#[derive(Debug, Default, Clone, Copy)]
pub struct RunCommandTool;

impl RunCommandTool {
    /// Parse `args` (optional array of strings) from the tool input.
    fn parse_args(input: &Value) -> Result<Vec<String>, ToolResult> {
        match input.get("args") {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::Array(items)) => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    match item.as_str() {
                        Some(s) => out.push(s.to_string()),
                        None => {
                            return Err(ToolResult::error(
                                "run_command `args` must be an array of strings",
                            ));
                        }
                    }
                }
                Ok(out)
            }
            Some(_) => Err(ToolResult::error(
                "run_command `args` must be an array of strings",
            )),
        }
    }

    /// Parse `timeout_secs` (optional integer), defaulting and clamping to the
    /// `[1, MAX_TIMEOUT_SECS]` range.
    fn parse_timeout_secs(input: &Value) -> Result<u64, ToolResult> {
        match input.get("timeout_secs") {
            None | Some(Value::Null) => Ok(DEFAULT_TIMEOUT_SECS),
            Some(value) => match value.as_u64() {
                Some(secs) => Ok(secs.clamp(1, MAX_TIMEOUT_SECS)),
                None => Err(ToolResult::error(
                    "run_command `timeout_secs` must be a positive integer",
                )),
            },
        }
    }
}

#[async_trait]
impl Tool for RunCommandTool {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "run_command"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "run_command",
            "description": "Run a program in the workspace root and capture its output. `command` is the program to execute directly — it is NOT interpreted by a shell, so shell features (pipes, globs, redirection, `&&`, `$VARS`) will not work unless you invoke a shell yourself: set command=\"sh\" and args=[\"-c\", \"<your shell line>\"]. Pass ordinary arguments as separate strings via `args`. A non-zero exit code or a timeout is returned as an error so you can react to it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The program to run. Not shell-interpreted. Use \"sh\" with args [\"-c\", \"...\"] when you need shell features."
                    },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Arguments passed verbatim to the program, each as its own string."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Maximum seconds before the command is killed. Default 300, clamped to a maximum of 600."
                    }
                },
                "required": ["command"]
            }
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        let Some(command) = input.get("command").and_then(Value::as_str) else {
            return ToolResult::error(
                "run_command requires a `command` string (the program to run; pass arguments via `args`)",
            );
        };
        let args = match Self::parse_args(&input) {
            Ok(args) => args,
            Err(result) => return result,
        };
        let timeout_secs = match Self::parse_timeout_secs(&input) {
            Ok(secs) => secs,
            Err(result) => return result,
        };

        let spec = ExecSpec::new(
            command.to_string(),
            args,
            ctx.workspace().root().to_path_buf(),
            std::time::Duration::from_secs(timeout_secs),
        );
        let outcome = run(&spec).await;
        let duration = format_duration(outcome.duration);

        if outcome.timed_out {
            let mut result = ToolResult::with_detail(
                format!("timed out after {timeout_secs}s"),
                format!(
                    "--- stdout ---\n{}\n--- stderr ---\n{}",
                    outcome.stdout, outcome.stderr
                ),
                ctx,
            );
            result.is_error = true;
            return result;
        }

        let code = match outcome.exit_code {
            Some(code) => code.to_string(),
            None => "unknown".to_string(),
        };
        let is_error = outcome.exit_code != Some(0);
        let mut result = ToolResult::with_detail(
            format!("exit {code} ({duration})"),
            format!(
                "--- stdout ---\n{}\n--- stderr ---\n{}",
                outcome.stdout, outcome.stderr
            ),
            ctx,
        );
        result.is_error = is_error;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::RunCommandTool;
    use crate::tool::{DETAIL_CAP, Tool, ToolCtx};
    use serde_json::json;

    #[tokio::test]
    async fn happy_path_reports_exit_zero_and_output() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(
                json!({ "command": "/bin/sh", "args": ["-c", "echo hi-there"] }),
                &ctx,
            )
            .await;
        assert!(!out.is_error);
        assert!(
            out.summary.starts_with("exit 0 ("),
            "summary: {}",
            out.summary
        );
        assert!(out.detail.unwrap().contains("hi-there"));
    }

    #[tokio::test]
    async fn failing_command_sets_is_error() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(
                json!({ "command": "/bin/sh", "args": ["-c", "exit 5"] }),
                &ctx,
            )
            .await;
        assert!(out.is_error);
        assert!(
            out.summary.starts_with("exit 5 ("),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn timeout_path_sets_is_error_and_summary() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(
                json!({ "command": "/bin/sh", "args": ["-c", "sleep 30"], "timeout_secs": 1 }),
                &ctx,
            )
            .await;
        assert!(out.is_error);
        assert!(
            out.summary.contains("timed out after 1s"),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn spawn_failure_reports_unknown_exit_as_error() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(json!({ "command": "/no/such/program-xyzzy" }), &ctx)
            .await;
        assert!(out.is_error);
        assert!(
            out.summary.starts_with("exit unknown ("),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn large_output_offloads() {
        let ctx = ToolCtx::stub();
        // Emit well over DETAIL_CAP characters so the detail is offloaded.
        let script = "i=0; while [ $i -lt 1000 ]; do echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; i=$((i+1)); done";
        let out = RunCommandTool
            .run(
                json!({ "command": "/bin/sh", "args": ["-c", script] }),
                &ctx,
            )
            .await;
        assert!(!out.is_error);
        assert!(out.offload_path.is_some(), "large output should offload");
        let detail = out.detail.expect("detail present");
        assert!(
            detail.contains("truncated"),
            "detail should be flagged truncated"
        );
        assert!(detail.chars().count() <= DETAIL_CAP + 200);
    }

    #[tokio::test]
    async fn missing_command_is_a_steering_error() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool.run(json!({}), &ctx).await;
        assert!(out.is_error);
        assert!(out.summary.contains("requires a `command`"));
    }

    #[tokio::test]
    async fn non_string_args_element_is_rejected() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(json!({ "command": "/bin/sh", "args": [1, 2] }), &ctx)
            .await;
        assert!(out.is_error);
        assert!(out.summary.contains("array of strings"));
    }

    #[tokio::test]
    async fn non_array_args_is_rejected() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(json!({ "command": "/bin/sh", "args": "oops" }), &ctx)
            .await;
        assert!(out.is_error);
        assert!(out.summary.contains("array of strings"));
    }

    #[tokio::test]
    async fn non_integer_timeout_is_rejected() {
        let ctx = ToolCtx::stub();
        let out = RunCommandTool
            .run(
                json!({ "command": "/bin/sh", "args": ["-c", "true"], "timeout_secs": "soon" }),
                &ctx,
            )
            .await;
        assert!(out.is_error);
        assert!(out.summary.contains("positive integer"));
    }

    #[tokio::test]
    async fn oversized_timeout_is_clamped_and_still_runs() {
        let ctx = ToolCtx::stub();
        // 100000 clamps to 600; the command still runs to completion quickly.
        let out = RunCommandTool
            .run(
                json!({ "command": "/bin/sh", "args": ["-c", "echo ok"], "timeout_secs": 100_000 }),
                &ctx,
            )
            .await;
        assert!(!out.is_error);
        assert!(out.detail.unwrap().contains("ok"));
    }

    #[test]
    fn schema_advertises_name_and_required_command() {
        let schema = RunCommandTool.schema();
        assert_eq!(schema["name"], "run_command");
        assert_eq!(schema["input_schema"]["required"][0], "command");
    }

    #[test]
    fn name_is_run_command() {
        assert_eq!(RunCommandTool.name(), "run_command");
    }
}
