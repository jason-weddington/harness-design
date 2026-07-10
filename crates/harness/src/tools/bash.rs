//! The `bash` tool: the bounded shell workhorse.
//!
//! Runs a shell command string in the workspace root with a clamped timeout,
//! and reports the exit code plus a bounded stdout/stderr extract. The command
//! is passed verbatim to `sh -c`, so all shell features — pipes, globs,
//! redirection, `&&`, `||`, `$VARS` — work directly without any wrapping.
//! A non-zero exit or a timeout is surfaced as `is_error` — a failing command
//! is *steering* the agent, not a harness failure.
//!
//! # Confinement posture
//!
//! File-tool path checks (`read_file`, `edit_file`, `list_files`) are
//! **steering**: they guide the model toward the workspace and away from the
//! host filesystem, but they are not a security boundary. Process isolation
//! rests on two mechanisms: the **dispatch host account** (an unprivileged OS
//! user with no access to credentials or sensitive paths), and **`env_clear`**
//! (`sh -c` runs under a minimal TERM/PATH/HOME environment — no ambient
//! secrets can leak via inherited variables). Reviewers and operators should
//! treat the host account and `env_clear` as the actual boundary, not the
//! file-tool guards.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::exec::{ExecSpec, format_duration, run};
use crate::tool::{Tool, ToolCtx, ToolResult};

/// Default timeout when the model omits `timeout_secs`.
const DEFAULT_TIMEOUT_SECS: u64 = 300;
/// Hard upper bound; larger requests are clamped down to this.
const MAX_TIMEOUT_SECS: u64 = 600;

/// The `bash` tool. See the module docs for the contract.
#[derive(Debug, Default, Clone, Copy)]
pub struct BashTool;

impl BashTool {
    /// Parse `timeout_secs` (optional integer), defaulting and clamping to the
    /// `[1, MAX_TIMEOUT_SECS]` range.
    fn parse_timeout_secs(input: &Value) -> Result<u64, ToolResult> {
        match input.get("timeout_secs") {
            None | Some(Value::Null) => Ok(DEFAULT_TIMEOUT_SECS),
            Some(value) => match value.as_u64() {
                Some(secs) => Ok(secs.clamp(1, MAX_TIMEOUT_SECS)),
                None => Err(ToolResult::error(
                    "bash `timeout_secs` must be a positive integer",
                )),
            },
        }
    }
}

#[async_trait]
impl Tool for BashTool {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "bash"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "bash",
            "description": "Run a shell command in the workspace root and capture its output. \
                `command` is a full shell command string passed to `sh -c` — shell features \
                (pipes, globs, redirection, `&&`, `||`, `$VARS`) work directly. \
                Use grep or find via this tool to locate code; prefer edit_file for file \
                mutations (its unique-match contract is safer than sed -i). \
                A non-zero exit code or a timeout is returned as an error so you can react to it.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run. Passed verbatim to `sh -c`. \
                            Pipes, globs, &&, ||, and $VARS all work."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Maximum seconds before the command is killed. \
                            Default 300, clamped to a maximum of 600."
                    }
                },
                "required": ["command"]
            }
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        let Some(command) = input.get("command").and_then(Value::as_str) else {
            return ToolResult::error("bash requires a `command` string (a shell command line)");
        };
        let timeout_secs = match Self::parse_timeout_secs(&input) {
            Ok(secs) => secs,
            Err(result) => return result,
        };

        let spec = ExecSpec::new(
            "sh",
            vec!["-c".to_string(), command.to_string()],
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
    use super::BashTool;
    use crate::tool::{DETAIL_CAP, Tool, ToolCtx};
    use serde_json::json;

    #[tokio::test]
    async fn happy_path_reports_exit_zero_and_output() {
        let ctx = ToolCtx::stub();
        let out = BashTool
            .run(json!({ "command": "echo hi-there" }), &ctx)
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
        let out = BashTool.run(json!({ "command": "exit 5" }), &ctx).await;
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
        let out = BashTool
            .run(json!({ "command": "sleep 30", "timeout_secs": 1 }), &ctx)
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
        // sh -c with a missing binary: sh itself starts, but the command fails.
        // Use a path that sh cannot resolve — non-zero exit.
        let out = BashTool
            .run(json!({ "command": "/no/such/program-xyzzy" }), &ctx)
            .await;
        assert!(out.is_error);
        // sh -c returns 127 for "command not found" — not "unknown".
        assert!(out.summary.starts_with("exit "), "summary: {}", out.summary);
    }

    #[tokio::test]
    async fn large_output_offloads() {
        let ctx = ToolCtx::stub();
        // Emit well over DETAIL_CAP characters so the detail is offloaded.
        let script = "i=0; while [ $i -lt 1000 ]; do echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; i=$((i+1)); done";
        let out = BashTool.run(json!({ "command": script }), &ctx).await;
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
        let out = BashTool.run(json!({}), &ctx).await;
        assert!(out.is_error);
        assert!(out.summary.contains("requires a `command`"));
    }

    #[tokio::test]
    async fn non_integer_timeout_is_rejected() {
        let ctx = ToolCtx::stub();
        let out = BashTool
            .run(json!({ "command": "true", "timeout_secs": "soon" }), &ctx)
            .await;
        assert!(out.is_error);
        assert!(out.summary.contains("positive integer"));
    }

    #[tokio::test]
    async fn oversized_timeout_is_clamped_and_still_runs() {
        let ctx = ToolCtx::stub();
        // 100000 clamps to 600; the command still runs to completion quickly.
        let out = BashTool
            .run(
                json!({ "command": "echo ok", "timeout_secs": 100_000 }),
                &ctx,
            )
            .await;
        assert!(!out.is_error);
        assert!(out.detail.unwrap().contains("ok"));
    }

    #[tokio::test]
    async fn shell_features_work_pipes_and_env() {
        let ctx = ToolCtx::stub();
        // Pipes and shell built-ins should work because we run via sh -c.
        let out = BashTool
            .run(json!({ "command": "echo hello | tr a-z A-Z" }), &ctx)
            .await;
        assert!(!out.is_error, "summary: {}", out.summary);
        assert!(out.detail.unwrap().contains("HELLO"));
    }

    #[test]
    fn schema_advertises_name_and_required_command() {
        let schema = BashTool.schema();
        assert_eq!(schema["name"], "bash");
        assert_eq!(schema["input_schema"]["required"][0], "command");
        // The args field must be gone.
        assert!(
            schema["input_schema"]["properties"].get("args").is_none(),
            "bash schema must not advertise an `args` field"
        );
    }

    #[test]
    fn name_is_bash() {
        assert_eq!(BashTool.name(), "bash");
    }
}
