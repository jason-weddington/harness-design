//! The `run_checks` tool: run the task's configured quality gates.
//!
//! This is the model-facing wrapper over [`ChecksRunner`](crate::exec::ChecksRunner),
//! the mechanical done-oracle. It takes no arguments — the checks are declared
//! by project config, not chosen by the model — and its result mirrors
//! `finish(done)` verification: the same checks decide both.

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::exec::{ChecksRunner, format_duration};
use crate::tool::{Tool, ToolCtx, ToolResult};

/// The `run_checks` tool, holding the [`ChecksRunner`] it delegates to.
#[derive(Debug, Clone)]
pub struct RunChecksTool {
    runner: ChecksRunner,
}

impl RunChecksTool {
    /// Build the tool over a configured [`ChecksRunner`].
    #[must_use]
    pub fn new(runner: ChecksRunner) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl Tool for RunChecksTool {
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "run_checks"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "run_checks",
            "description": "Run the task's configured quality checks (the project's declared gates). Takes no arguments — the checks are fixed by project config, not chosen by you. This is the mechanical done-oracle: when you finish with status `done`, the harness re-runs these exact checks to verify the claim, so a green result here is the bar you must clear.",
            "input_schema": {
                "type": "object",
                "properties": {}
            }
        })
    }

    async fn run(&self, _input: Value, ctx: &ToolCtx) -> ToolResult {
        let report = self.runner.run(ctx).await;
        let duration = format_duration(report.duration);

        let summary = if report.passed {
            format!("checks passed ({duration})")
        } else if report.timed_out {
            format!("checks FAILED: timed out ({duration})")
        } else {
            let code = match report.exit_code {
                Some(code) => code.to_string(),
                None => "unknown".to_string(),
            };
            format!("checks FAILED: exit {code} ({duration})")
        };

        let detail = match &report.offload_path {
            Some(path) => format!(
                "{}\n\n[full check output: {}]",
                report.excerpt,
                path.display()
            ),
            None => report.excerpt.clone(),
        };

        let mut result = ToolResult::with_detail(summary, detail, ctx);
        // The runner already offloaded the full combined output; advertise that
        // path (with_detail only offloads when the *excerpt* itself overflows).
        result.offload_path.clone_from(&report.offload_path);
        result.is_error = !report.passed;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::RunChecksTool;
    use crate::exec::{CheckCommand, ChecksRunner};
    use crate::tool::{Tool, ToolCtx};
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Duration;

    fn runner(script: &str, timeout: Duration) -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), script.to_string()],
            },
            PathBuf::from("/"),
            timeout,
        )
    }

    #[tokio::test]
    async fn passing_checks_report_ok() {
        let ctx = ToolCtx::stub();
        let tool = RunChecksTool::new(runner("exit 0", Duration::from_secs(10)));
        let out = tool.run(json!({}), &ctx).await;
        assert!(!out.is_error);
        assert!(
            out.summary.starts_with("checks passed ("),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn failing_checks_report_exit_code_and_error() {
        let ctx = ToolCtx::stub();
        let tool = RunChecksTool::new(runner("exit 4", Duration::from_secs(10)));
        let out = tool.run(json!({}), &ctx).await;
        assert!(out.is_error);
        assert!(
            out.summary.contains("checks FAILED: exit 4"),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn timed_out_checks_report_timeout() {
        let ctx = ToolCtx::stub();
        let tool = RunChecksTool::new(runner("sleep 30", Duration::from_millis(300)));
        let out = tool.run(json!({}), &ctx).await;
        assert!(out.is_error);
        assert!(
            out.summary.contains("checks FAILED: timed out"),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn spawn_failure_reports_unknown_exit() {
        let ctx = ToolCtx::stub();
        let tool = RunChecksTool::new(ChecksRunner::new(
            CheckCommand {
                program: "/no/such/checks-xyzzy".to_string(),
                args: vec![],
            },
            PathBuf::from("/"),
            Duration::from_secs(5),
        ));
        let out = tool.run(json!({}), &ctx).await;
        assert!(out.is_error);
        assert!(
            out.summary.contains("checks FAILED: exit unknown"),
            "summary: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn detail_notes_offload_path() {
        let ctx = ToolCtx::stub();
        let tool = RunChecksTool::new(runner("echo checkoutput", Duration::from_secs(10)));
        let out = tool.run(json!({}), &ctx).await;
        assert!(out.offload_path.is_some());
        let detail = out.detail.expect("detail present");
        assert!(detail.contains("checkoutput"));
        assert!(detail.contains("full check output"));
    }

    #[test]
    fn schema_takes_no_required_args() {
        let tool = RunChecksTool::new(runner("true", Duration::from_secs(1)));
        let schema = tool.schema();
        assert_eq!(schema["name"], "run_checks");
        assert!(schema["input_schema"].get("required").is_none());
    }

    #[test]
    fn name_is_run_checks() {
        let tool = RunChecksTool::new(runner("true", Duration::from_secs(1)));
        assert_eq!(tool.name(), "run_checks");
    }
}
