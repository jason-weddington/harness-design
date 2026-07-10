//! The process-execution seam: spawn a child process under a **clean
//! environment**, bound it with a timeout, and capture its output — plus the
//! [`ChecksRunner`] built on top of it, which is the harness's mechanical
//! *done-oracle*.
//!
//! Two layers live here:
//!
//! - [`ExecSpec`] / [`ExecOutcome`] / [`run`] — the raw exec primitive. Every
//!   model-driven command (`bash`) and the project's quality gates
//!   (`run_checks`) go through this one place, so the creds-hygiene guarantee
//!   below is enforced once.
//! - [`CheckCommand`] / [`CheckReport`] / [`ChecksRunner`] — the declared
//!   quality-gate command and its structured result. A later engine item calls
//!   [`ChecksRunner::run`] directly to verify a `finish(done)` claim against the
//!   *same* checks the `run_checks` tool exposes, so this is a first-class type,
//!   not a tool-internal detail.
//!
//! Linux-only by design (tests use `/bin/sh`).

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

use crate::tool::ToolCtx;

/// Bound on the [`CheckReport::excerpt`], in characters. The excerpt is the
/// **tail** of the combined output because test/compiler failures print at the
/// *end* of a run — the last few thousand characters are the signal, the head
/// is usually setup noise.
const CHECK_EXCERPT_CAP: usize = 4_000;

/// A process to run: the program, its arguments, the working directory, a
/// hard timeout, and any explicitly-forwarded environment variables.
///
/// `args` are passed to the program directly — they are **not** shell-word-split
/// or glob-expanded. To use shell features, make `program` the shell (`sh`) and
/// pass `-c` plus a script in `args`.
#[derive(Debug, Clone)]
pub struct ExecSpec {
    /// The program to execute (resolved via the child's `PATH`).
    pub program: String,
    /// Arguments passed verbatim to the program (no shell interpretation).
    pub args: Vec<String>,
    /// Working directory the child is spawned in.
    pub cwd: PathBuf,
    /// Wall-clock budget; on expiry the child is killed and reaped.
    pub timeout: Duration,
    /// Extra environment variables layered on top of the clean base
    /// environment (see [`run`]). Empty by default.
    pub extra_env: Vec<(String, String)>,
}

impl ExecSpec {
    /// A spec for `program` with the given `args`, run in `cwd` under `timeout`
    /// and no extra environment.
    #[must_use]
    pub fn new(
        program: impl Into<String>,
        args: Vec<String>,
        cwd: PathBuf,
        timeout: Duration,
    ) -> Self {
        Self {
            program: program.into(),
            args,
            cwd,
            timeout,
            extra_env: Vec::new(),
        }
    }
}

/// The result of running an [`ExecSpec`].
///
/// `exit_code` is `None` when the process produced no ordinary exit status
/// (killed by a signal, including our own timeout kill, or failed to spawn).
/// `stdout` / `stderr` are captured with lossy UTF-8 decoding — arbitrary child
/// output is not guaranteed to be valid UTF-8, and a decoding error must never
/// crash the harness.
#[derive(Debug, Clone)]
pub struct ExecOutcome {
    /// The process exit code, or `None` if it did not exit normally.
    pub exit_code: Option<i32>,
    /// Whether the run was terminated because it exceeded its timeout.
    pub timed_out: bool,
    /// Captured standard output (lossy UTF-8).
    pub stdout: String,
    /// Captured standard error (lossy UTF-8).
    pub stderr: String,
    /// Wall-clock time the run took.
    pub duration: Duration,
}

/// Drain an optional async pipe to end, decoding lossily. A read error yields
/// whatever was captured so far rather than failing — captured output is
/// best-effort, never a hard error surface.
async fn drain<R: AsyncRead + Unpin>(pipe: Option<R>) -> String {
    let mut buf = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut buf).await;
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Run `spec` to completion (or timeout) and capture its output.
///
/// **Clean environment (creds hygiene).** The harness process holds secrets —
/// `ANTHROPIC_API_KEY`, `AGENT_GTD_API_KEY`, and friends. A model-driven command
/// must NEVER see them. So the child is spawned with `env_clear()` and ONLY:
/// - `PATH` and `HOME`, inherited from the parent (so the shell and common
///   tools still resolve and behave),
/// - `TERM=dumb` (disable any color/interactive terminal behavior), and
/// - the caller's explicit `extra_env`.
///
/// Nothing else crosses the boundary. This is the single chokepoint where that
/// guarantee is enforced; the env-cleanliness test proves a parent marker var
/// is absent from the child while `PATH` survives.
///
/// On timeout the child is killed (`start_kill` + `wait` to reap the zombie),
/// `timed_out` is set, and captured output is dropped (the run is abandoned).
pub async fn run(spec: &ExecSpec) -> ExecOutcome {
    let start = Instant::now();

    let mut command = Command::new(&spec.program);
    command
        .args(&spec.args)
        .current_dir(&spec.cwd)
        .env_clear()
        .env("TERM", "dumb")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Some(path) = std::env::var_os("PATH") {
        command.env("PATH", path);
    }
    if let Some(home) = std::env::var_os("HOME") {
        command.env("HOME", home);
    }
    for (key, value) in &spec.extra_env {
        command.env(key, value);
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(err) => {
            return ExecOutcome {
                exit_code: None,
                timed_out: false,
                stdout: String::new(),
                stderr: format!("failed to spawn `{}`: {err}", spec.program),
                duration: start.elapsed(),
            };
        }
    };

    // Take the pipe handles so we can drain them concurrently with the wait —
    // draining after the wait risks deadlock when the child fills a pipe buffer
    // and blocks on write while we block on exit.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let collected = tokio::time::timeout(spec.timeout, async {
        tokio::join!(child.wait(), drain(stdout_pipe), drain(stderr_pipe))
    })
    .await;

    match collected {
        Ok((status, stdout, stderr)) => {
            let exit_code = status.ok().and_then(|status| status.code());
            ExecOutcome {
                exit_code,
                timed_out: false,
                stdout,
                stderr,
                duration: start.elapsed(),
            }
        }
        Err(_elapsed) => {
            // Timed out: kill the child and reap it so we leave no zombie.
            let _ = child.start_kill();
            let _ = child.wait().await;
            ExecOutcome {
                exit_code: None,
                timed_out: true,
                stdout: String::new(),
                stderr: String::new(),
                duration: start.elapsed(),
            }
        }
    }
}

/// The last `cap` characters of `text` (the whole thing if it is shorter),
/// counted by `char` so a multi-byte boundary is never split.
fn tail(text: &str, cap: usize) -> String {
    let total = text.chars().count();
    if total <= cap {
        return text.to_string();
    }
    text.chars().skip(total - cap).collect()
}

/// Format a duration as a compact `"<secs>.<hundredths>s"` string for summaries.
pub(crate) fn format_duration(duration: Duration) -> String {
    format!("{:.2}s", duration.as_secs_f64())
}

/// A declared quality-gate command: the program plus its arguments.
///
/// Its [`Display`](std::fmt::Display) renders the human-readable command line
/// (program followed by space-separated args) for prompts and logs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckCommand {
    /// The program to run (not shell-interpreted).
    pub program: String,
    /// Arguments passed verbatim to the program.
    pub args: Vec<String>,
}

impl std::fmt::Display for CheckCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.program)?;
        for arg in &self.args {
            write!(f, " {arg}")?;
        }
        Ok(())
    }
}

/// The structured result of running the task's quality checks.
///
/// `passed` is the mechanical done-signal: `exit_code == Some(0) && !timed_out`.
/// The full combined output is offloaded and pointed at by `offload_path`;
/// `excerpt` is the bounded tail for inline display. This is the evidence a
/// later engine item attaches to a verified `Done`.
///
/// [`PartialEq`] is derived so [`crate::run_record::Verification`] — which
/// wraps a report — can be equality-compared in tests; [`Eq`] is deliberately
/// not implied here (no float-in-`Duration` reason today, but keeping the door
/// open for a future field that isn't `Eq`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckReport {
    /// Whether the checks passed (clean exit, not timed out).
    pub passed: bool,
    /// The check command's exit code, or `None` if it did not exit normally.
    pub exit_code: Option<i32>,
    /// Whether the checks were killed for exceeding their timeout.
    pub timed_out: bool,
    /// The tail of the combined stdout+stderr, bounded to [`CHECK_EXCERPT_CAP`].
    pub excerpt: String,
    /// Path to the full, offloaded combined output.
    pub offload_path: Option<PathBuf>,
    /// Wall-clock time the checks took.
    pub duration: Duration,
}

/// Runs a [`CheckCommand`] in the workspace and produces a [`CheckReport`].
///
/// This is the harness's mechanical done-oracle: the `run_checks` tool wraps it,
/// and a later engine item calls [`ChecksRunner::run`] directly to verify a
/// `finish(done)` claim against the identical checks — so it is a first-class
/// type rather than a detail hidden inside the tool.
#[derive(Debug, Clone)]
pub struct ChecksRunner {
    command: CheckCommand,
    workspace_root: PathBuf,
    timeout: Duration,
}

impl ChecksRunner {
    /// Build a runner for `command`, executed in `workspace_root` under
    /// `timeout`.
    #[must_use]
    pub fn new(command: CheckCommand, workspace_root: PathBuf, timeout: Duration) -> Self {
        Self {
            command,
            workspace_root,
            timeout,
        }
    }

    /// The command this runner executes.
    #[must_use]
    pub fn command(&self) -> &CheckCommand {
        &self.command
    }

    /// The runner's command rendered as a human-readable display string
    /// ([`CheckCommand`]'s [`Display`](std::fmt::Display)). Handed to the
    /// prompt layer so the system prompt can announce which checks the
    /// harness will use to verify a `finish(done)` claim.
    #[must_use]
    pub fn command_display(&self) -> String {
        self.command.to_string()
    }

    /// Run the checks: execute via [`run`] in the workspace root, offload the
    /// full combined output through `ctx`, and return a [`CheckReport`] whose
    /// `excerpt` is the bounded **tail** of the combined output.
    pub async fn run(&self, ctx: &ToolCtx) -> CheckReport {
        let spec = ExecSpec {
            program: self.command.program.clone(),
            args: self.command.args.clone(),
            cwd: self.workspace_root.clone(),
            timeout: self.timeout,
            extra_env: Vec::new(),
        };
        let outcome = run(&spec).await;

        let combined = format!("{}{}", outcome.stdout, outcome.stderr);
        let excerpt = tail(&combined, CHECK_EXCERPT_CAP);
        let offload_path = Some(ctx.offload(&combined));
        let passed = outcome.exit_code == Some(0) && !outcome.timed_out;

        CheckReport {
            passed,
            exit_code: outcome.exit_code,
            timed_out: outcome.timed_out,
            excerpt,
            offload_path,
            duration: outcome.duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CheckCommand, CheckReport, ChecksRunner, ExecSpec, format_duration, run, tail};
    use crate::tool::ToolCtx;
    use crate::workspace::{DiskOffloadSink, Workspace};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    fn sh(script: &str) -> ExecSpec {
        ExecSpec::new(
            "/bin/sh",
            vec!["-c".to_string(), script.to_string()],
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
    }

    #[tokio::test]
    async fn captures_stdout_and_stderr_separately_with_exit_code() {
        let outcome = run(&sh("echo to-out; echo to-err 1>&2")).await;
        assert_eq!(outcome.exit_code, Some(0));
        assert!(!outcome.timed_out);
        assert!(
            outcome.stdout.contains("to-out"),
            "stdout: {:?}",
            outcome.stdout
        );
        assert!(
            !outcome.stdout.contains("to-err"),
            "stderr leaked into stdout"
        );
        assert!(
            outcome.stderr.contains("to-err"),
            "stderr: {:?}",
            outcome.stderr
        );
    }

    #[tokio::test]
    async fn nonzero_exit_is_reported() {
        let outcome = run(&sh("exit 7")).await;
        assert_eq!(outcome.exit_code, Some(7));
        assert!(!outcome.timed_out);
    }

    #[tokio::test]
    async fn spawn_failure_yields_none_exit_and_message() {
        let outcome = run(&ExecSpec::new(
            "/no/such/program-xyzzy",
            vec![],
            PathBuf::from("/"),
            Duration::from_secs(5),
        ))
        .await;
        assert_eq!(outcome.exit_code, None);
        assert!(!outcome.timed_out);
        assert!(
            outcome.stderr.contains("failed to spawn"),
            "stderr: {:?}",
            outcome.stderr
        );
    }

    #[tokio::test]
    async fn timeout_kills_a_long_sleep_promptly() {
        let mut spec = sh("sleep 30");
        spec.timeout = Duration::from_millis(300);
        let outcome = run(&spec).await;
        assert!(outcome.timed_out, "should have timed out");
        assert_eq!(outcome.exit_code, None);
        assert!(
            outcome.duration < Duration::from_secs(5),
            "kill should be prompt, took {:?}",
            outcome.duration
        );
    }

    #[tokio::test]
    async fn child_environment_is_clean_but_keeps_path() {
        // We cannot set a marker via `std::env::set_var` — it is `unsafe` in
        // edition 2024 and `unsafe_code` is forbidden project-wide. Instead we
        // use an existing parent env var (any one other than the three we
        // intentionally forward) as the marker: it must be absent from the
        // child, proving `env_clear()` took effect.
        let marker = std::env::vars()
            .map(|(key, _)| key)
            .find(|key| key != "PATH" && key != "HOME" && key != "TERM")
            .expect("test process has at least one non-forwarded env var");

        let outcome = run(&sh("env")).await;
        assert_eq!(outcome.exit_code, Some(0));

        let has_marker = outcome
            .stdout
            .lines()
            .any(|line| line.starts_with(&format!("{marker}=")));
        assert!(
            !has_marker,
            "marker `{marker}` leaked into the clean child env"
        );
        assert!(
            outcome.stdout.lines().any(|line| line.starts_with("PATH=")),
            "PATH must be forwarded to the child"
        );
        assert!(
            outcome.stdout.lines().any(|line| line == "TERM=dumb"),
            "TERM=dumb must be set on the child"
        );
    }

    #[tokio::test]
    async fn extra_env_passes_through() {
        let mut spec = sh("echo marker=$MY_MARKER");
        spec.extra_env = vec![("MY_MARKER".to_string(), "hello-world".to_string())];
        let outcome = run(&spec).await;
        assert!(
            outcome.stdout.contains("marker=hello-world"),
            "extra_env not forwarded: {:?}",
            outcome.stdout
        );
    }

    fn checks_ctx() -> ToolCtx {
        ToolCtx::stub()
    }

    #[tokio::test]
    async fn checks_runner_green_on_zero_exit() {
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        );
        let report = runner.run(&checks_ctx()).await;
        assert!(report.passed);
        assert_eq!(report.exit_code, Some(0));
        assert!(!report.timed_out);
        assert!(report.offload_path.is_some());
    }

    #[tokio::test]
    async fn checks_runner_red_on_failing_command() {
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 3".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        );
        let report = runner.run(&checks_ctx()).await;
        assert!(!report.passed);
        assert_eq!(report.exit_code, Some(3));
    }

    #[tokio::test]
    async fn checks_runner_red_on_timeout() {
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "sleep 30".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_millis(300),
        );
        let report = runner.run(&checks_ctx()).await;
        assert!(!report.passed);
        assert!(report.timed_out);
        assert_eq!(report.exit_code, None);
    }

    #[tokio::test]
    async fn checks_runner_none_exit_on_spawn_failure() {
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/no/such/checks-binary-xyzzy".to_string(),
                args: vec![],
            },
            PathBuf::from("/"),
            Duration::from_secs(5),
        );
        let report = runner.run(&checks_ctx()).await;
        assert!(!report.passed);
        assert!(!report.timed_out);
        assert_eq!(report.exit_code, None);
    }

    #[tokio::test]
    async fn checks_runner_excerpt_is_the_tail_of_long_output() {
        let script = "echo START_MARKER; i=0; while [ $i -lt 2000 ]; do echo padding-line-number-$i; i=$((i+1)); done; echo END_MARKER";
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), script.to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(30),
        );
        let report = runner.run(&checks_ctx()).await;
        assert!(report.passed);
        assert!(
            report.excerpt.contains("END_MARKER"),
            "excerpt should keep the tail (END_MARKER)"
        );
        assert!(
            !report.excerpt.contains("START_MARKER"),
            "excerpt should have dropped the head (START_MARKER)"
        );
        assert!(report.excerpt.chars().count() <= super::CHECK_EXCERPT_CAP);
    }

    #[tokio::test]
    async fn checks_runner_offloads_full_output_readable_back() {
        let root_dir = tempdir().expect("root tempdir");
        let offload_dir = tempdir().expect("offload tempdir");
        let offload_canon = offload_dir.path().canonicalize().expect("canon offload");
        let ws = Workspace::new(root_dir.path(), Some(offload_dir.path().to_path_buf()))
            .expect("valid roots");
        let sink = DiskOffloadSink::new(offload_canon);
        let ctx = ToolCtx::new(Arc::new(ws), Arc::new(sink));

        let script = "echo START_MARKER; i=0; while [ $i -lt 2000 ]; do echo padding-line-number-$i; i=$((i+1)); done; echo END_MARKER";
        let runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), script.to_string()],
            },
            root_dir.path().to_path_buf(),
            Duration::from_secs(30),
        );
        let report = runner.run(&ctx).await;
        let offload_path = report.offload_path.expect("offloaded");

        let resolved = ctx
            .workspace()
            .resolve_read(offload_path.to_str().expect("utf8"))
            .expect("offload path resolves for reading");
        let full = std::fs::read_to_string(&resolved).expect("read offloaded output");
        // The full output has both ends; the excerpt kept only the tail.
        assert!(full.contains("START_MARKER"));
        assert!(full.contains("END_MARKER"));
    }

    #[test]
    fn check_report_round_trips_through_serde() {
        let report = CheckReport {
            passed: false,
            exit_code: Some(1),
            timed_out: false,
            excerpt: "boom".to_string(),
            offload_path: Some(PathBuf::from("/tmp/offload-0000.txt")),
            duration: Duration::from_millis(1234),
        };
        let text = serde_json::to_string(&report).expect("serialize");
        let back: CheckReport = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back.passed, report.passed);
        assert_eq!(back.exit_code, report.exit_code);
        assert_eq!(back.timed_out, report.timed_out);
        assert_eq!(back.excerpt, report.excerpt);
        assert_eq!(back.offload_path, report.offload_path);
        assert_eq!(back.duration, report.duration);
    }

    #[test]
    fn checks_runner_command_display_matches_command_display() {
        // `command_display()` is the hook the prompt layer uses to include
        // the check command in the system prompt — it MUST agree, byte for
        // byte, with the `CheckCommand`'s own Display, so a wording drift
        // between prompt and command is impossible.
        let command = CheckCommand {
            program: "cargo".to_string(),
            args: vec!["nextest".to_string(), "run".to_string()],
        };
        let runner =
            ChecksRunner::new(command.clone(), PathBuf::from("/"), Duration::from_secs(10));
        assert_eq!(runner.command_display(), command.to_string());
        assert_eq!(runner.command_display(), "cargo nextest run");
    }

    #[test]
    fn check_command_display_renders_program_and_args() {
        let command = CheckCommand {
            program: "cargo".to_string(),
            args: vec![
                "nextest".to_string(),
                "run".to_string(),
                "--workspace".to_string(),
            ],
        };
        assert_eq!(command.to_string(), "cargo nextest run --workspace");

        let bare = CheckCommand {
            program: "true".to_string(),
            args: vec![],
        };
        assert_eq!(bare.to_string(), "true");
    }

    #[test]
    fn tail_returns_whole_string_when_short() {
        assert_eq!(tail("short", 4_000), "short");
    }

    #[test]
    fn tail_keeps_only_the_last_cap_chars() {
        let text: String = (0..100).map(|_| 'x').collect();
        let got = tail(&text, 10);
        assert_eq!(got.chars().count(), 10);
        assert!(got.chars().all(|c| c == 'x'));
    }

    #[test]
    fn format_duration_is_two_decimals() {
        assert_eq!(format_duration(Duration::from_millis(1500)), "1.50s");
    }
}
