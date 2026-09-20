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

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use async_trait::async_trait;
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
pub(crate) fn tail(text: &str, cap: usize) -> String {
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

/// Leg 3 of the completion contract: **the agent claimed**, **the checks
/// verified**, and **work demonstrably happened**. This type carries the third
/// leg — whether the workspace actually moved between the start of the run and
/// the moment the claim was made.
///
/// `Unobservable` mirrors [`crate::run_record::Verification::NoChecksConfigured`]:
/// the loop could not observe the workspace, so the claim was accepted on
/// trust rather than on evidence.
///
/// Standing rule: **no code may synthesize leg-3 evidence it did not actually
/// observe.** Any constructor that did not itself call [`observe_tree`] and
/// [`classify_change`] MUST use `Unobservable { reason }` — an external-harness
/// adapter that wrote `TreeChanged` would make the invariant false at exactly
/// the surface that mirrors the production path. Evidence produced by
/// [`classify_change`] over observations returned by the run's configured
/// [`ChangeObserver`] is legitimately observed — a provider observing its own
/// durable state is the provider-side analogue of [`observe_tree`] — and the
/// rule continues to forbid constructing [`ChangeEvidence::TreeChanged`] or
/// [`ChangeEvidence::TreeUnchanged`] without a real pair of observations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeEvidence {
    /// The workspace is not in the state this run started from.
    TreeChanged,
    /// The workspace is byte-for-byte in the state this run started from.
    TreeUnchanged,
    /// The workspace could not be observed; the claim was accepted on trust.
    Unobservable {
        /// Why the observation could not be made.
        reason: String,
    },
}

impl Default for ChangeEvidence {
    // Hand-written rather than `#[derive(Default)]` + `#[default]`: that
    // attribute cannot sit on a variant with fields.
    fn default() -> Self {
        Self::Unobservable {
            reason: "not recorded (record predates change evidence)".to_string(),
        }
    }
}

/// Wall-clock bound on a single [`observe_tree`] call from the engine's finish
/// precondition. `ralph` passes its own, longer `GIT_TIMEOUT` instead.
pub const TREE_OBSERVE_TIMEOUT: Duration = Duration::from_secs(30);

/// Bound on the `reason` text of [`TreeObservation::Unobservable`], in
/// characters.
const TREE_REASON_CAP: usize = 200;

/// A single observation of a git work tree: what `git status --porcelain`
/// reported, plus where `HEAD` pointed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TreeObservation {
    /// The work tree was observed.
    Observed {
        /// `git status --porcelain` stdout, trimmed. Stored **verbatim and
        /// uncapped** — comparison correctness beats memory; only the
        /// transcript rendering is capped.
        porcelain: String,
        /// `git rev-parse HEAD`, trimmed, when it exited 0; `None` otherwise.
        head: Option<String>,
    },
    /// The work tree could not be observed (not a work tree, `git` missing,
    /// or the status call timed out).
    Unobservable {
        /// Why the observation failed, capped at 200 characters.
        reason: String,
    },
}

/// The pluggable leg-3 observation seam: whatever produces the pair of
/// [`TreeObservation`]s the engine compares via [`classify_change`].
///
/// **Naming:** this trait is `ChangeObserver`, NOT `ChangeEvidence` — the
/// latter name is already taken by the classification enum in this module
/// (see [`ChangeEvidence`]), and reusing it would collide.
///
/// The engine calls `observe` at BOTH leg-3 sites: the run-start baseline and
/// every finish-time current-tree observation. The default implementation,
/// [`GitTreeObserver`], shells out to `git status --porcelain` exactly as the
/// engine always has; a consumer whose effects are not filesystem effects —
/// an agent whose tools write into a database-backed service over HTTP, say —
/// supplies its own observer via [`crate::engine::RunConfig::with_change_observer`]
/// and gets a real leg 3 (before != after) instead of a fail-open.
///
/// **Implementations must bound their own `observe` call.** A provider may
/// make a network call, and an unbounded network call inside the engine loop
/// would hang a run exactly as an unbounded child process would. Use a
/// timeout at least as tight as [`TREE_OBSERVE_TIMEOUT`] (30s) — the reference
/// bound the filesystem observer uses. Per-provider timeout configuration is
/// deferred: for now each implementation pins its own bound.
///
/// The `Debug` + `Send` + `Sync` supertraits are REQUIRED, not stylistic:
/// [`crate::engine::RunConfig`] derives `Debug` + `Clone` and holds the
/// observer as an `Arc<dyn ChangeObserver>` — the same shape as
/// `Arc<dyn Clock>`, whose trait carries the identical supertraits.
#[async_trait]
pub trait ChangeObserver: std::fmt::Debug + Send + Sync {
    /// Observe the durable state rooted at `root`, exactly once per call.
    ///
    /// `root` is the run's workspace root; a provider whose state does not
    /// live on the filesystem will typically ignore it, but it is passed so
    /// filesystem-backed implementations stay drop-in symmetric with
    /// [`observe_tree`].
    async fn observe(&self, root: &Path) -> TreeObservation;

    /// The telemetry label recorded on the `run_start` event's
    /// `config.change_observer` key — `"git"` for [`GitTreeObserver`],
    /// `"custom"` by default for any configured provider.
    fn label(&self) -> &'static str {
        "custom"
    }
}

/// The default [`ChangeObserver`] for every run that does not call
/// [`crate::engine::RunConfig::with_change_observer`]: filesystem
/// `git status --porcelain` semantics, delegating to [`observe_tree`] under
/// [`TREE_OBSERVE_TIMEOUT`] — the SAME function and the SAME bound the engine
/// used inline before this seam existed, so the default path is behaviorally
/// identical by construction.
#[derive(Debug, Clone, Copy)]
pub struct GitTreeObserver;

#[async_trait]
impl ChangeObserver for GitTreeObserver {
    async fn observe(&self, root: &Path) -> TreeObservation {
        observe_tree(root, TREE_OBSERVE_TIMEOUT).await
    }

    fn label(&self) -> &'static str {
        "git"
    }
}

/// Build the [`TreeObservation::Unobservable`] reason for a failed
/// `git status --porcelain`, truncated to [`TREE_REASON_CAP`] characters.
///
/// Pure and separated from [`observe_tree`] deliberately: driving the
/// timed-out branch through a real `git` invocation is a race —
/// `tokio::time::timeout` polls the inner future before checking the
/// deadline, so even a zero duration returns the completed child whenever
/// the process happens to be ready on the first poll. Testing the formatting
/// against a synthetic [`ExecOutcome`] covers both branches deterministically.
fn unobservable_reason(status: &ExecOutcome, timeout: Duration) -> String {
    let reason = if status.timed_out {
        format!(
            "git status --porcelain timed out after {}s",
            timeout.as_secs()
        )
    } else {
        format!(
            "git status --porcelain exited {:?}: {}",
            status.exit_code,
            status.stderr.trim()
        )
    };
    reason.chars().take(TREE_REASON_CAP).collect()
}

/// Observe the tree rooted at `root`, bounded by `timeout`.
///
/// The FIRST action is always the root `git status --porcelain`, which picks
/// the branch:
///
/// - **exit 0** — the root is a work tree. The result is the single-repo
///   [`TreeObservation::Observed`]: trimmed porcelain, plus
///   `git rev-parse HEAD`. `head` is `None` for **both** an unborn `HEAD`
///   and a failed or timed-out `git rev-parse HEAD` — deliberately
///   indistinguishable, because the `None` ↔ `Some` transition reads as
///   [`ChangeEvidence::TreeChanged`] in [`classify_change`] and therefore
///   fails **open** rather than producing a spurious rejection. Children are
///   NEVER scanned (this is also the behavior when a PARENT of `root` is a
///   repo and `root`'s own status exits 0).
/// - **non-zero AND a `.git` entry at `root`** (file or directory) — the
///   root is repo-shaped but unreadable (broken `gitdir:` file, `git`
///   missing, timeout): today's [`TreeObservation::Unobservable`] with
///   [`unobservable_reason`]. The child scan NEVER runs.
/// - **non-zero AND no `.git` entry at `root`** — a multi-repo workspace
///   root. The immediate (depth-1) child directories that each carry a
///   `.git` entry are observed through [`observe_single_repo`] with the SAME
///   `timeout`, sorted by file name, and combined by
///   [`combine_child_observations`]: any unobservable child makes the
///   combination [`TreeObservation::Unobservable`], otherwise the porcelain
///   is per-child `<name>/`-prefixed lines and the head is per-child
///   `<name> <head|none>` lines. When at least one child qualifies, exactly
///   one stderr warning names the discarded root reason and the child list,
///   so the branch taken is observable even when the combined result looks
///   clean.
/// - **non-zero and NO qualifying child** — byte-identical fallback to
///   [`unobservable_reason`]: genuinely nothing was observable, the same
///   fail-open as ever.
///
/// `timeout` is a parameter rather than a read of [`TREE_OBSERVE_TIMEOUT`]
/// so `ralph` can keep its own, longer git bound while sharing this
/// primitive; it applies **per repository**, so a multi-repo workspace costs
/// at most one timeout per child, never per workspace.
pub async fn observe_tree(root: &Path, timeout: Duration) -> TreeObservation {
    let status = run(&ExecSpec {
        program: "git".to_string(),
        args: vec!["status".to_string(), "--porcelain".to_string()],
        cwd: root.to_path_buf(),
        timeout,
        extra_env: Vec::new(),
    })
    .await;

    if status.exit_code == Some(0) {
        return observe_head(root, timeout, status).await;
    }

    // Repo-shaped root (has a `.git` entry, so worktree and submodule
    // checkouts count too) whose status failed: this is a single-repo
    // failure, not a workspace to decompose — never scan children.
    if has_git_entry(root) {
        return TreeObservation::Unobservable {
            reason: unobservable_reason(&status, timeout),
        };
    }

    // Non-repo root: a multi-repo workspace. Observe each depth-1 child
    // that is itself a repository.
    let names = discover_child_repos(root);
    if names.is_empty() {
        return TreeObservation::Unobservable {
            reason: unobservable_reason(&status, timeout),
        };
    }

    let mut children = Vec::with_capacity(names.len());
    for name in &names {
        let observation = observe_single_repo(&root.join(name), timeout).await;
        children.push((name.clone(), observation));
    }
    eprintln!(
        "{}",
        child_scan_warning(&unobservable_reason(&status, timeout), &names)
    );
    combine_child_observations(&children)
}

/// Whether `dir` directly contains a `.git` entry — file or directory, the
/// same pure check that qualifies a child (worktrees and submodules carry a
/// `gitdir:` FILE).
fn has_git_entry(dir: &Path) -> bool {
    dir.join(".git").exists()
}

/// The stderr warning printed when a non-repo root's failed
/// `git status --porcelain` is replaced by a depth-1 child scan: the branch
/// taken, the discarded root reason, and the discovered child list stay
/// observable even when the combined result is a clean-looking
/// [`TreeObservation::Observed`]. Pure so the exact wording is unit-pinned.
fn child_scan_warning(reason: &str, names: &[String]) -> String {
    format!(
        "warning: root git status failed ({reason}); observing {} child repos: {}",
        names.len(),
        names.join(", ")
    )
}

/// The immediate (depth-1) child directories of `root` that are git
/// repositories or worktrees, sorted by file name in byte order.
///
/// A child qualifies iff `Path::is_dir()` is true (symlinks to directories
/// count) AND it directly contains an entry named `.git` — file or
/// directory, so `git worktree add` checkouts and submodule checkouts
/// qualify alongside plain clones. No recursion: grandchildren are never
/// considered.
fn discover_child_repos(root: &Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() && has_git_entry(&path) {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    names.sort();
    names
}

/// Observe ONE child repository: the status + rev-parse body shared with a
/// successful [`observe_tree`] root observation, factored out so
/// [`observe_tree`]'s multi-repo branch can invoke it per child. A child
/// whose status fails is simply [`TreeObservation::Unobservable`] — it is
/// never re-scanned one level deeper.
async fn observe_single_repo(dir: &Path, timeout: Duration) -> TreeObservation {
    let status = run(&ExecSpec {
        program: "git".to_string(),
        args: vec!["status".to_string(), "--porcelain".to_string()],
        cwd: dir.to_path_buf(),
        timeout,
        extra_env: Vec::new(),
    })
    .await;

    if status.exit_code != Some(0) {
        return TreeObservation::Unobservable {
            reason: unobservable_reason(&status, timeout),
        };
    }
    observe_head(dir, timeout, status).await
}

/// The rev-parse half of a successful single-repo observation: `head` is
/// `None` for both an unborn `HEAD` and a failed or timed-out
/// `git rev-parse HEAD`, and the porcelain is the status stdout trimmed.
async fn observe_head(dir: &Path, timeout: Duration, status: ExecOutcome) -> TreeObservation {
    let head_outcome = run(&ExecSpec {
        program: "git".to_string(),
        args: vec!["rev-parse".to_string(), "HEAD".to_string()],
        cwd: dir.to_path_buf(),
        timeout,
        extra_env: Vec::new(),
    })
    .await;
    let head = if head_outcome.exit_code == Some(0) {
        Some(head_outcome.stdout.trim().to_string())
    } else {
        None
    };

    TreeObservation::Observed {
        porcelain: status.stdout.trim().to_string(),
        head,
    }
}

/// Combine the sorted `(name, observation)` pairs of a multi-repo workspace
/// into ONE [`TreeObservation`], the shape [`classify_change`] already
/// understands.
///
/// - **any child `Unobservable`** — the combination is
///   [`TreeObservation::Unobservable`] with the FIRST failing child in
///   sorted order, reason `"<name>: <child reason>"`, truncated to
///   [`TREE_REASON_CAP`] characters **after** prefixing. An unreadable repo
///   must never let an uncertain workspace read clean.
/// - **all children `Observed`** — the porcelain is each child's porcelain
///   lines prefixed with `"<name>/"`, joined with `'\n'` and trimmed (an
///   all-clean set yields the empty string, matching a clean single repo);
///   `head` is `Some` with one `"<name> <head|none>"` line per child. Heads
///   MUST be encoded — `git status --porcelain` does not show commits, so
///   only the head line catches committed-only work in a child.
fn combine_child_observations(children: &[(String, TreeObservation)]) -> TreeObservation {
    if let Some((name, TreeObservation::Unobservable { reason })) = children
        .iter()
        .find(|(_, o)| matches!(o, TreeObservation::Unobservable { .. }))
    {
        debug_assert!(
            children
                .iter()
                .take_while(|(child_name, _)| child_name != name)
                .all(|(_, o)| matches!(o, TreeObservation::Observed { .. })),
            "the named child must be the FIRST Unobservable in sorted order"
        );
        let prefixed = format!("{name}: {reason}");
        return TreeObservation::Unobservable {
            reason: prefixed.chars().take(TREE_REASON_CAP).collect(),
        };
    }

    debug_assert!(
        children
            .iter()
            .all(|(_, o)| matches!(o, TreeObservation::Observed { .. })),
        "the Observed branch must only run when no child was Unobservable"
    );
    let mut porcelain_lines = Vec::new();
    let mut head_lines = Vec::new();
    for (name, observation) in children {
        let TreeObservation::Observed { porcelain, head } = observation else {
            continue;
        };
        // A clean child contributes NO porcelain lines — an empty porcelain
        // must not degenerate into a bare `"<name>/"` line.
        if !porcelain.is_empty() {
            for line in porcelain.split('\n') {
                porcelain_lines.push(format!("{name}/{line}"));
            }
        }
        head_lines.push(format!("{name} {}", head.as_deref().unwrap_or("none")));
    }
    TreeObservation::Observed {
        porcelain: porcelain_lines.join("\n").trim().to_string(),
        head: Some(head_lines.join("\n")),
    }
}

/// Classify a pair of observations into [`ChangeEvidence`]: *the workspace is
/// not in the state this run started from*.
///
/// The comparison is **baseline-relative**, never a bare `current`-dirtiness
/// test. A pre-existing dirty entry (e.g. the untracked
/// `<run-id>-attachments/` directory the dispatch worker stages inside the
/// clone root before launching the agent) appears in **both** observations and
/// cancels; any agent edit changes the text.
///
/// Either side being `Unobservable` fails **open** — the loop could not
/// establish what the run started from or ended at, so the claim is accepted
/// on trust.
///
/// Two accepted holes: a workspace mutated by something other than the agent
/// between the two observations reads as changed, and an agent that edits a
/// file and reverts it byte-for-byte reads as unchanged.
///
/// This is **not** [`crate::engine::RunStats::tree_dirty`], which a single
/// read-only `bash` call latches and which any agent could therefore defeat.
#[must_use]
pub fn classify_change(baseline: &TreeObservation, current: &TreeObservation) -> ChangeEvidence {
    match (baseline, current) {
        // Fail open, preferring `current`'s reason when both are unobservable.
        (_, TreeObservation::Unobservable { reason })
        | (TreeObservation::Unobservable { reason }, _) => ChangeEvidence::Unobservable {
            reason: reason.clone(),
        },
        (
            TreeObservation::Observed {
                porcelain: base_porcelain,
                head: base_head,
            },
            TreeObservation::Observed {
                porcelain: cur_porcelain,
                head: cur_head,
            },
        ) => {
            if base_porcelain == cur_porcelain && base_head == cur_head {
                ChangeEvidence::TreeUnchanged
            } else {
                ChangeEvidence::TreeChanged
            }
        }
    }
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

/// Build a [`ChecksRunner`] from a shell gate-command string, or `None` if the
/// command is empty or whitespace-only.
///
/// A whitespace-only `gate_command` is treated as empty — closing the
/// `/bin/sh -c ' '` exits-0 rubber-stamp false-Done vector.
///
/// `/bin/sh -c` is used (NOT direct exec) so the gate string can contain
/// shell operators like `&&` and pipes. Shared by talos production dispatch
/// (`crates/talos/src/main.rs`'s `build_checks_runner`) and the tier-2
/// `agent_gate_command` path (`crate::mined_eval`), so both surfaces build
/// the identical `/bin/sh -c <cmd>` shape from one place.
#[must_use]
pub fn shell_checks_runner(
    gate_command: &str,
    workspace_root: PathBuf,
    timeout: Duration,
) -> Option<ChecksRunner> {
    if gate_command.trim().is_empty() {
        return None;
    }
    Some(ChecksRunner::new(
        CheckCommand {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), gate_command.to_string()],
        },
        workspace_root,
        timeout,
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        ChangeEvidence, ChangeObserver, CheckCommand, CheckReport, ChecksRunner, ExecOutcome,
        ExecSpec, GitTreeObserver, TREE_OBSERVE_TIMEOUT, TreeObservation, classify_change,
        format_duration, observe_tree, run, shell_checks_runner, tail,
    };
    use crate::tool::ToolCtx;
    use crate::workspace::{DiskOffloadSink, Workspace};
    use std::path::{Path, PathBuf};
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

    // ---- shell_checks_runner -----------------------------------------------

    #[test]
    fn shell_checks_runner_empty_command_is_none() {
        assert!(shell_checks_runner("", PathBuf::from("/"), Duration::from_secs(1)).is_none());
    }

    #[test]
    fn shell_checks_runner_whitespace_only_command_is_none() {
        assert!(shell_checks_runner("   ", PathBuf::from("/"), Duration::from_secs(1)).is_none());
    }

    #[test]
    fn shell_checks_runner_builds_sh_c_shape() {
        let runner = shell_checks_runner("a && b", PathBuf::from("/"), Duration::from_secs(1))
            .expect("non-empty command builds a runner");
        assert_eq!(runner.command().program, "/bin/sh");
        assert_eq!(
            runner.command().args,
            vec!["-c".to_string(), "a && b".to_string()]
        );
        assert_eq!(runner.command_display(), "/bin/sh -c a && b");
    }

    /// Run `git` with `args` in `dir`, asserting it succeeded.
    fn git(dir: &std::path::Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            // Git exports these into every hook process, so a test that shells
            // out to `git` inherits them when the suite runs from a pre-commit
            // hook and not when it runs directly. That difference is invisible
            // until it bites: `GIT_INDEX_FILE` arrives as the RELATIVE path
            // `.git/index`, `git worktree add` then resolves it inside the new
            // worktree — whose `.git` is a FILE, not a directory — and the
            // command dies with `index file open failed: Not a directory`.
            // The suite passed the dispatch host's `gate_command` (run
            // directly) and failed the identical command under `lefthook`.
            // Clear the ambient repo pointers so these fixtures always describe
            // their own temp repos.
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_PREFIX")
            .env_remove("GIT_COMMON_DIR")
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@example.com")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@example.com")
            .output()
            .expect("git runs");
        assert!(out.status.success(), "git {args:?} failed: {out:?}");
    }

    #[tokio::test]
    async fn observe_tree_non_work_tree_is_unobservable() {
        let dir = tempdir().expect("tempdir");
        let obs = observe_tree(dir.path(), Duration::from_secs(30)).await;
        match obs {
            TreeObservation::Unobservable { reason } => {
                assert!(reason.contains("git status"), "reason was {reason}");
                assert!(reason.contains("exited"), "reason was {reason}");
            }
            other @ TreeObservation::Observed { .. } => {
                panic!("expected Unobservable, got {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn observe_tree_untracked_file_is_observed_dirty() {
        let dir = tempdir().expect("tempdir");
        git(dir.path(), &["init", "-q"]);
        std::fs::write(dir.path().join("a.txt"), "hi").expect("write");
        match observe_tree(dir.path(), Duration::from_secs(30)).await {
            TreeObservation::Observed { porcelain, .. } => {
                assert!(!porcelain.is_empty(), "porcelain was empty");
            }
            other @ TreeObservation::Unobservable { .. } => {
                panic!("expected Observed, got {other:?}")
            }
        }
    }

    #[tokio::test]
    async fn observe_tree_clean_committed_tree_has_head_and_empty_porcelain() {
        let dir = tempdir().expect("tempdir");
        git(dir.path(), &["init", "-q"]);
        std::fs::write(dir.path().join("a.txt"), "hi").expect("write");
        git(dir.path(), &["add", "."]);
        git(dir.path(), &["commit", "-qm", "init"]);
        match observe_tree(dir.path(), Duration::from_secs(30)).await {
            TreeObservation::Observed { porcelain, head } => {
                assert!(porcelain.is_empty(), "porcelain was {porcelain:?}");
                assert!(head.is_some(), "head was None");
            }
            other @ TreeObservation::Unobservable { .. } => {
                panic!("expected Observed, got {other:?}")
            }
        }
    }

    /// Drives the timed-out reason branch against a synthetic
    /// [`ExecOutcome`] rather than a real `git` invocation. The previous
    /// version of this test passed `Duration::from_millis(0)` to
    /// `observe_tree` and asserted the timeout fired; that is a race —
    /// `tokio::time::timeout` polls the inner future before checking the
    /// deadline, so a `git status` that is ready on the first poll returns
    /// `Observed` and the test fails intermittently. It failed a pre-commit
    /// run on 2026-09-18 having passed the identical suite moments earlier.
    #[test]
    fn unobservable_reason_timed_out_branch_says_timed_out() {
        let reason = super::unobservable_reason(
            &ExecOutcome {
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                duration: Duration::from_secs(30),
                timed_out: true,
            },
            Duration::from_secs(30),
        );
        assert!(reason.contains("timed out"), "reason was {reason}");
        assert!(reason.contains("git status"), "reason was {reason}");
        assert!(reason.contains("30s"), "reason was {reason}");
    }

    fn observed(porcelain: &str, head: Option<&str>) -> TreeObservation {
        TreeObservation::Observed {
            porcelain: porcelain.to_string(),
            head: head.map(ToString::to_string),
        }
    }

    fn unobservable(reason: &str) -> TreeObservation {
        TreeObservation::Unobservable {
            reason: reason.to_string(),
        }
    }

    #[test]
    fn classify_change_current_unobservable_fails_open_with_current_reason() {
        let evidence = classify_change(&observed("", Some("a")), &unobservable("cur"));
        assert_eq!(
            evidence,
            ChangeEvidence::Unobservable {
                reason: "cur".to_string()
            }
        );
    }

    #[test]
    fn classify_change_baseline_unobservable_fails_open_not_changed() {
        let evidence = classify_change(&unobservable("base"), &observed("?? x", Some("a")));
        assert_eq!(
            evidence,
            ChangeEvidence::Unobservable {
                reason: "base".to_string()
            }
        );
    }

    #[test]
    fn classify_change_both_unobservable_prefers_current_reason() {
        let evidence = classify_change(&unobservable("base"), &unobservable("cur"));
        assert_eq!(
            evidence,
            ChangeEvidence::Unobservable {
                reason: "cur".to_string()
            }
        );
    }

    #[test]
    fn classify_change_identical_dirty_observations_are_unchanged() {
        let evidence = classify_change(
            &observed("?? run-1-attachments/", Some("a")),
            &observed("?? run-1-attachments/", Some("a")),
        );
        assert_eq!(evidence, ChangeEvidence::TreeUnchanged);
    }

    #[test]
    fn classify_change_differing_porcelain_is_changed() {
        let evidence = classify_change(&observed("", Some("a")), &observed("?? x", Some("a")));
        assert_eq!(evidence, ChangeEvidence::TreeChanged);
    }

    #[test]
    fn classify_change_differing_head_is_changed() {
        assert_eq!(
            classify_change(&observed("", Some("a")), &observed("", Some("b"))),
            ChangeEvidence::TreeChanged
        );
        assert_eq!(
            classify_change(&observed("", None), &observed("", Some("b"))),
            ChangeEvidence::TreeChanged
        );
    }

    #[test]
    fn change_evidence_default_is_unobservable() {
        match ChangeEvidence::default() {
            ChangeEvidence::Unobservable { reason } => {
                assert!(reason.contains("predates"), "reason was {reason}");
            }
            other => panic!("expected Unobservable, got {other:?}"),
        }
    }

    /// The default observer must be a byte-for-byte DELEGATE, not a
    /// re-implementation: on a real git repo after a real edit, both the
    /// porcelain and the head must match [`observe_tree`]'s output exactly.
    #[tokio::test]
    async fn git_tree_observer_delegates_to_observe_tree_on_a_real_git_repo() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let out = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(root)
            .output()
            .expect("git init runs");
        assert!(out.status.success(), "git init failed: {out:?}");
        std::fs::write(root.join("seed.txt"), "v1\n").expect("write");
        std::fs::write(root.join("edited.txt"), "edited\n").expect("write");

        let via_observer = GitTreeObserver.observe(root).await;
        let direct = observe_tree(root, TREE_OBSERVE_TIMEOUT).await;
        assert_eq!(via_observer, direct);
        match &via_observer {
            TreeObservation::Observed { porcelain, head: _ } => {
                assert!(porcelain.contains("seed.txt"), "porcelain was {porcelain}");
                assert!(
                    porcelain.contains("edited.txt"),
                    "porcelain was {porcelain}"
                );
            }
            TreeObservation::Unobservable { reason: _ } => {
                panic!("expected Observed on a git repo")
            }
        }
    }

    /// Same delegation invariant on the fail-open side: a bare non-git
    /// directory yields `Unobservable` with the SAME reason string both ways.
    #[tokio::test]
    async fn git_tree_observer_delegates_to_observe_tree_on_a_non_git_dir() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();

        let via_observer = GitTreeObserver.observe(root).await;
        let direct = observe_tree(root, TREE_OBSERVE_TIMEOUT).await;
        assert_eq!(via_observer, direct);
        match &via_observer {
            TreeObservation::Unobservable { reason } => {
                assert!(
                    reason.contains("git status --porcelain"),
                    "reason was {reason}"
                );
            }
            TreeObservation::Observed { .. } => {
                panic!("expected Unobservable on a non-git dir")
            }
        }
    }

    #[test]
    fn git_tree_observer_labels_itself_git_and_trait_default_is_custom() {
        use async_trait::async_trait;

        struct Unlabeled;
        impl std::fmt::Debug for Unlabeled {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Unlabeled")
            }
        }
        #[async_trait]
        impl ChangeObserver for Unlabeled {
            async fn observe(&self, _root: &Path) -> TreeObservation {
                TreeObservation::Unobservable {
                    reason: "n/a".to_string(),
                }
            }
        }

        assert_eq!(GitTreeObserver.label(), "git");
        assert_eq!(Unlabeled.label(), "custom");
    }

    // ---- multi-repo workspace observation (combined child repos) ----------

    #[test]
    fn combine_children_prefixes_porcelain_lines_and_joins_heads() {
        let children = vec![
            ("cleanr".to_string(), observed("?? c.txt", Some("cafe"))),
            ("photoqueue".to_string(), observed(" M src/x.py", None)),
        ];
        let combined = super::combine_child_observations(&children);
        assert_eq!(
            combined,
            observed(
                "cleanr/?? c.txt\nphotoqueue/ M src/x.py",
                Some("cleanr cafe\nphotoqueue none"),
            ),
            "porcelain lines carry a `{{name}}/` prefix (the leading space of a \
             modified entry survives) and heads are encoded `{{name}} <head|none>` \
             because `git status --porcelain` does not show commits"
        );
    }

    #[test]
    fn combine_children_all_clean_yields_empty_porcelain_but_joined_head() {
        let children = vec![
            ("a".to_string(), observed("", Some("aa"))),
            ("b".to_string(), observed("", Some("bb"))),
        ];
        let combined = super::combine_child_observations(&children);
        assert_eq!(
            combined,
            observed("", Some("a aa\nb bb")),
            "EMPTY porcelain but NON-EMPTY joined head — the joined head is the \
             only signal a committed-only change can ride on"
        );
    }

    #[test]
    fn combine_children_first_unobservable_is_prefixed_then_capped() {
        let long_reason = "r".repeat(300);
        let children = vec![
            ("observed".to_string(), observed("?? x", Some("h"))),
            ("longchild".to_string(), unobservable(&long_reason)),
        ];
        let combined = super::combine_child_observations(&children);
        match combined {
            TreeObservation::Unobservable { reason } => {
                assert_eq!(reason.chars().count(), super::TREE_REASON_CAP);
                assert!(
                    reason.starts_with("longchild: "),
                    "the cap applies to the FINAL prefixed string; got {reason:?}"
                );
            }
            other @ TreeObservation::Observed { .. } => {
                panic!("expected Unobservable, got {other:?}")
            }
        }
    }

    #[test]
    fn combine_children_mixed_observability_is_unobservable_in_sorted_order() {
        // `a` sorts first and IS observed-and-changed; `b` could not be
        // observed. The first FAILING child in sorted order names the reason —
        // an unreadable repo must not let an uncertain workspace read clean.
        let children = vec![
            ("a".to_string(), observed("?? new", Some("a"))),
            ("b".to_string(), unobservable("r")),
        ];
        let combined = super::combine_child_observations(&children);
        assert_eq!(combined, unobservable("b: r"));
        assert_eq!(
            classify_change(&combined, &combined),
            ChangeEvidence::Unobservable {
                reason: "b: r".to_string()
            },
            "an uncertain workspace must never classify TreeChanged or TreeUnchanged"
        );
    }

    #[test]
    fn combined_children_cancel_pre_existing_dirt_baseline_relatively() {
        // Child `b` carries the SAME untracked entry at baseline and at exit
        // (pre-existing dirt cancels); the two runs differ only in child `a`.
        let baseline_children = vec![
            ("a".to_string(), observed("", Some("a"))),
            ("b".to_string(), observed("?? pre-existing", Some("b"))),
        ];
        let baseline = super::combine_child_observations(&baseline_children);

        // (i) a run that adds a NEW untracked file only in `a` changed.
        let changed_children = vec![
            ("a".to_string(), observed("?? new", Some("a"))),
            ("b".to_string(), observed("?? pre-existing", Some("b"))),
        ];
        let changed = super::combine_child_observations(&changed_children);
        assert_eq!(
            classify_change(&baseline, &changed),
            ChangeEvidence::TreeChanged
        );

        // (ii) a run that touches nothing in any child did not.
        assert_eq!(
            classify_change(&baseline, &baseline),
            ChangeEvidence::TreeUnchanged
        );
    }

    /// Pins the exit-128 reason branch deterministically, the same way
    /// [`unobservable_reason_timed_out_branch_says_timed_out`] pins the
    /// timeout branch. The stderr tail is git-version-dependent, which is
    /// why the live-git tests only assert contains-style.
    #[test]
    fn unobservable_reason_exit_128_says_exited_with_trimmed_stderr() {
        let reason = super::unobservable_reason(
            &ExecOutcome {
                exit_code: Some(128),
                stdout: String::new(),
                stderr: "fatal: not a git repository (or any of the parent directories): .git\n"
                    .to_string(),
                duration: Duration::from_secs(30),
                timed_out: false,
            },
            Duration::from_secs(30),
        );
        assert_eq!(
            reason,
            "git status --porcelain exited Some(128): fatal: not a git \
             repository (or any of the parent directories): .git"
        );
    }

    /// Live-git fallback on a bare non-repo directory: contains-style, the
    /// same assertion shape as
    /// [`observe_tree_non_work_tree_is_unobservable`] (the exact stderr tail
    /// is git-version-dependent).
    #[tokio::test]
    async fn observe_tree_bare_non_repo_root_reason_names_the_failed_status() {
        let dir = tempdir().expect("tempdir");
        match observe_tree(dir.path(), Duration::from_secs(30)).await {
            TreeObservation::Unobservable { reason } => {
                assert!(
                    reason.contains("git status --porcelain exited"),
                    "reason was {reason}"
                );
                assert!(
                    reason.contains("fatal: not a git repository"),
                    "reason was {reason}"
                );
            }
            other @ TreeObservation::Observed { .. } => {
                panic!("expected Unobservable, got {other:?}")
            }
        }
    }

    /// A repo-shaped root (its OWN status exits 0) keeps today's
    /// single-repo observation even when it CONTAINS a `.git`-bearing child
    /// repo: the porcelain is exactly the root's own `git status --porcelain`
    /// output (no `<child>/`-prefixed child lines) and the head is the
    /// root's own.
    #[tokio::test]
    async fn observe_tree_repo_root_with_child_repo_observes_only_the_root() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        git(root, &["init", "-q"]);
        std::fs::write(root.join("seed.txt"), "seed\n").expect("write");
        git(root, &["add", "."]);
        git(root, &["commit", "-qm", "seed"]);
        let child = root.join("child");
        std::fs::create_dir(&child).expect("child dir");
        git(&child, &["init", "-q"]);
        std::fs::write(child.join("u.txt"), "untracked\n").expect("write");

        // The root's own status, straight from git: the untracked child
        // directory shows up as `?? child/` — a ROOT line, not a prefixed
        // child line.
        let own_status = std::process::Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(root)
            .output()
            .expect("git status runs");
        assert!(own_status.status.success());
        let own_porcelain = String::from_utf8(own_status.stdout)
            .expect("utf8")
            .trim()
            .to_string();
        let own_head = {
            let out = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(root)
                .output()
                .expect("rev-parse runs");
            assert!(out.status.success());
            String::from_utf8(out.stdout)
                .expect("utf8 sha")
                .trim()
                .to_string()
        };

        match observe_tree(root, Duration::from_secs(30)).await {
            TreeObservation::Observed { porcelain, head } => {
                assert_eq!(
                    porcelain, own_porcelain,
                    "porcelain must be the root's OWN status: {porcelain:?}"
                );
                assert!(
                    !porcelain.contains("child/?? u.txt"),
                    "the child's own lines are never merged into a repo-shaped \
                     root's observation: {porcelain:?}"
                );
                assert_eq!(head, Some(own_head), "head is the root's own commit");
            }
            other @ TreeObservation::Unobservable { .. } => {
                panic!("expected Observed, got {other:?}")
            }
        }
    }

    /// A repo-shaped root whose git is BROKEN (here: a `.git` file pointing
    /// at a nonexistent gitdir, exit 128) returns today's `Unobservable`
    /// with today's reason — the child scan NEVER runs for a repo-shaped
    /// root, even when a valid child repo sits right there.
    #[tokio::test]
    async fn observe_tree_repo_shaped_root_with_broken_git_never_scans_children() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(root.join(".git"), "gitdir: /nonexistent\n").expect("write .git file");
        let child = root.join("child");
        std::fs::create_dir(&child).expect("child dir");
        git(&child, &["init", "-q"]);
        std::fs::write(child.join("u.txt"), "untracked\n").expect("write");

        match observe_tree(root, Duration::from_secs(30)).await {
            TreeObservation::Unobservable { reason } => {
                assert!(
                    reason.contains("git status --porcelain exited Some(128)"),
                    "reason was {reason}"
                );
                assert!(
                    !reason.starts_with("child:"),
                    "the child scan must not run for a repo-shaped root: {reason:?}"
                );
            }
            other @ TreeObservation::Observed { .. } => {
                panic!("expected Unobservable, got {other:?}")
            }
        }
    }

    /// A qualifying child that is itself broken is reported with a
    /// `<name>:`-prefixed reason and is NEVER re-scanned one level deeper:
    /// `broken/inner/` (a real repo) must not rescue `broken/`'s
    /// unobservable status.
    #[tokio::test]
    async fn observe_tree_broken_child_is_prefixed_and_never_re_scanned_deeper() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let broken = root.join("broken");
        std::fs::create_dir(&broken).expect("broken dir");
        std::fs::write(broken.join(".git"), "gitdir: /nonexistent\n").expect("write .git file");
        let inner = broken.join("inner");
        std::fs::create_dir(&inner).expect("inner dir");
        git(&inner, &["init", "-q"]);

        match observe_tree(root, Duration::from_secs(30)).await {
            TreeObservation::Unobservable { reason } => {
                assert!(
                    reason.starts_with("broken: git status --porcelain exited"),
                    "reason was {reason}"
                );
            }
            other @ TreeObservation::Observed { .. } => {
                panic!("expected Unobservable, got {other:?}")
            }
        }
    }

    /// The depth-1 scan does not recurse: `outer/` qualifies (its own
    /// `.git`), `outer/inner/` (also a real repo) is never reached, so the
    /// combined porcelain has exactly one line and the head is exactly
    /// `outer <outer's sha>`. Sorting is visible in the same output.
    #[tokio::test]
    async fn observe_tree_depth_one_scan_does_not_recurse_into_grandchildren() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let outer = root.join("outer");
        std::fs::create_dir(&outer).expect("outer dir");
        git(&outer, &["init", "-q"]);
        std::fs::write(outer.join("seed.txt"), "seed\n").expect("write");
        git(&outer, &["add", "."]);
        git(&outer, &["commit", "-qm", "seed"]);
        std::fs::write(outer.join("u.txt"), "untracked\n").expect("write");
        let inner = outer.join("inner");
        std::fs::create_dir(&inner).expect("inner dir");
        git(&inner, &["init", "-q"]);
        std::fs::write(inner.join("v.txt"), "untracked\n").expect("write");
        // `inner/` must not appear in `outer`'s own porcelain (git would list
        // the untracked directory), or the exact one-line pin below would
        // carry a second line mentioning `inner`. Ignore it from `outer`.
        std::fs::write(outer.join(".gitignore"), "inner/\n").expect("write gitignore");
        git(&outer, &["add", ".gitignore"]);
        git(&outer, &["commit", "-qm", "ignore inner"]);

        let outer_head = {
            let out = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(&outer)
                .output()
                .expect("rev-parse runs");
            assert!(out.status.success());
            String::from_utf8(out.stdout)
                .expect("utf8 sha")
                .trim()
                .to_string()
        };

        match observe_tree(root, Duration::from_secs(30)).await {
            TreeObservation::Observed { porcelain, head } => {
                assert_eq!(
                    porcelain,
                    format!("outer/?? u.txt"),
                    "exactly one depth-1 line; `inner` never scanned"
                );
                assert_eq!(
                    head,
                    Some(format!("outer {outer_head}")),
                    "head lines are `<name> <sha>` per observed child"
                );
            }
            other @ TreeObservation::Unobservable { .. } => {
                panic!("expected Observed, got {other:?}")
            }
        }
    }

    /// Committed-only work in one child repo of a multi-repo workspace: BOTH
    /// porcelains stay empty (each child is clean), so the moved `a <sha>`
    /// HEAD line alone carries the signal — exactly what `head` catches in a
    /// single repo today.
    #[tokio::test]
    async fn observe_tree_commit_only_change_in_a_child_repo_reads_as_changed() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        for name in ["a", "b"] {
            let child = root.join(name);
            std::fs::create_dir(&child).expect("child dir");
            git(&child, &["init", "-q"]);
            std::fs::write(child.join("seed.txt"), "seed\n").expect("write");
            git(&child, &["add", "."]);
            git(&child, &["commit", "-qm", "seed"]);
        }

        let baseline = observe_tree(root, Duration::from_secs(30)).await;
        match &baseline {
            TreeObservation::Observed { porcelain, head } => {
                assert!(porcelain.is_empty(), "both children clean: {porcelain:?}");
                assert!(head.is_some(), "heads must be encoded even when clean");
            }
            other @ TreeObservation::Unobservable { .. } => {
                panic!("expected Observed, got {other:?}")
            }
        }

        // Commit-ONLY change in `a`: edit the tracked file, `git add .`,
        // commit. `b` untouched. A throwaway tempdir fixture, never a git
        // command against the repository working tree.
        let a_head_before = match &baseline {
            TreeObservation::Observed { head, .. } => head.clone().expect("baseline head"),
            TreeObservation::Unobservable { .. } => unreachable!("baseline was observed"),
        };
        let child_a = root.join("a");
        std::fs::write(child_a.join("seed.txt"), "seed\nmore\n").expect("write");
        git(&child_a, &["add", "."]);
        git(&child_a, &["commit", "-qm", "work"]);

        let current = observe_tree(root, Duration::from_secs(30)).await;
        match (&baseline, &current) {
            (
                TreeObservation::Observed { porcelain: bp, .. },
                TreeObservation::Observed {
                    porcelain: cp,
                    head,
                    ..
                },
            ) => {
                assert_eq!(bp, cp, "both porcelains stay empty — heads carry it");
                assert_ne!(
                    head,
                    &Some(a_head_before),
                    "the `a <sha>` head line must move"
                );
            }
            _ => panic!("expected both sides Observed"),
        }
        assert_eq!(
            classify_change(&baseline, &current),
            ChangeEvidence::TreeChanged,
            "the joined-head encoding catches committed work in multi-repo workspaces"
        );
    }

    /// The two discovery branches a cheaper implementation most easily
    /// misses: (i) a SYMLINK to a repo directory outside the root qualifies
    /// (`Path::is_dir` follows symlinks and the target is never scanned as
    /// a root child), and (ii) a `git worktree add` checkout qualifies via
    /// its `.git` FILE while its source repo at depth ≥ 2 does not (depth-1
    /// only). Combined porcelain and heads are pinned in sorted order.
    #[tokio::test]
    async fn observe_tree_discovers_symlink_children_and_worktree_children() {
        let outside = tempdir().expect("outside tempdir");
        let target = outside.path().join("real-repo-target");
        std::fs::create_dir(&target).expect("target dir");
        git(&target, &["init", "-q"]);
        std::fs::write(target.join("u.txt"), "untracked\n").expect("write");

        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        std::os::unix::fs::symlink(&target, root.join("link")).expect("symlink child");

        // Source repo at depth 2 (`outer/src`): neither `outer` nor anything
        // else at depth 1 exposes a direct `.git` entry.
        let src = root.join("outer").join("src");
        std::fs::create_dir_all(&src).expect("src dir");
        git(&src, &["init", "-q"]);
        std::fs::write(src.join("s.txt"), "seed\n").expect("write");
        git(&src, &["add", "."]);
        git(&src, &["commit", "-qm", "seed"]);
        git(
            &src,
            &[
                "worktree",
                "add",
                "-q",
                root.join("wt").to_str().expect("utf8"),
            ],
        );
        std::fs::write(root.join("wt").join("w.txt"), "untracked\n").expect("write");
        let wt_meta = std::fs::metadata(root.join("wt").join(".git")).expect("worktree .git");
        assert!(
            wt_meta.is_file(),
            "`git worktree add` produces a `gitdir:` FILE, not a directory"
        );

        match observe_tree(root, Duration::from_secs(30)).await {
            TreeObservation::Observed { porcelain, head } => {
                assert_eq!(
                    porcelain, "link/?? u.txt\nwt/?? w.txt",
                    "sorted (`link` before `wt`), each line `{{name}}/`-prefixed; \
                     `outer` (no depth-1 `.git`) never qualifies"
                );
                let head = head.expect("heads are encoded");
                assert!(
                    head.starts_with("link none\nwt "),
                    "`link` has no commits (head `none`), `wt` shares the source \
                     repo's HEAD: {head:?}"
                );
            }
            other @ TreeObservation::Unobservable { .. } => {
                panic!("expected Observed, got {other:?}")
            }
        }
    }

    /// The depth-1 discovery rule itself, pinned without subprocesses beyond
    /// the fixture setup: a child qualifies iff it is a directory (symlinks
    /// to directories count) AND directly contains a `.git` entry (file or
    /// directory); the result is sorted by file name in byte order.
    #[test]
    fn discover_child_repos_is_sorted_and_qualifies_dirs_with_a_git_entry() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();

        std::fs::create_dir(root.join("zz-repo")).expect("dir");
        std::fs::create_dir(root.join("zz-repo").join(".git")).expect("git dir entry");
        std::fs::create_dir(root.join("aa-repo")).expect("dir");
        std::fs::write(root.join("aa-repo").join(".git"), "gitdir: elsewhere\n")
            .expect("git file entry");
        std::fs::create_dir(root.join("plain")).expect("plain dir, no .git");
        std::fs::write(root.join("afile"), "not a dir\n").expect("file entry");
        // A `.git`-bearing child that is itself a FILE does not qualify.
        std::fs::write(root.join("not-a-dir-with-git"), "x\n").expect("file");
        // A symlink to a directory counts as a directory.
        std::os::unix::fs::symlink(root.join("zz-repo"), root.join("mm-link")).expect("symlink");

        let names = super::discover_child_repos(root);
        assert_eq!(names, vec!["aa-repo", "mm-link", "zz-repo"]);
    }

    #[test]
    fn child_scan_warning_names_reason_count_and_sorted_names() {
        let warning = super::child_scan_warning(
            "git status --porcelain exited Some(128): fatal: not a git repository",
            &["a".to_string(), "b".to_string()],
        );
        assert_eq!(
            warning,
            "warning: root git status failed (git status --porcelain exited Some(128): \
             fatal: not a git repository); observing 2 child repos: a, b"
        );
    }
}
