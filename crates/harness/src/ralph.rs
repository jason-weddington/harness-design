//! The Ralph outer loop: re-invoke the inner [`engine::run`] with a FRESH
//! CONTEXT each iteration so the model never hits the context-degradation
//! "dumb zone". Durable state lives OUTSIDE the context window — the codebase
//! on disk, the git history, and a notes/progress file the agent
//! reads-then-appends — so each outer pass can start cold and still make
//! forward progress.
//!
//! Each outer iteration does EXACTLY ONE unit of work, then the HARNESS owns
//! a per-iteration git commit (a deliberate ralph-mode-only exception to the
//! worker-owns-git contract) — but ONLY for a green `Finished(Done)` outcome.
//! A non-green inner outcome, or a green outcome whose per-iteration `git
//! commit` exited non-zero (e.g. a rejecting pre-commit hook), is REVERTED to
//! the last green commit (`git reset --hard HEAD` then `git clean -fd`; the
//! iteration's PROGRESS.md append is discarded — a deliberate clean do-over;
//! ignored files such as `target/` survive — `git clean` never uses
//! `-x`/`-X`), and the loop retries with a fresh context. After
//! `max_do_overs` CONSECUTIVE such do-overs (a green commit resets the count)
//! the loop terminates with [`RalphTerminal::DoOversExhausted`] (exit 20, the
//! completing iteration appended). An inner [`LoopOutcome::BackendError`] is
//! EXEMPT from the do-over counter (recorded + loop continues, tree reverted
//! if dirty, counter untouched) to preserve overnight resilience. The stop
//! condition is a COMMAND (exit 0 = objective met), kept SEPARATE from the
//! inner per-iteration gate ([`RalphConfig::inner_checks`]) — they are never
//! collapsed. Circuit breakers are a max-outer-iterations cap, a total
//! wall-clock budget (reusing the existing [`Clock`] seam), a stuck-detector,
//! and the consecutive-do-over cap.
//!
//! Ralph is ORTHOGONAL to finish-recovery / the stop-cold nudge: those keep
//! operating unchanged inside each inner [`engine::run`] pass. [`run_ralph`]
//! wraps [`engine::run`] and lets them be.
//!
//! [`Clock`]: crate::time::Clock

use std::sync::Arc;
use std::time::Duration;

use crate::engine::{self, LoopOutcome, RunConfig, RunResult};
use crate::exec::{self, CheckCommand, ExecSpec};
use crate::model::ModelBackend;
use crate::prompt;
use crate::run_record::Disposition;
use crate::time::{Clock, SystemClock};
use crate::tool::ToolCtx;
use crate::tools;

/// Default stuck-detection threshold K: how many consecutive iterations may
/// produce no progress (no git diff OUTSIDE the notes file) before the outer
/// loop gives up with [`RalphTerminal::Stuck`]. Mirrors
/// [`engine::DEFAULT_STATIC_TREE_K`].
pub const DEFAULT_STUCK_K: u32 = 3;

/// Default consecutive-do-over cap: how many consecutive do-overs (a non-green
/// inner outcome that left the tree dirty, or a green outcome whose per-
/// iteration commit was rejected — e.g. by a failing pre-commit hook — both
/// reverted to the last green commit) the outer loop tolerates before
/// terminating with [`RalphTerminal::DoOversExhausted`]. An inner
/// [`LoopOutcome::BackendError`] is EXEMPT (recorded + loop continues, counter
/// untouched) so an overnight run rides out transient backend blips.
pub const DEFAULT_MAX_DO_OVERS: u32 = 3;

/// Default timeout for the outer stop-command. The stop-command is the
/// objective-met oracle; five minutes is generous for a check suite while
/// still bounding a wedged command.
pub const DEFAULT_STOP_COMMAND_TIMEOUT: Duration = Duration::from_mins(5);

/// Timeout for the harness-owned git invocations (status / add / commit /
/// the revert `git reset --hard HEAD` + `git clean -fd`). Git operations on
/// a local work tree are fast; a minute is a generous ceiling that still
/// bounds a wedged repo.
const GIT_TIMEOUT: Duration = Duration::from_mins(1);

/// Configuration for one call to [`run_ralph`]: the objective, the
/// stop-command oracle, the circuit breakers, and the inner-run knobs.
///
/// The harness commits ONLY green `Finished(Done)` outcomes; a non-green
/// inner outcome or a green commit that exited non-zero is reverted to the
/// last green commit and retried as a do-over. [`Self::max_do_overs`] caps
/// consecutive do-overs before [`RalphTerminal::DoOversExhausted`] fires; a
/// green commit resets the count; an inner [`LoopOutcome::BackendError`] is
/// exempt (recorded + continued, counter untouched).
///
/// Derives `Debug` and `Clone` exactly like [`RunConfig`] — `Arc<dyn Clock>`
/// is both `Debug` (via the [`Clock`] supertrait) and `Clone` (via
/// `Arc::clone`).
#[derive(Debug, Clone)]
pub struct RalphConfig {
    /// The high-level objective the ralph loop is working toward. Rendered
    /// verbatim into each per-iteration prompt and into each commit message.
    pub objective: String,
    /// The stop-command oracle: exit `0` = objective met. DISTINCT from
    /// [`Self::inner_checks`] (the inner per-iteration gate) — never
    /// collapsed.
    pub stop_command: CheckCommand,
    /// Wall-clock budget for one invocation of the stop-command.
    pub stop_command_timeout: Duration,
    /// Workspace-relative path of the notes/progress file the agent
    /// reads-then-appends each iteration. Read at the top of every outer
    /// pass (a missing file yields an empty string, never a panic) and named
    /// in the rendered prompt so the agent appends to the exact file
    /// [`run_ralph`] reads next.
    pub notes_file: String,
    /// Hard cap on outer iterations. After this many passes complete
    /// without [`RalphTerminal::StopConditionMet`] or [`RalphTerminal::Stuck`],
    /// the loop terminates with [`RalphTerminal::MaxIterationsExhausted`].
    pub max_outer_iterations: u32,
    /// Stuck-detection threshold K: this many consecutive iterations with no
    /// progress (no git diff OUTSIDE the notes file) terminate with
    /// [`RalphTerminal::Stuck`].
    pub stuck_k: u32,
    /// Consecutive-do-over cap. A "do-over" is an iteration whose tree did
    /// NOT advance to a new green commit: either a non-green inner outcome
    /// (NOT a [`LoopOutcome::BackendError`], which is exempt) that left the
    /// tree dirty, or a green `Finished(Done)` whose per-iteration `git commit`
    /// exited non-zero (e.g. a rejecting pre-commit hook). Either way the tree
    /// is reverted to the last green commit and the loop tries again with a
    /// fresh context. After this many CONSECUTIVE do-overs (a green commit
    /// resets the count to zero) the loop terminates with
    /// [`RalphTerminal::DoOversExhausted`] (exit 20). A green `Finished(Done)`
    /// that produced no changes leaves the count UNCHANGED.
    pub max_do_overs: u32,
    /// Total wall-clock budget in seconds. `0` means unbounded — mirroring
    /// [`RunConfig::wall_clock_secs`]. When non-zero, the loop terminates with
    /// [`RalphTerminal::TimeBudgetExhausted`] once elapsed ≥ this value.
    pub wall_clock_secs: u64,
    /// The clock used for the outer wall-clock budget. The SAME `Arc` is
    /// threaded into each inner [`RunConfig`] so there is one clock seam,
    /// not two.
    pub clock: Arc<dyn Clock>,
    /// Inner-run iteration cap handed to each fresh [`RunConfig`].
    pub inner_max_iterations: u32,
    /// The inner per-iteration GATE — the [`exec::ChecksRunner`] the inner
    /// [`engine::run`] uses to verify a `finish(done)` claim. DISTINCT from
    /// [`Self::stop_command`]; `None` means no automated inner verification.
    pub inner_checks: Option<exec::ChecksRunner>,
}

impl RalphConfig {
    /// Build a config with the given `objective`, `stop_command`, outer
    /// iteration cap, and inner iteration cap. Defaults `stuck_k` to
    /// [`DEFAULT_STUCK_K`], `max_do_overs` to [`DEFAULT_MAX_DO_OVERS`],
    /// `stop_command_timeout` to
    /// [`DEFAULT_STOP_COMMAND_TIMEOUT`], `notes_file` to `"PROGRESS.md"`,
    /// `wall_clock_secs` to `0` (unbounded), `clock` to a [`SystemClock`],
    /// and `inner_checks` to `None`.
    #[must_use]
    pub fn new(
        objective: impl Into<String>,
        stop_command: CheckCommand,
        max_outer_iterations: u32,
        inner_max_iterations: u32,
    ) -> Self {
        Self {
            objective: objective.into(),
            stop_command,
            stop_command_timeout: DEFAULT_STOP_COMMAND_TIMEOUT,
            notes_file: "PROGRESS.md".to_string(),
            max_outer_iterations,
            stuck_k: DEFAULT_STUCK_K,
            max_do_overs: DEFAULT_MAX_DO_OVERS,
            wall_clock_secs: 0,
            clock: Arc::new(SystemClock),
            inner_max_iterations,
            inner_checks: None,
        }
    }

    /// Override the stuck-detection threshold K ([`DEFAULT_STUCK_K`] by
    /// default). See [`Self::stuck_k`].
    #[must_use]
    pub fn with_stuck_k(mut self, stuck_k: u32) -> Self {
        self.stuck_k = stuck_k;
        self
    }

    /// Override the consecutive-do-over cap ([`DEFAULT_MAX_DO_OVERS`] by
    /// default). See [`Self::max_do_overs`].
    #[must_use]
    pub fn with_max_do_overs(mut self, max_do_overs: u32) -> Self {
        self.max_do_overs = max_do_overs;
        self
    }

    /// Override the stop-command timeout
    /// ([`DEFAULT_STOP_COMMAND_TIMEOUT`] by default).
    #[must_use]
    pub fn with_stop_command_timeout(mut self, timeout: Duration) -> Self {
        self.stop_command_timeout = timeout;
        self
    }

    /// Override the notes/progress file path (`"PROGRESS.md"` by default).
    #[must_use]
    pub fn with_notes_file(mut self, notes_file: impl Into<String>) -> Self {
        self.notes_file = notes_file.into();
        self
    }

    /// Set the total wall-clock budget in seconds (`0` = unbounded, the
    /// default). See [`Self::wall_clock_secs`].
    #[must_use]
    pub fn with_wall_clock_secs(mut self, secs: u64) -> Self {
        self.wall_clock_secs = secs;
        self
    }

    /// Inject a custom [`Clock`] implementation. Use
    /// [`crate::time::FakeClock`] in tests for zero-sleep deterministic
    /// timing; production code uses the [`SystemClock`] default set by
    /// [`Self::new`].
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Wire an inner per-iteration gate ([`exec::ChecksRunner`]) the inner
    /// [`engine::run`] uses to verify a `finish(done)` claim. `None` by
    /// default. DISTINCT from [`Self::stop_command`].
    #[must_use]
    pub fn with_inner_checks(mut self, checks: exec::ChecksRunner) -> Self {
        self.inner_checks = Some(checks);
        self
    }
}

/// Why the Ralph outer loop stopped.
///
/// Derives `PartialEq`/`Eq` (no `BackendError` payload here — an inner
/// backend error is RECORDED on the iteration outcome and the loop
/// continues, never a terminal) so tests can assert on it directly.
///
/// Commit contract: ralph commits ONLY green `Finished(Done)` outcomes; any
/// non-green inner outcome (or a green outcome whose `git commit` exited
/// non-zero, e.g. a rejecting pre-commit hook) is reverted to the last green
/// commit (`git reset --hard HEAD` then `git clean -fd`; ignored files such as
/// `target/` survive — `git clean` never uses `-x`/`-X`), the iteration's
/// PROGRESS.md append is discarded (a deliberate clean do-over), and the loop
/// retries with a fresh context. After `max_do_overs` CONSECUTIVE such do-
/// overs (a green commit resets the count) the loop terminates with
/// [`Self::DoOversExhausted`] (a task-side terminal, exit 20 — the completing
/// iteration IS appended, like [`Self::Stuck`]/[`Self::MaxIterationsExhausted`]).
/// An inner [`LoopOutcome::BackendError`] is EXEMPT from the do-over counter
/// (recorded + loop continues, tree reverted if dirty, counter untouched) to
/// preserve overnight resilience.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RalphTerminal {
    /// The stop-command exited `0` (and did not time out) after some
    /// iteration — the objective is met.
    StopConditionMet,
    /// `max_outer_iterations` outer passes completed without meeting the
    /// stop condition or hitting another terminal.
    MaxIterationsExhausted,
    /// `stuck_k` consecutive iterations produced no progress (no git diff
    /// OUTSIDE the notes file).
    Stuck,
    /// The outer wall-clock budget elapsed (only evaluated when
    /// `wall_clock_secs > 0`).
    TimeBudgetExhausted,
    /// `max_do_overs` CONSECUTIVE do-overs elapsed: `max_do_overs` iterations
    /// in a row failed to advance to a new green commit (a non-green inner
    /// outcome that left the tree dirty, or a green `Finished(Done)` whose
    /// `git commit` exited non-zero — both reverted to the last green commit).
    /// A task-side terminal (exit 20), distinct from [`Self::Error`]: the
    /// completing iteration IS appended before this terminal fires, so the
    /// do-over count the breaker tests can be inspected.
    DoOversExhausted,
    /// A hard harness-level failure: a git-status / git-add / revert-command
    /// (`git reset --hard HEAD` or `git clean -fd`) invocation whose
    /// `exit_code != Some(0)` (or that failed to spawn), or a `git commit`
    /// spawn failure, or a stop-command spawn failure. The `String` payload
    /// describes the failure; the partial iteration's [`RalphIterationOutcome`]
    /// is NOT appended to [`RalphReport::iterations`] on this terminal (a
    /// `git commit` that RUNS but exits non-zero — e.g. a rejecting pre-commit
    /// hook — is NOT an `Error`: it triggers a revert + do-over instead).
    Error(String),
}

/// One outer pass, recorded on [`RalphReport::iterations`].
///
/// Derives `Debug` only — it embeds [`LoopOutcome`], which is not `PartialEq`
/// (its [`crate::model::BackendError`] variant doesn't compare), mirroring
/// [`crate::eval::TrialResult`].
#[derive(Debug)]
pub struct RalphIterationOutcome {
    /// 0-based outer iteration index.
    pub iteration: u32,
    /// The inner [`engine::run`] terminal outcome for this pass. An inner
    /// `BackendError` is recorded here and the outer loop continues.
    pub inner_outcome: LoopOutcome,
    /// The inner [`engine::RunStats`] accumulated over this pass.
    pub inner_stats: engine::RunStats,
    /// Whether the post-inner `git status --porcelain` (full, including the
    /// notes file) was non-empty.
    pub made_changes: bool,
    /// Whether a commit was created this iteration.
    pub committed: bool,
}

/// The full result of one [`run_ralph`] call: the objective, the terminal
/// reason, and the ordered per-iteration outcomes.
///
/// Derives `Debug` only — it inherits non-`PartialEq` from
/// [`RalphIterationOutcome`] (and through it [`LoopOutcome`]), mirroring
/// [`crate::eval::EvalReport`].
#[derive(Debug)]
pub struct RalphReport {
    /// Copy of [`RalphConfig::objective`] — so a report can be read in
    /// isolation.
    pub objective: String,
    /// Why the outer loop stopped.
    pub terminal: RalphTerminal,
    /// Per-iteration detail — one entry per outer pass, in order. The
    /// aggregate accessors derive from this vec, so it stays the single
    /// source of truth.
    pub iterations: Vec<RalphIterationOutcome>,
}

impl RalphReport {
    /// How many outer iterations ran — `iterations.len()`.
    #[must_use]
    pub fn outer_iterations(&self) -> u32 {
        // `iterations.len()` is `usize`; on a 64-bit target that is wider than
        // the `u32` return. The vec is bounded by `max_outer_iterations`
        // (a `u32`), so the value always fits — the truncation lint is
        // satisfied by construction, not by luck.
        #[allow(clippy::cast_possible_truncation)]
        let count = self.iterations.len() as u32;
        count
    }

    /// Sum of every iteration's inner `RunStats::iterations`. The `u32` per
    /// iteration is widened with `u64::from` before summing so a long ralph
    /// run can't overflow.
    #[must_use]
    pub fn total_inner_iterations(&self) -> u64 {
        self.iterations
            .iter()
            .map(|o| u64::from(o.inner_stats.iterations))
            .sum()
    }
}

/// Run the Ralph outer loop: re-invoke the inner [`engine::run`] with a
/// FRESH CONTEXT each iteration until the stop-command oracle reports the
/// objective met, or a circuit breaker fires.
///
/// Commit only green finishes: the harness commits ONLY when the inner
/// outcome is a green `Finished(Done)` AND the tree changed. A non-green
/// inner outcome (or a green outcome whose `git commit` exited non-zero, e.g.
/// a rejecting pre-commit hook) is REVERTED to the last green commit (`git
/// reset --hard HEAD` then `git clean -fd`; the iteration's PROGRESS.md
/// append is discarded — a deliberate clean do-over; ignored files such as
/// `target/` survive — `git clean` never uses `-x`/`-X`), and the loop
/// retries with a fresh context as a do-over. After `config.max_do_overs`
/// CONSECUTIVE do-overs (a green commit resets the count to zero; a green
/// `Finished(Done)` that produced no changes leaves it untouched) the loop
/// terminates with [`RalphTerminal::DoOversExhausted`] (exit 20, the
/// completing iteration appended). An inner [`LoopOutcome::BackendError`] is
/// EXEMPT — recorded + loop continues exactly as today, tree reverted if
/// dirty, do-over counter untouched (overnight resilience).
///
/// Per outer iteration, in order: read the current notes content (a missing
/// file yields an empty string, never a panic); render the ralph prompt;
/// build a FRESH inner [`RunConfig`] (fresh context by construction —
/// [`engine::run`] seeds `initial_messages` from `config.task` every call);
/// run the inner loop; capture `is_green` once (before the commit decision
/// — `result.outcome` is MOVED into the appended iteration at the
/// stop-command step, so it cannot be re-matched at the breaker step);
/// detect changes via `git status --porcelain`; if green AND changed, stage
/// and commit; revert-to-green on a non-green-with-changes or
/// green-commit-failure iteration; evaluate the stop-command; evaluate the
/// breakers. The inner per-iteration GATE ([`RalphConfig::inner_checks`]) and
/// the outer STOP-COMMAND are two distinct commands and are never
/// conflated — the inner gate result never decides outer termination, and
/// the stop-command is never passed as the inner-run `checks`.
///
/// `run_ralph` does NOT run `git init` — it assumes the workspace root is
/// already a git work tree (the caller/test sets that up). It does NOT
/// thread `messages` between inner runs.
#[allow(clippy::too_many_lines)]
pub async fn run_ralph(
    backend: &impl ModelBackend,
    ctx: &ToolCtx,
    config: &RalphConfig,
) -> RalphReport {
    let root = ctx.workspace().root().to_path_buf();
    let loop_start = config.clock.now();
    let objective = config.objective.clone();
    let mut iterations: Vec<RalphIterationOutcome> = Vec::new();
    let mut stuck_counter: u32 = 0;
    // Do-over counter (three-way rule): RESET to 0 after a green
    // `Finished(Done)` commit succeeds (exit 0); INCREMENT on a non-green
    // inner outcome that is NOT `BackendError(_)` (the BackendError
    // exemption) or a green outcome whose commit exited non-zero (after the
    // revert); UNCHANGED on a green no-change pass or an exempt
    // `BackendError(_)` pass.
    let mut do_over_counter: u32 = 0;

    let mut i: u32 = 0;
    while i < config.max_outer_iterations {
        // (1) Read the current notes content. A missing file yields `Err`,
        // which `unwrap_or_default()` collapses to an empty `String` —
        // NEVER a panic. `Workspace` has no `resolve()` for this; `root()`
        // plus a plain join is the right path (the notes file is
        // workspace-relative, authored by the agent).
        let notes = std::fs::read_to_string(root.join(&config.notes_file)).unwrap_or_default();

        // (2) Render the ralph prompt with the objective, the notes, and the
        // exact notes filename the agent must append to next iteration.
        let task = prompt::render_ralph_prompt(&config.objective, &notes, &config.notes_file);

        // (3) Build a FRESH inner RunConfig for this pass. A fresh
        // `RunConfig` per pass IS fresh context by construction —
        // `engine::run` seeds `initial_messages` from `config.task` every
        // call. Reuse the SAME injected clock (one seam, not two).
        let registry = tools::standard_registry(config.inner_checks.clone());
        let mut inner_config =
            RunConfig::new(task, config.inner_max_iterations).with_clock(config.clock.clone());
        if let Some(runner) = config.inner_checks.clone() {
            inner_config = inner_config.with_checks(runner);
        }

        // (4) Run the inner loop with fresh context.
        let result: RunResult = engine::run(backend, &registry, ctx, &inner_config).await;

        // (4b) Capture green-once. `result.outcome` is MOVED into the
        // appended `RalphIterationOutcome` at the stop-command step below, so
        // it CANNOT be re-matched at the breaker step — reuse this bool for
        // the commit gate, the do-over counter, and the stuck gate. An inner
        // `BackendError(_)` is non-green but exempt from the do-over counter,
        // so capture its discriminant once here too (no payload move).
        let is_green = matches!(
            result.outcome,
            LoopOutcome::Finished(Disposition::Done { .. })
        );
        let is_backend_error = matches!(result.outcome, LoopOutcome::BackendError(_));

        // (5) Detect changes via the FULL `git status --porcelain` (INCLUDING
        // the notes file — the journal commits even on a notes-only pass).
        let status = exec::run(&ExecSpec::new(
            "git",
            vec!["status".to_string(), "--porcelain".to_string()],
            root.clone(),
            GIT_TIMEOUT,
        ))
        .await;
        if status.exit_code != Some(0) {
            return RalphReport {
                objective,
                terminal: RalphTerminal::Error(format!(
                    "git status exited {:?}: {}",
                    status.exit_code,
                    status.stderr.trim()
                )),
                iterations,
            };
        }
        let made_changes = !status.stdout.trim().is_empty();

        // Stuck progress signal: `git status --porcelain` EXCLUDING the notes
        // file. A notes-only pass still commits (full status non-empty) but
        // STILL counts toward stuck (progress signal empty). Distinct from
        // the commit decision above, which uses the full status.
        let progress = exec::run(&ExecSpec::new(
            "git",
            vec![
                "status".to_string(),
                "--porcelain".to_string(),
                "--".to_string(),
                ".".to_string(),
                format!(":(exclude){}", config.notes_file),
            ],
            root.clone(),
            GIT_TIMEOUT,
        ))
        .await;
        if progress.exit_code != Some(0) {
            return RalphReport {
                objective,
                terminal: RalphTerminal::Error(format!(
                    "git status (progress signal) exited {:?}: {}",
                    progress.exit_code,
                    progress.stderr.trim()
                )),
                iterations,
            };
        }
        let progress_empty = progress.stdout.trim().is_empty();

        // (6) Commit only green finishes. The harness commits ONLY when the
        // inner outcome is a green `Finished(Done)` AND the tree changed; a
        // non-green inner outcome, or a green commit the repo's pre-commit
        // hook rejected, leaves the tree dirty and is reverted below (a
        // clean do-over). `git add -A` failure remains an infra
        // `RalphTerminal::Error`; a `git commit` that RUNS but exits non-zero
        // (e.g. a rejecting pre-commit hook) triggers revert + do-over
        // instead of `Error` (a commit spawn failure is still an infra
        // `RalphTerminal::Error`). The commit uses an explicit identity so it
        // works in a repo with no global git config.
        let mut committed = false;
        let mut commit_failed = false;
        if is_green && made_changes {
            let add = exec::run(&ExecSpec::new(
                "git",
                vec!["add".to_string(), "-A".to_string()],
                root.clone(),
                GIT_TIMEOUT,
            ))
            .await;
            if add.exit_code != Some(0) {
                return RalphReport {
                    objective,
                    terminal: RalphTerminal::Error(format!(
                        "git add exited {:?}: {}",
                        add.exit_code,
                        add.stderr.trim()
                    )),
                    iterations,
                };
            }
            // Commit message pinned format: `ralph: iteration {n} — {objective}`,
            // with {n} the 0-based iteration and {objective} verbatim (NOT
            // truncated).
            let message = format!("ralph: iteration {i} — {}", config.objective);
            let commit = exec::run(&ExecSpec::new(
                "git",
                vec![
                    "-c".to_string(),
                    "user.name=talos-ralph".to_string(),
                    "-c".to_string(),
                    "user.email=talos-ralph@localhost".to_string(),
                    "commit".to_string(),
                    "-m".to_string(),
                    message,
                ],
                root.clone(),
                GIT_TIMEOUT,
            ))
            .await;
            match commit.exit_code {
                None => {
                    // spawn failure — infra Error (not a do-over).
                    return RalphReport {
                        objective,
                        terminal: RalphTerminal::Error(format!(
                            "git commit failed to spawn: {}",
                            commit.stderr.trim()
                        )),
                        iterations,
                    };
                }
                Some(0) => {
                    committed = true;
                }
                Some(_) => {
                    // commit ran but was rejected (e.g. a failing pre-commit
                    // hook) — revert the tree and do over.
                    commit_failed = true;
                }
            }
        }

        // (6b) Revert-to-green. On a non-green inner outcome that left the
        // tree dirty, OR a green outcome whose commit exited non-zero, reset
        // the work tree to the last green commit: FIRST `git reset --hard
        // HEAD`, THEN `git clean -fd`. `git clean` MUST NOT use `-x`/`-X` —
        // ignored files such as `target/` survive, and `.git` is never
        // touched by `git clean`. The revert discards this iteration's
        // PROGRESS.md append too (a deliberate clean do-over). A single
        // error-mapping arm covers both commands: the first non-zero exit
        // returns an infra `RalphTerminal::Error` naming the failing git
        // command (before the stop-command append, same shape as the
        // git-status / git-add errors above).
        if made_changes && (!is_green || commit_failed) {
            for (args, label) in [
                (
                    vec![
                        "reset".to_string(),
                        "--hard".to_string(),
                        "HEAD".to_string(),
                    ],
                    "git reset --hard HEAD",
                ),
                (
                    vec!["clean".to_string(), "-fd".to_string()],
                    "git clean -fd",
                ),
            ] {
                let out = exec::run(&ExecSpec::new("git", args, root.clone(), GIT_TIMEOUT)).await;
                if out.exit_code != Some(0) {
                    return RalphReport {
                        objective,
                        terminal: RalphTerminal::Error(format!(
                            "{label} exited {:?}: {}",
                            out.exit_code,
                            out.stderr.trim()
                        )),
                        iterations,
                    };
                }
            }
        }

        // (7) Evaluate the stop-command. StopConditionMet is checked FIRST
        // among post-iteration terminals so a just-satisfied objective wins
        // even on the final allowed iteration. A timeout or non-zero exit
        // means the objective is not yet met — fall through to the breakers.
        // A spawn failure (exit_code None && !timed_out) is a hard Error that
        // does NOT append the partial iteration's outcome. The outcome is
        // appended ONLY once we know this iteration is not terminating in an
        // Error — so StopConditionMet appends, the fall-through appends, and
        // every Error branch returns before appending.
        let stop = exec::run(&ExecSpec::new(
            config.stop_command.program.clone(),
            config.stop_command.args.clone(),
            root.clone(),
            config.stop_command_timeout,
        ))
        .await;
        if stop.timed_out {
            // objective not yet met — record the iteration, then breakers.
            iterations.push(RalphIterationOutcome {
                iteration: i,
                inner_outcome: result.outcome,
                inner_stats: result.stats,
                made_changes,
                committed,
            });
        } else {
            match stop.exit_code {
                Some(0) => {
                    iterations.push(RalphIterationOutcome {
                        iteration: i,
                        inner_outcome: result.outcome,
                        inner_stats: result.stats,
                        made_changes,
                        committed,
                    });
                    return RalphReport {
                        objective,
                        terminal: RalphTerminal::StopConditionMet,
                        iterations,
                    };
                }
                None => {
                    return RalphReport {
                        objective,
                        terminal: RalphTerminal::Error(format!(
                            "stop_command failed to spawn: {}",
                            stop.stderr.trim()
                        )),
                        iterations,
                    };
                }
                Some(_) => {
                    // non-zero — objective not yet met; record the iteration,
                    // then breakers.
                    iterations.push(RalphIterationOutcome {
                        iteration: i,
                        inner_outcome: result.outcome,
                        inner_stats: result.stats,
                        made_changes,
                        committed,
                    });
                }
            }
        }

        // (8) Evaluate breakers. Precedence among post-iteration terminals:
        // StopConditionMet (already returned above) -> DoOversExhausted ->
        // Stuck -> MaxIterationsExhausted -> TimeBudgetExhausted -> continue.
        //
        // Do-over counter (three-way rule): RESET to 0 after a green
        // `Finished(Done)` commit succeeds (exit 0); INCREMENT on a non-green
        // inner outcome that is NOT `BackendError(_)` (the BackendError
        // exemption) or a green outcome whose commit exited non-zero (after
        // the revert); UNCHANGED on a green no-change pass or an exempt
        // `BackendError(_)` pass.
        if is_green && committed {
            do_over_counter = 0;
        } else if (is_green && commit_failed) || (!is_green && !is_backend_error) {
            do_over_counter += 1;
        }
        if do_over_counter >= config.max_do_overs {
            return RalphReport {
                objective,
                terminal: RalphTerminal::DoOversExhausted,
                iterations,
            };
        }

        // Stuck reconciliation (no double-count): stuck_counter moves ONLY
        // on a green iteration that did NOT fail its commit — i.e.
        // `is_green && (committed || !made_changes)`. On any non-green
        // iteration (including the exempt `BackendError`) OR a green-commit-
        // failure iteration, stuck_counter is left UNCHANGED (the do-over
        // counter owns that failure), so stuck and do-over never both move
        // on a failure iteration. Green-inert (finish-only, no changes) and
        // green-notes-only passes still increment stuck (progress empty).
        if is_green && (committed || !made_changes) {
            if progress_empty {
                stuck_counter += 1;
            } else {
                stuck_counter = 0;
            }
        }
        if stuck_counter >= config.stuck_k {
            return RalphReport {
                objective,
                terminal: RalphTerminal::Stuck,
                iterations,
            };
        }

        if i + 1 >= config.max_outer_iterations {
            return RalphReport {
                objective,
                terminal: RalphTerminal::MaxIterationsExhausted,
                iterations,
            };
        }

        if config.wall_clock_secs != 0 {
            let elapsed = config
                .clock
                .now()
                .duration_since(loop_start)
                .unwrap_or(Duration::ZERO);
            if elapsed.as_secs() >= config.wall_clock_secs {
                return RalphReport {
                    objective,
                    terminal: RalphTerminal::TimeBudgetExhausted,
                    iterations,
                };
            }
        }

        i += 1;
    }

    // `max_outer_iterations == 0`: no passes ran. Terminate with
    // MaxIterationsExhausted (zero iterations, by definition exhausted).
    RalphReport {
        objective,
        terminal: RalphTerminal::MaxIterationsExhausted,
        iterations,
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_STUCK_K, RalphConfig, RalphReport, RalphTerminal, run_ralph};
    use crate::engine::{FINISH_TOOL_NAME, LoopOutcome};
    use crate::exec::{CheckCommand, ChecksRunner, ExecSpec, run as exec_run};
    use crate::model::{
        AssistantTurn, BackendError, ContentBlock, Message, ModelBackend, StopReason, TerminalKind,
        ToolCallRequest, TurnRequest, Usage,
    };
    use crate::run_record::Disposition;
    use crate::test_support::MockBackend;
    use crate::time::FakeClock;
    use crate::tool::ToolCtx;
    use crate::workspace::Workspace;
    use async_trait::async_trait;
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};
    use tempfile::TempDir;

    // ---- test helpers (mirror engine.rs's scripting helpers) ------------

    fn usage() -> Usage {
        Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn tool_call(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
        ContentBlock::ToolCall(ToolCallRequest {
            id: id.to_string(),
            name: name.to_string(),
            input,
        })
    }

    fn turn_with(content: Vec<ContentBlock>, stop_reason: StopReason) -> AssistantTurn {
        AssistantTurn {
            content,
            stop_reason,
            usage: usage(),
        }
    }

    fn finish_call(id: &str, input: serde_json::Value) -> AssistantTurn {
        turn_with(
            vec![tool_call(id, FINISH_TOOL_NAME, input)],
            StopReason::ToolUse,
        )
    }

    fn edit_create_call(id: &str, path: &str, content: &str) -> AssistantTurn {
        turn_with(
            vec![tool_call(
                id,
                "edit_file",
                serde_json::json!({
                    "path": path,
                    "old_string": "",
                    "new_string": content,
                }),
            )],
            StopReason::ToolUse,
        )
    }

    fn bash_call(id: &str, command: &str) -> AssistantTurn {
        turn_with(
            vec![tool_call(
                id,
                "bash",
                serde_json::json!({ "command": command }),
            )],
            StopReason::ToolUse,
        )
    }

    fn finish_done(id: &str) -> AssistantTurn {
        finish_call(
            id,
            serde_json::json!({ "disposition": "done", "summary": "ok" }),
        )
    }

    /// A no-tool assistant turn that ends the turn (`StopReason::EndTurn`). The
    /// inner loop sees no tool call to dispatch and stops with
    /// `LoopOutcome::StoppedWithoutFinish` — a non-green inner outcome that,
    /// if it left the tree dirty, triggers a revert + do-over.
    fn no_tool_turn(text: &str) -> AssistantTurn {
        turn_with(
            vec![ContentBlock::Text(text.to_string())],
            StopReason::EndTurn,
        )
    }

    /// Build a `ToolCtx` rooted at a git work tree the test sets up.
    fn ctx_for(root: &std::path::Path) -> ToolCtx {
        let workspace = Workspace::new(root, None).expect("workspace");
        ToolCtx::new(Arc::new(workspace), Arc::new(crate::tool::StubOffloadSink))
    }

    /// `git init` a temp dir into a work tree (no global git config needed;
    /// commits pass `-c user.name/-c user.email`).
    fn git_init(root: &std::path::Path) {
        std::process::Command::new("git")
            .arg("init")
            .current_dir(root)
            .output()
            .expect("git init");
    }

    /// Count commits reachable from HEAD via the harness's own exec seam
    /// (the same clean-env child PATH the stop-command uses). Returns `None`
    /// when HEAD is unborn (no commits yet).
    async fn commit_count(root: &std::path::Path) -> Option<u32> {
        let out = exec_run(&ExecSpec::new(
            "git",
            vec![
                "rev-list".to_string(),
                "--count".to_string(),
                "HEAD".to_string(),
            ],
            root.to_path_buf(),
            Duration::from_secs(10),
        ))
        .await;
        if out.exit_code != Some(0) {
            return None;
        }
        out.stdout.trim().parse::<u32>().ok()
    }

    /// A stop-command that exits `0` once at least `n` commits exist in the
    /// repo (and non-zero before that). Resolves `/bin/sh` and `git` via the
    /// child PATH that `exec::run` preserves.
    fn stop_when_n_commits(n: u32) -> CheckCommand {
        CheckCommand {
            program: "/bin/sh".to_string(),
            args: vec![
                "-c".to_string(),
                format!("test \"$(git rev-list --count HEAD 2>/dev/null || echo 0)\" -ge {n}"),
            ],
        }
    }

    /// A stop-command that always exits non-zero (objective never met).
    fn stop_never() -> CheckCommand {
        CheckCommand {
            program: "/bin/sh".to_string(),
            args: vec!["-c".to_string(), "exit 1".to_string()],
        }
    }

    // ===================================================================
    // Scenario 1: HAPPY PATH
    // ===================================================================

    #[tokio::test]
    async fn happy_path_edits_finish_and_stops_when_target_commits_exist() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // Two outer iterations, each: edit a distinct file then finish(done).
        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "work0.txt", "first\n"),
            finish_done("f0"),
            edit_create_call("e1", "work1.txt", "second\n"),
            finish_done("f1"),
        ]);

        let config = RalphConfig::new(
            "happy-path-objective",
            stop_when_n_commits(2),
            5, // max_outer_iterations
            4, // inner_max_iterations
        );

        let report = run_ralph(&backend, &ctx, &config).await;

        assert_eq!(
            report.terminal,
            RalphTerminal::StopConditionMet,
            "terminal must be StopConditionMet; got {:?}",
            report.terminal
        );
        assert_eq!(
            report.outer_iterations(),
            2,
            "exactly two outer iterations must run; got {}",
            report.outer_iterations()
        );
        assert_eq!(
            commit_count(&root_path).await,
            Some(2),
            "exactly two ralph commits must exist in git log"
        );
        // Every iteration made changes and committed.
        for (n, it) in report.iterations.iter().enumerate() {
            assert!(it.made_changes, "iteration {n} must report made_changes");
            assert!(it.committed, "iteration {n} must report committed");
            assert!(
                matches!(
                    it.inner_outcome,
                    LoopOutcome::Finished(Disposition::Done { .. })
                ),
                "iteration {n} inner outcome must be Finished(Done); got {:?}",
                it.inner_outcome
            );
        }
        // Aggregate accessor: each inner run took 2 turns (edit + finish).
        assert_eq!(
            report.total_inner_iterations(),
            4,
            "total_inner_iterations must sum inner turns across passes"
        );
    }

    // ===================================================================
    // FRESH CONTEXT across iterations
    // ===================================================================

    #[tokio::test]
    async fn fresh_context_proven_across_iterations() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "a.txt", "x\n"),
            finish_done("f0"),
            edit_create_call("e1", "b.txt", "y\n"),
            finish_done("f1"),
        ]);

        let config = RalphConfig::new("FRESH_CONTEXT_SENTINEL", stop_when_n_commits(2), 5, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(report.terminal, RalphTerminal::StopConditionMet);
        assert_eq!(report.outer_iterations(), 2);

        // messages_seen accumulates one entry per `turn` call, in order.
        // iter 0: turn 0 (edit) -> [task_seed]; turn 1 (finish) -> [task_seed,
        //   assistant_edit, tool_result].
        // iter 1: turn 2 (edit) -> [task_seed] (FRESH — no iter-0 assistant
        //   turns carried over).
        let seen = backend.messages_seen();
        assert!(
            seen.len() >= 3,
            "need at least 3 turn calls to assert across two iterations; got {}",
            seen.len()
        );
        let iter1_first = &seen[2];
        assert_eq!(
            iter1_first.len(),
            1,
            "iteration 2's first turn must see exactly ONE message (the freshly-rendered \
             task seed), proving fresh context — not a continuation carrying iteration-1 \
             assistant turns; got {iter1_first:?}"
        );
        // That single message is the task seed (a User::Text containing the
        // objective sentinel). Extract its text and assert it carries the
        // objective, NOT iteration-1's assistant tool-call content.
        let text = match &iter1_first[0] {
            Message::User { content } => match &content[0] {
                crate::model::UserBlock::Text(t) => t.clone(),
                other @ crate::model::UserBlock::ToolResult { .. } => {
                    panic!("expected Text block, got {other:?}")
                }
            },
            other @ Message::Assistant { .. } => panic!("expected User message, got {other:?}"),
        };
        assert!(
            text.contains("FRESH_CONTEXT_SENTINEL"),
            "iteration 2's task seed must contain the objective sentinel; got:\n{text}"
        );
        // And it must NOT carry iteration-1's assistant edit turn (a fresh
        // context has only the task seed; the seed text has no tool calls).
        assert!(
            !text.contains("edit_file"),
            "fresh task seed must not leak prior assistant tool-call content"
        );
    }

    // ===================================================================
    // Scenario 2: MAX ITERATIONS
    // ===================================================================

    #[tokio::test]
    async fn max_iterations_exhausted_when_stop_never_met() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // Each iteration edits a distinct file + finishes; stop_command
        // always exits 1, so the loop runs to the cap.
        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "w0.txt", "0\n"),
            finish_done("f0"),
            edit_create_call("e1", "w1.txt", "1\n"),
            finish_done("f1"),
            edit_create_call("e2", "w2.txt", "2\n"),
            finish_done("f2"),
        ]);

        let config = RalphConfig::new("max-iter-objective", stop_never(), 3, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::MaxIterationsExhausted,
            "terminal must be MaxIterationsExhausted; got {:?}",
            report.terminal
        );
        assert_eq!(
            report.outer_iterations(),
            3,
            "iterations.len() == max_outer_iterations"
        );
    }

    // ===================================================================
    // Scenario 3a: STUCK-INERT (no edits at all)
    // ===================================================================

    #[tokio::test]
    async fn stuck_inert_when_agent_never_edits() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // The agent calls ONLY finish(done) — no file edit at all. Each inner
        // run returns Finished(Done) after one turn; no changes; no commit;
        // the stuck counter increments each pass.
        let backend = MockBackend::from_turns(vec![
            finish_done("f0"),
            finish_done("f1"),
            finish_done("f2"),
            finish_done("f3"),
        ]);

        let config = RalphConfig::new("stuck-inert-objective", stop_never(), 10, 4).with_stuck_k(3);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::Stuck,
            "terminal must be Stuck; got {:?}",
            report.terminal
        );
        assert_eq!(
            report.outer_iterations(),
            DEFAULT_STUCK_K,
            "Stuck must fire after stuck_k passes"
        );
        // ZERO commits: every pass had no changes.
        for it in &report.iterations {
            assert!(!it.made_changes, "inert pass must report no changes");
            assert!(!it.committed, "inert pass must not commit");
        }
    }

    // ===================================================================
    // Scenario 3b: STUCK-NOTES-ONLY (journal but no code progress)
    // ===================================================================

    #[tokio::test]
    async fn stuck_notes_only_commits_but_still_counts_toward_stuck() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // The agent appends ONLY to the notes file each pass (via bash) — no
        // code change. Each pass COMMITS (full status non-empty) but the
        // notes-excluded progress signal is empty, so stuck still fires.
        let backend = MockBackend::from_turns(vec![
            bash_call("b0", "printf 'iter 0\\n' >> PROGRESS.md"),
            finish_done("f0"),
            bash_call("b1", "printf 'iter 1\\n' >> PROGRESS.md"),
            finish_done("f1"),
            bash_call("b2", "printf 'iter 2\\n' >> PROGRESS.md"),
            finish_done("f2"),
        ]);

        let config = RalphConfig::new("stuck-notes-objective", stop_never(), 10, 4).with_stuck_k(3);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::Stuck,
            "notes-only journaling must NOT defeat stuck; got {:?}",
            report.terminal
        );
        assert_eq!(report.outer_iterations(), 3, "Stuck after stuck_k passes");
        // Each pass committed (the journal), proving the notes file is
        // excluded only from the stuck signal, not from the commit.
        for it in &report.iterations {
            assert!(it.made_changes, "notes-only pass must report made_changes");
            assert!(it.committed, "notes-only pass must commit the journal");
        }
        assert_eq!(
            commit_count(&root_path).await,
            Some(3),
            "the journal: one commit per notes-only pass"
        );
    }

    // ===================================================================
    // Scenario 4: TWO-COMMAND SEPARATION
    // ===================================================================

    #[tokio::test]
    async fn inner_gate_green_while_stop_command_nonzero_continues() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // The inner gate is a ChecksRunner whose command exits 0 (`true`), so
        // finish(done) is VERIFIED GREEN and the inner run returns
        // Finished(Done). The stop_command still exits non-zero, so the
        // outer loop CONTINUES to the next iteration rather than terminating.
        let green_runner = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            root_path.clone(),
            Duration::from_secs(10),
        );

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "g0.txt", "0\n"),
            finish_done("f0"),
            edit_create_call("e1", "g1.txt", "1\n"),
            finish_done("f1"),
        ]);

        let config = RalphConfig::new("two-command-objective", stop_never(), 2, 4)
            .with_inner_checks(green_runner);

        let report = run_ralph(&backend, &ctx, &config).await;

        // The inner gate was green (Finished(Done) verified), yet the loop
        // did NOT terminate on it — it continued past iteration 0.
        assert_eq!(
            report.outer_iterations(),
            2,
            "the outer loop must CONTINUE past the green inner gate; a green inner \
             gate must NOT terminate the outer loop"
        );
        for it in &report.iterations {
            assert!(
                matches!(
                    it.inner_outcome,
                    LoopOutcome::Finished(Disposition::Done { .. })
                ),
                "inner gate must be green (Finished Done); got {:?}",
                it.inner_outcome
            );
        }
        // And it bottoms out on MaxIterationsExhausted (stop never met), NOT
        // StopConditionMet — the inner gate result never decides outer
        // termination.
        assert_eq!(
            report.terminal,
            RalphTerminal::MaxIterationsExhausted,
            "outer loop must terminate on the breaker, not the inner gate"
        );
    }

    // ===================================================================
    // Scenario 5: TIME BUDGET
    // ===================================================================

    /// A scripted backend that ALSO advances a shared manual [`FakeClock`] by
    /// a fixed delta on every `turn` call. The advance is EXPLICIT (driven by
    /// the test's script, not by inner `now()` tick counts), so the outer
    /// elapsed arithmetic stays decoupled from how many `now()` reads the
    /// inner loop makes — the rationale for using a MANUAL clock here. A
    /// single `run_ralph` `await` cannot be interrupted to call
    /// `FakeClock::advance` between iterations, so the explicit advance rides
    /// on the one thing that does run between ralph's per-iteration clock
    /// reads: the inner `turn` call. Each turn returns a `finish(done)` so the
    /// inner run completes in one turn.
    struct AdvancingBackend {
        clock: Arc<FakeClock>,
        advance: Duration,
    }

    #[async_trait]
    impl ModelBackend for AdvancingBackend {
        async fn turn(&self, _req: &TurnRequest<'_>) -> Result<AssistantTurn, BackendError> {
            self.clock.advance(self.advance);
            Ok(finish_done("adv"))
        }
    }

    #[tokio::test]
    async fn time_budget_exhausted_when_clock_advances_past_budget() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // MANUAL FakeClock (no auto-advance): only the backend's explicit
        // `.advance()` moves it. Each inner turn advances by 5s; the wall
        // budget is 3s, so after iteration 0 (one turn → +5s) elapsed=5>=3.
        let clock = Arc::new(FakeClock::new(UNIX_EPOCH));
        let backend = AdvancingBackend {
            clock: clock.clone(),
            advance: Duration::from_secs(5),
        };

        let config = RalphConfig::new("time-budget-objective", stop_never(), 10, 4)
            .with_stuck_k(100) // disable stuck so the time breaker wins
            .with_wall_clock_secs(3)
            .with_clock(clock);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::TimeBudgetExhausted,
            "terminal must be TimeBudgetExhausted; got {:?}",
            report.terminal
        );
        // At least one iteration completed before the budget fired.
        assert!(
            !report.iterations.is_empty(),
            "TimeBudgetExhausted must follow at least one completed iteration"
        );
    }

    // ===================================================================
    // Scenario 6: ERROR PATHS
    // ===================================================================

    #[tokio::test]
    async fn error_stop_command_spawn_failure_does_not_append_iteration() {
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // A valid edit + finish (so the iteration completes and commits),
        // then a stop_command that FAILS TO SPAWN (program does not exist) =>
        // exit_code None && !timed_out => RalphTerminal::Error.
        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "x.txt", "x\n"),
            finish_done("f0"),
        ]);

        let config = RalphConfig::new(
            "error-spawn-objective",
            CheckCommand {
                program: "/no/such/program-xyzzy-ralph".to_string(),
                args: vec![],
            },
            5,
            4,
        );

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::Error(_)),
            "terminal must be Error; got {:?}",
            report.terminal
        );
        // The partial iteration's outcome must NOT be appended.
        assert!(
            report.iterations.is_empty(),
            "an Error terminal must NOT append the partial iteration's outcome; got {} entries",
            report.iterations.len()
        );
    }

    #[tokio::test]
    async fn error_git_status_nonzero_when_root_is_not_a_work_tree() {
        // A temp dir that is NOT a git work tree: `git status --porcelain`
        // exits 128 => RalphTerminal::Error (no iteration appended).
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        // Deliberately NO git init.
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![finish_done("f0")]);

        let config = RalphConfig::new("error-status-objective", stop_never(), 5, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::Error(ref e) if e.contains("git status")),
            "terminal must be a git-status Error; got {:?}",
            report.terminal
        );
        assert!(
            report.iterations.is_empty(),
            "no iteration must be appended on a git-status Error"
        );
    }

    #[tokio::test]
    async fn error_git_add_fails_via_stale_index_lock() {
        // `git status` SUCCEEDS with a stale `.git/index.lock` present, but
        // `git add -A` FAILS (exit 128) => RalphTerminal::Error. Isolates the
        // git-add error path from the git-status error path.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        // An initial empty commit so HEAD is born (rev-list works in the
        // stop_command and the repo is a valid work tree).
        std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@l",
                "commit",
                "--allow-empty",
                "-m",
                "init",
            ])
            .current_dir(&root_path)
            .output()
            .expect("initial commit");
        // Plant a stale index lock: `git status` reads it fine, `git add` fails
        // to write it.
        std::fs::write(root_path.join(".git").join("index.lock"), b"").expect("index.lock");
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "new.txt", "n\n"),
            finish_done("f0"),
        ]);

        let config = RalphConfig::new("error-add-objective", stop_never(), 5, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::Error(ref e) if e.contains("git add")),
            "terminal must be a git-add Error; got {:?}",
            report.terminal
        );
        assert!(
            report.iterations.is_empty(),
            "no iteration must be appended on a git-add Error"
        );
    }

    #[tokio::test]
    async fn error_git_commit_fails_via_failing_pre_commit_hook() {
        // Under the new contract a failing pre-commit hook on a GREEN outcome
        // triggers revert + do-over, NOT an immediate `RalphTerminal::Error`.
        // With the default `max_do_overs` (3), three consecutive green commits
        // rejected by the hook exhaust the do-over budget: the loop terminates
        // with `DoOversExhausted`, and only the init commit lands (every green
        // commit is rejected by the hook and reverted, so no code lands).
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@l",
                "commit",
                "--allow-empty",
                "-m",
                "init",
            ])
            .current_dir(&root_path)
            .output()
            .expect("initial commit");
        // Plant a failing pre-commit hook.
        let hook = root_path.join(".git").join("hooks").join("pre-commit");
        std::fs::write(&hook, b"#!/bin/sh\nexit 1\n").expect("write hook");
        make_executable(&hook);
        let ctx = ctx_for(&root_path);

        // Three iterations of [edit a distinct file, finish(done)]: each is a
        // green `Finished(Done)` whose commit the hook rejects, then reverts.
        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "c0.txt", "c\n"),
            finish_done("f0"),
            edit_create_call("e1", "c1.txt", "c\n"),
            finish_done("f1"),
            edit_create_call("e2", "c2.txt", "c\n"),
            finish_done("f2"),
        ]);

        let config = RalphConfig::new("error-commit-objective", stop_never(), 5, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::DoOversExhausted),
            "terminal must be DoOversExhausted; got {:?}",
            report.terminal
        );
        assert_eq!(
            report.outer_iterations(),
            3,
            "exactly three do-overs must run; got {}",
            report.outer_iterations()
        );
        assert_eq!(
            commit_count(&root_path).await,
            Some(1),
            "only the init commit must land; every green commit is rejected and reverted"
        );
    }

    // ===================================================================
    // Do-over contract: revert-to-green + N consecutive do-overs
    // ===================================================================

    #[tokio::test]
    async fn non_green_reverts_does_not_commit_and_removes_untracked_file() {
        // (a) A non-green inner outcome (`StoppedWithoutFinish` from a no-tool
        // `EndTurn`) that left the tree dirty is reverted: nothing commits,
        // and the untracked file the agent created is removed by `git clean
        // -fd`. With `max_ralph_iterations == 1` the loop bottoms out on
        // `MaxIterationsExhausted` (one do-over, below the default cap of 3).
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@l",
                "commit",
                "--allow-empty",
                "-m",
                "init",
            ])
            .current_dir(&root_path)
            .output()
            .expect("initial commit");
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "newfile.txt", "x\n"),
            no_tool_turn("stopping"),
        ]);

        let config = RalphConfig::new("revert-objective", stop_never(), 1, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::MaxIterationsExhausted,
            "a single do-over below the cap must bottom out on MaxIterations; got {:?}",
            report.terminal
        );
        assert_eq!(report.outer_iterations(), 1, "exactly one outer iteration");
        assert!(
            !report.iterations[0].committed,
            "the non-green iteration must NOT report committed"
        );
        assert_eq!(
            commit_count(&root_path).await,
            Some(1),
            "only the init commit must exist — nothing committed this iteration"
        );
        assert!(
            !root_path.join("newfile.txt").exists(),
            "the untracked file must be removed by `git clean -fd`"
        );
    }

    #[tokio::test]
    async fn three_consecutive_non_green_exhausts_do_overs() {
        // (b) Three consecutive non-green do-overs (each a no-tool `EndTurn`
        // that leaves the tree dirty) terminate with `DoOversExhausted` once
        // the counter reaches `max_do_overs` (3). `max_ralph_iterations` is
        // large enough that `MaxIterations` does not fire first.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@l",
                "commit",
                "--allow-empty",
                "-m",
                "init",
            ])
            .current_dir(&root_path)
            .output()
            .expect("initial commit");
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "n0.txt", "x\n"),
            no_tool_turn("stop"),
            edit_create_call("e1", "n1.txt", "x\n"),
            no_tool_turn("stop"),
            edit_create_call("e2", "n2.txt", "x\n"),
            no_tool_turn("stop"),
        ]);

        let config = RalphConfig::new("doover-objective", stop_never(), 10, 4).with_max_do_overs(3);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::DoOversExhausted),
            "terminal must be DoOversExhausted; got {:?}",
            report.terminal
        );
        assert_eq!(
            report.outer_iterations(),
            3,
            "DoOversExhausted must fire after exactly three do-overs; got {}",
            report.outer_iterations()
        );
    }

    #[tokio::test]
    async fn green_commit_between_failures_resets_do_over_counter() {
        // (c) A green commit between failures resets the do-over counter:
        // fail, fail, green, fail, fail, fail -> DoOversExhausted fires only
        // after the 3rd consecutive post-reset do-over (six outer iterations).
        // commit_count is init + the single green commit (2).
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@l",
                "commit",
                "--allow-empty",
                "-m",
                "init",
            ])
            .current_dir(&root_path)
            .output()
            .expect("initial commit");
        let ctx = ctx_for(&root_path);

        // Distinct filenames per iteration: fail = edit + no-tool stop;
        // green = edit + finish(done). 12 turns total.
        let backend = MockBackend::from_turns(vec![
            edit_create_call("f0", "f0.txt", "x\n"),
            no_tool_turn("stop"),
            edit_create_call("f1", "f1.txt", "x\n"),
            no_tool_turn("stop"),
            edit_create_call("g", "g.txt", "x\n"),
            finish_done("gf"),
            edit_create_call("f3", "f3.txt", "x\n"),
            no_tool_turn("stop"),
            edit_create_call("f4", "f4.txt", "x\n"),
            no_tool_turn("stop"),
            edit_create_call("f5", "f5.txt", "x\n"),
            no_tool_turn("stop"),
        ]);

        let config = RalphConfig::new("reset-objective", stop_never(), 10, 4).with_max_do_overs(3);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::DoOversExhausted),
            "terminal must be DoOversExhausted; got {:?}",
            report.terminal
        );
        assert_eq!(
            report.outer_iterations(),
            6,
            "DoOversExhausted must fire only after the 3rd consecutive post-reset do-over; got {}",
            report.outer_iterations()
        );
        assert_eq!(
            commit_count(&root_path).await,
            Some(2),
            "init + the single green commit; got {:?}",
            commit_count(&root_path).await
        );
    }

    #[tokio::test]
    async fn revert_command_failure_returns_error_before_append() {
        // (e) A revert-command failure is an infra `RalphTerminal::Error`
        // that returns BEFORE the stop-command append (so no iteration is
        // appended). Reproduce the stale-index-lock trick: `git status`
        // succeeds (read-only) but the revert's `git reset --hard HEAD`
        // fails (exit 128) because `.git/index.lock` is held.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        std::process::Command::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@l",
                "commit",
                "--allow-empty",
                "-m",
                "init",
            ])
            .current_dir(&root_path)
            .output()
            .expect("initial commit");
        // Plant a stale index lock: `git status` reads it fine, `git reset`
        // fails to write it.
        std::fs::write(root_path.join(".git").join("index.lock"), b"").expect("index.lock");
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "x.txt", "x\n"),
            no_tool_turn("stop"),
        ]);

        let config = RalphConfig::new("revert-error-objective", stop_never(), 5, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert!(
            matches!(report.terminal, RalphTerminal::Error(ref e) if e.contains("git reset")),
            "terminal must be a git-reset Error; got {:?}",
            report.terminal
        );
        assert!(
            report.iterations.is_empty(),
            "a revert Error must NOT append the partial iteration's outcome; got {} entries",
            report.iterations.len()
        );
    }

    // ===================================================================
    // Branch coverage: stop_command timeout-vs-spawn split + missing notes
    // ===================================================================

    #[tokio::test]
    async fn stop_command_timeout_falls_through_not_error() {
        // A stop_command that sleeps longer than its timeout => timed_out =>
        // fall through to breakers (NOT Error). The loop runs to max_outer.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        let backend = MockBackend::from_turns(vec![finish_done("f0"), finish_done("f1")]);

        let config = RalphConfig::new(
            "timeout-objective",
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "sleep 30".to_string()],
            },
            2,
            4,
        )
        .with_stuck_k(100) // disable stuck so max-iter wins
        .with_stop_command_timeout(Duration::from_millis(200));

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::MaxIterationsExhausted,
            "a stop_command TIMEOUT must fall through to breakers, not Error; got {:?}",
            report.terminal
        );
        assert_eq!(report.outer_iterations(), 2);
    }

    #[tokio::test]
    async fn missing_notes_file_yields_empty_string_no_panic() {
        // No PROGRESS.md exists at start. The first iteration reads it via
        // `unwrap_or_default()` => empty string, never a panic; the rendered
        // prompt still carries the objective and notes filename. The loop
        // runs normally (happy path) and stops on the stop-command.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);
        assert!(
            !root_path.join("PROGRESS.md").exists(),
            "precondition: notes file absent"
        );

        let backend = MockBackend::from_turns(vec![
            edit_create_call("e0", "m.txt", "m\n"),
            finish_done("f0"),
        ]);

        let config = RalphConfig::new("missing-notes-objective", stop_when_n_commits(1), 5, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(report.terminal, RalphTerminal::StopConditionMet);
        assert_eq!(report.outer_iterations(), 1);
        // The first message the model saw must carry the objective (rendered
        // from an empty notes string, proving the missing file collapsed to
        // empty and the prompt still rendered).
        let seen = backend.messages_seen();
        assert!(!seen.is_empty());
        let text = match &seen[0][0] {
            Message::User { content } => match &content[0] {
                crate::model::UserBlock::Text(t) => t.clone(),
                other @ crate::model::UserBlock::ToolResult { .. } => {
                    panic!("expected Text, got {other:?}")
                }
            },
            other @ Message::Assistant { .. } => panic!("expected User, got {other:?}"),
        };
        assert!(
            text.contains("missing-notes-objective"),
            "prompt must render from empty notes; got:\n{text}"
        );
    }

    #[tokio::test]
    async fn max_outer_iterations_zero_yields_max_iterations_with_no_passes() {
        // Edge: max_outer_iterations == 0 => no passes run, MaxIterationsExhausted.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);
        let backend = MockBackend::from_turns(vec![]);

        let config = RalphConfig::new("zero-objective", stop_never(), 0, 4);

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(report.terminal, RalphTerminal::MaxIterationsExhausted);
        assert_eq!(report.outer_iterations(), 0);
    }

    #[tokio::test]
    async fn inner_backend_error_is_recorded_and_loop_continues() {
        // An inner LoopOutcome::BackendError is RECORDED on the iteration
        // outcome and the outer loop CONTINUES (overnight-resilience over
        // fail-fast), bottoming out on MaxIterationsExhausted.
        let root = TempDir::new().expect("tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        git_init(&root_path);
        let ctx = ctx_for(&root_path);

        // Every inner turn is a BackendError(Terminal). The inner run surfaces
        // BackendError after one turn; no changes; no commit; the stuck
        // counter increments. With stuck_k large, the loop runs to the cap.
        let backend = MockBackend::new(vec![
            Err(BackendError::Terminal {
                kind: TerminalKind::Other,
                message: "boom".to_string(),
            }),
            Err(BackendError::Terminal {
                kind: TerminalKind::Other,
                message: "boom".to_string(),
            }),
        ]);

        let config =
            RalphConfig::new("backend-error-objective", stop_never(), 2, 4).with_stuck_k(100); // disable stuck so max-iter wins

        let report = run_ralph(&backend, &ctx, &config).await;
        assert_eq!(
            report.terminal,
            RalphTerminal::MaxIterationsExhausted,
            "a recorded inner BackendError must NOT terminate the outer loop"
        );
        assert_eq!(report.outer_iterations(), 2);
        for it in &report.iterations {
            assert!(
                matches!(it.inner_outcome, LoopOutcome::BackendError(_)),
                "inner BackendError must be recorded on the iteration outcome; got {:?}",
                it.inner_outcome
            );
            assert!(!it.committed, "a BackendError pass makes no changes");
        }
    }

    /// Make a path executable on Unix (best-effort; tests run on Linux).
    fn make_executable(path: &std::path::Path) {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mut perm = meta.permissions();
            perm.set_mode(0o755);
            let _ = std::fs::set_permissions(path, perm);
        }
    }

    /// Sanity: the public types are Debug as the ACs pin (and `RalphTerminal`
    /// is PartialEq/Eq so tests assert on it directly).
    #[test]
    fn ralph_types_derive_as_pinned() {
        let terminal = RalphTerminal::StopConditionMet;
        assert_eq!(terminal, RalphTerminal::StopConditionMet);
        let _ = format!("{terminal:?}");
        let err = RalphTerminal::Error("e".to_string());
        assert_eq!(err, RalphTerminal::Error("e".to_string()));
        // Clone is available on the config and terminal.
        let _ = terminal.clone();

        let config = RalphConfig::new("o", stop_never(), 1, 1);
        let _cloned: RalphConfig = config.clone();
        let _ = format!("{config:?}");
    }

    /// `RalphReport` aggregate accessors over an empty report.
    #[test]
    fn ralph_report_accessors_on_empty() {
        let report = RalphReport {
            objective: "o".to_string(),
            terminal: RalphTerminal::MaxIterationsExhausted,
            iterations: vec![],
        };
        assert_eq!(report.outer_iterations(), 0);
        assert_eq!(report.total_inner_iterations(), 0);
    }
}
