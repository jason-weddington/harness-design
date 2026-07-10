//! Eval harness: the measuring rig wrapped around the agent loop.
//!
//! An [`EvalTask`] is a task prompt plus a predicate over the loop's terminal
//! [`LoopOutcome`] that decides whether a trial passed. [`run_eval`] runs `k`
//! independent trials against a [`model::ModelBackend`] and reports the count
//! of passes and the derived rate as an [`EvalReport`].
//!
//! ## Why this exists
//!
//! The point is to make every future loop / model / prompt change *measurable*:
//! once a task has a predicate and a baseline pass-rate at `k`, any
//! regression shows up as a number, not a feeling.
//!
//! ## Pass^k framing
//!
//! `EvalReport::passes` is the count of trials that satisfied the predicate
//! out of `EvalReport::trials = k`; `EvalReport::pass_rate = passes / k` is
//! the comparable number when `k` changes between runs. The name "pass^k" is
//! field-of-evals shorthand for "we ran k trials and counted passes" — that
//! is exactly the math here.
//!
//! ## System prompt sourcing
//!
//! The `EvalTask` deliberately does NOT carry a `system` field: the canonical
//! system prompt is now rendered by [`crate::prompt`] inside the engine, from
//! the registered tool set (plus the check command, when configured). Every
//! trial and every real run render the same prompt from the same source of
//! truth — a wording drift between eval and prod is impossible.
//!
//! ## Per-trial workspace isolation
//!
//! Code-editing trials are **stateful** — an `edit_file` in trial 1 must not be
//! visible to trial 2. So [`run_eval`] takes a per-trial *environment factory*
//! ([`TrialEnv`]) and calls it once per trial: each trial gets a FRESH workspace
//! (a recursive copy of a source dir into a new scratch dir via
//! [`copy_dir_recursive`]), a fresh offload dir, a fresh
//! [`ToolCtx`](crate::tool::ToolCtx), and a fresh
//! [`ToolRegistry`](crate::tool::ToolRegistry) whose checks run in that trial's
//! own workspace. Trials run **sequentially** — no parallelism, which keeps the
//! model simple and is friendly to provider rate limits.
//!
//! ## What this is NOT (deferred)
//!
//! - Statistical rigor beyond pass^k counting, multi-provider / model-routing
//!   eval, multi-fixture suites, git-worktree isolation (plain dir copy is the
//!   v1 mechanism — fixtures aren't repos), and CI wiring of the live eval all
//!   land later.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::engine::{self, LoopOutcome, RunConfig, RunResult, RunStats};
use crate::exec::{CheckCommand, ChecksRunner};
use crate::model::ModelBackend;
use crate::run_record::{Disposition, Verification};
use crate::tool::{ToolCtx, ToolRegistry};
use crate::tools::standard_registry;
use crate::workspace::{DiskOffloadSink, Workspace};

/// Generous timeout for the coding-fix eval's `cargo test`: the first compile of
/// a fresh crate copy is slow on a small host, so allow a wide margin.
const CODING_CHECK_TIMEOUT: Duration = Duration::from_mins(3);

/// Predicate that judges whether a single trial's [`LoopOutcome`] is a pass.
///
/// A boxed `Fn` rather than a generic parameter keeps [`EvalTask`] storable in
/// a `Vec` and constructible by helpers like [`finish_task`]. The `Send +
/// Sync` bounds match what backends already require so the predicate doesn't
/// pin the future to a single thread.
pub type SuccessPredicate = Box<dyn Fn(&LoopOutcome) -> bool + Send + Sync>;

/// One eval task: the seed prompt plus how to judge whether it passed.
///
/// All fields are public — the helper [`finish_task`] is just a constructor,
/// and downstream callers are expected to build their own tasks with struct
/// literals. There is no `Debug` impl because [`SuccessPredicate`] is opaque.
pub struct EvalTask {
    /// Human-readable label; flows through to [`EvalReport::task_name`].
    pub name: String,
    /// The seed user message: the actual task description handed to the agent.
    /// The canonical system prompt is rendered by [`crate::prompt`] inside
    /// the engine — see the module docs.
    pub task: String,
    /// Predicate over the terminal [`LoopOutcome`] that decides pass/fail.
    pub success: SuccessPredicate,
}

/// One trial's per-trial detail: which trial, whether it passed the task's
/// success predicate, and the mechanical [`RunStats`] the engine accumulated
/// alongside its terminal [`LoopOutcome`].
///
/// This is what [`EvalReport::trial_results`] carries — the per-trial fidelity
/// the aggregate `passes / trials` collapses away. A comparable-across-runs
/// number (`pass_rate`) stays on [`EvalReport`]; the per-trial detail lets
/// callers compare backends/models on iterations and tokens spent, not just
/// pass/fail.
///
/// Not `PartialEq` — inherited from [`LoopOutcome`], whose
/// [`crate::model::BackendError`] variant is a runtime error that doesn't
/// compare.
#[derive(Debug)]
pub struct TrialResult {
    /// Zero-based trial index — matches the `i` in `run_eval`'s trial loop.
    pub trial: u32,
    /// Whether [`EvalTask::success`] accepted this trial's terminal outcome.
    pub passed: bool,
    /// Result of the out-of-band holdout re-gate run after the trial loop
    /// terminated. `None` means no holdout re-gate ran (either the fixture had
    /// no top-level `holdout/` dir, or the trial env carried no
    /// [`ChecksRunner`]). `Some(true)` / `Some(false)` is the re-gate outcome.
    pub holdout_passed: Option<bool>,
    /// The engine's terminal [`LoopOutcome`] for this trial.
    pub outcome: LoopOutcome,
    /// Mechanical stats the engine accumulated over this trial's run.
    pub stats: RunStats,
}

/// The result of an eval run. "Pass^k" framing: `passes` out of `trials = k`,
/// plus `pass_rate = passes / trials` for cross-`k` comparison.
///
/// The per-trial detail — one [`TrialResult`] per trial, in order — is
/// preserved on [`Self::trial_results`]. Aggregate views over the per-trial
/// stats are exposed as accessor methods
/// ([`Self::mean_iterations`], [`Self::min_iterations`],
/// [`Self::max_iterations`], [`Self::total_input_tokens`],
/// [`Self::total_output_tokens`], [`Self::total_wall_clock`]) rather than
/// baked-in fields, so the source of truth is one place (`trial_results`).
///
/// Not `PartialEq` — inherited from [`TrialResult`] and [`LoopOutcome`], whose
/// [`crate::model::BackendError`] variant doesn't compare.
#[derive(Debug)]
pub struct EvalReport {
    /// Copy of [`EvalTask::name`] — so a report can be printed in isolation.
    pub task_name: String,
    /// `k`: the number of independent trials that were run.
    pub trials: u32,
    /// How many trials' outcomes satisfied [`EvalTask::success`].
    pub passes: u32,
    /// `passes / trials` as an f64. `0.0` when `trials == 0` (no division by
    /// zero in the report).
    pub pass_rate: f64,
    /// Per-trial detail — one entry per trial, in order (trial index matches
    /// the vec position). Aggregate accessors on this struct derive from this
    /// vec, so it stays the single source of truth.
    pub trial_results: Vec<TrialResult>,
}

impl EvalReport {
    /// Mean iterations across every trial. Returns `0.0` for an empty report
    /// (no trials) so the accessor never yields `NaN`.
    #[must_use]
    pub fn mean_iterations(&self) -> f64 {
        if self.trial_results.is_empty() {
            return 0.0;
        }
        let sum: u64 = self
            .trial_results
            .iter()
            .map(|t| u64::from(t.stats.iterations))
            .sum();
        // `u64` → `f64` and `usize` → `f64`: an eval with hundreds of trials
        // can't approach `f64`'s precision limit; the pedantic-clippy-clean
        // form `as` for the divisor is fine here for a stats mean.
        #[allow(clippy::cast_precision_loss)]
        {
            sum as f64 / self.trial_results.len() as f64
        }
    }

    /// Minimum iterations across every trial. `None` when no trials ran.
    #[must_use]
    pub fn min_iterations(&self) -> Option<u32> {
        self.trial_results.iter().map(|t| t.stats.iterations).min()
    }

    /// Maximum iterations across every trial. `None` when no trials ran.
    #[must_use]
    pub fn max_iterations(&self) -> Option<u32> {
        self.trial_results.iter().map(|t| t.stats.iterations).max()
    }

    /// Sum of every trial's `input_tokens`.
    #[must_use]
    pub fn total_input_tokens(&self) -> u64 {
        self.trial_results
            .iter()
            .map(|t| t.stats.input_tokens)
            .sum()
    }

    /// Sum of every trial's `output_tokens`.
    #[must_use]
    pub fn total_output_tokens(&self) -> u64 {
        self.trial_results
            .iter()
            .map(|t| t.stats.output_tokens)
            .sum()
    }

    /// Sum of every trial's `wall_clock`.
    #[must_use]
    pub fn total_wall_clock(&self) -> Duration {
        self.trial_results.iter().map(|t| t.stats.wall_clock).sum()
    }

    /// Count of trials where the holdout re-gate ran and passed
    /// (`holdout_passed == Some(true)`). Returns `0` for an empty report.
    ///
    /// # Examples
    ///
    /// ```
    /// # use harness::eval::EvalReport;
    /// let report = EvalReport {
    ///     task_name: "t".to_string(),
    ///     trials: 0,
    ///     passes: 0,
    ///     pass_rate: 0.0,
    ///     trial_results: Vec::new(),
    /// };
    /// assert_eq!(report.holdout_passes(), 0);
    /// ```
    #[must_use]
    pub fn holdout_passes(&self) -> u32 {
        self.trial_results
            .iter()
            .filter(|t| t.holdout_passed == Some(true))
            .map(|_| 1u32)
            .sum()
    }

    /// Count of trials where the self-gate passed (`passed == true`) but the
    /// holdout re-gate failed (`holdout_passed == Some(false)`). These are
    /// trials the agent claimed done and the in-workspace gate accepted, yet
    /// the sealed holdout exposed as incomplete. Returns `0` for an empty
    /// report.
    ///
    /// # Examples
    ///
    /// ```
    /// # use harness::eval::EvalReport;
    /// let report = EvalReport {
    ///     task_name: "t".to_string(),
    ///     trials: 0,
    ///     passes: 0,
    ///     pass_rate: 0.0,
    ///     trial_results: Vec::new(),
    /// };
    /// assert_eq!(report.false_dones(), 0);
    /// ```
    #[must_use]
    pub fn false_dones(&self) -> u32 {
        self.trial_results
            .iter()
            .filter(|t| t.passed && t.holdout_passed == Some(false))
            .map(|_| 1u32)
            .sum()
    }
}

/// One trial's fresh, isolated execution environment.
///
/// [`run_eval`] obtains one of these per trial from its `env_factory`, so a
/// stateful trial (one that edits files) never leaks writes into the next trial.
/// The `_scratch` dirs are held for the life of the value and deleted on drop —
/// the trial's workspace is torn down as soon as the trial ends.
///
/// There is no `Debug` impl: [`ToolCtx`] wraps a `dyn` offload sink.
pub struct TrialEnv {
    /// The trial's tool registry (wired to `ctx`'s workspace).
    pub tools: ToolRegistry,
    /// The trial's run context: its own workspace + offload sink.
    pub ctx: ToolCtx,
    /// The checks the harness re-runs to verify a `finish(done)` claim, with a
    /// cwd pointing at this trial's own workspace. `None` disables verification
    /// (a `finish(done)` then yields [`Verification::NoChecksConfigured`]).
    pub checks: Option<ChecksRunner>,
    /// The fixture's `holdout/` source dir to copy into the trial workspace
    /// after the agent loop terminates, so the out-of-band holdout re-gate can
    /// run. `None` when the fixture has no top-level `holdout/` directory (the
    /// common case for legacy fixtures; holdout re-gate is skipped entirely).
    pub holdout_src: Option<PathBuf>,
    /// Scratch dirs kept alive for the trial and removed on drop. Private
    /// because callers never inspect it — they build a `TrialEnv` via a factory.
    _scratch: Vec<ScratchDir>,
}

/// Run `task` `k` independent times and report how many trials the
/// [`EvalTask::success`] predicate accepted.
///
/// `env_factory` is invoked **once per trial** to produce a fresh, isolated
/// [`TrialEnv`] (see the module docs on per-trial isolation); the trial then
/// runs [`engine::run`] with a [`RunConfig`] carrying the task's seed prompt,
/// the given `max_iterations`, and the env's checks. Trials are **sequential**.
///
/// `on_trial` is called with each trial's [`TrialResult`] as it completes —
/// the live example uses it to stream per-trial one-liners that include the
/// [`RunStats`]. Pass `|_| {}` to ignore it. The full per-trial vec is also
/// preserved on [`EvalReport::trial_results`] for downstream aggregate work.
///
/// `pass_rate` is `0.0` when `k == 0` so a misuse never produces `NaN`; the
/// trial loop runs zero times, so neither the factory nor the backend is
/// touched in that case.
///
/// # Panics
/// Panics if the holdout re-gate copy fails (broken-host semantics — the trial
/// workspace or holdout source directory is unreadable/unwritable after the
/// agent loop ran). This condition indicates a broken host, not a recoverable
/// eval failure.
pub async fn run_eval(
    task: &EvalTask,
    backend: &impl ModelBackend,
    env_factory: impl Fn() -> TrialEnv,
    k: u32,
    max_iterations: u32,
    mut on_trial: impl FnMut(&TrialResult),
) -> EvalReport {
    let mut passes: u32 = 0;
    let mut trial_results: Vec<TrialResult> = Vec::with_capacity(k as usize);
    for i in 0..k {
        let env = env_factory();
        let mut config = RunConfig::new(task.task.clone(), max_iterations);
        if let Some(checks) = env.checks.clone() {
            config = config.with_checks(checks);
        }
        let RunResult { outcome, stats } =
            engine::run(backend, &env.tools, &env.ctx, &config).await;
        // Bind the per-trial boolean to a name distinct from `passes` so
        // clippy's `similar_names` lint has no bait; the struct field stays
        // `passed` on the way in.
        let is_pass = (task.success)(&outcome);
        if is_pass {
            passes += 1;
        }
        // Holdout re-gate: copy the sealed holdout dir into the trial workspace
        // and re-run the gate out-of-band.  Runs BEFORE TrialResult is
        // constructed and BEFORE on_trial is called, so the callback always
        // observes the final holdout_passed value.  Skipped (→ None) when the
        // fixture had no holdout/ dir or the env carries no ChecksRunner.
        let holdout_passed =
            if let (Some(holdout_src), Some(checks)) = (&env.holdout_src, &env.checks) {
                copy_dir_recursive(holdout_src, env.ctx.workspace().root())
                    .expect("copy holdout/ contents into trial workspace");
                let report = checks.run(&env.ctx).await;
                Some(report.passed)
            } else {
                None
            };
        let trial = TrialResult {
            trial: i,
            passed: is_pass,
            holdout_passed,
            outcome,
            stats,
        };
        on_trial(&trial);
        trial_results.push(trial);
        // `env` — and with it this trial's scratch workspace — drops here.
    }
    let pass_rate = if k == 0 {
        0.0
    } else {
        // u32 → f64 is lossless; f64::from is the pedantic-clippy-clean form.
        f64::from(passes) / f64::from(k)
    };
    EvalReport {
        task_name: task.name.clone(),
        trials: k,
        passes,
        pass_rate,
        trial_results,
    }
}

/// An owned scratch directory that deletes itself on drop.
///
/// A tiny, std-only stand-in for `tempfile::TempDir` so the *library* carries no
/// extra runtime dependency (`tempfile` stays a dev-dependency, used only by
/// tests). Uniqueness comes from the process id plus a monotonic counter, so two
/// scratch dirs never collide within a process.
#[derive(Debug)]
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// Create a fresh, uniquely-named directory under the system temp dir.
    fn new(prefix: &str) -> std::io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("harness-eval-{prefix}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    /// The directory's path.
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        // Best-effort teardown: a failed cleanup must never panic a trial.
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// List every immediate subdirectory of `fixtures_root`, sorted by path.
///
/// The multi-fixture coding eval calls this to enumerate every fixture crate
/// under `fixtures/` in a deterministic order — one eval per directory. Only
/// directory entries are returned; stray files (`README.md`, `.gitignore`,
/// etc.) at that level are ignored. The sort is by full path, which — because
/// every returned entry lives under the same parent — is equivalent to a sort
/// by directory name.
///
/// # Errors
/// Propagates `std::fs` errors from `read_dir` and per-entry metadata reads.
pub fn discover_fixtures(fixtures_root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(fixtures_root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// Recursively copy the contents of `src` into `dst`, creating `dst` (and any
/// nested subdirectories) as needed.
///
/// Plain `std::fs`, copying every entry — it skips nothing, which is correct for
/// eval fixtures (self-contained cargo packages with no `.git` or build
/// artifacts). Files are copied byte-for-byte; directories recurse.
///
/// # Errors
/// Propagates any `std::fs` error (unreadable source, unwritable destination,
/// etc.).
pub fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let dst_path = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

/// The trivial-smoke-test trial environment: the standard v1 toolset, a stub
/// context, and NO checks. Used by [`finish_task`] runs, where a `finish(done)`
/// is accepted on trust ([`Verification::NoChecksConfigured`]).
///
/// This is a plain `fn` so it can be passed directly as `run_eval`'s
/// `env_factory` (a `fn` item implements `Fn`).
#[must_use]
pub fn finish_env() -> TrialEnv {
    TrialEnv {
        tools: standard_registry(None),
        ctx: ToolCtx::stub(),
        checks: None,
        holdout_src: None,
        _scratch: Vec::new(),
    }
}

/// Copy the contents of a fixture source dir into a trial workspace, excluding
/// four entries AT THE FIXTURE ROOT ONLY: the eval-only `task.json` and
/// `holdout/` (sealed from the agent under eval), plus the on-disk build
/// artifacts `target/` and `Cargo.lock` (running `cargo test` inside a
/// committed fixture leaves both behind; copying them would bloat every trial
/// workspace and leak host build state). Entries nested deeper in the tree
/// (e.g. `src/task.json`) are copied verbatim — the exclusion is scoped to the
/// top level.
///
/// # Errors
/// Propagates any `std::fs` error (unreadable source, unwritable destination).
fn copy_fixture_into_workspace(fixture_src: &Path, workspace_root: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(workspace_root)?;
    for entry in std::fs::read_dir(fixture_src)? {
        let entry = entry?;
        let name = entry.file_name();
        // Skip top-level task.json (eval-only spec), holdout/ (sealed holdout
        // dir — the agent under eval must never see it), and local build
        // artifacts (target/, Cargo.lock).
        if name == "task.json" || name == "holdout" || name == "target" || name == "Cargo.lock" {
            continue;
        }
        let dst_path = workspace_root.join(&name);
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

/// Build one fresh coding-fix trial environment: copy the fixture into a new
/// scratch workspace (excluding top-level `task.json` and `holdout/`), wire a
/// fresh offload dir + [`ToolCtx`], and a `cargo test` [`ChecksRunner`] whose
/// cwd IS this trial's workspace.
///
/// # Panics
/// Panics if the scratch dirs cannot be created, the fixture cannot be copied,
/// or the workspace roots cannot be canonicalized — all of which indicate a
/// broken host environment, not a recoverable eval condition.
fn build_coding_env(fixture_src: &Path) -> TrialEnv {
    let workspace_scratch =
        ScratchDir::new("workspace").expect("create trial workspace scratch dir");
    copy_fixture_into_workspace(fixture_src, workspace_scratch.path())
        .expect("copy fixture into trial workspace");
    let offload_scratch = ScratchDir::new("offload").expect("create trial offload scratch dir");

    let workspace = Workspace::new(
        workspace_scratch.path(),
        Some(offload_scratch.path().to_path_buf()),
    )
    .expect("trial workspace roots are freshly-created dirs");
    // Canonicalized root — this is where the copied fixture lives and where the
    // checks (and file tools) operate.
    let workspace_root = workspace.root().to_path_buf();
    let offload_canon = offload_scratch
        .path()
        .canonicalize()
        .expect("offload scratch dir canonicalizes");

    let ctx = ToolCtx::new(
        Arc::new(workspace),
        Arc::new(DiskOffloadSink::new(offload_canon)),
    );

    let checks = ChecksRunner::new(
        CheckCommand {
            program: "cargo".to_string(),
            args: vec!["test".to_string()],
        },
        workspace_root,
        CODING_CHECK_TIMEOUT,
    );
    let tools = standard_registry(Some(checks.clone()));

    // Set holdout_src iff the fixture carries a top-level `holdout/` dir —
    // after the agent loop the contents will be merged into the workspace for
    // the out-of-band holdout re-gate.
    let holdout_src = {
        let d = fixture_src.join("holdout");
        d.is_dir().then_some(d)
    };

    TrialEnv {
        tools,
        ctx,
        checks: Some(checks),
        holdout_src,
        _scratch: vec![workspace_scratch, offload_scratch],
    }
}

/// The coding-fix eval: prove the harness can autonomously FIX a failing test in
/// a real (tiny) Rust crate, where "done" is HARNESS-VERIFIED (`cargo test` came
/// back green), not model-claimed.
///
/// Returns the [`EvalTask`] plus a per-trial environment factory (pass it
/// straight to [`run_eval`]). Each factory call copies `fixture_src` into a
/// fresh scratch workspace (excluding top-level `task.json` and `holdout/`) and
/// wires a `cargo test` [`ChecksRunner`] rooted there — so a trial's edits are
/// isolated and its verification is real.
///
/// **Prompt routing:** if `fixture_src/task.json` exists it is deserialized into
/// a [`crate::task_spec::TaskSpec`] and the prompt is rendered via
/// [`crate::prompt::render_task_prompt_from_spec`] (the production path); otherwise
/// the legacy literal prompt is used. Both routes share the same success
/// predicate.
///
/// The success predicate accepts **only** a
/// [`LoopOutcome::Finished`]`(`[`FinishDisposition::Done`]`)` whose verification
/// is [`Verification::Checks`] with `report.passed == true`. The vacuous
/// [`Verification::NoChecksConfigured`] path does NOT count — an unverified
/// `done` claim is a failure here, by design.
///
/// # Panics
/// Panics if `fixture_src/task.json` exists but cannot be read or is not valid
/// [`crate::task_spec::TaskSpec`] JSON — a broken fixture is a broken host.
pub fn coding_fix_task(fixture_src: &Path) -> (EvalTask, impl Fn() -> TrialEnv) {
    let task_json_path = fixture_src.join("task.json");
    let task_prompt = if task_json_path.exists() {
        let json_str = std::fs::read_to_string(&task_json_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", task_json_path.display()));
        let spec: crate::task_spec::TaskSpec = serde_json::from_str(&json_str)
            .unwrap_or_else(|e| panic!("parse {} as TaskSpec: {e}", task_json_path.display()));
        crate::prompt::render_task_prompt_from_spec(&spec)
    } else {
        "The test suite in this Rust crate fails. Find the bug, fix it, \
               and make the tests pass."
            .to_string()
    };

    let task = EvalTask {
        name: "coding_fix".to_string(),
        task: task_prompt,
        success: Box::new(|outcome| {
            matches!(
                outcome,
                LoopOutcome::Finished(Disposition::Done {
                    verification: Verification::Checks(report),
                    ..
                }) if report.passed
            )
        }),
    };

    let fixture_src = fixture_src.to_path_buf();
    let factory = move || build_coding_env(&fixture_src);
    (task, factory)
}

/// The simplest possible eval task: prove the agent can invoke the `finish`
/// tool with `disposition="done"` to terminate a run.
///
/// This is the smoke-test rung: any working loop + backend + finish-tool
/// wiring should hit `pass_rate ≈ 1.0` here. The predicate matches
/// **any** [`LoopOutcome::Finished`] carrying [`FinishDisposition::Done`] —
/// including the `NoChecksConfigured` variant, which is what the eval yields
/// today (no [`crate::exec::ChecksRunner`] is wired). Every other terminal
/// outcome (blocked, failed, hit max iterations, stopped without finishing,
/// backend error) is a failure.
#[must_use]
pub fn finish_task() -> EvalTask {
    EvalTask {
        name: "finish".to_string(),
        task: "Acknowledge and finish.".to_string(),
        success: Box::new(|outcome| {
            matches!(outcome, LoopOutcome::Finished(Disposition::Done { .. }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EvalReport, EvalTask, TrialEnv, TrialResult, build_coding_env, coding_fix_task,
        copy_dir_recursive, discover_fixtures, finish_env, finish_task, run_eval,
    };
    use crate::engine::{FINISH_TOOL_NAME, FinishTool, LoopOutcome, RunStats};
    use crate::exec::{CheckCommand, ChecksRunner};
    use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
    use crate::run_record::{Disposition, FailureMode, Verification};
    use crate::test_support::MockBackend;
    use crate::tool::{EchoTool, ToolCtx, ToolRegistry};
    use crate::tools::standard_registry;
    use crate::workspace::{DiskOffloadSink, Workspace};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tempfile::tempdir;

    /// A trial-env factory (a plain `fn`) for the finish smoke tests: the
    /// echo+finish registry and a stub context, no checks. Passable directly as
    /// `run_eval`'s `env_factory`.
    fn echo_finish_env() -> TrialEnv {
        TrialEnv {
            tools: registry(),
            ctx: ToolCtx::stub(),
            checks: None,
            holdout_src: None,
            _scratch: Vec::new(),
        }
    }

    /// A tool-use turn that CREATEs `path` with `contents` via `edit_file`.
    fn edit_create_turn(call_id: &str, path: &str, contents: &str) -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: call_id.to_string(),
                name: "edit_file".to_string(),
                input: json!({ "path": path, "old_string": "", "new_string": contents }),
            })],
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    fn usage() -> Usage {
        Usage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        }
    }

    fn finish_done_turn() -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: json!({ "disposition": "done", "summary": "ok" }),
            })],
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    /// A tool-use turn that calls `echo` (not finish), so the loop keeps
    /// drawing turns until it either finishes or hits `max_iterations`.
    fn non_finish_turn() -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-echo".to_string(),
                name: "echo".to_string(),
                input: json!({}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: usage(),
        }
    }

    fn registry() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register("echo", Arc::new(EchoTool));
        r.register(FINISH_TOOL_NAME, Arc::new(FinishTool));
        r
    }

    #[tokio::test]
    async fn always_finishing_script_yields_pass_rate_one() {
        // k trials × 1 turn each (the finish call terminates immediately).
        let k: u32 = 5;
        let script: Vec<AssistantTurn> = (0..k).map(|_| finish_done_turn()).collect();
        let backend = MockBackend::from_turns(script);
        let task = finish_task();

        let report = run_eval(&task, &backend, echo_finish_env, k, 10, |_| {}).await;

        assert_eq!(report.task_name, task.name);
        assert_eq!(report.trials, k);
        assert_eq!(report.passes, k);
        // f64::from(u32) is lossless; an exact equality is safe here.
        assert!((report.pass_rate - 1.0).abs() < f64::EPSILON);
        // And the backend was called exactly once per trial.
        assert_eq!(backend.calls(), k);
    }

    #[tokio::test]
    async fn never_finishing_script_yields_pass_rate_zero() {
        // k trials × max_iter turns of `echo` (no finish), so every trial
        // terminates as LoopOutcome::MaxIterations — never Finished(Done).
        let k: u32 = 3;
        let max_iter: u32 = 2;
        let script: Vec<AssistantTurn> = (0..(k * max_iter)).map(|_| non_finish_turn()).collect();
        let backend = MockBackend::from_turns(script);
        let task = finish_task();

        let report = run_eval(&task, &backend, echo_finish_env, k, max_iter, |_| {}).await;

        assert_eq!(report.task_name, "finish");
        assert_eq!(report.trials, k);
        assert_eq!(report.passes, 0);
        assert!((report.pass_rate - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn mixed_script_yields_correct_fraction() {
        // 4 trials, alternating success / failure. Failure here is a plain
        // tool_use that isn't finish — each such trial hits max_iterations
        // after `max_iter` draws.
        let max_iter: u32 = 1;
        // Alternating turn 0..3 — comment per element documents what each
        // trial does so the expected pass count is self-evident.
        let script: Vec<AssistantTurn> = vec![
            finish_done_turn(), // trial 1: 1 turn, finishes
            non_finish_turn(),  // trial 2: 1 turn, hits max_iter
            finish_done_turn(), // trial 3: 1 turn, finishes
            non_finish_turn(),  // trial 4: 1 turn, hits max_iter
        ];
        let backend = MockBackend::from_turns(script);
        let task = finish_task();

        let k: u32 = 4;
        let report = run_eval(&task, &backend, echo_finish_env, k, max_iter, |_| {}).await;

        assert_eq!(report.trials, k);
        assert_eq!(report.passes, 2);
        assert!((report.pass_rate - 0.5).abs() < f64::EPSILON);

        // Per-trial detail: one entry per trial in order, with correct
        // passed-flag alternation and the trial index matching the vec
        // position.
        assert_eq!(report.trial_results.len(), k as usize);
        let flags: Vec<bool> = report.trial_results.iter().map(|t| t.passed).collect();
        assert_eq!(flags, vec![true, false, true, false]);
        for (i, t) in report.trial_results.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let expected: u32 = i as u32;
            assert_eq!(t.trial, expected, "trial index matches vec position");
        }
    }

    #[tokio::test]
    async fn k_zero_short_circuits_and_yields_zero_pass_rate() {
        // No trials run → backend is never called and pass_rate is 0.0
        // (not NaN). The empty script proves the loop body never executes.
        let backend = MockBackend::from_turns(vec![]);
        let task = finish_task();

        let report = run_eval(&task, &backend, echo_finish_env, 0, 5, |_| {}).await;

        assert_eq!(report.trials, 0);
        assert_eq!(report.passes, 0);
        assert!((report.pass_rate - 0.0).abs() < f64::EPSILON);
        assert_eq!(backend.calls(), 0, "k=0 must not touch the backend");
    }

    #[test]
    fn finish_task_predicate_accepts_only_finished_done() {
        // The whole point of the finish_task predicate: Done means pass, every
        // other terminal outcome means fail. Cover each non-Done outcome
        // explicitly so the contract is locked in.
        let task = finish_task();

        // NoChecksConfigured is the eval-path Done: verify it passes.
        assert!((task.success)(&LoopOutcome::Finished(Disposition::Done {
            summary: "ok".to_string(),
            verification: Verification::NoChecksConfigured,
        })));

        assert!(!(task.success)(&LoopOutcome::Finished(
            Disposition::Blocked {
                decision_needed: "which API?".to_string(),
            }
        )));
        assert!(!(task.success)(&LoopOutcome::Finished(
            Disposition::Failed {
                mode: FailureMode::Loop,
                summary: "tool errored".to_string(),
            }
        )));
        assert!(!(task.success)(&LoopOutcome::StoppedWithoutFinish));
        assert!(!(task.success)(&LoopOutcome::MaxIterations));
        // BackendError is also not a Done outcome.
        assert!(!(task.success)(&LoopOutcome::BackendError(
            crate::model::BackendError::Terminal {
                kind: crate::model::TerminalKind::Auth,
                message: "no creds".to_string(),
            }
        )));
    }

    #[test]
    fn finish_task_carries_non_empty_task_prompt() {
        let task = finish_task();
        assert_eq!(task.name, "finish");
        assert!(!task.task.is_empty());
    }

    #[test]
    fn eval_task_fields_are_publicly_constructible() {
        // A custom predicate via struct-literal construction — proves the
        // public-field shape works for non-built-in tasks.
        let task = EvalTask {
            name: "custom".to_string(),
            task: "do something".to_string(),
            success: Box::new(|outcome| matches!(outcome, LoopOutcome::MaxIterations)),
        };
        assert_eq!(task.name, "custom");
        assert!((task.success)(&LoopOutcome::MaxIterations));
        assert!(!(task.success)(&LoopOutcome::Finished(Disposition::Done {
            summary: String::new(),
            verification: Verification::NoChecksConfigured,
        })));
    }

    #[test]
    fn eval_report_is_debug_and_carries_trial_results() {
        // `EvalReport` no longer derives `Clone`/`PartialEq` — its
        // `trial_results` field carries a `LoopOutcome` whose `BackendError`
        // variant is a runtime error that doesn't compare. Debug is what a
        // caller actually needs for logging; the per-trial vec is what the
        // aggregate accessors derive from.
        let r = EvalReport {
            task_name: "finish".to_string(),
            trials: 3,
            passes: 1,
            pass_rate: 1.0 / 3.0,
            trial_results: Vec::new(),
        };
        let printed = format!("{r:?}");
        assert!(printed.contains("EvalReport"));
        assert!(printed.contains("task_name"));
        assert!(printed.contains("trial_results"));
    }

    /// A finish(done) turn with an explicit non-zero `Usage` — for tests that
    /// pin per-trial `RunStats` values through the eval harness.
    fn finish_done_turn_with_usage(input_tokens: u32, output_tokens: u32) -> AssistantTurn {
        AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: json!({ "disposition": "done", "summary": "ok" }),
            })],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens,
                output_tokens,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        }
    }

    #[tokio::test]
    async fn eval_report_aggregates_pin_mean_min_max_iterations_and_token_totals() {
        // Three trials, each drawing a KNOWN, distinct number of turns and
        // token counts. After the eval finishes we assert every aggregate
        // accessor on the report matches the hand-computed values exactly.
        //
        // Trial layout (each trial is one script prefix):
        //   trial 0: echo, echo, finish   → 3 iterations
        //     usage per turn:  (10, 1), (20, 2), (30, 3)
        //   trial 1: echo, finish         → 2 iterations
        //     usage per turn:  (40, 4), (50, 5)
        //   trial 2: finish               → 1 iteration
        //     usage per turn:  (60, 6)
        //
        // Totals:
        //   iterations: [3, 2, 1] → mean 2.0, min 1, max 3
        //   input_tokens per trial: [60, 90, 60] → total 210
        //   output_tokens per trial: [6, 9, 6]   → total 21
        let script = vec![
            // trial 0
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    input: json!({ "i": 1 }),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 1,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            },
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c2".to_string(),
                    name: "echo".to_string(),
                    input: json!({ "i": 2 }),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 20,
                    output_tokens: 2,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            },
            finish_done_turn_with_usage(30, 3),
            // trial 1
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c1".to_string(),
                    name: "echo".to_string(),
                    input: json!({ "i": 3 }),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 40,
                    output_tokens: 4,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            },
            finish_done_turn_with_usage(50, 5),
            // trial 2
            finish_done_turn_with_usage(60, 6),
        ];
        let backend = MockBackend::from_turns(script);
        let task = finish_task();

        // Callback receives every TrialResult in order — verify by recording.
        let seen: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_c = Arc::clone(&seen);
        let report = run_eval(
            &task,
            &backend,
            echo_finish_env,
            3,
            5,
            move |t: &TrialResult| {
                seen_c.lock().expect("seen lock").push(t.trial);
            },
        )
        .await;

        // pass_rate is unchanged shape — every trial passes here (finish tool
        // is available), so passes = 3, pass_rate = 1.0.
        assert_eq!(report.trials, 3);
        assert_eq!(report.passes, 3);
        assert!((report.pass_rate - 1.0).abs() < f64::EPSILON);

        // Per-trial detail: exact order, exact per-trial stats.
        assert_eq!(report.trial_results.len(), 3);
        assert_eq!(
            report
                .trial_results
                .iter()
                .map(|t| t.trial)
                .collect::<Vec<_>>(),
            vec![0, 1, 2],
        );
        assert_eq!(report.trial_results[0].stats.iterations, 3);
        assert_eq!(report.trial_results[0].stats.input_tokens, 60);
        assert_eq!(report.trial_results[0].stats.output_tokens, 6);
        assert_eq!(report.trial_results[1].stats.iterations, 2);
        assert_eq!(report.trial_results[1].stats.input_tokens, 90);
        assert_eq!(report.trial_results[1].stats.output_tokens, 9);
        assert_eq!(report.trial_results[2].stats.iterations, 1);
        assert_eq!(report.trial_results[2].stats.input_tokens, 60);
        assert_eq!(report.trial_results[2].stats.output_tokens, 6);

        // Aggregate accessors match the hand-computed values EXACTLY.
        assert!((report.mean_iterations() - 2.0).abs() < f64::EPSILON);
        assert_eq!(report.min_iterations(), Some(1));
        assert_eq!(report.max_iterations(), Some(3));
        assert_eq!(report.total_input_tokens(), 60 + 90 + 60);
        assert_eq!(report.total_output_tokens(), 6 + 9 + 6);
        // total_wall_clock sums the three per-trial durations. We can't pin an
        // exact value, but it must equal `sum(trial.stats.wall_clock)` — the
        // accessor's contract.
        let expected_total: Duration = report
            .trial_results
            .iter()
            .map(|t| t.stats.wall_clock)
            .sum();
        assert_eq!(report.total_wall_clock(), expected_total);

        // Callback ordering: every trial's index was observed, in order,
        // exactly once.
        assert_eq!(*seen.lock().expect("seen lock"), vec![0, 1, 2]);
    }

    #[test]
    fn eval_report_aggregate_accessors_handle_empty_trial_results() {
        // A k=0 (or otherwise empty) report is a valid, in-band state: the
        // aggregates must not `NaN`, panic, or `unwrap` — the same contract
        // `pass_rate = 0.0` already implements for the pass fraction.
        let empty = EvalReport {
            task_name: "empty".to_string(),
            trials: 0,
            passes: 0,
            pass_rate: 0.0,
            trial_results: Vec::new(),
        };
        assert!(
            (empty.mean_iterations() - 0.0).abs() < f64::EPSILON,
            "empty mean is 0.0, not NaN"
        );
        assert_eq!(empty.min_iterations(), None);
        assert_eq!(empty.max_iterations(), None);
        assert_eq!(empty.total_input_tokens(), 0);
        assert_eq!(empty.total_output_tokens(), 0);
        assert_eq!(empty.total_wall_clock(), Duration::ZERO);
    }

    #[test]
    fn trial_result_is_debug_and_carries_run_stats() {
        // TrialResult carries the engine's RunStats verbatim — pin the shape
        // via Debug so downstream logging never silently loses a field.
        let t = TrialResult {
            trial: 7,
            passed: true,
            holdout_passed: Some(false),
            outcome: LoopOutcome::MaxIterations,
            stats: RunStats {
                iterations: 4,
                input_tokens: 111,
                output_tokens: 22,
                wall_clock: Duration::from_millis(50),
            },
        };
        let printed = format!("{t:?}");
        assert!(printed.contains("TrialResult"));
        assert!(printed.contains("trial: 7"));
        assert!(printed.contains("passed: true"));
        assert!(printed.contains("holdout_passed: Some(false)"));
        assert!(printed.contains("MaxIterations"));
        assert!(printed.contains("RunStats"));
    }

    #[test]
    fn finish_env_wires_finish_tool_and_no_checks() {
        let env = finish_env();
        assert!(env.checks.is_none());
        assert!(
            env.tools.get(FINISH_TOOL_NAME).is_some(),
            "finish_env must register the finish tool"
        );
    }

    #[test]
    fn discover_fixtures_lists_subdirs_sorted_and_skips_files() {
        // Build a mini fixtures root: three fake fixture crates + one stray
        // file that must be skipped. Names are chosen so filesystem order is
        // unlikely to match sorted order — we're testing the sort.
        let root = tempdir().expect("root tempdir");
        std::fs::create_dir(root.path().join("zebra")).expect("mkdir zebra");
        std::fs::create_dir(root.path().join("alpha")).expect("mkdir alpha");
        std::fs::create_dir(root.path().join("middle")).expect("mkdir middle");
        std::fs::write(root.path().join("README.md"), "not a fixture\n").expect("write readme");

        let dirs = discover_fixtures(root.path()).expect("discover");
        let names: Vec<String> = dirs
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            vec![
                "alpha".to_string(),
                "middle".to_string(),
                "zebra".to_string()
            ],
            "directories must be returned sorted; stray files must be dropped"
        );
    }

    #[test]
    fn discover_fixtures_finds_the_four_committed_fixtures() {
        // The repo's real `fixtures/` root, discovered from this crate's
        // manifest dir. The four v1 fixtures (`broken-adder`, `interval-merge`,
        // `lru-cache`, `text-preview`) must be present, in dir form. The suite
        // grows over time, so extra fixtures are permitted — containment, not
        // exact equality (sortedness is pinned by the tempdir test above).
        let fixtures_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("fixtures");
        let dirs = discover_fixtures(&fixtures_root).expect("discover fixtures dir");
        let names: Vec<String> = dirs
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        for expected in [
            "broken-adder",
            "interval-merge",
            "lru-cache",
            "text-preview",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "v1 fixture `{expected}` missing from discovered set {names:?}",
            );
        }
    }

    #[test]
    fn copy_dir_recursive_round_trips_nested_structure() {
        let src = tempdir().expect("src tempdir");
        std::fs::write(src.path().join("top.txt"), "top").expect("write top");
        std::fs::create_dir_all(src.path().join("a/b")).expect("mkdir a/b");
        std::fs::write(src.path().join("a/one.txt"), "one").expect("write one");
        std::fs::write(src.path().join("a/b/two.txt"), "two").expect("write two");

        let dst = tempdir().expect("dst tempdir");
        let dst_root = dst.path().join("copy");
        copy_dir_recursive(src.path(), &dst_root).expect("recursive copy");

        assert_eq!(
            std::fs::read_to_string(dst_root.join("top.txt")).unwrap(),
            "top"
        );
        assert_eq!(
            std::fs::read_to_string(dst_root.join("a/one.txt")).unwrap(),
            "one"
        );
        assert_eq!(
            std::fs::read_to_string(dst_root.join("a/b/two.txt")).unwrap(),
            "two"
        );
    }

    /// A trivially-green `ChecksRunner` (`/bin/sh -c exit 0`) rooted at `cwd`, so
    /// a scripted `finish(done)` verifies fast without invoking cargo.
    fn green_checks(cwd: PathBuf) -> ChecksRunner {
        ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 0".to_string()],
            },
            cwd,
            Duration::from_secs(10),
        )
    }

    #[tokio::test]
    async fn each_trial_gets_a_fresh_workspace_copy() {
        // A source dir (with nested content) copied fresh into each trial.
        let src = tempdir().expect("src tempdir");
        std::fs::write(src.path().join("original.txt"), "seed").expect("write original");
        std::fs::create_dir(src.path().join("nested")).expect("mkdir nested");
        std::fs::write(src.path().join("nested/deep.txt"), "deep").expect("write deep");

        // A persistent holder so each trial's workspace survives past `run_eval`
        // for inspection (the recording factory below does NOT auto-clean).
        let holder = tempdir().expect("holder tempdir");
        let holder_path = holder.path().to_path_buf();
        let src_path = src.path().to_path_buf();
        let recorded: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
        let rec = Arc::clone(&recorded);

        let factory = move || {
            let idx = rec.lock().expect("rec lock").len();
            let ws_dir = holder_path.join(format!("trial-{idx}"));
            copy_dir_recursive(&src_path, &ws_dir).expect("copy source into trial workspace");
            let offload_dir = holder_path.join(format!("offload-{idx}"));
            std::fs::create_dir_all(&offload_dir).expect("mkdir offload");
            rec.lock().expect("rec lock").push(ws_dir.clone());

            let workspace = Workspace::new(&ws_dir, Some(offload_dir.clone())).expect("workspace");
            let offload_canon = offload_dir.canonicalize().expect("canon offload");
            let ctx = ToolCtx::new(
                Arc::new(workspace),
                Arc::new(DiskOffloadSink::new(offload_canon)),
            );
            let checks = green_checks(ws_dir.canonicalize().expect("canon ws"));
            let tools = standard_registry(Some(checks.clone()));
            TrialEnv {
                tools,
                ctx,
                checks: Some(checks),
                holdout_src: None,
                _scratch: Vec::new(),
            }
        };

        // Trial 1: create `marker.txt`, then finish(done) → checks green → Done.
        // Trial 2: finish(done) immediately. A shared reused workspace would let
        // trial 2 see trial 1's marker; fresh copies must not.
        let backend = MockBackend::from_turns(vec![
            edit_create_turn("c-edit", "marker.txt", "planted\n"),
            finish_done_turn(),
            finish_done_turn(),
        ]);
        // Reuse the real coding-fix predicate (Checks-verified Done).
        let (task, _ignored_factory) = coding_fix_task(src.path());

        let report = run_eval(&task, &backend, factory, 2, 3, |_| {}).await;
        assert_eq!(report.passes, 2, "both trials verify green");

        let dirs = recorded.lock().expect("rec lock").clone();
        assert_eq!(dirs.len(), 2, "factory invoked exactly once per trial");
        assert!(
            dirs[0].join("marker.txt").exists(),
            "trial 1 wrote its marker"
        );
        assert!(
            !dirs[1].join("marker.txt").exists(),
            "trial 2 got a FRESH copy — trial 1's marker must NOT be present"
        );
        // The fresh copy still carries the (recursively copied) source content.
        assert!(dirs[1].join("original.txt").exists());
        assert!(dirs[1].join("nested/deep.txt").exists());
    }

    #[test]
    fn coding_fix_task_env_copies_fixture_and_wires_cargo_test_checks() {
        // A stand-in fixture crate (never actually built here).
        let src = tempdir().expect("src tempdir");
        std::fs::write(
            src.path().join("Cargo.toml"),
            "[package]\nname = \"x\"\nedition = \"2024\"\n",
        )
        .expect("write Cargo.toml");
        std::fs::create_dir(src.path().join("src")).expect("mkdir src");
        std::fs::write(src.path().join("src/lib.rs"), "// stub\n").expect("write lib");

        let (task, factory) = coding_fix_task(src.path());
        assert_eq!(task.name, "coding_fix");
        assert!(task.task.to_lowercase().contains("fix"));

        let env = factory();
        // The fixture was copied into this trial's own workspace.
        let root = env.ctx.workspace().root().to_path_buf();
        assert!(
            root.join("Cargo.toml").exists(),
            "fixture Cargo.toml copied"
        );
        assert!(root.join("src/lib.rs").exists(), "fixture src copied");
        // Checks are `cargo test`, rooted at the trial workspace.
        let checks = env.checks.as_ref().expect("checks wired");
        assert_eq!(checks.command().program, "cargo");
        assert_eq!(checks.command().args, vec!["test".to_string()]);
        assert_eq!(
            checks.command_display(),
            "cargo test",
            "checks must be `cargo test`"
        );
        // run_checks is registered (the with-checks toolset).
        assert!(env.tools.get("run_checks").is_some());

        // A second invocation yields a DISTINCT, independent workspace.
        let env2 = factory();
        assert_ne!(
            env.ctx.workspace().root(),
            env2.ctx.workspace().root(),
            "each trial gets its own scratch workspace"
        );
    }

    #[tokio::test]
    async fn coding_fix_predicate_requires_harness_verified_done() {
        let (task, _factory) = coding_fix_task(Path::new("/unused-for-predicate-only"));

        // Accepts a Checks-verified GREEN Done — built via a REAL ChecksRunner on
        // a trivially-green command (no cargo, no network).
        let green = green_checks(PathBuf::from("/")).run(&ToolCtx::stub()).await;
        assert!(green.passed);
        assert!(
            (task.success)(&LoopOutcome::Finished(Disposition::Done {
                summary: "fixed".to_string(),
                verification: Verification::Checks(green),
            })),
            "a green Checks-verified Done must pass"
        );

        // Rejects the vacuous NoChecksConfigured Done — the whole point.
        assert!(
            !(task.success)(&LoopOutcome::Finished(Disposition::Done {
                summary: "claimed".to_string(),
                verification: Verification::NoChecksConfigured,
            })),
            "an unverified (NoChecksConfigured) Done must NOT count"
        );

        // Rejects a RED Checks Done (checks ran but failed).
        let red = ChecksRunner::new(
            CheckCommand {
                program: "/bin/sh".to_string(),
                args: vec!["-c".to_string(), "exit 1".to_string()],
            },
            PathBuf::from("/"),
            Duration::from_secs(10),
        )
        .run(&ToolCtx::stub())
        .await;
        assert!(!red.passed);
        assert!(
            !(task.success)(&LoopOutcome::Finished(Disposition::Done {
                summary: "lie".to_string(),
                verification: Verification::Checks(red),
            })),
            "a red Checks Done must NOT count"
        );

        // Rejects every non-Done terminal outcome.
        assert!(!(task.success)(&LoopOutcome::Finished(
            Disposition::Failed {
                mode: FailureMode::Loop,
                summary: "boom".to_string(),
            }
        )));
        assert!(!(task.success)(&LoopOutcome::Finished(
            Disposition::Blocked {
                decision_needed: "which?".to_string(),
            }
        )));
        assert!(!(task.success)(&LoopOutcome::MaxIterations));
        assert!(!(task.success)(&LoopOutcome::StoppedWithoutFinish));
    }

    // ── NEW TESTS: holdout mechanics, task.json routing, schema validation ──

    /// `build_coding_env` must exclude top-level `task.json` and `holdout/`
    /// from the workspace copy, but NOT exclude nested `src/task.json`.
    /// `holdout_src` must be `Some` iff the fixture has a top-level `holdout/`.
    #[test]
    fn build_coding_env_excludes_root_entries_but_not_nested() {
        let src = tempdir().expect("src tempdir");
        // Top-level task.json — must be excluded from workspace
        let valid_spec = r#"{
            "title": "T",
            "description": "D",
            "acceptance_criteria": [],
            "files_to_modify": [],
            "gate_command": "cargo test"
        }"#;
        std::fs::write(src.path().join("task.json"), valid_spec).expect("write task.json");
        // Top-level holdout/ — must be excluded from workspace
        std::fs::create_dir_all(src.path().join("holdout/tests")).expect("mkdir holdout/tests");
        std::fs::write(src.path().join("holdout/tests/holdout.rs"), "// holdout\n")
            .expect("write holdout.rs");
        // Regular source — must be present
        std::fs::create_dir_all(src.path().join("src")).expect("mkdir src");
        std::fs::write(src.path().join("src/lib.rs"), "// lib\n").expect("write lib.rs");
        // Nested src/task.json — must NOT be excluded (only root is excluded)
        std::fs::write(src.path().join("src/task.json"), "// nested\n")
            .expect("write src/task.json");
        // Top-level build artifacts — must be excluded from workspace
        std::fs::create_dir_all(src.path().join("target/debug")).expect("mkdir target/debug");
        std::fs::write(src.path().join("target/debug/junk.bin"), "junk\n").expect("write junk");
        std::fs::write(src.path().join("Cargo.lock"), "# lock\n").expect("write Cargo.lock");

        let env = build_coding_env(src.path());
        let root = env.ctx.workspace().root().to_path_buf();

        assert!(
            !root.join("task.json").exists(),
            "top-level task.json must be excluded from workspace"
        );
        assert!(
            !root.join("holdout").exists(),
            "top-level holdout/ must be excluded from workspace"
        );
        assert!(
            !root.join("target").exists(),
            "top-level target/ build dir must be excluded from workspace"
        );
        assert!(
            !root.join("Cargo.lock").exists(),
            "top-level Cargo.lock must be excluded from workspace"
        );
        assert!(
            root.join("src/lib.rs").exists(),
            "src/lib.rs must be present in workspace"
        );
        assert!(
            root.join("src/task.json").exists(),
            "nested src/task.json must NOT be excluded from workspace"
        );
        assert!(
            env.holdout_src.is_some(),
            "holdout_src must be Some when fixture has a holdout/ dir"
        );
    }

    /// Without a top-level `holdout/`, `holdout_src` must be `None`.
    #[test]
    fn build_coding_env_holdout_src_is_none_when_no_holdout_dir() {
        let src = tempdir().expect("src tempdir");
        std::fs::write(src.path().join("dummy.txt"), "x\n").expect("write dummy");
        let env = build_coding_env(src.path());
        assert!(
            env.holdout_src.is_none(),
            "holdout_src must be None when fixture has no holdout/ dir"
        );
    }

    /// `coding_fix_task` routes the prompt through `render_task_prompt_from_spec`
    /// when `task.json` exists, and uses the legacy literal otherwise.
    #[test]
    fn coding_fix_task_routes_prompt_by_task_json_presence() {
        // With task.json: prompt must contain the spec's title + an AC sentinel.
        let src_with = tempdir().expect("src with task.json");
        let spec_json = r#"{
            "title": "Sentinel Task Title",
            "description": "Does something.",
            "acceptance_criteria": ["sentinel-acceptance-criterion"],
            "files_to_modify": [],
            "gate_command": "cargo test"
        }"#;
        std::fs::write(src_with.path().join("task.json"), spec_json).expect("write task.json");
        let (task_with, _) = coding_fix_task(src_with.path());
        assert!(
            task_with.task.contains("Sentinel Task Title"),
            "spec-routed prompt must contain the spec title; got:\n{}",
            task_with.task,
        );
        assert!(
            task_with.task.contains("sentinel-acceptance-criterion"),
            "spec-routed prompt must contain the acceptance criterion; got:\n{}",
            task_with.task,
        );

        // Without task.json: must use the exact legacy literal.
        let src_without = tempdir().expect("src without task.json");
        let (task_without, _) = coding_fix_task(src_without.path());
        assert_eq!(
            task_without.task,
            "The test suite in this Rust crate fails. Find the bug, fix it, \
               and make the tests pass.",
            "without task.json the legacy literal must be used verbatim"
        );
    }

    /// A malformed (non-TaskSpec) `task.json` must cause a panic whose message
    /// names the fixture path (or the task.json path).
    #[test]
    #[should_panic(expected = "task.json")]
    fn coding_fix_task_panics_on_malformed_task_json() {
        let src = tempdir().expect("src tempdir");
        std::fs::write(src.path().join("task.json"), "{ not valid json {{{{")
            .expect("write malformed task.json");
        // `let _ =` discards the (never-reached) return value; the panic
        // propagates before any result is produced.
        let _ = coding_fix_task(src.path());
    }

    /// For a task.json-bearing fixture, `env.checks.command_display()` must
    /// still be `"cargo test"` — `spec.gate_command` is rendered into the
    /// prompt only, never parsed or executed.
    #[test]
    fn coding_fix_task_env_uses_cargo_test_for_spec_fixture() {
        let src = tempdir().expect("src tempdir");
        // gate_command is a different command to confirm it is NOT used.
        let spec_json = r#"{
            "title": "T",
            "description": "D",
            "acceptance_criteria": [],
            "files_to_modify": [],
            "gate_command": "cargo nextest run --workspace"
        }"#;
        std::fs::write(src.path().join("task.json"), spec_json).expect("write task.json");
        let (_, factory) = coding_fix_task(src.path());
        let env = factory();
        let checks = env.checks.as_ref().expect("checks must be wired");
        assert_eq!(
            checks.command_display(),
            "cargo test",
            "trial ChecksRunner must always be `cargo test`, never the spec's gate_command"
        );
    }

    /// Exact-count matrix for `holdout_passes()` and `false_dones()`.
    #[test]
    fn eval_report_holdout_accessor_exact_count_matrix() {
        // Build TrialResults covering all relevant combinations:
        //  (passed=true,  holdout=Some(true))  → holdout_passes: +1; false_dones: 0
        //  (passed=true,  holdout=Some(false)) → holdout_passes:  0; false_dones: +1
        //  (passed=false, holdout=Some(false)) → holdout_passes:  0; false_dones:  0
        //  (passed=false, holdout=Some(true))  → holdout_passes: +1; false_dones:  0
        //  (passed=true,  holdout=None)        → holdout_passes:  0; false_dones:  0
        let mk = |trial: u32, passed: bool, holdout_passed: Option<bool>| TrialResult {
            trial,
            passed,
            holdout_passed,
            outcome: LoopOutcome::MaxIterations,
            stats: RunStats {
                iterations: 0,
                input_tokens: 0,
                output_tokens: 0,
                wall_clock: Duration::ZERO,
            },
        };
        let report = EvalReport {
            task_name: "matrix".to_string(),
            trials: 5,
            passes: 3,
            pass_rate: 0.6,
            trial_results: vec![
                mk(0, true, Some(true)),
                mk(1, true, Some(false)),
                mk(2, false, Some(false)),
                mk(3, false, Some(true)),
                mk(4, true, None),
            ],
        };
        assert_eq!(
            report.holdout_passes(),
            2,
            "holdout_passes must count Some(true) regardless of passed"
        );
        assert_eq!(
            report.false_dones(),
            1,
            "false_dones must count passed==true AND holdout==Some(false)"
        );
    }

    /// Empty `EvalReport` → both new accessors return 0.
    #[test]
    fn eval_report_holdout_accessors_zero_for_empty_report() {
        let empty = EvalReport {
            task_name: "empty".to_string(),
            trials: 0,
            passes: 0,
            pass_rate: 0.0,
            trial_results: Vec::new(),
        };
        assert_eq!(empty.holdout_passes(), 0);
        assert_eq!(empty.false_dones(), 0);
    }

    /// Schema-validation: every `fixtures/*/task.json` must deserialize to
    /// `TaskSpec`. Non-vacuous since the task-spec-shaped fixture wave landed
    /// (four task.json files at time of writing); also guards every future
    /// fixture's task.json at the merge gate.
    #[test]
    fn fixture_task_json_files_are_valid_task_specs() {
        let fixtures_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("fixtures");
        let dirs = discover_fixtures(&fixtures_root).expect("discover fixtures");
        for dir in &dirs {
            let task_json = dir.join("task.json");
            if task_json.exists() {
                let json_str = std::fs::read_to_string(&task_json)
                    .unwrap_or_else(|e| panic!("read {}: {e}", task_json.display()));
                let result = serde_json::from_str::<crate::task_spec::TaskSpec>(&json_str);
                assert!(
                    result.is_ok(),
                    "task.json at {} must be a valid TaskSpec: {:?}",
                    task_json.display(),
                    result.unwrap_err(),
                );
            }
        }
    }

    /// Backward-compat: a fixture without `task.json` or `holdout/` must yield
    /// `holdout_passed == None` in every trial, and `on_trial` must observe
    /// `None` (proving the callback fires AFTER the re-gate decision).
    #[tokio::test]
    async fn holdout_passed_is_none_for_fixture_without_holdout() {
        // echo_finish_env has holdout_src = None and checks = None → None.
        let backend = MockBackend::from_turns(vec![finish_done_turn()]);
        let task = finish_task();

        let observed: Arc<Mutex<Vec<Option<bool>>>> = Arc::new(Mutex::new(Vec::new()));
        let obs_c = Arc::clone(&observed);
        let report = run_eval(&task, &backend, echo_finish_env, 1, 5, move |t| {
            obs_c.lock().expect("lock").push(t.holdout_passed);
        })
        .await;

        assert_eq!(
            report.trial_results[0].holdout_passed, None,
            "holdout_passed must be None when env has no holdout_src"
        );
        assert_eq!(
            *observed.lock().expect("lock"),
            vec![None],
            "on_trial must observe holdout_passed=None"
        );
    }

    /// Holdout re-gate: when `env.holdout_src` is set and `env.checks` is a
    /// green runner, the re-gate runs BEFORE `on_trial` and
    /// `holdout_passed == Some(true)`. The holdout content is merged into the
    /// workspace before the gate fires.
    #[tokio::test]
    async fn holdout_re_gate_runs_and_on_trial_observes_result() {
        use super::ScratchDir;

        // Build a "holdout source" dir with one canary file.
        let holdout_holder = tempdir().expect("holdout holder");
        let holdout_src_dir = holdout_holder.path().join("holdout");
        std::fs::create_dir(&holdout_src_dir).expect("mkdir holdout");
        std::fs::write(holdout_src_dir.join("canary.txt"), "canary").expect("write canary");

        let holdout_src_path = holdout_src_dir.clone();
        let factory = move || {
            let ws_scratch = ScratchDir::new("holdout-gate-ws").expect("ws scratch");
            let off_scratch = ScratchDir::new("holdout-gate-off").expect("off scratch");
            let workspace =
                Workspace::new(ws_scratch.path(), Some(off_scratch.path().to_path_buf()))
                    .expect("workspace");
            let ws_root = workspace.root().to_path_buf();
            let off_canon = off_scratch.path().canonicalize().expect("canon offload");
            let ctx = ToolCtx::new(
                Arc::new(workspace),
                Arc::new(DiskOffloadSink::new(off_canon)),
            );
            let checks = green_checks(ws_root);
            let tools = standard_registry(Some(checks.clone()));
            TrialEnv {
                tools,
                ctx,
                checks: Some(checks),
                holdout_src: Some(holdout_src_path.clone()),
                _scratch: vec![ws_scratch, off_scratch],
            }
        };

        let task = finish_task();
        let backend = MockBackend::from_turns(vec![finish_done_turn()]);

        let observed: Arc<Mutex<Option<Option<bool>>>> = Arc::new(Mutex::new(None));
        let obs_c = Arc::clone(&observed);
        let report = run_eval(&task, &backend, factory, 1, 5, move |t| {
            *obs_c.lock().expect("lock") = Some(t.holdout_passed);
        })
        .await;

        assert_eq!(
            report.trial_results[0].holdout_passed,
            Some(true),
            "holdout re-gate must have run (green checks) → Some(true)"
        );
        assert_eq!(
            *observed.lock().expect("lock"),
            Some(Some(true)),
            "on_trial must observe holdout_passed AFTER the re-gate ran"
        );
    }

    #[test]
    fn no_fixture_crate_is_a_workspace_member() {
        // Prove `exclude = ["fixtures/*"]` took effect: NO fixture crate under
        // `fixtures/` is a member of the harness workspace, so the project's
        // gates never build or test any of them. `cargo metadata` is offline
        // (no compile, --no-deps).
        //
        // The check is generic — the assertion looks at every member's package
        // id and rejects any whose path passes through a `fixtures/` segment.
        // That covers `broken-adder` (v0) plus the v1 fixtures
        // (`interval-merge`, `lru-cache`, `text-preview`) and any future
        // fixture added to `fixtures/` without touching this test.
        let cargo = env!("CARGO");
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("Cargo.toml");
        let output = std::process::Command::new(cargo)
            .args([
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
            ])
            .arg(&manifest)
            .output()
            .expect("run cargo metadata");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let meta: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("parse cargo metadata json");
        let members = meta["workspace_members"]
            .as_array()
            .expect("workspace_members is an array");
        // Generic guard: no member id may mention `fixtures/` (that would be
        // a fixture crate leaking into the workspace).
        for m in members {
            let s = m.as_str().unwrap_or_default();
            assert!(
                !s.contains("/fixtures/") && !s.contains("\\fixtures\\"),
                "no fixture crate may be a workspace member; found `{s}` in members: {members:?}",
            );
        }
        // Belt-and-suspenders: the four v1 fixture package names must not
        // appear either — a stronger check than path-substring matching in
        // case some future cargo-metadata format changes the id shape.
        for banned in [
            "broken-adder",
            "interval-merge",
            "lru-cache",
            "text-preview",
        ] {
            assert!(
                members
                    .iter()
                    .all(|m| !m.as_str().unwrap_or_default().contains(banned)),
                "fixture `{banned}` must NOT be a workspace member; members: {members:?}",
            );
        }
        // Sanity: the harness crate IS a member (proves metadata actually ran).
        assert!(
            members
                .iter()
                .any(|m| m.as_str().unwrap_or_default().contains("harness")),
            "the harness crate should be a workspace member"
        );
    }
}
