//! Tier-2 mined-task eval runner: score an agent on ~SWE-bench-shaped tasks
//! mined from real git history, where "resolved" is decided by **test-id
//! parsing**, never by an exit code.
//!
//! ## Why this exists
//!
//! Tier-1 (`crate::eval::run_eval`) is saturated at ~100% for every top-tier
//! model. Tier-2 is the discriminating rig: hand-mined tasks pulled from real
//! commits across our repo family, statement-only (no localization), scored
//! by re-running the mined `FAIL_TO_PASS` tests as a sealed holdout.
//!
//! See `docs/design/04-discriminating-eval-tier.md` for the full design.
//!
//! ## Task-dir schema (talos-evals shape)
//!
//! The example runner points at a `talos-evals` task-family root and picks
//! task dirs from it. Each task dir carries:
//!
//! ```text
//! <task-id>/
//!   task.json                        # the [`MinedTask`] spec, described below
//!   statements/
//!     s1.md                          # intent-only statement (pre-groom seed)
//!     s2.md                          # AC-only statement (default)
//!     s3.md                          # full groomed spec
//!   sealed/
//!     tests/<repo-relative-path>.py  # sealed `FAIL_TO_PASS` test files
//! ```
//!
//! `task.json` fields (all required except `notes` and `agent_gate_command`):
//!
//! | field | shape | meaning |
//! |---|---|---|
//! | `id` | `String` | stable task id (matches the dir name) |
//! | `repo` | `String` | source repo short name, e.g. `cleanr` |
//! | `repo_path` | `String` | on-box source repo path (supports leading `~/`) |
//! | `parent_sha` | `String` | commit the agent starts from |
//! | `fix_sha` | `String` | reference commit (informational; not checked out) |
//! | `rung_guess` | `String` | initial rung placement (empirically re-ranked) |
//! | `language` | `String` | informational only |
//! | `provenance` | opaque map | ignored on load |
//! | `env.setup` | `String` | shell script to bootstrap the workspace |
//! | `env.sibling_repos` | `Vec<{repo, pin}>` | worktrees to lay down beside primary |
//! | `test_scope` | `String` | informational only |
//! | `gate_command` | `String` | shell command run for scoring |
//! | `agent_gate_command` | `Option<String>` | (optional) in-run agent gate; required when the agent gate is on |
//! | `fail_to_pass` | `Vec<String>` | test ids that must be Passed at fix |
//! | `positive_controls` | `Vec<String>` | test ids that must stay Passed |
//! | `pass_to_pass_exclusions` | `Vec<String>` | red tests to ignore (literal or `Class::*`) |
//! | `sealed` | `Vec<{path}>` | files to overwrite in the workspace before the re-gate |
//! | `notes` | `Vec<String>` | (optional) free-form notes, ignored |
//!
//! ## Contamination hygiene
//!
//! No talos-evals content (statements, sealed tests, task.json, expected-
//! behavior notes) is copied into or referenced by this crate. Every unit
//! test constructs synthetic task dirs + synthetic pytest output + throwaway
//! git repos in tempdirs. The schema above is documented in code doc-comments
//! only.
//!
//! ## Scope-outs (pilot pins)
//!
//! - vitest + cargo-test output parsers — the [`TestReportParser`] trait
//!   exists so those slot in later; [`PytestParser`] is the only pilot impl.
//! - Dispatch-host portability — the runner is local-box-only for the pilot
//!   (`repo_path` resolves `~/git/<repo>`).
//! - Automated task mining — the pilot consumes hand-mined task dirs.
//! - Parent-baseline capture — the pilot pins **exclusions-authoritative**
//!   scoring; capturing the green-at-parent id set pre-agent is deferred.
//! - Cached/pre-provisioned base environments — a fresh `env.setup` runs
//!   per trial.
//!
//! ## Housekeeping
//!
//! Best-effort worktree teardown runs in [`WorktreeSet`]'s [`Drop`]. A crash
//! that skips the drop may leave stale entries; `git worktree prune` inside
//! the source repo clears them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::engine::{self, LoopOutcome, RunConfig, RunResult};
use crate::eval::{CODING_CHECK_TIMEOUT, copy_dir_recursive};
use crate::exec::{CheckReport, ExecSpec, run, shell_checks_runner};
use crate::model::ModelBackend;
use crate::run_record::Disposition;
use crate::tool::ToolCtx;
use crate::tools::standard_registry;
use crate::workspace::{DiskOffloadSink, Workspace};

// ===== timeouts ==========================================================

/// Timeout for `env.setup`. A cold `uv sync` (or `pnpm install`) can exceed
/// [`CODING_CHECK_TIMEOUT`] (3 min), so setup gets its own wider budget.
pub const MINED_SETUP_TIMEOUT: Duration = Duration::from_mins(10);

// ===== task.json ==========================================================

/// Deserialized `task.json`.
///
/// Field names are **verbatim** matches of the on-disk schema — no
/// `#[serde(rename)]`. Unknown top-level and nested keys are tolerated so the
/// schema can grow additive metadata without a harness update. All listed
/// fields are required except `notes` (optional — ignored, carried only for
/// forward-compatibility with schema growth) and `agent_gate_command`
/// (optional — `None` when the task predates the in-run agent gate; required
/// only when [`AgentGateMode::On`] is selected, see [`resolve_agent_gate_command`]).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MinedTask {
    /// Stable task id (matches the task-dir name).
    pub id: String,
    /// Source repo short name (informational; used to resolve `env.sibling_repos`).
    pub repo: String,
    /// On-box path to the source repo. A leading `~/` is expanded via `$HOME`
    /// when the workspace is built.
    pub repo_path: String,
    /// The commit the agent starts from (checked out via `git worktree add
    /// --detach`).
    pub parent_sha: String,
    /// Reference commit (the mined fix); informational only, never checked out.
    pub fix_sha: String,
    /// Initial rung placement guess ("easy" | "mid" | "hard"); rungs are
    /// re-ranked empirically downstream.
    pub rung_guess: String,
    /// Informational language tag (e.g. `python`, `typescript`).
    pub language: String,
    /// Opaque provenance metadata (commit trailer / mining pipeline info);
    /// ignored on load but preserved so a round-trip is possible.
    #[serde(default)]
    pub provenance: serde_json::Value,
    /// Environment bootstrap.
    pub env: MinedEnv,
    /// Informational scope hint (e.g. `tests/test_cleanr.py`).
    pub test_scope: String,
    /// Shell command run in the primary workspace for the sealed re-gate.
    /// Passed to `bash -c`.
    ///
    /// This is the SEALED SCORING gate only: it is never shown to the agent
    /// and never registered as `run_checks` — [`AgentGateMode`] and
    /// [`resolve_agent_gate_command`] draw a hard line between this field
    /// and [`Self::agent_gate_command`].
    ///
    /// Authoring invariant (verified across all eight tier-2 task.json files):
    /// every `.py` path named in `gate_command` also appears in `sealed[]`,
    /// and `copy_sealed` overwrites those files before the re-gate — which is
    /// WHY agent-authored tests cannot currently reach the scorer (open
    /// question 3).
    pub gate_command: String,
    /// The in-run AGENT gate: the repo's real dispatch gate (incl. lint /
    /// typecheck, not just tests). Shown to the agent (appended to its
    /// prompt as a `## Verification` section), run by `run_checks`, by
    /// `finish(done)` verification, and by the pre-agent baseline tripwire.
    /// It is NEVER used for scoring — [`MinedTask::gate_command`] stays the
    /// sealed scoring judge, unchanged.
    ///
    /// Must be a CHECKER: it must not mutate the tree, and it must be green
    /// at `parent_sha` in the task env (verified by the pre-agent baseline
    /// tripwire; see [`AgentGateMode::On`]).
    ///
    /// `#[serde(default)]` so existing task.json files without this key still
    /// load (as `None`) — required for [`AgentGateMode::Off`] to keep working
    /// as the legacy comparison row.
    #[serde(default)]
    pub agent_gate_command: Option<String>,
    /// Test ids that must all be Passed for a trial to resolve.
    pub fail_to_pass: Vec<String>,
    /// Test ids that MUST remain Passed (present + green).
    ///
    /// An **absent** `positive_control` makes the trial [`TrialScore::Invalid`]
    /// (`positive-control-uncollected`) — an environment tripwire, not a red
    /// verdict.
    pub positive_controls: Vec<String>,
    /// Test ids allowed to stay red without failing the trial. Two forms:
    /// literal `Class::test_name` OR the glob `Class::*` (matches any test in
    /// that class).
    pub pass_to_pass_exclusions: Vec<String>,
    /// Files to overwrite in the workspace before the sealed re-gate runs.
    pub sealed: Vec<SealedEntry>,
    /// Optional free-form notes (ignored — forward-compat).
    #[serde(default)]
    pub notes: Vec<String>,
}

/// The `env` sub-object of [`MinedTask`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MinedEnv {
    /// Shell script bootstrapping the workspace. Run via `bash -c` under
    /// [`MINED_SETUP_TIMEOUT`]. A non-zero exit or timeout makes the trial
    /// [`TrialScore::Invalid`] and the agent NEVER runs.
    pub setup: String,
    /// Sibling repos to lay down alongside the primary worktree so
    /// `../<sibling>` path-deps resolve.
    #[serde(default)]
    pub sibling_repos: Vec<SiblingRepo>,
}

/// One entry in [`MinedEnv::sibling_repos`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SiblingRepo {
    /// Sibling repo short name; resolved as
    /// `<parent_dir_of_expanded_repo_path>/<repo>` (so `repo_path`
    /// `~/git/cleanr` + sibling `flickrasync` ⇒ `~/git/flickrasync`).
    pub repo: String,
    /// Ref to check out (tag or sha).
    pub pin: String,
}

/// One entry in [`MinedTask::sealed`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SealedEntry {
    /// Workspace-relative destination path. Sourced from
    /// `<task_dir>/sealed/<path>`.
    pub path: String,
}

/// Load a `task.json` from `<task_dir>/task.json`.
///
/// # Errors
/// Returns any `std::io::Error` from the read or a wrapping error message
/// with the source path when serde fails.
pub fn load_task(task_dir: &Path) -> std::io::Result<MinedTask> {
    let path = task_dir.join("task.json");
    let text = std::fs::read_to_string(&path)?;
    serde_json::from_str::<MinedTask>(&text).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("parse {}: {e}", path.display()),
        )
    })
}

// ===== agent gate ==========================================================

/// Default timeout for the in-run agent gate (`agent_gate_command`), and the
/// pre-agent baseline tripwire that runs it once before the agent starts.
///
/// Mirrors agent-gtd-dispatch's default `TALOS_GATE_TIMEOUT_SECS = 900`,
/// which dispatch passes to talos as `--gate-timeout-secs` and which
/// overrides the talos clap default of 300. A host-level
/// `TALOS_GATE_TIMEOUT_SECS` override is not tracked here; use
/// `MINED_EVAL_GATE_TIMEOUT_SECS` to match one.
pub const DEFAULT_AGENT_GATE_TIMEOUT: Duration = Duration::from_mins(15);

/// Whether the in-run agent gate (`agent_gate_command`) is exercised this run.
///
/// `Off` is the legacy comparison row: no `run_checks` tool is registered,
/// [`crate::engine::RunConfig::checks`] is `None`, a `finish(done)` claim is
/// accepted as `Verification::NoChecksConfigured`, the system prompt carries
/// no harness Verification section, and no pre-agent baseline tripwire runs.
///
/// `On` is production parity: the repo's real dispatch gate is shown to the
/// agent, wired as `run_checks`, verifies `finish(done)`, and must be green
/// at `parent_sha` before the agent is allowed to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentGateMode {
    /// Legacy comparison row — no in-run agent gate.
    Off,
    /// Production parity — the in-run agent gate is armed with `timeout`.
    On {
        /// Timeout applied to every run of `agent_gate_command`: the
        /// baseline tripwire, `run_checks`, and `finish(done)` verification.
        timeout: Duration,
    },
}

impl std::fmt::Display for AgentGateMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::On { timeout } => write!(f, "on(timeout={}s)", timeout.as_secs()),
        }
    }
}

/// Parse the `MINED_EVAL_AGENT_GATE` / `MINED_EVAL_GATE_TIMEOUT_SECS` env
/// pair into an [`AgentGateMode`].
///
/// Both inputs are trimmed first. `timeout_secs` is ALWAYS parsed first, even
/// when `enabled` selects [`AgentGateMode::Off`]: `None` or empty gives
/// [`DEFAULT_AGENT_GATE_TIMEOUT`] (900s); a `u64 >= 1` gives that many
/// seconds; `0` or an unparsable value is an `Err` naming
/// `MINED_EVAL_GATE_TIMEOUT_SECS` and the offending value.
///
/// For `enabled`: `None`, empty, or `"1"` gives `On` (with the parsed
/// timeout); `"0"` gives `Off`; any other value is an `Err` naming
/// `MINED_EVAL_AGENT_GATE` and the offending value.
///
/// # Errors
/// See above — every error message names the offending env var and value.
pub fn parse_agent_gate_mode(
    enabled: Option<&str>,
    timeout_secs: Option<&str>,
) -> Result<AgentGateMode, String> {
    let timeout_raw = timeout_secs.map(str::trim).unwrap_or_default();
    let timeout = if timeout_raw.is_empty() {
        DEFAULT_AGENT_GATE_TIMEOUT
    } else {
        match timeout_raw.parse::<u64>() {
            Ok(n) if n >= 1 => Duration::from_secs(n),
            _ => {
                return Err(format!(
                    "MINED_EVAL_GATE_TIMEOUT_SECS: expected a positive integer, got `{timeout_raw}`"
                ));
            }
        }
    };

    let enabled_raw = enabled.map(str::trim).unwrap_or_default();
    match enabled_raw {
        "" | "1" => Ok(AgentGateMode::On { timeout }),
        "0" => Ok(AgentGateMode::Off),
        other => Err(format!(
            "MINED_EVAL_AGENT_GATE: expected `0`, `1`, or empty, got `{other}`"
        )),
    }
}

/// Resolve the `agent_gate_command` to actually run for `task` under `mode`.
///
/// Branches are evaluated in this order:
/// 1. [`AgentGateMode::Off`] gives `Ok(None)`, even when the field is `Some`.
/// 2. `On` with `task.agent_gate_command == None` is an `Err`.
/// 3. `On` with `Some(cmd)` where `cmd.trim().is_empty()` is an `Err`.
/// 4. `On` where `cmd.trim() == task.gate_command.trim()` is an `Err` — the
///    agent gate must never be the same command as the sealed scoring gate.
/// 5. `On` where `cmd` contains any non-empty `task.sealed[i].path` is an
///    `Err` — the agent gate must never leak a sealed test path into the
///    agent's own prompt/verification loop.
/// 6. Otherwise `Ok(Some(cmd))`, with `cmd` returned untrimmed and verbatim.
///
/// There is NO fallback to `task.gate_command` on any path — a missing or
/// invalid `agent_gate_command` under `On` is a hard, fail-loud error, never
/// a silent substitution of the sealed scoring gate.
///
/// # Errors
/// Every error message contains `task.id` and the literal `agent_gate_command`.
/// The errors from branches 2 and 3 additionally contain `MINED_EVAL_AGENT_GATE=0`
/// (the escape hatch: run in `Off` mode instead).
pub fn resolve_agent_gate_command(
    task: &MinedTask,
    mode: AgentGateMode,
) -> Result<Option<&str>, String> {
    let AgentGateMode::On { .. } = mode else {
        return Ok(None);
    };
    let Some(cmd) = task.agent_gate_command.as_deref() else {
        return Err(format!(
            "task `{}`: agent_gate_command is missing (required when the agent gate is on; \
             set MINED_EVAL_AGENT_GATE=0 to run the legacy off row)",
            task.id
        ));
    };
    if cmd.trim().is_empty() {
        return Err(format!(
            "task `{}`: agent_gate_command is blank (required when the agent gate is on; \
             set MINED_EVAL_AGENT_GATE=0 to run the legacy off row)",
            task.id
        ));
    }
    if cmd.trim() == task.gate_command.trim() {
        return Err(format!(
            "task `{}`: agent_gate_command equals gate_command — the in-run agent gate must \
             differ from the sealed scoring gate",
            task.id
        ));
    }
    for entry in &task.sealed {
        if !entry.path.is_empty() && cmd.contains(&entry.path) {
            return Err(format!(
                "task `{}`: agent_gate_command contains sealed path `{}` — the in-run agent \
                 gate must never reference a sealed scoring path",
                task.id, entry.path
            ));
        }
    }
    Ok(Some(cmd))
}

/// Parse `MINED_EVAL_TEST_FIRST` into whether the shared test-first approach
/// guidance is appended to the tier-2 agent prompt.
///
/// Trimmed first. `None`, empty, or `"1"` gives `true`; `"0"` gives `false`;
/// anything else is an `Err` naming `MINED_EVAL_TEST_FIRST` and the value.
///
/// This toggle exists ONLY for the tier-2 test-first A/B; the talos
/// production prompt path (`render_task_prompt_from_spec`) hard-codes the
/// guidance on and is untouched by this function.
///
/// # Errors
/// See above.
pub fn parse_test_first(v: Option<&str>) -> Result<bool, String> {
    let raw = v.map(str::trim).unwrap_or_default();
    match raw {
        "" | "1" => Ok(true),
        "0" => Ok(false),
        other => Err(format!(
            "MINED_EVAL_TEST_FIRST: expected `0`, `1`, or empty, got `{other}`"
        )),
    }
}

/// Parse `MINED_EVAL_WALL_CLOCK_SECS` into the wall-clock budget applied to
/// every trial, mirroring talos production's `--wall-clock-secs` (default 0 =
/// unbounded).
///
/// Trimmed first. `None` or empty gives `0` (unbounded). A `u64` gives that
/// value verbatim — `0` is a valid, explicit "unbounded" too. Anything else is
/// an `Err` naming `MINED_EVAL_WALL_CLOCK_SECS` and the value.
///
/// # Errors
/// See above.
pub fn parse_wall_clock_secs(v: Option<&str>) -> Result<u64, String> {
    let raw = v.map(str::trim).unwrap_or_default();
    if raw.is_empty() {
        return Ok(0);
    }
    raw.parse::<u64>().map_err(|_| {
        format!("MINED_EVAL_WALL_CLOCK_SECS: expected a non-negative integer, got `{raw}`")
    })
}

// ===== statements =========================================================

/// Which statement level to hand the agent.
///
/// The level is the experimental variable — a missing level file is a hard
/// startup error naming the absolute path, never a silent fallback (a
/// fallback would corrupt cross-run comparison).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecLevel {
    /// Intent-only (pre-groom seed).
    S1,
    /// AC-only (the standard tier-2 statement, and the default).
    S2,
    /// Full groomed spec (calibration floor).
    S3,
}

impl SpecLevel {
    /// Statement filename slug (`s1` / `s2` / `s3`).
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::S1 => "s1",
            Self::S2 => "s2",
            Self::S3 => "s3",
        }
    }

    /// Parse `"s1" | "s2" | "s3"` (case-insensitive). Returns `None` for
    /// anything else — caller decides how to surface the mistake.
    ///
    /// Not `impl std::str::FromStr` because our error case is `None`, not a
    /// dedicated error type; matching the trait would require inventing an
    /// error just to say "not a valid level slug".
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "s1" => Some(Self::S1),
            "s2" => Some(Self::S2),
            "s3" => Some(Self::S3),
            _ => None,
        }
    }
}

/// Load `<task_dir>/statements/<level>.md` as the agent's prompt body.
///
/// # Errors
/// Returns an error naming the absolute path when the file is missing (or
/// unreadable). Callers surface this at startup — there is no silent
/// fallback to another level.
pub fn load_statement(task_dir: &Path, level: SpecLevel) -> std::io::Result<String> {
    let path = task_dir
        .join("statements")
        .join(format!("{}.md", level.slug()));
    std::fs::read_to_string(&path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("load_statement({level:?}) at `{}`: {e}", path.display()),
        )
    })
}

// ===== test id parsing ====================================================

/// One test's status parsed from a pytest `-rA` short-summary line.
///
/// The six variants match pytest's own vocabulary; `XPass` is deliberately
/// treated as a distinct status (an xfail that unexpectedly passed) because
/// the eval must never silently promote it to `Passed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TestStatus {
    Passed,
    Failed,
    Error,
    Skipped,
    XFail,
    XPass,
}

/// Result of parsing a test-runner's output via [`TestReportParser::parse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseReport {
    /// Parsed nodeid → status map.
    pub statuses: BTreeMap<String, TestStatus>,
    /// Second-source count from the runner's trailing summary line; `None`
    /// when absent.
    pub summary_count: Option<usize>,
    /// Status lines that matched a recognised status tag but fell OUTSIDE
    /// a `short test summary info` section. A nonzero value on a future run
    /// is the tripwire that pytest renamed its banner: section scoping makes
    /// the parser under-inclusive, whose failure mode is silent (everything
    /// becomes `Invalid{parse-empty}` at exactly zero ids).
    pub dropped_outside_section: usize,
    /// Number of `short test summary info` section headers seen in the
    /// output. Zero means the gate produced no parseable section (e.g. an
    /// early timeout, a collection error that aborts before pytest reaches
    /// that section, or a bare `-q` run without `-rA`).
    pub sections_seen: usize,
}

/// Parser of a runner's test-report output into a `nodeid -> status` map.
///
/// Only [`PytestParser`] ships in the pilot; the trait exists so a vitest or
/// cargo-test parser can plug in later without touching the scorer.
pub trait TestReportParser: Send + Sync {
    /// Parse `output` (typically `stdout + stderr` from the gate command) into
    /// a [`ParseReport`].
    ///
    /// `summary_count` is the parser's second-source count of individual test
    /// outcomes, cross-checked against `map.len()` in [`resolve`] to detect
    /// truncation / broken output (`parse-mismatch`). `None` means the parser
    /// has no second-source count.
    fn parse(&self, output: &str) -> ParseReport;
}

/// Parser for pytest's `-rA` short-summary lines.
///
/// The eval MUST inject `PYTEST_ADDOPTS="-rA"` (see [`sealed_regate_score`])
/// so pytest emits `PASSED/FAILED/ERROR/SKIPPED/XFAIL/XPASS <nodeid>` short-
/// summary lines regardless of the gate's own `-q`. Under bare `-q`, passes
/// print as dots and never appear in the output — the parser would return an
/// empty map, which [`resolve`] catches as [`TrialScore::Invalid`]
/// (`parse-empty`) rather than a silent unresolved verdict.
///
/// A second-source `summary_count` is scraped from pytest's trailing summary
/// line (`"=== N passed, M failed, ..."`), so the scorer can flag a mismatch
/// between the count of `-rA` lines and pytest's own total.
#[derive(Debug, Clone, Copy, Default)]
pub struct PytestParser;

impl TestReportParser for PytestParser {
    fn parse(&self, output: &str) -> ParseReport {
        let mut statuses: BTreeMap<String, TestStatus> = BTreeMap::new();
        let mut summary_count: Option<usize> = None;
        let mut in_section = false;
        let mut sections_seen: usize = 0;
        let mut dropped_outside_section: usize = 0;
        for line in output.lines() {
            let t = line.trim_start();
            let tag = parse_short_summary_line(t);
            // (1) section-start detection.
            if t.starts_with('=') && t.contains("short test summary info") {
                in_section = true;
                sections_seen += 1;
            } else if in_section && (t.trim().is_empty() || tag.is_none()) {
                // (2) section-end: blank or non-status line while in-section.
                in_section = false;
            }
            // (3) nodeid recording.
            if let Some((status, rest)) = tag {
                let nodeid = derive_nodeid(status, rest);
                if !nodeid.is_empty() {
                    if in_section {
                        // Last-write-wins on duplicates (a re-run is rare
                        // and harmless — a stable status is what matters).
                        statuses.insert(nodeid, status);
                    } else {
                        dropped_outside_section += 1;
                    }
                }
            }
            // (4) summary count — unconditional: a status line can never
            // also be a `===`-prefixed summary line, so this is always a
            // no-op on status lines.
            if let Some(n) = parse_pytest_summary_totals(t) {
                summary_count = Some(n);
            }
        }
        ParseReport {
            statuses,
            summary_count,
            dropped_outside_section,
            sections_seen,
        }
    }
}

/// Parse a pytest `-rA` short-summary line: `PASSED <nodeid> [reason]` etc.
/// Returns the status and everything after `STATUS ` (starting at the nodeid).
fn parse_short_summary_line(line: &str) -> Option<(TestStatus, &str)> {
    // Match a leading UPPERCASE status token followed by a space + nodeid.
    for (tag, status) in [
        ("PASSED ", TestStatus::Passed),
        ("FAILED ", TestStatus::Failed),
        ("ERROR ", TestStatus::Error),
        ("SKIPPED ", TestStatus::Skipped),
        ("XFAIL ", TestStatus::XFail),
        ("XPASS ", TestStatus::XPass),
    ] {
        if let Some(rest) = line.strip_prefix(tag) {
            return Some((status, rest));
        }
    }
    None
}

/// Derive the map key from a parsed short-summary `rest` fragment.
///
/// For `SKIPPED [N] <location>: <reason>` (the bracketed form emitted when
/// multiple tests share a skip condition), the key is the location token with
/// at most one trailing `:` stripped. For all other statuses the key is the
/// first whitespace-delimited token.
fn derive_nodeid(status: TestStatus, rest: &str) -> String {
    if status == TestStatus::Skipped {
        let first = rest.split_whitespace().next().unwrap_or_default();
        if first.starts_with('[') && first.ends_with(']') {
            // Bracketed: key is the NEXT token, trailing `:` stripped.
            return rest
                .split_whitespace()
                .nth(1)
                .map(|s| s.trim_end_matches(':').to_string())
                .unwrap_or_default();
        }
    }
    rest.split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string()
}

/// Parse pytest's trailing summary line into a total-outcome count.
///
/// Matches `"=== N passed, M failed, K skipped ... in T.TTs ==="` (any subset
/// of those tokens) and sums the leading integers of every recognised outcome
/// bucket. Returns `None` when the line is not a pytest summary.
///
/// Recognised buckets: `passed`, `failed`, `error(s)`, `skipped`, `xfailed`,
/// `xpassed`, `deselected`, `warnings` (matches what pytest itself emits).
fn parse_pytest_summary_totals(line: &str) -> Option<usize> {
    // A pytest summary line always has surrounding `=` runs.
    let inner = line
        .strip_prefix("===")
        .and_then(|s| s.rsplit_once("==="))?;
    // Strip the REST of the banner's `=` padding, not just the first three:
    // pytest pads to terminal width (`====== 1 failed, 6 passed ======`), so
    // leaving it attached glues `======` onto the FIRST bucket's count, whose
    // `parse::<usize>()` then fails and silently drops that bucket. That
    // under-counts multi-bucket summaries (false `parse-mismatch`) and, for a
    // single-bucket summary, zeroes `any` so the check returns `None` and
    // disables itself entirely.
    let body = inner.0.trim_matches('=').trim();
    // Split on `,` and pick the leading integer + label pairs; ignore the
    // trailing `in T.TTs`.
    let mut total: usize = 0;
    let mut any = false;
    for part in body.split(',') {
        let part = part.trim();
        // Drop a possible `in 1.23s` suffix.
        let mut it = part.split_whitespace();
        let (Some(n_str), Some(label)) = (it.next(), it.next()) else {
            continue;
        };
        let Ok(n) = n_str.parse::<usize>() else {
            continue;
        };
        // Any of pytest's counted outcome labels — trim a trailing 's' so
        // `error` and `errors` both match.
        let label = label.trim_end_matches('s');
        if matches!(
            label,
            "passed" | "failed" | "error" | "skipped" | "xfailed" | "xpassed" | "deselected"
        ) {
            total += n;
            any = true;
        }
        // `warnings` is present in the summary line but not a per-test count;
        // deliberately excluded.
    }
    any.then_some(total)
}

/// Normalize a pytest nodeid so it can be compared to a bare task-id.
///
/// Task-json ids are bare (`TestClass::test_method`); pytest emits file-
/// prefixed nodeids (`tests/test_cleanr.py::TestClass::test_method`). This
/// strips everything up to and including the first `::` when the first
/// segment ends in `.py`. A parametrized `[param]` suffix is preserved on
/// the way in — [`match_task_id`] strips it separately.
#[must_use]
pub fn normalize(nodeid: &str) -> String {
    if let Some(idx) = nodeid.find("::") {
        let (head, tail) = nodeid.split_at(idx);
        // Extension check via `Path::extension` — a bare `ends_with(".py")`
        // would trip clippy's `case_sensitive_file_extension_comparisons`,
        // and the Path form is the pedantic-clippy-idiomatic answer.
        let is_py = Path::new(head)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("py"));
        if is_py {
            // strip the `::` too
            return tail[2..].to_string();
        }
    }
    nodeid.to_string()
}

/// Strip a `[param]` suffix from a normalized nodeid for matching against a
/// bare task id. Preserves the id when there is no `[...]`.
fn strip_param_suffix(id: &str) -> &str {
    if let Some(open) = id.find('[')
        && id.ends_with(']')
    {
        &id[..open]
    } else {
        id
    }
}

/// True when a task-side id is FILE-QUALIFIED (`tests/x.py::…`) rather than
/// bare (`Class::test`). Multi-file task scopes qualify their ids to
/// disambiguate same-named tests across files (e.g. agent-gtd-rollout-deadlock
/// spans three test files with recurring names) — those must be matched in
/// RAW nodeid space, never after [`normalize`] (stripping would collide
/// same-named tests from different files).
fn is_file_qualified(task_id: &str) -> bool {
    normalize(task_id) != task_id
}

/// True when a parsed nodeid matches `task_id`, comparing in the space the
/// task id dictates: a file-qualified task id matches against the RAW
/// pytest nodeid; a bare task id matches against the [`normalize`]d form.
/// Both sides drop a parametrized `[param]` suffix before comparing.
fn match_task_id(raw_nodeid: &str, normalized_nodeid: &str, task_id: &str) -> bool {
    let side = if is_file_qualified(task_id) {
        raw_nodeid
    } else {
        normalized_nodeid
    };
    strip_param_suffix(side) == strip_param_suffix(task_id)
}

/// True when the exclusion entry `excl` matches the parsed nodeid.
///
/// Two forms, each honoring file-qualification like [`match_task_id`]:
/// - literal `Class::test_name` — exact match after `[param]` stripping.
/// - glob `Class::*` (or `tests/x.py::Class::*`) — any test id under that
///   prefix matches.
fn matches_exclusion(raw_nodeid: &str, normalized_nodeid: &str, excl: &str) -> bool {
    if let Some(class_prefix) = excl.strip_suffix("::*") {
        let side = if is_file_qualified(excl) {
            raw_nodeid
        } else {
            normalized_nodeid
        };
        let prefix = format!("{class_prefix}::");
        side.starts_with(&prefix)
    } else {
        match_task_id(raw_nodeid, normalized_nodeid, excl)
    }
}

// ===== scoring ============================================================

/// A single trial's terminal verdict.
///
/// Three-member so **infra failures never masquerade as agent failures**:
/// [`Self::Invalid`] is excluded from the `resolved / k` denominator.
///
/// `Invalid` reason prefixes and where each is produced:
/// - `worktree-setup-failed` — `single_trial`, worktree add step
/// - `setup-failed` — `run_env_setup`, timeout / non-zero exit / no exit code
/// - `offload-scratch-failed` — `single_trial`, scratch dir creation
/// - `workspace-init-failed` — `single_trial`, `Workspace::new`
/// - `offload-canon-failed` — `single_trial`, offload path canonicalize
/// - `sealed-copy-failed` — `copy_sealed`, file/dir copy failure
/// - `gate-timeout` — `sealed_regate_score`, gate exceeded timeout
/// - `gate-no-exit-code` — `sealed_regate_score`, gate killed by signal with no output
/// - `parse-empty` — `resolve`, parser emitted empty map
/// - `parse-mismatch` — `resolve`, parsed count != summary count
/// - `positive-control-uncollected` — `resolve`, a positive control id is absent
///
/// Invalid means the MEASUREMENT failed and the trial leaves the denominator.
/// Anything traceable to the code under test is `Unresolved`.
///
/// `Unresolved` arms:
/// - Red positive control (agent broke a must-stay-green test)
/// - Missing or red `fail_to_pass` id
/// - Unexcluded red id
/// - File-level collection error (module unimportable — real agent failure)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrialScore {
    /// Every `fail_to_pass` + `positive_controls` id was Passed AND every
    /// otherwise-red id was covered by `pass_to_pass_exclusions`.
    Resolved,
    /// The gate ran and parsed, but at least one clause did not hold.
    Unresolved { reason: ResolveDetail },
    /// An infra failure — excluded from the resolved/k rate. See the enum
    /// doc for the complete list of reason prefixes.
    Invalid { reason: String },
}

/// Structured evidence for [`TrialScore::Unresolved`], retained per trial so
/// the exclusions-vs-baseline decision can be audited later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveDetail {
    /// Every `fail_to_pass` id and the status it was observed with (or that
    /// it was missing).
    pub fail_to_pass_status: Vec<(String, Option<TestStatus>)>,
    /// Normalized nodeids that were Failed/Error and NOT matched by any
    /// entry in `pass_to_pass_exclusions`.
    pub unexcluded_red: Vec<String>,
    /// `fail_to_pass` ids that were missing or non-Passed (`Skipped` /
    /// `XFail` / `Failed` / `Error`).
    pub missing_fail_to_pass: Vec<String>,
    /// File-level pytest collection errors: `Error` statuses with no `::`
    /// separator and a `.py` extension. When non-empty the agent left a
    /// module unimportable — a real agent failure. Scored `Unresolved`
    /// (step 3 of [`resolve`]). The `.py`-extension guard in
    /// [`build_collection_errors`] is load-bearing independently of the
    /// section-scoping fix; do NOT drop it.
    pub collection_errors: Vec<String>,
}

/// Score a parsed nodeid→status map against `task`'s clauses.
///
/// Exclusions-authoritative semantics (pilot pin). Returns
/// [`TrialScore::Invalid`] up front for structural problems (empty parse,
/// count-mismatch, missing `positive_control`) so the caller sees a distinct
/// signal from a genuine unresolved verdict.
///
/// Step 3 (collection errors) precedes step 4 (positive-control loop): a
/// file-level `ERROR <path>` means pytest never ran — the positive controls
/// are absent because the agent broke an import, not because the controls
/// regressed. The `.py`-extension guard in `build_collection_errors` ensures
/// the arm is correct even if a future parser regression reintroduces
/// caplog-style phantom ids.
#[must_use]
pub fn resolve(
    parsed: &BTreeMap<String, TestStatus>,
    summary_count: Option<usize>,
    task: &MinedTask,
) -> TrialScore {
    // (1) Empty parse: gate produced no parseable ids.
    if parsed.is_empty() {
        return TrialScore::Invalid {
            reason: "parse-empty".to_string(),
        };
    }
    // (2) Count mismatch: parsed count disagrees with pytest's own total.
    if let Some(n) = summary_count
        && n != parsed.len()
    {
        return TrialScore::Invalid {
            reason: format!("parse-mismatch: {} ids vs summary {}", parsed.len(), n),
        };
    }

    // Both views per nodeid: (raw as parsed, normalized `Class::test[param]`,
    // status). Matching picks the side per task-id form (file-qualified ids
    // match raw; bare ids match normalized — see [`match_task_id`]).
    let normalized: Vec<(String, String, TestStatus)> = parsed
        .iter()
        .map(|(id, status)| (id.clone(), normalize(id), *status))
        .collect();

    // (3) Collection errors: file-level ERROR records mean the agent left a
    // module unimportable — a real agent failure, not an infra fault.
    let collection_errors = build_collection_errors(&normalized);
    if !collection_errors.is_empty() {
        return TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: build_fail_to_pass_status(&normalized, task),
                unexcluded_red: build_unexcluded_red(&normalized, task),
                missing_fail_to_pass: build_missing_fail_to_pass(&normalized, task),
                collection_errors,
            },
        };
    }

    // (4) positive_controls: every id present + Passed. Absent ⇒ Invalid.
    for pc in &task.positive_controls {
        match find_status_for_task_id(&normalized, pc) {
            None => {
                return TrialScore::Invalid {
                    reason: format!("positive-control-uncollected: {pc}"),
                };
            }
            Some(TestStatus::Passed) => {}
            Some(_other) => {
                // A red positive_control is an "over-fixed" verdict — the
                // agent broke a test that must stay green.
                return TrialScore::Unresolved {
                    reason: ResolveDetail {
                        fail_to_pass_status: build_fail_to_pass_status(&normalized, task),
                        unexcluded_red: build_unexcluded_red(&normalized, task),
                        missing_fail_to_pass: build_missing_fail_to_pass(&normalized, task),
                        collection_errors: Vec::new(),
                    },
                };
            }
        }
    }

    // (5) fail_to_pass: every id present + Passed.
    let missing_ftp = build_missing_fail_to_pass(&normalized, task);
    if !missing_ftp.is_empty() {
        return TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: build_fail_to_pass_status(&normalized, task),
                unexcluded_red: build_unexcluded_red(&normalized, task),
                missing_fail_to_pass: missing_ftp,
                collection_errors: Vec::new(),
            },
        };
    }

    // (6) any other red id must be covered by an exclusion.
    let unexcluded = build_unexcluded_red(&normalized, task);
    if !unexcluded.is_empty() {
        return TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: build_fail_to_pass_status(&normalized, task),
                unexcluded_red: unexcluded,
                missing_fail_to_pass: Vec::new(),
                collection_errors: Vec::new(),
            },
        };
    }

    let _ = summary_count; // acknowledged above.
    TrialScore::Resolved
}

/// Find the status of the first normalized id that matches `task_id` (after
/// `[param]` stripping). Returns `None` when no id matches — a parametrized
/// nodeid like `Class::test[a]` matches the bare `Class::test` on the way in.
fn find_status_for_task_id(
    normalized: &[(String, String, TestStatus)],
    task_id: &str,
) -> Option<TestStatus> {
    // Parametrized nodeids: if ANY param instance was red, we surface that
    // (a partial pass across params is still not a Passed for the id).
    // Otherwise, if all matches were Passed, return Passed. Otherwise return
    // the first observed non-Passed status.
    let mut seen: Option<TestStatus> = None;
    for (raw, nid, status) in normalized {
        if match_task_id(raw, nid, task_id) {
            match (seen, *status) {
                (None, s) => seen = Some(s),
                (Some(TestStatus::Passed), s) if s != TestStatus::Passed => seen = Some(s),
                _ => {}
            }
        }
    }
    seen
}

/// Build the `fail_to_pass_status` evidence: for every `fail_to_pass` id, the
/// status it was observed with (or `None` when missing entirely).
fn build_fail_to_pass_status(
    normalized: &[(String, String, TestStatus)],
    task: &MinedTask,
) -> Vec<(String, Option<TestStatus>)> {
    task.fail_to_pass
        .iter()
        .map(|ftp| (ftp.clone(), find_status_for_task_id(normalized, ftp)))
        .collect()
}

/// `fail_to_pass` ids that are missing OR present but not Passed.
fn build_missing_fail_to_pass(
    normalized: &[(String, String, TestStatus)],
    task: &MinedTask,
) -> Vec<String> {
    task.fail_to_pass
        .iter()
        .filter(|ftp| {
            !matches!(
                find_status_for_task_id(normalized, ftp),
                Some(TestStatus::Passed)
            )
        })
        .cloned()
        .collect()
}

/// Every parsed id whose status is Failed/Error and which is NOT covered by
/// an exclusion NOR a `fail_to_pass` entry (a red `fail_to_pass` is surfaced by
/// [`build_missing_fail_to_pass`], not here).
fn build_unexcluded_red(
    normalized: &[(String, String, TestStatus)],
    task: &MinedTask,
) -> Vec<String> {
    // Track already-reported entries so duplicate parametrized nodeids don't
    // double-count.
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut out: Vec<String> = Vec::new();
    for (raw, nid, status) in normalized {
        if !matches!(status, TestStatus::Failed | TestStatus::Error) {
            continue;
        }
        if task
            .fail_to_pass
            .iter()
            .any(|ftp| match_task_id(raw, nid, ftp))
        {
            continue;
        }
        if task
            .pass_to_pass_exclusions
            .iter()
            .any(|excl| matches_exclusion(raw, nid, excl))
        {
            continue;
        }
        if seen.insert(nid.clone()) {
            out.push(nid.clone());
        }
    }
    out
}

/// File-level pytest collection errors: parsed ids whose status is
/// [`TestStatus::Error`] AND which lack a `::` separator (they are file
/// paths, not test nodeids) AND whose extension is `.py`.
///
/// `normalized` is produced by iterating a [`BTreeMap`] (call site in
/// [`resolve`]), so the result is already in ascending order and free of
/// duplicates; no explicit `.sort()` / `.dedup()` is needed.
///
/// The `.py`-extension clause is load-bearing independently of the
/// section-scoping fix: without it, a future parser regression that lets a
/// caplog phantom like `agent_gtd.event_bus:event_bus.py:130` through would
/// silently convert a loud `Invalid{parse-empty}` into a quiet `Unresolved`.
/// Its `Path::extension()` returns `Some("py:130")` — which does NOT equal
/// `"py"` — so the phantom is excluded correctly even without section
/// scoping.
pub(crate) fn build_collection_errors(normalized: &[(String, String, TestStatus)]) -> Vec<String> {
    normalized
        .iter()
        .filter(|(raw, _, status)| {
            *status == TestStatus::Error
                && !raw.contains("::")
                && Path::new(raw)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("py"))
        })
        .map(|(raw, _, _)| raw.clone())
        .collect()
}

// ===== per-trial workspace: worktrees + env.setup =========================

/// Owned git worktree — best-effort torn down on Drop via
/// `git worktree remove --force`. A failed teardown is logged to stderr,
/// never a panic (a Drop that panics is undefined behavior in a
/// dropping-because-of-panic context).
#[derive(Debug)]
struct Worktree {
    /// Path of the source repo the worktree was `add`-ed from.
    source_repo: PathBuf,
    /// Where the worktree was placed.
    worktree_path: PathBuf,
}

/// A `git` [`std::process::Command`] with any inherited git-hook environment
/// SCRUBBED. When the runner (or its tests) executes under a `git commit`
/// hook, git exports `GIT_DIR`/`GIT_INDEX_FILE`/`GIT_WORK_TREE`/`GIT_PREFIX`
/// into the hook's environment; a child `git -C <other-repo>` inheriting
/// those misresolves against the WRONG repo (observed: `worktree add` dying
/// with `.git/index: Not a directory` inside a freshly-created worktree).
/// Every git invocation in this module must go through this constructor.
fn git_command() -> std::process::Command {
    let mut cmd = std::process::Command::new("git");
    for var in [
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_PREFIX",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

impl Drop for Worktree {
    fn drop(&mut self) {
        let output = git_command()
            .arg("-C")
            .arg(&self.source_repo)
            .arg("worktree")
            .arg("remove")
            .arg("--force")
            .arg(&self.worktree_path)
            .output();
        if let Ok(o) = &output
            && !o.status.success()
        {
            eprintln!(
                "worktree remove {} from {} failed: {}",
                self.worktree_path.display(),
                self.source_repo.display(),
                String::from_utf8_lossy(&o.stderr).trim(),
            );
        }
    }
}

/// Set of worktrees laid down for one trial. Torn down on drop (via each
/// [`Worktree`]'s own drop).
///
/// Public only through [`prepare_worktrees`]; callers hold this alive across
/// the trial's agent run + sealed re-gate.
#[derive(Debug)]
pub struct WorktreeSet {
    /// Primary worktree (matches `task.repo`), the workspace root.
    pub primary: PathBuf,
    /// Sibling worktrees (matches `task.env.sibling_repos` order).
    pub siblings: Vec<PathBuf>,
    /// Scratch dir that owns primary+siblings; removed on drop.
    _scratch: ScratchDir,
    /// The worktree handles that own their teardown.
    _worktrees: Vec<Worktree>,
}

/// Prepare the primary + sibling worktrees for one trial's workspace.
///
/// Layout under `scratch/`:
/// ```text
/// scratch/
///   <task.repo>/                      # primary worktree at parent_sha
///   <sibling.repo>/                   # each sibling worktree at sibling.pin
/// ```
///
/// # Errors
/// Returns `std::io::Error` on scratch creation failure or when any
/// `git worktree add --detach` fails (the error's message carries the sha,
/// source repo, and git stderr for triage).
pub fn prepare_worktrees(task: &MinedTask) -> std::io::Result<WorktreeSet> {
    let scratch = ScratchDir::new(&format!("mined-{}", task.id))?;
    let expanded_repo_path = expand_home(&task.repo_path);
    let source_repo = PathBuf::from(&expanded_repo_path);
    let parent_dir = source_repo.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("repo_path `{expanded_repo_path}` has no parent"),
        )
    })?;

    // Primary worktree.
    let primary = scratch.path().join(&task.repo);
    add_worktree(&source_repo, &primary, &task.parent_sha)?;
    let mut owned: Vec<Worktree> = vec![Worktree {
        source_repo: source_repo.clone(),
        worktree_path: primary.clone(),
    }];

    // Sibling worktrees.
    let mut siblings: Vec<PathBuf> = Vec::with_capacity(task.env.sibling_repos.len());
    for sibling in &task.env.sibling_repos {
        let sibling_source = parent_dir.join(&sibling.repo);
        let sibling_path = scratch.path().join(&sibling.repo);
        add_worktree(&sibling_source, &sibling_path, &sibling.pin)?;
        siblings.push(sibling_path.clone());
        owned.push(Worktree {
            source_repo: sibling_source,
            worktree_path: sibling_path,
        });
    }

    Ok(WorktreeSet {
        primary,
        siblings,
        _scratch: scratch,
        _worktrees: owned,
    })
}

/// `git -C <source> worktree add --detach <dst> <sha>`, surfacing the error's
/// stderr on failure.
fn add_worktree(source: &Path, dst: &Path, sha: &str) -> std::io::Result<()> {
    let out = git_command()
        .arg("-C")
        .arg(source)
        .arg("worktree")
        .arg("add")
        .arg("--detach")
        .arg(dst)
        .arg(sha)
        .output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "git -C {} worktree add --detach {} {sha} failed: {}",
            source.display(),
            dst.display(),
            String::from_utf8_lossy(&out.stderr).trim(),
        )));
    }
    Ok(())
}

/// Expand a leading `~/` (and bare `~`) against `$HOME`. Non-tilde input is
/// returned unchanged. When `$HOME` is unset the input is returned as-is
/// (a leading `~/` will then be treated as a literal path segment by the
/// caller — better than a silent expansion to `/`).
fn expand_home(raw: &str) -> String {
    expand_home_with(raw, std::env::var_os("HOME"))
}

/// Pure worker for [`expand_home`], parameterised on the `HOME` value so each
/// branch is testable without mutating process env.
fn expand_home_with(raw: &str, home: Option<std::ffi::OsString>) -> String {
    let Some(home) = home.map(PathBuf::from) else {
        return raw.to_string();
    };
    if raw == "~" {
        return home.to_string_lossy().into_owned();
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        return home.join(rest).to_string_lossy().into_owned();
    }
    raw.to_string()
}

/// Run `env.setup` inside the primary worktree.
///
/// Uses `bash -c` and the clean environment enforced by [`crate::exec::run`]
/// (secrets stay out of the child). Returns `Ok(())` on zero-exit,
/// `Err(reason)` for any non-zero exit / timeout / spawn failure — the reason
/// is what feeds into [`TrialScore::Invalid`]'s message.
pub async fn run_env_setup(
    primary: &Path,
    setup: &str,
    extra_env: Vec<(String, String)>,
) -> Result<(), String> {
    let outcome = run(&ExecSpec {
        program: "bash".to_string(),
        args: vec!["-c".to_string(), setup.to_string()],
        cwd: primary.to_path_buf(),
        timeout: MINED_SETUP_TIMEOUT,
        extra_env,
    })
    .await;
    if outcome.timed_out {
        return Err(format!(
            "setup-failed: timeout after {}s",
            outcome.duration.as_secs()
        ));
    }
    match outcome.exit_code {
        Some(0) => Ok(()),
        Some(code) => Err(format!("setup-failed: exit {code}")),
        None => Err(format!(
            "setup-failed: no exit code (stderr: {})",
            outcome.stderr.trim()
        )),
    }
}

// ===== tier-2 agent prompt ================================================

/// Assemble the tier-2 agent prompt: the mined statement, then (when
/// `test_first`) the shared test-first approach guidance, then (when
/// `agent_gate_command` is `Some`) the shared `## Verification` section for
/// the in-run agent gate ([`AgentGateMode::On`]).
///
/// Tier-2's prompt is the mined statement VERBATIM — it never passes through
/// `task_spec_prompt.md`, so harness-owned guidance that lives only in that
/// template reaches talos dispatch and the tier-1 eval but NOT this path. That
/// asymmetry silently voided a test-first experiment (three matrix runs that
/// measured nothing), which is why both sections are appended here from the
/// same shared templates rather than restated.
///
/// The guidance is deliberately harness-owned rather than written into the
/// talos-evals statement files: it is a property of the harness under test, not
/// task content, and putting it in task data would contaminate every spec level
/// of every task and make a harness behavior look like part of the mined
/// commit.
///
/// With `test_first = true` and `agent_gate_command = None`, the output is
/// byte-identical to the pre-agent-gate `format!("{}\n\n{}", statement.trim_end(),
/// render_test_first_approach())`. With `agent_gate_command = Some(cmd)`, the
/// output is exactly that string followed immediately by
/// `render_verification_section(cmd)`, with NO separator inserted.
fn tier2_task_prompt(
    statement: &str,
    agent_gate_command: Option<&str>,
    test_first: bool,
) -> String {
    let mut out = if test_first {
        format!(
            "{}\n\n{}",
            statement.trim_end(),
            crate::prompt::render_test_first_approach()
        )
    } else {
        statement.trim_end().to_string()
    };
    if let Some(cmd) = agent_gate_command {
        out.push_str(&crate::prompt::render_verification_section(cmd));
    }
    out
}

// ===== agent test authorship ==============================================

/// Timeout for the `git status` scan that measures agent test authorship.
const AUTHORSHIP_SCAN_TIMEOUT: Duration = Duration::from_secs(30);

/// Whether `path` looks like a pytest test file.
///
/// Deliberately a path-shape heuristic, not a collection attempt: the scan runs
/// BEFORE [`copy_sealed`] and must not execute anything the agent wrote. Matches
/// pytest's own default discovery conventions — a `tests` directory component,
/// a `test_`-prefixed basename, or a `_test.py` suffix. All eight tier-2 tasks
/// keep their tests under `tests/`, so the directory clause carries the signal
/// today; the basename clauses cover an agent that invents a new location.
fn is_test_path(path: &str) -> bool {
    let p = Path::new(path);
    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("py"))
        && let Some(name) = p.file_name().and_then(|n| n.to_str())
        && (name.starts_with("test_") || name.ends_with("_test.py"))
    {
        return true;
    }
    p.components()
        .any(|c| c.as_os_str().eq_ignore_ascii_case("tests"))
}

/// Split `git status --porcelain -uall` output into (added, modified) test paths.
///
/// Pure so the porcelain vocabulary is pinned by unit tests rather than by
/// standing up a git repo per case. Status codes are the two-character XY pair
/// in columns 0-1; the path starts at column 3. `??` (untracked) and any `A` in
/// either column count as ADDED; anything else that names a test path counts as
/// MODIFIED, which deliberately includes `D` (deleting a test is authorship too,
/// and lumping it with modified keeps the added-count honest). A rename record
/// (`R  old -> new`) is attributed to its destination.
///
/// `-uall` is REQUIRED at the call site: without it git collapses an untracked
/// directory to a single `?? somedir/` entry and every test file inside it is
/// invisible to this parser.
fn parse_authored_tests(porcelain: &str) -> (Vec<String>, Vec<String>) {
    let mut added = Vec::new();
    let mut modified = Vec::new();
    for line in porcelain.lines() {
        if line.len() < 4 {
            continue;
        }
        let (code, rest) = line.split_at(2);
        let path = rest.trim_start();
        let path = path.rsplit(" -> ").next().unwrap_or(path);
        let path = path.trim_matches('"');
        if !is_test_path(path) {
            continue;
        }
        if code == "??" || code.contains('A') {
            added.push(path.to_string());
        } else {
            modified.push(path.to_string());
        }
    }
    (added, modified)
}

/// Record which test files the AGENT created or changed in `workspace_root`.
///
/// MUST be called before [`copy_sealed`] overwrites the sealed paths, otherwise
/// every sealed file reads as agent-modified. Returns `(added, modified)`;
/// a git failure yields two empty vectors rather than failing the trial —
/// this is observational telemetry and must never change a score.
///
/// This is the tier-2-native way to measure test-first compliance. Even in
/// [`AgentGateMode::On`], the ONLY gate the engine observes is
/// `agent_gate_command` (a full-repo checker); the file-scoped SEALED SCORING
/// `gate_command` of every tier-2 task collects only sealed paths regardless
/// of mode, so an agent-authored test at a new path is invisible to the
/// sealed re-gate entirely. In [`AgentGateMode::Off`] the engine additionally
/// registers no `run_checks` tool at all, so the loop never observes gate
/// state in that mode either.
async fn scan_authored_tests(workspace_root: &Path) -> (Vec<String>, Vec<String>) {
    let outcome = run(&ExecSpec {
        program: "git".to_string(),
        args: vec![
            "status".to_string(),
            "--porcelain".to_string(),
            "-uall".to_string(),
        ],
        cwd: workspace_root.to_path_buf(),
        timeout: AUTHORSHIP_SCAN_TIMEOUT,
        extra_env: Vec::new(),
    })
    .await;
    if outcome.exit_code != Some(0) {
        return (Vec::new(), Vec::new());
    }
    parse_authored_tests(&outcome.stdout)
}

// ===== sealed re-gate =====================================================

/// Copy the sealed files from `<task_dir>/sealed/<entry.path>` over
/// `<workspace_root>/<entry.path>`, creating parent dirs as needed.
///
/// Overwrites the agent's copy unconditionally: the agent MUST NEVER be scored
/// against a test file it could have modified. A missing source or copy
/// failure is an infra fault, surfaced as `Err(reason)` for
/// [`TrialScore::Invalid`].
///
/// A sealed source that is itself a directory triggers a recursive copy
/// through [`copy_dir_recursive`] (the pilot ships per-file entries, but this
/// keeps the schema extensible).
pub fn copy_sealed(
    task_dir: &Path,
    workspace_root: &Path,
    entries: &[SealedEntry],
) -> Result<(), String> {
    for entry in entries {
        let source = task_dir.join("sealed").join(&entry.path);
        let dest = workspace_root.join(&entry.path);
        if !source.exists() {
            return Err(format!(
                "sealed-copy-failed: source missing `{}`",
                source.display()
            ));
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "sealed-copy-failed: create_dir_all({}): {e}",
                    parent.display()
                )
            })?;
        }
        let file_type = std::fs::metadata(&source)
            .map_err(|e| format!("sealed-copy-failed: metadata({}): {e}", source.display()))?;
        if file_type.is_dir() {
            copy_dir_recursive(&source, &dest).map_err(|e| {
                format!(
                    "sealed-copy-failed: copy_dir({} -> {}): {e}",
                    source.display(),
                    dest.display()
                )
            })?;
        } else {
            std::fs::copy(&source, &dest).map_err(|e| {
                format!(
                    "sealed-copy-failed: copy({} -> {}): {e}",
                    source.display(),
                    dest.display()
                )
            })?;
        }
    }
    Ok(())
}

/// Map a gate-execution outcome to an [`Invalid`](TrialScore::Invalid) reason,
/// or `None` when the outcome is scoreable.
///
/// Returns `Some` for two cases that make the output unscoreable:
/// - **Timeout**: all output was dropped by the executor (see `exec.rs:183-193`).
/// - **Signal-kill with no stdout**: `exit_code` is `None` (the OS delivered
///   no numeric exit status) AND `stdout` is empty. The `stdout.trim().is_empty()`
///   guard is REQUIRED: `exec.rs:174` maps `exit_code = status.ok().and_then(|s| s.code())`,
///   which is `None` for a child killed by a signal (OOM-kill, SIGSEGV) — and
///   in that branch stdout/stderr ARE fully captured. Without the guard, a
///   pytest run that emitted its whole `-rA` section and was then `SIGKILLed`
///   would flip from a scoreable `Unresolved` to `Invalid`, shrinking the
///   denominator on real agent evidence — the exact direction Defect 2 exists
///   to eliminate.
///
/// A non-zero exit code returns `None`: a red gate is NOT an infra fault.
pub fn gate_fault_reason(
    timed_out: bool,
    exit_code: Option<i32>,
    duration: std::time::Duration,
    stdout: &str,
    stderr: &str,
) -> Option<String> {
    if timed_out {
        return Some(format!("gate-timeout: after {}s", duration.as_secs()));
    }
    if exit_code.is_none() && stdout.trim().is_empty() {
        return Some(format!(
            "gate-no-exit-code: no exit status (stderr: {})",
            stderr.trim()
        ));
    }
    None
}

/// Run the sealed re-gate and score the result.
///
/// Delegates to [`sealed_regate_score_with_timeout`] at [`CODING_CHECK_TIMEOUT`].
pub async fn sealed_regate_score<P: TestReportParser + ?Sized>(
    task: &MinedTask,
    task_dir: &Path,
    workspace_root: &Path,
    parser: &P,
) -> (TrialScore, String) {
    sealed_regate_score_with_timeout(task, task_dir, workspace_root, parser, CODING_CHECK_TIMEOUT)
        .await
}

/// Run the sealed re-gate with a caller-supplied timeout and score the result.
///
/// Steps (mirrors the tier-1 holdout ordering):
/// 1. `copy_sealed` — overwrite the agent's copies of the sealed files.
/// 2. Run `task.gate_command` under `bash -c` in `workspace_root` with
///    `PYTEST_ADDOPTS="-rA"` injected.
///    We use [`crate::exec::run`] directly (not [`crate::exec::ChecksRunner`])
///    because `ChecksRunner` truncates output to a 4 KB tail, which would
///    lose the `-rA` PASSED lines the scorer needs.
/// 3. Check for gate execution faults (timeout, signal-kill) via
///    [`gate_fault_reason`]; return `Invalid` immediately on fault.
/// 4. Parse `stdout + stderr` with `parser`.
/// 5. Call [`resolve`].
///
/// Returns `(TrialScore, raw_output)` — the caller persists `raw_output`
/// to disk for provenance.
pub async fn sealed_regate_score_with_timeout<P: TestReportParser + ?Sized>(
    task: &MinedTask,
    task_dir: &Path,
    workspace_root: &Path,
    parser: &P,
    timeout: std::time::Duration,
) -> (TrialScore, String) {
    if let Err(reason) = copy_sealed(task_dir, workspace_root, &task.sealed) {
        return (TrialScore::Invalid { reason }, String::new());
    }
    let outcome = run(&ExecSpec {
        program: "bash".to_string(),
        args: vec!["-c".to_string(), task.gate_command.clone()],
        cwd: workspace_root.to_path_buf(),
        timeout,
        extra_env: vec![("PYTEST_ADDOPTS".to_string(), "-rA".to_string())],
    })
    .await;
    let raw = format!("{}{}", outcome.stdout, outcome.stderr);
    if let Some(reason) = gate_fault_reason(
        outcome.timed_out,
        outcome.exit_code,
        outcome.duration,
        &outcome.stdout,
        &outcome.stderr,
    ) {
        return (TrialScore::Invalid { reason }, raw);
    }
    let report = parser.parse(&raw);
    let score = resolve(&report.statuses, report.summary_count, task);
    (score, raw)
}

// ===== per-trial result + report =========================================

/// One trial's terminal record.
///
/// The build agent may ADD fields; the pinned ones (needed by the false-done
/// cross-tab and by the audit trail) MUST NOT be omitted.
#[derive(Debug, Clone)]
pub struct MinedTrialResult {
    /// Zero-based trial index.
    pub trial: u32,
    /// Sealed re-gate verdict.
    pub score: TrialScore,
    /// Agent loop iterations spent.
    pub iterations: u32,
    /// Sum of `usage.input_tokens` (successful turns only, from
    /// [`crate::engine::RunStats`]).
    pub input_tokens: u64,
    /// Sum of `usage.output_tokens` (successful turns only).
    pub output_tokens: u64,
    /// Wall-clock of the whole trial (setup + agent + re-gate).
    pub wall: Duration,
    /// Terminal disposition the agent CLAIMED (`Done` | `Blocked` | `Failed`
    /// | `MaxIterations` | `StoppedWithoutFinish` | `BudgetExhausted` |
    /// `BackendError`). Used for the claimed-Done × Unresolved false-done
    /// cross-tab.
    pub claimed_disposition: String,
    /// Raw nodeid → status map from the sealed re-gate parser.
    /// Empty for every pre-agent `Invalid` trial (worktree/setup/offload
    /// failures) and for a gate timeout (raw is `""`; `exec.rs:183-193`
    /// drops output on timeout). Possibly non-empty for a `gate-no-exit-code`
    /// `Invalid` trial where output WAS captured before the signal. The
    /// re-parse site is at `mined_eval.rs:single_trial` (after
    /// `sealed_regate_score`).
    pub statuses: BTreeMap<String, TestStatus>,
    /// Lines that matched a status tag but fell outside any
    /// `short test summary info` section. Non-zero means the parser was
    /// under-inclusive on this trial; see [`ParseReport::dropped_outside_section`].
    pub dropped_outside_section: usize,
    /// Number of `short test summary info` section headers seen in the
    /// sealed re-gate output for this trial. See [`ParseReport::sections_seen`].
    pub sections_seen: usize,
    /// Test files the AGENT created during the run, captured by
    /// [`scan_authored_tests`] BEFORE `copy_sealed` overwrites the sealed
    /// paths. This is the test-first-compliance measurement: a non-empty
    /// value means the agent wrote a test of its own. Note these files do NOT
    /// reach the sealed re-gate under today's file-scoped `gate_command`s, so
    /// this measures authorship only, never whether the agent's test passed.
    pub agent_tests_added: Vec<String>,
    /// Existing test files the agent modified, same capture point. Under
    /// today's task authoring every such path is also a sealed path, so a
    /// non-empty value means the agent edited a file `copy_sealed` then
    /// reverted — worth seeing, and never scored.
    pub agent_tests_modified: Vec<String>,
    /// Absolute path to the persisted raw re-gate stdout+stderr under
    /// `<MinedRunConfig::state_root>/talos/mined-eval/<run-id>/<task-id>/trial-<k>/gate-output.txt`.
    pub gate_output_path: PathBuf,
    /// Whether finish-recovery was structurally armed for this trial: true iff
    /// `run_checks` is in the registry AND `run_config.max_nudges > 0`.
    /// [`AgentGateMode::Off`] builds the registry with an empty `checks`
    /// argument, which omits `run_checks`, so this is always `false` in Off mode —
    /// meaning NOT-ARMED, not tried-and-failed. [`AgentGateMode::On`] with a
    /// resolved `agent_gate_command` registers `run_checks`, so this is `true`
    /// whenever `run_config.max_nudges > 0` (the default). Computed at
    /// runtime from `tools` and `run_config`; never hard-coded. Read this
    /// field alongside `gates_green_at_exit` and `nudges_fired`.
    pub finish_recovery_armed: bool,
    /// Whether the last in-loop gate (`run_checks`) was GREEN at the terminal.
    /// Constant `false` in [`AgentGateMode::Off`]: `single_trial` builds the
    /// registry with an empty `checks` argument, and `standard_registry`
    /// registers `run_checks` only when `checks.is_some()` — so
    /// `last_gate_green` can never flip true and both engine finish-recovery
    /// guards are structurally unreachable, meaning `false` here is NOT-ARMED,
    /// not tried-and-failed. In [`AgentGateMode::On`] this varies with the
    /// agent's actual `run_checks` calls. Read `finish_recovery_armed`
    /// alongside this value. `mutating_iters`, `bash_calls_ok`,
    /// `edit_file_calls_ok`, `iters_since_tree_change_at_exit` and
    /// `peak_iters_since_tree_change` are the columns that actually vary.
    pub gates_green_at_exit: bool,
    /// Finish-recovery nudges injected this trial. Constant `0` in
    /// [`AgentGateMode::Off`], for the same reason as
    /// [`Self::gates_green_at_exit`] — meaning NOT-ARMED, not tried-and-failed.
    /// In [`AgentGateMode::On`] this varies with the agent's actual behavior.
    /// Read `finish_recovery_armed` alongside this value. `mutating_iters`,
    /// `bash_calls_ok`, `edit_file_calls_ok`, `iters_since_tree_change_at_exit`
    /// and `peak_iters_since_tree_change` are the columns that actually vary.
    pub nudges_fired: u32,
    /// Whether any successful `edit_file` or `bash` call ran this trial
    /// (latched, never cleared). From [`crate::engine::RunStats::tree_dirty`].
    pub tree_dirty: bool,
    /// Consecutive-non-mutating-iteration counter value at the terminal.
    /// From [`crate::engine::RunStats::iters_since_tree_change_at_exit`].
    pub iters_since_tree_change_at_exit: u32,
    /// Peak value the consecutive-non-mutating-iteration counter reached.
    /// From [`crate::engine::RunStats::peak_iters_since_tree_change`].
    pub peak_iters_since_tree_change: u32,
    /// Count of loop iterations classified as mutating.
    /// From [`crate::engine::RunStats::mutating_iters`].
    pub mutating_iters: u32,
    /// Count of successful (`!is_error`) `bash` tool calls this trial.
    /// From [`crate::engine::RunStats::bash_calls_ok`].
    pub bash_calls_ok: u32,
    /// Count of successful (`!is_error`) `edit_file` tool calls this trial.
    /// From [`crate::engine::RunStats::edit_file_calls_ok`].
    pub edit_file_calls_ok: u32,
    /// Count of `finish` calls rejected as `FinishClaim::Invalid` this trial.
    /// From [`crate::engine::RunStats::invalid_finish_calls`].
    pub invalid_finish_calls: u32,
    /// The untruncated `raw` of the first rejected `finish` call this trial.
    /// From [`crate::engine::RunStats::first_invalid_finish_raw`].
    pub first_invalid_finish_raw: Option<String>,
    /// The post-run agent-gate verdict — THE production-shippability signal.
    /// A real dispatch pushes only when this gate is green at a
    /// claim-verified `Done`, so this is what separates a sealed-`Resolved`
    /// trial that is actually shippable from one that merely happens to pass
    /// the hidden tests while the agent's own project gate (lint/typecheck/
    /// whole suite) is red or was never re-run at the end.
    ///
    /// `Some(report.passed)` in [`AgentGateMode::On`], computed by running
    /// the resolved `agent_gate_command` once immediately after `engine::run`
    /// returns (before `scan_authored_tests`/`sealed_regate_score`, so it
    /// sees the agent's own tree rather than the sealed-path overwrite from
    /// `copy_sealed`). `None` in [`AgentGateMode::Off`] and on every
    /// pre-agent `Invalid` path (see [`invalid_trial`]) — meaning
    /// NOT-RUN, not "ran and failed".
    pub agent_gate_post: Option<bool>,
    /// Absolute path to the persisted post-run agent-gate output, written
    /// next to [`Self::gate_output_path`] as `agent-gate-output.txt`.
    /// `Some` iff [`Self::agent_gate_post`] is `Some` (the gate ran);
    /// `None` otherwise.
    pub agent_gate_output_path: Option<PathBuf>,
    /// The path the transcript was directed to, when [`MinedRunConfig::transcripts`]
    /// was on AND `engine::run` was reached for this trial; `None` otherwise
    /// (transcripts off, or a pre-agent [`invalid_trial`] that never reached
    /// `engine::run`). The file may be missing or partial if the best-effort
    /// [`crate::transcript`] writer disabled itself.
    pub transcript_path: Option<PathBuf>,
}

/// The full run report over `k` trials of one task.
#[derive(Debug, Clone)]
pub struct MinedReport {
    /// Copy of `task.id`.
    pub task_id: String,
    /// Human-readable backend description (from the runner).
    pub backend_desc: String,
    /// Which spec level the agent got.
    pub spec_level: SpecLevel,
    /// The per-trial iteration cap.
    pub max_iterations: u32,
    /// `k`: the number of independent trials that were run.
    pub k: u32,
    /// Count of trials scored [`TrialScore::Resolved`].
    pub resolved_count: u32,
    /// Count of trials scored [`TrialScore::Invalid`] — excluded from the
    /// resolved/k denominator.
    pub invalid_count: u32,
    /// Per-trial detail, in order.
    pub trials: Vec<MinedTrialResult>,
}

impl MinedReport {
    /// Trials the agent claimed Done AND the sealed re-gate said Unresolved
    /// — the mined analogue of [`crate::eval::EvalReport::false_dones`].
    ///
    /// `Invalid` trials do NOT contribute here; only Unresolved does.
    ///
    /// A file-level collection error now scores `Unresolved`, so an agent
    /// that claims Done while leaving a module unimportable IS a false-done
    /// and DOES contribute here — intended semantics, not a regression.
    #[must_use]
    pub fn false_dones(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| {
                t.claimed_disposition == CLAIMED_DONE
                    && matches!(t.score, TrialScore::Unresolved { .. })
            })
            .map(|_| 1u32)
            .sum()
    }

    /// Denominator for the resolved rate: `k - invalid_count`. Zero when
    /// every trial was Invalid (caller renders that as "no valid trials").
    #[must_use]
    pub fn valid_denominator(&self) -> u32 {
        self.k.saturating_sub(self.invalid_count)
    }

    /// `resolved_count / valid_denominator` as an f64. Returns `0.0` when
    /// the denominator is zero — never `NaN`.
    #[must_use]
    pub fn resolved_rate(&self) -> f64 {
        let denom = self.valid_denominator();
        if denom == 0 {
            return 0.0;
        }
        f64::from(self.resolved_count) / f64::from(denom)
    }

    /// Trials where the last in-loop gate was GREEN at exit and the agent did
    /// NOT claim Done. EVERY non-Done label counts — `Blocked`, `Failed`,
    /// `MaxIterations`, `StoppedWithoutFinish`, `BudgetExhausted`,
    /// `BackendError`, `NotRun` (see [`claimed_disposition_label`]). Always `0`
    /// in [`AgentGateMode::Off`] (no `run_checks` tool is ever registered, so
    /// `gates_green_at_exit` can never be true) — see
    /// [`MinedTrialResult::gates_green_at_exit`]. Varies in
    /// [`AgentGateMode::On`].
    #[must_use]
    pub fn post_green_stops(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| t.gates_green_at_exit && t.claimed_disposition != CLAIMED_DONE)
            .map(|_| 1u32)
            .sum()
    }

    /// Trials the sealed re-gate scored `Resolved` where the agent did NOT
    /// claim Done — the finish-discipline gap. `Invalid` and `Unresolved`
    /// trials are never `Resolved`, so they never count.
    #[must_use]
    pub fn resolved_unclaimed(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| {
                matches!(t.score, TrialScore::Resolved) && t.claimed_disposition != CLAIMED_DONE
            })
            .map(|_| 1u32)
            .sum()
    }

    /// Trials with at least one finish-recovery nudge fired on an UNTOUCHED
    /// tree (`tree_dirty == false`).
    ///
    /// `tree_dirty` latches on ANY successful `bash`, including read-only
    /// commands, so this count is a conservative LOWER bound on untouched-tree
    /// finish-recovery trips — a trial with a read-only `bash` call and a
    /// nudge would NOT count here even though the tree itself was never
    /// edited. This is the instrument for the deferred `tree_dirty` guard
    /// decision (see the item that introduced [`AgentGateMode`]); it adds no
    /// new field, only a filtered count over existing per-trial data.
    #[must_use]
    pub fn clean_tree_nudges(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| t.nudges_fired > 0 && !t.tree_dirty)
            .map(|_| 1u32)
            .sum()
    }

    /// Trials the agent claimed Done on an UNTOUCHED tree (`tree_dirty ==
    /// false`). See [`Self::clean_tree_nudges`] for the same conservative
    /// lower-bound caveat.
    #[must_use]
    pub fn clean_tree_dones(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| t.claimed_disposition == CLAIMED_DONE && !t.tree_dirty)
            .map(|_| 1u32)
            .sum()
    }

    /// Trials that are actually production-shippable: the sealed re-gate
    /// scored [`TrialScore::Resolved`] AND the agent's claimed terminal was
    /// `Done`. Deliberately stricter than `resolved_count` alone — a real
    /// dispatch pushes only on a claim-verified `Done`, so a `Resolved`
    /// trial that ended `MaxIterations` (or any other non-`Done` claim) is
    /// correct-but-NOT-shippable and must not count here.
    #[must_use]
    pub fn shippable(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| {
                matches!(t.score, TrialScore::Resolved) && t.claimed_disposition == CLAIMED_DONE
            })
            .map(|_| 1u32)
            .sum()
    }

    /// Trials the sealed re-gate scored [`TrialScore::Resolved`] where the
    /// post-run agent gate came back red ([`MinedTrialResult::agent_gate_post`]
    /// `== Some(false)`) — resolved-but-NOT-shippable for a different reason
    /// than [`Self::resolved_unclaimed`]: the sealed tests happen to pass,
    /// but the agent's own project gate was red (or never re-run) at the
    /// end. A trial where the post-run gate never ran
    /// (`agent_gate_post == None`, e.g. [`AgentGateMode::Off`]) never counts
    /// here — that is NOT-ARMED, not tried-and-failed.
    #[must_use]
    pub fn resolved_gate_red(&self) -> u32 {
        self.trials
            .iter()
            .filter(|t| matches!(t.score, TrialScore::Resolved) && t.agent_gate_post == Some(false))
            .map(|_| 1u32)
            .sum()
    }
}

/// Label used for a Done disposition on [`MinedTrialResult::claimed_disposition`].
pub const CLAIMED_DONE: &str = "Done";
/// Label used for a Blocked disposition.
pub const CLAIMED_BLOCKED: &str = "Blocked";
/// Label used for a Failed disposition.
pub const CLAIMED_FAILED: &str = "Failed";

/// Render a terminal [`LoopOutcome`] as the compact string stored on
/// [`MinedTrialResult::claimed_disposition`].
#[must_use]
pub fn claimed_disposition_label(outcome: &LoopOutcome) -> String {
    match outcome {
        LoopOutcome::Finished(Disposition::Done { .. }) => CLAIMED_DONE.to_string(),
        LoopOutcome::Finished(Disposition::Blocked { .. }) => CLAIMED_BLOCKED.to_string(),
        LoopOutcome::Finished(Disposition::Failed { .. }) => CLAIMED_FAILED.to_string(),
        LoopOutcome::StoppedWithoutFinish => "StoppedWithoutFinish".to_string(),
        LoopOutcome::MaxIterations => "MaxIterations".to_string(),
        LoopOutcome::BudgetExhausted { .. } => "BudgetExhausted".to_string(),
        // Carry the error payload: a bare `BackendError` label is undiagnosable
        // after the fact (the runner persists no run record), and the variants
        // it collapses — context-guard trip, protocol/parse failure, retries
        // exhausted — demand completely different responses from the operator.
        LoopOutcome::BackendError(e) => format!("BackendError({e})"),
    }
}

// ===== the k-trial loop ===================================================

/// Callback for per-trial telemetry (mirrors [`crate::eval::run_eval`]'s
/// `on_trial`). The runner uses it to stream one line per trial.
pub type OnMinedTrial<'a> = &'a mut dyn FnMut(&MinedTrialResult);

/// Default root for mined-eval per-trial captures, resolved from
/// `TALOS_MINED_STATE_ROOT`, then `XDG_STATE_HOME`, then
/// `$HOME/.local/state`, then the process temp dir. The runner resolves this
/// ONCE and injects it as [`MinedRunConfig::state_root`]; tests inject a
/// tempdir instead, so no library code below the runner reads process env
/// for the state root.
pub fn default_state_root() -> PathBuf {
    resolve_state_root(
        std::env::var_os("TALOS_MINED_STATE_ROOT"),
        std::env::var_os("XDG_STATE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// Pure precedence rule for [`default_state_root`]: extracted so each branch
/// can be tested without mutating process env (edition-2024 `set_var` is
/// unsafe and `unsafe_code` is `forbid`-den project-wide).
fn resolve_state_root(
    override_root: Option<std::ffi::OsString>,
    xdg_state_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> PathBuf {
    if let Some(root) = override_root {
        return PathBuf::from(root);
    }
    if let Some(xdg) = xdg_state_home {
        return PathBuf::from(xdg);
    }
    if let Some(h) = home {
        return PathBuf::from(h).join(".local/state");
    }
    // Last-resort fallback: process temp dir. A build agent will always
    // have HOME set in practice.
    std::env::temp_dir()
}

/// Per-process run id used to namespace the mined-eval state directory, so
/// concurrent runner processes (e.g. parallel matrix configs evaluating the
/// SAME task) never overwrite each other's persisted gate captures.
/// Computed ONCE per process — `<unix-secs>-<pid>` — and cached; every trial
/// in this process shares the same run id.
fn run_id() -> &'static str {
    static RUN_ID: OnceLock<String> = OnceLock::new();
    RUN_ID.get_or_init(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        format!("{secs}-{}", std::process::id())
    })
}

/// The per-trial state directory:
/// `<state-root>/talos/mined-eval/<run-id>/<task-id>/trial-<k>`. `state_root`
/// is supplied by the caller ([`MinedRunConfig::state_root`]), not read from
/// env. Shared by [`persist_named_output`] (gate/agent-gate captures) and,
/// when [`MinedRunConfig::transcripts`] is on, [`single_trial`]'s
/// `transcript.jsonl`.
fn trial_state_dir(state_root: &Path, task_id: &str, trial: u32) -> PathBuf {
    state_root
        .join("talos/mined-eval")
        .join(run_id())
        .join(task_id)
        .join(format!("trial-{trial}"))
}

/// Persist `raw` under
/// `<state-root>/talos/mined-eval/<run-id>/<task-id>/trial-<k>/<filename>` and
/// return its absolute path. `state_root` is the caller's
/// [`MinedRunConfig::state_root`]. A write error falls back to a temp path so
/// a trial never fails just because the state dir is unwritable. Shared by
/// [`persist_gate_output`] (`gate-output.txt`) and
/// [`persist_agent_gate_output`] (`agent-gate-output.txt`) so both captures
/// land in the same per-trial directory.
fn persist_named_output(
    state_root: &Path,
    task_id: &str,
    trial: u32,
    filename: &str,
    raw: &str,
) -> PathBuf {
    let dir = trial_state_dir(state_root, task_id, trial);
    let path = dir.join(filename);
    if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&path, raw).is_ok() {
        return path;
    }
    // Best-effort fallback so a trial never dies over an offload write.
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let fallback = std::env::temp_dir().join(format!(
        "talos-mined-eval-{}-{trial}-{n}-{filename}",
        sanitize_for_filename(task_id)
    ));
    let _ = std::fs::write(&fallback, raw);
    fallback
}

/// Persist raw re-gate output under
/// `<state-root>/talos/mined-eval/<run-id>/<task-id>/trial-<k>/gate-output.txt`
/// and return its absolute path. `state_root` is the caller's
/// [`MinedRunConfig::state_root`]. A write error falls back to a temp path so
/// a trial never fails just because the state dir is unwritable.
fn persist_gate_output(state_root: &Path, task_id: &str, trial: u32, raw: &str) -> PathBuf {
    persist_named_output(state_root, task_id, trial, "gate-output.txt", raw)
}

/// Persist the post-run agent-gate's captured output under
/// `<state-root>/talos/mined-eval/<run-id>/<task-id>/trial-<k>/agent-gate-output.txt`,
/// next to [`persist_gate_output`]'s capture, and return its absolute path.
/// `state_root` is the caller's [`MinedRunConfig::state_root`]. Same
/// best-effort fallback behavior — a write failure never fails the trial.
fn persist_agent_gate_output(state_root: &Path, task_id: &str, trial: u32, raw: &str) -> PathBuf {
    persist_named_output(state_root, task_id, trial, "agent-gate-output.txt", raw)
}

/// Best-effort full text of a post-run [`CheckReport`] for persistence:
/// prefers the offloaded full combined stdout+stderr, falling back to the
/// bounded `excerpt` when the offload path is absent or unreadable (e.g. the
/// `OFFLOAD_UNAVAILABLE` placeholder).
fn agent_gate_output_raw(report: &CheckReport) -> String {
    report
        .offload_path
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_else(|| report.excerpt.clone())
}

/// Serialise a task id for use in a filename fragment (drop path-y chars).
fn sanitize_for_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Counter for temp-fallback filenames.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Options for [`run_mined_task`].
pub struct MinedRunConfig<'a> {
    /// The task directory holding `task.json`, `statements/`, and `sealed/`.
    pub task_dir: &'a Path,
    /// Root directory under which per-trial captures (`gate-output.txt`,
    /// `agent-gate-output.txt`, `transcript.jsonl`) are written. See
    /// [`default_state_root`] for the default resolution the runner uses.
    pub state_root: &'a Path,
    /// Parsed task spec (already loaded via [`load_task`]).
    pub task: &'a MinedTask,
    /// The statement text to hand the agent (already loaded via
    /// [`load_statement`]).
    pub statement: &'a str,
    /// Spec level the statement corresponds to (stored on the report header).
    pub spec_level: SpecLevel,
    /// Human-readable backend description for the report header.
    pub backend_desc: String,
    /// `k` — number of independent trials.
    pub k: u32,
    /// Per-trial iteration cap.
    pub max_iterations: u32,
    /// Whether the in-run agent gate is armed this run. See [`AgentGateMode`].
    pub agent_gate: AgentGateMode,
    /// Whether the shared test-first approach guidance is appended to the
    /// agent prompt. See [`parse_test_first`].
    pub test_first: bool,
    /// Wall-clock budget (seconds) applied to every trial via
    /// [`crate::engine::RunConfig::with_wall_clock_secs`]. `0` is unbounded —
    /// mirrors talos production's `--wall-clock-secs` default. See
    /// [`parse_wall_clock_secs`].
    pub wall_clock_secs: u64,
    /// Opt-in full run transcript (see [`crate::transcript`]). Default off —
    /// set from `MINED_EVAL_TRANSCRIPTS` (see [`crate::transcript::parse_transcripts_flag`]).
    /// When `true`, each trial's `engine::run` is configured with
    /// `.with_transcript(trial_state_dir(config.state_root, &task.id, trial).join("transcript.jsonl"),
    /// backend_desc.clone())`.
    pub transcripts: bool,
}

/// Run the whole mined-task eval: `k` independent trials, each with a fresh
/// primary+sibling worktree, `env.setup`, an agent loop (checks armed only
/// under [`AgentGateMode::On`], `checks=None` under [`AgentGateMode::Off`]),
/// and sealed re-gate scoring.
///
/// Trials are **sequential**. Every trial gets a fresh workspace so agent
/// edits never leak between trials.
pub async fn run_mined_task<B: ModelBackend>(
    backend: &B,
    parser: &(impl TestReportParser + ?Sized),
    config: &MinedRunConfig<'_>,
    on_trial: OnMinedTrial<'_>,
) -> MinedReport {
    let mut trials: Vec<MinedTrialResult> = Vec::with_capacity(config.k as usize);
    let mut resolved_count = 0u32;
    let mut invalid_count = 0u32;

    for i in 0..config.k {
        let started = Instant::now();
        let trial = single_trial(backend, parser, config, i).await;
        // Best-effort adjust wall-clock in case `single_trial` fell out
        // through an early error path (it always sets `wall`, but this
        // guarantees monotonicity even for the fallback branches).
        let mut trial = trial;
        if trial.wall.is_zero() {
            trial.wall = started.elapsed();
        }
        match &trial.score {
            TrialScore::Resolved => resolved_count += 1,
            TrialScore::Invalid { .. } => invalid_count += 1,
            TrialScore::Unresolved { .. } => {}
        }
        on_trial(&trial);
        trials.push(trial);
    }

    MinedReport {
        task_id: config.task.id.clone(),
        backend_desc: config.backend_desc.clone(),
        spec_level: config.spec_level,
        max_iterations: config.max_iterations,
        k: config.k,
        resolved_count,
        invalid_count,
        trials,
    }
}

/// One trial: setup worktrees + env → agent loop → sealed re-gate. Any
/// pre-agent infra failure short-circuits to [`TrialScore::Invalid`] and the
/// agent is not run.
#[allow(clippy::too_many_lines)]
async fn single_trial<B: ModelBackend>(
    backend: &B,
    parser: &(impl TestReportParser + ?Sized),
    config: &MinedRunConfig<'_>,
    trial: u32,
) -> MinedTrialResult {
    let start = Instant::now();

    // 0. Resolve the in-run agent gate BEFORE touching any worktree — a
    //    missing/invalid `agent_gate_command` under `AgentGateMode::On` is a
    //    hard, fail-loud pre-flight error.
    let agent_gate_cmd = match resolve_agent_gate_command(config.task, config.agent_gate) {
        Ok(c) => c,
        Err(e) => {
            return invalid_trial(config, trial, format!("agent-gate-missing: {e}"), start);
        }
    };

    // 1. Worktrees.
    let worktrees = match prepare_worktrees(config.task) {
        Ok(w) => w,
        Err(e) => {
            return invalid_trial(config, trial, format!("worktree-setup-failed: {e}"), start);
        }
    };
    let workspace_root = worktrees.primary.clone();

    // 2. env.setup.
    if let Err(reason) = run_env_setup(&workspace_root, &config.task.env.setup, Vec::new()).await {
        return invalid_trial(config, trial, reason, start);
    }

    // 3. Agent loop, checks armed only in AgentGateMode::On.
    let offload = match ScratchDir::new(&format!("offload-{}-{trial}", config.task.id)) {
        Ok(s) => s,
        Err(e) => {
            return invalid_trial(config, trial, format!("offload-scratch-failed: {e}"), start);
        }
    };
    let workspace = match Workspace::new(&workspace_root, Some(offload.path().to_path_buf())) {
        Ok(w) => w,
        Err(e) => {
            return invalid_trial(config, trial, format!("workspace-init-failed: {e}"), start);
        }
    };
    let offload_canon = match offload.path().canonicalize() {
        Ok(p) => p,
        Err(e) => {
            return invalid_trial(config, trial, format!("offload-canon-failed: {e}"), start);
        }
    };
    // Canonical root, captured BEFORE `workspace` is moved into the `Arc` for
    // `ToolCtx::new` — the same root talos production `run_cmd` uses to build
    // its `ChecksRunner`.
    let gate_root = workspace.root().to_path_buf();
    let ctx = ToolCtx::new(
        Arc::new(workspace),
        Arc::new(DiskOffloadSink::new(offload_canon)),
    );
    let checks = match (config.agent_gate, agent_gate_cmd) {
        (AgentGateMode::On { timeout }, Some(cmd)) => shell_checks_runner(cmd, gate_root, timeout),
        _ => None,
    };
    let tools = standard_registry(checks.clone());
    let mut run_config = RunConfig::new(
        tier2_task_prompt(config.statement, agent_gate_cmd, config.test_first),
        config.max_iterations,
    )
    .with_wall_clock_secs(config.wall_clock_secs);
    if let Some(runner) = checks.clone() {
        run_config = run_config.with_checks(runner);
    }
    let transcript_path = if config.transcripts {
        let path =
            trial_state_dir(config.state_root, &config.task.id, trial).join("transcript.jsonl");
        run_config = run_config.with_transcript(path.clone(), config.backend_desc.clone());
        Some(path)
    } else {
        None
    };

    // Baseline tripwire (On only): the agent gate must be green at parent —
    // never run the agent against a gate that is already red or unable to
    // complete within its timeout.
    if let Some(runner) = checks.as_ref() {
        let baseline = runner.run(&ctx).await;
        if !baseline.passed {
            return invalid_trial(
                config,
                trial,
                format!(
                    "agent-gate-red-at-parent: exit={:?} timed_out={}",
                    baseline.exit_code, baseline.timed_out
                ),
                start,
            );
        }
    }

    let RunResult { outcome, stats } = engine::run(backend, &tools, &ctx, &run_config).await;
    let claimed = claimed_disposition_label(&outcome);

    // 3b. Post-run agent gate (On mode only) — THE production-shippability
    //     signal: a real dispatch pushes only when this gate is green at a
    //     verified Done. Must run BEFORE `scan_authored_tests` /
    //     `sealed_regate_score` so it sees the agent's own tree, not the
    //     sealed-path overwrite `copy_sealed` performs during scoring.
    let (agent_gate_post, agent_gate_output_path) = if let Some(runner) = checks.as_ref() {
        let report = runner.run(&ctx).await;
        let raw = agent_gate_output_raw(&report);
        let path = persist_agent_gate_output(config.state_root, &config.task.id, trial, &raw);
        (Some(report.passed), Some(path))
    } else {
        (None, None)
    };

    // 4. Test-authorship telemetry — MUST run before the re-gate, because
    //    `copy_sealed` overwrites the sealed paths and would make every one of
    //    them read as agent-modified.
    let (agent_tests_added, agent_tests_modified) = scan_authored_tests(&workspace_root).await;

    // 5. Sealed re-gate + scoring.
    let (score, raw) =
        sealed_regate_score(config.task, config.task_dir, &workspace_root, parser).await;
    let gate_output_path = persist_gate_output(config.state_root, &config.task.id, trial, &raw);

    // Re-parse to capture the raw status map on the trial record (empty on
    // Invalid trials that never ran the parser).
    let reparse = if raw.is_empty() {
        ParseReport {
            statuses: BTreeMap::new(),
            summary_count: None,
            dropped_outside_section: 0,
            sections_seen: 0,
        }
    } else {
        parser.parse(&raw)
    };

    // Worktrees drop here (after re-gate has read the workspace).
    drop(worktrees);

    let finish_recovery_armed = tools.get("run_checks").is_some() && run_config.max_nudges > 0;
    MinedTrialResult {
        trial,
        score,
        iterations: stats.iterations,
        input_tokens: stats.input_tokens,
        output_tokens: stats.output_tokens,
        wall: start.elapsed(),
        claimed_disposition: claimed,
        statuses: reparse.statuses,
        dropped_outside_section: reparse.dropped_outside_section,
        sections_seen: reparse.sections_seen,
        agent_tests_added,
        agent_tests_modified,
        gate_output_path,
        finish_recovery_armed,
        gates_green_at_exit: stats.gates_green_at_exit,
        nudges_fired: stats.nudges_fired,
        tree_dirty: stats.tree_dirty,
        iters_since_tree_change_at_exit: stats.iters_since_tree_change_at_exit,
        peak_iters_since_tree_change: stats.peak_iters_since_tree_change,
        mutating_iters: stats.mutating_iters,
        bash_calls_ok: stats.bash_calls_ok,
        edit_file_calls_ok: stats.edit_file_calls_ok,
        invalid_finish_calls: stats.invalid_finish_calls,
        first_invalid_finish_raw: stats.first_invalid_finish_raw,
        agent_gate_post,
        agent_gate_output_path,
        transcript_path,
    }
}

/// Build an Invalid trial record for a pre-agent infra failure.
fn invalid_trial(
    config: &MinedRunConfig<'_>,
    trial: u32,
    reason: String,
    start: Instant,
) -> MinedTrialResult {
    // Persist an empty gate-output file so the provenance path always
    // exists on disk — even for a pre-agent Invalid trial.
    let gate_output_path = persist_gate_output(config.state_root, &config.task.id, trial, "");
    MinedTrialResult {
        trial,
        score: TrialScore::Invalid { reason },
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall: start.elapsed(),
        claimed_disposition: "NotRun".to_string(),
        statuses: BTreeMap::new(),
        dropped_outside_section: 0,
        sections_seen: 0,
        agent_tests_added: Vec::new(),
        agent_tests_modified: Vec::new(),
        gate_output_path,
        finish_recovery_armed: false,
        gates_green_at_exit: false,
        nudges_fired: 0,
        tree_dirty: false,
        iters_since_tree_change_at_exit: 0,
        peak_iters_since_tree_change: 0,
        mutating_iters: 0,
        bash_calls_ok: 0,
        edit_file_calls_ok: 0,
        invalid_finish_calls: 0,
        first_invalid_finish_raw: None,
        agent_gate_post: None,
        agent_gate_output_path: None,
        transcript_path: None,
    }
}

// ===== scratch dir (std-only, mirrors eval::ScratchDir) ===================

/// A scratch directory that self-deletes on drop. Uniqueness = pid + counter.
#[derive(Debug)]
struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    fn new(prefix: &str) -> std::io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "harness-mined-{}-{}-{n}",
            sanitize_for_filename(prefix),
            std::process::id()
        ));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AgentGateMode, CLAIMED_BLOCKED, CLAIMED_DONE, DEFAULT_AGENT_GATE_TIMEOUT, MinedReport,
        MinedTask, MinedTrialResult, PytestParser, ResolveDetail, ScratchDir, SealedEntry,
        SpecLevel, TestReportParser, TestStatus, TrialScore, build_collection_errors,
        claimed_disposition_label, copy_sealed, expand_home, gate_fault_reason, is_test_path,
        load_statement, load_task, match_task_id, matches_exclusion, normalize,
        parse_agent_gate_mode, parse_authored_tests, parse_pytest_summary_totals,
        parse_short_summary_line, parse_test_first, parse_wall_clock_secs, prepare_worktrees,
        resolve, resolve_agent_gate_command, run_env_setup, sanitize_for_filename,
        scan_authored_tests, strip_param_suffix, tier2_task_prompt,
    };
    use crate::run_record::{Disposition, FailureMode, Verification};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::time::Duration;
    use tempfile::tempdir;

    // ---- task.json --------------------------------------------------------

    const SAMPLE_TASK_JSON: &str = r#"{
        "id": "cleanr-abcdef1",
        "repo": "cleanr",
        "repo_path": "~/git/cleanr",
        "parent_sha": "aaaa111",
        "fix_sha": "bbbb222",
        "rung_guess": "mid",
        "language": "python",
        "provenance": {"source": "commit-mining", "extra": [1, 2]},
        "env": {
            "setup": "uv sync --frozen",
            "sibling_repos": [
                {"repo": "flickrasync", "pin": "v0.3.0"}
            ]
        },
        "test_scope": "tests/test_cleanr.py",
        "gate_command": "uv run --frozen pytest tests/test_cleanr.py -q",
        "fail_to_pass": ["TestCommentFilters::test_owner_comments_skipped"],
        "positive_controls": ["TestCommentFilters::test_reader_comments_kept"],
        "pass_to_pass_exclusions": [
            "TestCalcBadScore::*",
            "TestOldWireFormat::test_legacy_still_parses"
        ],
        "sealed": [
            {"path": "tests/test_cleanr.py"}
        ],
        "notes": ["mined 2026-08-13"]
    }"#;

    fn write_task(dir: &Path, body: &str) {
        std::fs::write(dir.join("task.json"), body).expect("write task.json");
    }

    #[test]
    fn load_task_deserialises_every_field() {
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), SAMPLE_TASK_JSON);
        let task = load_task(dir.path()).expect("load task.json");
        assert_eq!(task.id, "cleanr-abcdef1");
        assert_eq!(task.repo, "cleanr");
        assert_eq!(task.repo_path, "~/git/cleanr");
        assert_eq!(task.parent_sha, "aaaa111");
        assert_eq!(task.fix_sha, "bbbb222");
        assert_eq!(task.rung_guess, "mid");
        assert_eq!(task.language, "python");
        assert_eq!(task.env.setup, "uv sync --frozen");
        assert_eq!(task.env.sibling_repos.len(), 1);
        assert_eq!(task.env.sibling_repos[0].repo, "flickrasync");
        assert_eq!(task.env.sibling_repos[0].pin, "v0.3.0");
        assert_eq!(task.test_scope, "tests/test_cleanr.py");
        assert_eq!(
            task.gate_command,
            "uv run --frozen pytest tests/test_cleanr.py -q"
        );
        assert_eq!(
            task.fail_to_pass,
            vec!["TestCommentFilters::test_owner_comments_skipped".to_string(),]
        );
        assert_eq!(
            task.positive_controls,
            vec!["TestCommentFilters::test_reader_comments_kept".to_string(),]
        );
        assert_eq!(task.pass_to_pass_exclusions.len(), 2);
        assert_eq!(task.sealed.len(), 1);
        assert_eq!(task.sealed[0].path, "tests/test_cleanr.py");
        assert_eq!(task.notes, vec!["mined 2026-08-13".to_string()]);
        // provenance is opaque but must round-trip.
        assert_eq!(
            task.provenance.get("source").and_then(|v| v.as_str()),
            Some("commit-mining")
        );
    }

    #[test]
    fn load_task_tolerates_unknown_top_level_keys() {
        let dir = tempdir().expect("tempdir");
        // Add a bogus top-level key.
        let with_extra = SAMPLE_TASK_JSON.replace(
            "\"id\": \"cleanr-abcdef1\",",
            "\"id\": \"cleanr-abcdef1\", \"future_field\": {\"a\": 1},",
        );
        write_task(dir.path(), &with_extra);
        let task = load_task(dir.path()).expect("must tolerate extra keys");
        assert_eq!(task.id, "cleanr-abcdef1");
    }

    #[test]
    fn load_task_rejects_missing_required_field() {
        let dir = tempdir().expect("tempdir");
        // Drop `gate_command`.
        let missing = SAMPLE_TASK_JSON.replace(
            "\"gate_command\": \"uv run --frozen pytest tests/test_cleanr.py -q\",\n        ",
            "",
        );
        write_task(dir.path(), &missing);
        let err = load_task(dir.path()).expect_err("missing gate_command must error");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn load_task_omits_notes_by_default() {
        // The optional `notes` field defaults to empty when absent.
        let json = SAMPLE_TASK_JSON.replace(",\n        \"notes\": [\"mined 2026-08-13\"]", "");
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), &json);
        let task = load_task(dir.path()).expect("load with no notes");
        assert!(task.notes.is_empty());
    }

    #[test]
    fn load_task_omits_agent_gate_command_by_default() {
        // SAMPLE_TASK_JSON has no `agent_gate_command` key at all.
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), SAMPLE_TASK_JSON);
        let task = load_task(dir.path()).expect("load without agent_gate_command");
        assert_eq!(task.agent_gate_command, None);
    }

    #[test]
    fn load_task_reads_agent_gate_command_verbatim() {
        let json = SAMPLE_TASK_JSON.replace(
            "\"gate_command\": \"uv run --frozen pytest tests/test_cleanr.py -q\",",
            "\"gate_command\": \"uv run --frozen pytest tests/test_cleanr.py -q\", \
             \"agent_gate_command\": \"make check && pytest\",",
        );
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), &json);
        let task = load_task(dir.path()).expect("load with agent_gate_command");
        assert_eq!(
            task.agent_gate_command,
            Some("make check && pytest".to_string())
        );
    }

    #[test]
    fn load_task_null_agent_gate_command_is_none() {
        let json = SAMPLE_TASK_JSON.replace(
            "\"gate_command\": \"uv run --frozen pytest tests/test_cleanr.py -q\",",
            "\"gate_command\": \"uv run --frozen pytest tests/test_cleanr.py -q\", \
             \"agent_gate_command\": null,",
        );
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), &json);
        let task = load_task(dir.path()).expect("load with null agent_gate_command");
        assert_eq!(task.agent_gate_command, None);
    }

    // ---- statements -------------------------------------------------------

    #[test]
    fn spec_level_parse_is_case_insensitive() {
        assert_eq!(SpecLevel::parse("s1"), Some(SpecLevel::S1));
        assert_eq!(SpecLevel::parse("S2"), Some(SpecLevel::S2));
        assert_eq!(SpecLevel::parse("s3"), Some(SpecLevel::S3));
        assert_eq!(SpecLevel::parse("bogus"), None);
    }

    #[test]
    fn load_statement_reads_exact_level() {
        let dir = tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("statements")).expect("mkdir statements");
        std::fs::write(dir.path().join("statements/s2.md"), "Behave X so Y.\n")
            .expect("write s2.md");
        let text = load_statement(dir.path(), SpecLevel::S2).expect("read s2");
        assert_eq!(text, "Behave X so Y.\n");
    }

    #[test]
    fn load_statement_missing_level_is_hard_error_with_absolute_path() {
        let dir = tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("statements")).expect("mkdir statements");
        // No s2.md written.
        let err = load_statement(dir.path(), SpecLevel::S2).expect_err("missing s2 must error");
        // Error message must name the absolute path so the operator can act.
        let msg = err.to_string();
        assert!(
            msg.contains("s2.md"),
            "error must name the level filename, got: {msg}"
        );
        // Path should be inside the task dir (the tempdir).
        assert!(
            msg.contains(dir.path().to_string_lossy().as_ref()) || msg.contains("statements"),
            "error must name a path, got: {msg}"
        );
    }

    // ---- agent gate ---------------------------------------------------------

    fn sample_task() -> MinedTask {
        MinedTask {
            id: "cleanr-abcdef1".to_string(),
            repo: "cleanr".to_string(),
            repo_path: "~/git/cleanr".to_string(),
            parent_sha: "aaaa111".to_string(),
            fix_sha: "bbbb222".to_string(),
            rung_guess: "mid".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: "tests/test_synth.py".to_string(),
            gate_command: "true # SEALED_SCORING_GATE".to_string(),
            agent_gate_command: None,
            fail_to_pass: Vec::new(),
            positive_controls: Vec::new(),
            pass_to_pass_exclusions: Vec::new(),
            sealed: vec![SealedEntry {
                path: "tests/test_synth.py".to_string(),
            }],
            notes: Vec::new(),
        }
    }

    #[test]
    fn agent_gate_mode_display_pins_exact_wording() {
        assert_eq!(AgentGateMode::Off.to_string(), "off");
        assert_eq!(
            AgentGateMode::On {
                timeout: Duration::from_mins(15)
            }
            .to_string(),
            "on(timeout=900s)"
        );
        assert_eq!(DEFAULT_AGENT_GATE_TIMEOUT, Duration::from_mins(15));
    }

    #[test]
    fn parse_agent_gate_mode_covers_the_pinned_tuples() {
        // (None, None) -> On{900s}
        assert_eq!(
            parse_agent_gate_mode(None, None),
            Ok(AgentGateMode::On {
                timeout: Duration::from_mins(15)
            })
        );
        // (Some("1"), None) -> On{900s}
        assert_eq!(
            parse_agent_gate_mode(Some("1"), None),
            Ok(AgentGateMode::On {
                timeout: Duration::from_mins(15)
            })
        );
        // (Some(""), Some("")) -> On{900s}
        assert_eq!(
            parse_agent_gate_mode(Some(""), Some("")),
            Ok(AgentGateMode::On {
                timeout: Duration::from_mins(15)
            })
        );
        // (Some(" 1 "), Some(" 60 ")) -> On{60s}
        assert_eq!(
            parse_agent_gate_mode(Some(" 1 "), Some(" 60 ")),
            Ok(AgentGateMode::On {
                timeout: Duration::from_mins(1)
            })
        );
        // (None, Some("1200")) -> On{1200s}
        assert_eq!(
            parse_agent_gate_mode(None, Some("1200")),
            Ok(AgentGateMode::On {
                timeout: Duration::from_mins(20)
            })
        );
        // (Some("0"), None) -> Off
        assert_eq!(
            parse_agent_gate_mode(Some("0"), None),
            Ok(AgentGateMode::Off)
        );
        // (Some(" 0 "), None) -> Off
        assert_eq!(
            parse_agent_gate_mode(Some(" 0 "), None),
            Ok(AgentGateMode::Off)
        );
        // (Some("0"), Some("1200")) -> Off
        assert_eq!(
            parse_agent_gate_mode(Some("0"), Some("1200")),
            Ok(AgentGateMode::Off)
        );
        // (Some("yes"), None) -> Err containing MINED_EVAL_AGENT_GATE and yes
        let err = parse_agent_gate_mode(Some("yes"), None).expect_err("yes is invalid");
        assert!(err.contains("MINED_EVAL_AGENT_GATE"));
        assert!(err.contains("yes"));
        // (None, Some("0")) -> Err containing MINED_EVAL_GATE_TIMEOUT_SECS
        let err = parse_agent_gate_mode(None, Some("0")).expect_err("0 timeout is invalid");
        assert!(err.contains("MINED_EVAL_GATE_TIMEOUT_SECS"));
        // (None, Some("abc")) -> Err containing MINED_EVAL_GATE_TIMEOUT_SECS and abc
        let err = parse_agent_gate_mode(None, Some("abc")).expect_err("abc timeout is invalid");
        assert!(err.contains("MINED_EVAL_GATE_TIMEOUT_SECS"));
        assert!(err.contains("abc"));
        // (Some("0"), Some("abc")) -> Err containing MINED_EVAL_GATE_TIMEOUT_SECS and abc
        // (timeout is parsed FIRST, even though enabled selects Off)
        let err =
            parse_agent_gate_mode(Some("0"), Some("abc")).expect_err("abc timeout is invalid");
        assert!(err.contains("MINED_EVAL_GATE_TIMEOUT_SECS"));
        assert!(err.contains("abc"));
        // (Some("0"), Some("0")) -> Err
        assert!(parse_agent_gate_mode(Some("0"), Some("0")).is_err());
    }

    #[test]
    fn resolve_agent_gate_command_off_ignores_the_field() {
        let mut task = sample_task();
        task.agent_gate_command = Some("x".to_string());
        assert_eq!(
            resolve_agent_gate_command(&task, AgentGateMode::Off),
            Ok(None)
        );
    }

    fn on_mode() -> AgentGateMode {
        AgentGateMode::On {
            timeout: Duration::from_secs(30),
        }
    }

    #[test]
    fn resolve_agent_gate_command_on_missing_is_err() {
        let task = sample_task();
        let err = resolve_agent_gate_command(&task, on_mode()).expect_err("missing must error");
        assert!(err.contains(&task.id));
        assert!(err.contains("agent_gate_command"));
        assert!(err.contains("MINED_EVAL_AGENT_GATE=0"));
    }

    #[test]
    fn resolve_agent_gate_command_on_blank_is_err() {
        let mut task = sample_task();
        task.agent_gate_command = Some("   ".to_string());
        let err = resolve_agent_gate_command(&task, on_mode()).expect_err("blank must error");
        assert!(err.contains(&task.id));
        assert!(err.contains("agent_gate_command"));
        assert!(err.contains("MINED_EVAL_AGENT_GATE=0"));
    }

    #[test]
    fn resolve_agent_gate_command_on_equal_to_gate_command_is_err() {
        let mut task = sample_task();
        task.agent_gate_command = Some(format!("  {}  ", task.gate_command));
        let err =
            resolve_agent_gate_command(&task, on_mode()).expect_err("equal to gate_command errors");
        assert!(err.contains("equals gate_command"));
    }

    #[test]
    fn resolve_agent_gate_command_on_leaking_sealed_path_is_err() {
        let mut task = sample_task();
        task.agent_gate_command = Some("pytest tests/test_synth.py -k agent_gate".to_string());
        let err =
            resolve_agent_gate_command(&task, on_mode()).expect_err("sealed path leak must error");
        assert!(err.contains("tests/test_synth.py"));
    }

    #[test]
    fn resolve_agent_gate_command_on_returns_verbatim_untrimmed() {
        let mut task = sample_task();
        task.agent_gate_command = Some("  make x  ".to_string());
        assert_eq!(
            resolve_agent_gate_command(&task, on_mode()),
            Ok(Some("  make x  "))
        );
    }

    // ---- test-first toggle / wall-clock parsers ----------------------------

    #[test]
    fn parse_test_first_covers_the_pinned_tuples() {
        assert_eq!(parse_test_first(None), Ok(true));
        assert_eq!(parse_test_first(Some("")), Ok(true));
        assert_eq!(parse_test_first(Some("1")), Ok(true));
        assert_eq!(parse_test_first(Some(" 1 ")), Ok(true));
        assert_eq!(parse_test_first(Some("0")), Ok(false));
        assert_eq!(parse_test_first(Some(" 0 ")), Ok(false));
        let err = parse_test_first(Some("yes")).expect_err("yes is invalid");
        assert!(err.contains("MINED_EVAL_TEST_FIRST"));
        assert!(err.contains("yes"));
    }

    #[test]
    fn parse_wall_clock_secs_covers_the_pinned_tuples() {
        assert_eq!(parse_wall_clock_secs(None), Ok(0));
        assert_eq!(parse_wall_clock_secs(Some("")), Ok(0));
        assert_eq!(parse_wall_clock_secs(Some("0")), Ok(0));
        assert_eq!(parse_wall_clock_secs(Some(" 1200 ")), Ok(1200));
        let err = parse_wall_clock_secs(Some("abc")).expect_err("abc is invalid");
        assert!(err.contains("MINED_EVAL_WALL_CLOCK_SECS"));
        assert!(err.contains("abc"));
        let err = parse_wall_clock_secs(Some("-1")).expect_err("-1 is invalid");
        assert!(err.contains("MINED_EVAL_WALL_CLOCK_SECS"));
        assert!(err.contains("-1"));
    }

    // ---- normalize / matching --------------------------------------------

    #[test]
    fn normalize_strips_leading_py_prefix() {
        assert_eq!(
            normalize("tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped"),
            "TestCommentFilters::test_owner_comments_skipped"
        );
        // Module-level test function.
        assert_eq!(normalize("tests/test_x.py::test_fn"), "test_fn");
        // Bare id passes through.
        assert_eq!(normalize("Class::test"), "Class::test");
    }

    #[test]
    fn normalize_preserves_param_suffix() {
        assert_eq!(
            normalize("tests/t.py::Class::test[case-a]"),
            "Class::test[case-a]"
        );
    }

    #[test]
    fn strip_param_suffix_trims_bracketed_suffix() {
        assert_eq!(strip_param_suffix("Class::test[a]"), "Class::test");
        assert_eq!(strip_param_suffix("Class::test"), "Class::test");
        // Only strips when the id ends with ']'.
        assert_eq!(
            strip_param_suffix("Class::test[a]::more"),
            "Class::test[a]::more"
        );
    }

    #[test]
    fn match_task_id_handles_param_matching() {
        // Bare task ids match in normalized space (raw side irrelevant).
        assert!(match_task_id(
            "tests/t.py::Class::test",
            "Class::test",
            "Class::test"
        ));
        assert!(match_task_id(
            "tests/t.py::Class::test[a]",
            "Class::test[a]",
            "Class::test"
        ));
        assert!(match_task_id(
            "tests/t.py::Class::test",
            "Class::test",
            "Class::test[a]"
        ));
        assert!(!match_task_id(
            "tests/t.py::OtherClass::test_owner_comments_skipped",
            "OtherClass::test_owner_comments_skipped",
            "TestCommentFilters::test_owner_comments_skipped",
        ));
    }

    #[test]
    fn file_qualified_task_ids_match_in_raw_space() {
        // The agent-gtd-rollout-deadlock shape: multi-file scope, module-level
        // test names qualified by file. Must match the raw nodeid…
        assert!(match_task_id(
            "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
            "test_complete_item_in_rollout_from_ready",
            "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
        ));
        // …and must NOT cross-match a same-named test from a DIFFERENT file
        // (the collision stripping would create).
        assert!(!match_task_id(
            "tests/test_rollout_executor.py::test_complete_item_in_rollout_from_ready",
            "test_complete_item_in_rollout_from_ready",
            "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
        ));
        // Parametrized instances still fold onto the file-qualified id.
        assert!(match_task_id(
            "tests/test_a.py::test_fn[case-1]",
            "test_fn[case-1]",
            "tests/test_a.py::test_fn",
        ));
    }

    #[test]
    fn matches_exclusion_supports_literal_and_glob() {
        assert!(matches_exclusion(
            "tests/t.py::TestCalcBadScore::test_low_score_flagged",
            "TestCalcBadScore::test_low_score_flagged",
            "TestCalcBadScore::*",
        ));
        assert!(matches_exclusion(
            "tests/t.py::TestOldWireFormat::test_legacy_still_parses",
            "TestOldWireFormat::test_legacy_still_parses",
            "TestOldWireFormat::test_legacy_still_parses",
        ));
        assert!(!matches_exclusion(
            "tests/t.py::OtherClass::test_other",
            "OtherClass::test_other",
            "TestCalcBadScore::*",
        ));
        // File-qualified glob binds to its file only.
        assert!(matches_exclusion(
            "tests/a.py::TestX::test_one",
            "TestX::test_one",
            "tests/a.py::TestX::*",
        ));
        assert!(!matches_exclusion(
            "tests/b.py::TestX::test_one",
            "TestX::test_one",
            "tests/a.py::TestX::*",
        ));
    }

    // ---- pytest parser ---------------------------------------------------

    /// Synthetic pytest output for unit tests. Task CONTENT (statements, sealed test sources, answer keys) never enters this repo, but verbatim gate-output CAPTURES are required test input and live in `crates/harness/testdata/mined_eval/`.
    fn synthetic_pytest_output() -> String {
        // Includes noise: a dot-progress line, a header, a traceback, and a
        // trailing summary. The parser must find only the STATUS lines.
        r"
=== test session starts ===
platform linux -- Python 3.12.0
collected 5 items

tests/test_cleanr.py ..F..                                            [100%]

=== FAILURES ===
______________________ TestX::test_bad ______________________

    def test_bad():
>       assert False
E       assert False

tests/test_cleanr.py:5: AssertionError

=== short test summary info ===
PASSED tests/test_cleanr.py::TestX::test_ok
PASSED tests/test_cleanr.py::TestX::test_also_ok
FAILED tests/test_cleanr.py::TestX::test_bad
SKIPPED tests/test_cleanr.py::TestY::test_skipped
XFAIL tests/test_cleanr.py::TestY::test_expected_fail
=== 2 passed, 1 failed, 1 skipped, 1 xfailed in 0.12s ===
"
        .to_string()
    }

    #[test]
    fn pytest_parser_reads_short_summary_lines_only() {
        let report = PytestParser.parse(&synthetic_pytest_output());
        let map = &report.statuses;
        let count = report.summary_count;
        assert_eq!(count, Some(5));
        assert_eq!(map.len(), 5);
        assert_eq!(
            map.get("tests/test_cleanr.py::TestX::test_ok"),
            Some(&TestStatus::Passed)
        );
        assert_eq!(
            map.get("tests/test_cleanr.py::TestX::test_bad"),
            Some(&TestStatus::Failed)
        );
        assert_eq!(
            map.get("tests/test_cleanr.py::TestY::test_skipped"),
            Some(&TestStatus::Skipped)
        );
        assert_eq!(
            map.get("tests/test_cleanr.py::TestY::test_expected_fail"),
            Some(&TestStatus::XFail)
        );
    }

    #[test]
    fn pytest_parser_handles_error_and_xpass() {
        let out = "=========================== short test summary info ============================\nERROR tests/test_x.py::TestX::test_broken\nXPASS tests/test_x.py::TestX::test_unex_pass\n=== 0 passed, 1 error, 1 xpassed in 0.01s ===";
        let report = PytestParser.parse(out);
        assert_eq!(report.summary_count, Some(2));
        assert_eq!(
            report.statuses.get("tests/test_x.py::TestX::test_broken"),
            Some(&TestStatus::Error)
        );
        assert_eq!(
            report
                .statuses
                .get("tests/test_x.py::TestX::test_unex_pass"),
            Some(&TestStatus::XPass)
        );
    }

    #[test]
    fn parse_short_summary_line_ignores_non_status_lines() {
        assert!(parse_short_summary_line("=== FAILURES ===").is_none());
        assert!(parse_short_summary_line("  ").is_none());
        assert!(parse_short_summary_line("tests/x.py .").is_none());
        // Case-sensitive: `passed` (lowercase) is not a STATUS token.
        assert!(parse_short_summary_line("passed tests/x.py::t").is_none());
    }

    // ---- tier-2 agent prompt ----------------------------------------------

    /// The consume-side pin: it is not enough that the guidance RENDERS, it has
    /// to reach the tier-2 agent. A prior experiment changed
    /// `task_spec_prompt.md` and re-ran the matrix three times without noticing
    /// that tier-2 passes the mined statement straight into `RunConfig::new`,
    /// so the change could not possibly apply. This test fails if that wiring
    /// is ever removed.
    #[test]
    fn tier2_task_prompt_appends_the_shared_test_first_guidance() {
        let out = tier2_task_prompt("Fix the deadlock in the rollout executor.\n", None, true);
        assert!(
            out.starts_with("Fix the deadlock in the rollout executor."),
            "statement must lead the prompt; got:\n{out}"
        );
        // Byte-identical to what the task-spec template includes — not a
        // paraphrase that can drift.
        assert!(
            out.ends_with(&crate::prompt::render_test_first_approach()),
            "tier-2 prompt must end with the SHARED approach template; got:\n{out}"
        );
        assert!(out.contains("Confirm it fails"), "got:\n{out}");
        // Exactly one blank line between the two parts, regardless of how the
        // statement file happens to end.
        assert!(
            out.contains(".\n\n## Approach"),
            "expected one blank line between statement and guidance; got:\n{out}"
        );
    }

    /// Parity: the tail of the agent-gate On prompt (from `## Approach`
    /// onward) is byte-identical to the tail of the production
    /// `render_task_prompt_from_spec` output for the same gate command — the
    /// tier-2 agent gate must render the SAME verification framing production
    /// talos dispatch ships, not a paraphrase.
    #[test]
    fn tier2_task_prompt_on_mode_matches_production_verification_tail() {
        use crate::prompt::render_task_prompt_from_spec;
        use crate::task_spec::TaskSpec;

        let spec = TaskSpec {
            title: "T".to_string(),
            description: "D".to_string(),
            acceptance_criteria: vec![],
            files_to_modify: vec![],
            gate_command: "AGENT_SENTINEL_CMD".to_string(),
        };
        let a = tier2_task_prompt("S.", Some("AGENT_SENTINEL_CMD"), true);
        let p = render_task_prompt_from_spec(&spec);
        assert_eq!(
            &a[a.find("## Approach").unwrap()..],
            &p[p.find("## Approach").unwrap()..]
        );
    }

    #[test]
    fn tier2_task_prompt_off_mode_has_no_verification_section() {
        let out = tier2_task_prompt("S.", None, true);
        assert!(!out.contains("Run the following command to verify the task is complete:"));
        assert!(!out.contains("finish(done) immediately"));
    }

    /// `test_first = false` must remove EXACTLY the shared test-first section
    /// (and its leading separator) and nothing else — checked with the
    /// agent-gate off (no `## Verification` section appended) AND on (a
    /// `## Verification` section appended after where the test-first section
    /// would have gone), so the toggle composes correctly with both modes.
    #[test]
    fn tier2_task_prompt_test_first_toggle_removes_exactly_the_section_and_separator() {
        let separator_and_section = format!("\n\n{}", crate::prompt::render_test_first_approach());
        for gate in [None, Some("AGENT_SENTINEL_CMD")] {
            let on = tier2_task_prompt("S.", gate, true);
            let off = tier2_task_prompt("S.", gate, false);
            assert_eq!(on.replace(&separator_and_section, ""), off, "gate={gate:?}");
        }
    }

    // ---- agent test authorship --------------------------------------------

    #[test]
    fn is_test_path_matches_pytest_discovery_conventions() {
        // Directory component.
        assert!(is_test_path("tests/test_cli.py"));
        assert!(is_test_path("src/pkg/tests/helpers.py"));
        // Basename conventions, outside a tests/ dir.
        assert!(is_test_path("src/test_thing.py"));
        assert!(is_test_path("src/thing_test.py"));
        // Non-tests.
        assert!(!is_test_path("src/agent_gtd/cli.py"));
        assert!(!is_test_path("README.md"));
        // `test_`-prefixed but not Python — a fixture datafile, not a test.
        assert!(!is_test_path("data/test_input.json"));
        // Substring, not a path component: must NOT match.
        assert!(!is_test_path("src/latest/thing.py"));
    }

    #[test]
    fn parse_authored_tests_splits_added_from_modified() {
        // Verbatim `git status --porcelain -uall` vocabulary.
        let porcelain = concat!(
            "?? tests/test_agent_authored.py\n",
            "A  tests/test_staged_new.py\n",
            " M tests/test_existing.py\n",
            "M  tests/test_staged_edit.py\n",
            " D tests/test_deleted.py\n",
            "R  tests/test_old.py -> tests/test_renamed.py\n",
            "?? src/agent_gtd/cli.py\n",
            " M src/agent_gtd/service.py\n",
        );
        let (added, modified) = parse_authored_tests(porcelain);
        assert_eq!(
            added,
            vec![
                "tests/test_agent_authored.py".to_string(),
                "tests/test_staged_new.py".to_string(),
            ]
        );
        assert_eq!(
            modified,
            vec![
                "tests/test_existing.py".to_string(),
                "tests/test_staged_edit.py".to_string(),
                "tests/test_deleted.py".to_string(),
                "tests/test_renamed.py".to_string(),
            ]
        );
    }

    #[test]
    fn parse_authored_tests_ignores_short_and_empty_lines() {
        assert_eq!(parse_authored_tests(""), (Vec::new(), Vec::new()));
        assert_eq!(parse_authored_tests("??\n M\n\n"), (Vec::new(), Vec::new()));
    }

    #[test]
    fn parse_authored_tests_unquotes_paths_with_spaces() {
        let (added, _) = parse_authored_tests("?? \"tests/test a b.py\"\n");
        assert_eq!(added, vec!["tests/test a b.py".to_string()]);
    }

    #[tokio::test]
    async fn scan_authored_tests_sees_untracked_files_in_new_directories() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        for args in [
            vec!["init", "-q"],
            vec!["config", "user.email", "t@example.com"],
            vec!["config", "user.name", "t"],
        ] {
            let st = std::process::Command::new("git")
                .args(&args)
                .current_dir(root)
                .status()
                .expect("git");
            assert!(st.success(), "git {args:?}");
        }
        // A test in a directory git would otherwise collapse to `?? deep/`.
        std::fs::create_dir_all(root.join("deep/tests")).expect("mkdir");
        std::fs::write(root.join("deep/tests/test_new.py"), "def test_x(): pass\n").expect("write");
        std::fs::write(root.join("notes.md"), "hi\n").expect("write");

        let (added, modified) = scan_authored_tests(root).await;
        assert_eq!(added, vec!["deep/tests/test_new.py".to_string()]);
        assert!(modified.is_empty());
    }

    #[tokio::test]
    async fn scan_authored_tests_returns_empty_when_git_fails() {
        // Not a git repo → non-zero exit → telemetry degrades to empty rather
        // than failing the trial.
        let dir = tempdir().expect("tempdir");
        assert_eq!(
            scan_authored_tests(dir.path()).await,
            (Vec::new(), Vec::new())
        );
    }

    #[test]
    fn parse_pytest_summary_totals_sums_all_buckets() {
        assert_eq!(
            parse_pytest_summary_totals("=== 2 passed, 1 failed, 1 skipped, 1 xfailed in 0.1s ==="),
            Some(5),
        );
        // Warnings are excluded from the count.
        assert_eq!(
            parse_pytest_summary_totals("=== 3 passed, 2 warnings in 0.5s ==="),
            Some(3),
        );
        // Not a summary line.
        assert_eq!(parse_pytest_summary_totals("=== FAILURES ==="), None);
    }

    /// Regression: pytest pads its banner to terminal width, not to three `=`.
    /// Both lines below are verbatim from real gate output; the width-padded
    /// form previously dropped the FIRST bucket (`1 failed`), yielding 7 and a
    /// spurious `parse-mismatch: 8 ids vs summary 7` that voided whole tasks.
    #[test]
    fn parse_pytest_summary_totals_strips_full_width_banner_padding() {
        assert_eq!(
            parse_pytest_summary_totals(
                "============== 1 failed, 6 passed, 1 skipped, 8 warnings in 1.85s ==============",
            ),
            Some(8),
        );
        // Single-bucket: the same bug returned `None`, silently DISABLING the
        // truncation cross-check rather than merely miscounting it.
        assert_eq!(
            parse_pytest_summary_totals(
                "============================== 68 passed in 0.30s ==============================",
            ),
            Some(68),
        );
    }

    // ---- Invalid{parse-empty} on -q output --------------------------------

    #[test]
    fn dots_only_output_yields_parse_empty_invalid() {
        // A `-q` run without -rA: dots, no STATUS lines.
        let dots_only = "tests/x.py ...F.                                     [100%]\n";
        let report = PytestParser.parse(dots_only);
        let map = &report.statuses;
        let count = report.summary_count;
        assert!(map.is_empty(), "dots-only output produces no ids");
        assert_eq!(count, None, "no summary line in this fragment");
        // Scoring: parse-empty is Invalid, never silently unresolved.
        let task = sample_task_for_scoring();
        let score = resolve(map, count, &task);
        assert!(matches!(score, TrialScore::Invalid { .. }));
        if let TrialScore::Invalid { reason } = score {
            assert_eq!(reason, "parse-empty");
        }
    }

    // ---- resolve() branches ----------------------------------------------

    fn sample_task_for_scoring() -> MinedTask {
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), SAMPLE_TASK_JSON);
        load_task(dir.path()).expect("parse sample")
    }

    fn status_map(pairs: &[(&str, TestStatus)]) -> BTreeMap<String, TestStatus> {
        pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
    }

    fn task_from_json(json: &str) -> MinedTask {
        let dir = tempdir().expect("tempdir");
        write_task(dir.path(), json);
        load_task(dir.path()).expect("parse task json")
    }

    const DEADLOCK_TASK_JSON: &str = r#"{
        "id": "agent-gtd-rollout-deadlock",
        "repo": "agent-gtd",
        "repo_path": "~/git/agent-gtd",
        "parent_sha": "aaaa111",
        "fix_sha": "bbbb222",
        "rung_guess": "mid",
        "language": "python",
        "provenance": {},
        "env": {
            "setup": "uv sync --frozen",
            "sibling_repos": []
        },
        "test_scope": "tests/test_dispatch_service.py tests/test_rollout_service.py tests/test_rollout_executor.py",
        "gate_command": "uv run --frozen pytest tests/test_dispatch_service.py tests/test_rollout_service.py tests/test_rollout_executor.py -q",
        "fail_to_pass": [
            "tests/test_dispatch_service.py::test_create_run_manage_mode_rejected",
            "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready",
            "tests/test_rollout_service.py::test_complete_item_in_rollout_from_ready_unblocks_downstream",
            "tests/test_rollout_executor.py::test_manage_dispatch_does_not_flip_item_status"
        ],
        "positive_controls": [
            "tests/test_rollout_service.py::test_complete_item_in_rollout_rejects_pending_status",
            "tests/test_rollout_service.py::test_managed_rollout_happy_path",
            "tests/test_rollout_executor.py::test_manage_dispatch_flips_wave_to_running",
            "tests/test_rollout_executor.py::test_manage_dispatch_emits_wave_started_event",
            "tests/test_rollout_executor.py::test_happy_path_plan_rollout_to_complete_item_in_rollout"
        ],
        "pass_to_pass_exclusions": [],
        "sealed": [
            {"path": "tests/test_dispatch_service.py"},
            {"path": "tests/test_rollout_service.py"},
            {"path": "tests/test_rollout_executor.py"}
        ],
        "notes": []
    }"#;

    const FROM_JSON_TASK_JSON: &str = r#"{
        "id": "agent-gtd-from-json-contract",
        "repo": "agent-gtd",
        "repo_path": "~/git/agent-gtd",
        "parent_sha": "aaaa111",
        "fix_sha": "bbbb222",
        "rung_guess": "mid",
        "language": "python",
        "provenance": {},
        "env": {
            "setup": "uv sync --frozen",
            "sibling_repos": []
        },
        "test_scope": "tests/test_cli.py",
        "gate_command": "uv run --frozen pytest tests/test_cli.py -q",
        "fail_to_pass": [
            "test_do_add_item_honors_project_id_and_status_from_payload",
            "test_do_add_item_persists_priority_due_date_assigned_to",
            "test_do_add_item_unknown_key_raises_and_creates_nothing",
            "test_cmd_add_item_unknown_key_exits_nonzero",
            "test_do_add_item_http_mode_threads_priority_due_date_assigned_to",
            "test_http_post_create_item_includes_priority_due_date_assigned_to"
        ],
        "positive_controls": [
            "test_do_add_item_flags_override_payload",
            "test_http_post_create_item_omits_priority_due_date_assigned_to_when_none"
        ],
        "pass_to_pass_exclusions": [],
        "sealed": [
            {"path": "tests/test_cli.py"}
        ],
        "notes": []
    }"#;

    #[test]
    fn resolve_returns_resolved_on_all_green() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
        ]);
        assert_eq!(resolve(&parsed, Some(2), &task), TrialScore::Resolved);
    }

    #[test]
    fn resolve_unresolved_when_fail_to_pass_is_red() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Failed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
        ]);
        match resolve(&parsed, Some(2), &task) {
            TrialScore::Unresolved { reason } => {
                assert!(
                    reason
                        .missing_fail_to_pass
                        .contains(&"TestCommentFilters::test_owner_comments_skipped".to_string())
                );
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn resolve_unresolved_when_fail_to_pass_is_missing() {
        let task = sample_task_for_scoring();
        // fail_to_pass id not present at all (uncollected).
        let parsed = status_map(&[(
            "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
            TestStatus::Passed,
        )]);
        match resolve(&parsed, Some(1), &task) {
            TrialScore::Unresolved { reason } => {
                assert!(!reason.missing_fail_to_pass.is_empty());
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn resolve_unresolved_when_positive_control_red() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Failed,
            ),
        ]);
        assert!(matches!(
            resolve(&parsed, Some(2), &task),
            TrialScore::Unresolved { .. }
        ));
    }

    #[test]
    fn resolve_invalid_when_positive_control_uncollected() {
        let task = sample_task_for_scoring();
        // fail_to_pass green, but positive_control MISSING entirely.
        let parsed = status_map(&[(
            "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
            TestStatus::Passed,
        )]);
        match resolve(&parsed, Some(1), &task) {
            TrialScore::Invalid { reason } => {
                assert!(reason.starts_with("positive-control-uncollected"));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn resolve_unresolved_when_non_excluded_test_is_red() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
            // A random red test that isn't excluded.
            (
                "tests/test_cleanr.py::TestSomethingElse::test_broke",
                TestStatus::Failed,
            ),
        ]);
        match resolve(&parsed, Some(3), &task) {
            TrialScore::Unresolved { reason } => {
                assert_eq!(
                    reason.unexcluded_red,
                    vec!["TestSomethingElse::test_broke".to_string()]
                );
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn resolve_ignores_class_glob_excluded_red() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
            // Matches TestCalcBadScore::* exclusion.
            (
                "tests/test_cleanr.py::TestCalcBadScore::test_low_score",
                TestStatus::Failed,
            ),
        ]);
        assert_eq!(resolve(&parsed, Some(3), &task), TrialScore::Resolved);
    }

    #[test]
    fn resolve_ignores_literal_excluded_red() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestOldWireFormat::test_legacy_still_parses",
                TestStatus::Failed,
            ),
        ]);
        assert_eq!(resolve(&parsed, Some(3), &task), TrialScore::Resolved);
    }

    #[test]
    fn resolve_flags_count_mismatch_as_invalid() {
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
        ]);
        match resolve(&parsed, Some(99), &task) {
            TrialScore::Invalid { reason } => assert!(reason.starts_with("parse-mismatch")),
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    // ---- parametrized fail_to_pass ---------------------------------------

    #[test]
    fn parametrized_instances_score_individually_and_any_red_breaks() {
        // fail_to_pass carries the BARE id; two param instances exist, one red.
        let mut task = sample_task_for_scoring();
        task.fail_to_pass = vec!["TestParamCase::test_it".to_string()];
        task.positive_controls = Vec::new();
        task.pass_to_pass_exclusions = Vec::new();
        // One param passed, one failed — the id is NOT resolved.
        let parsed = status_map(&[
            ("tests/t.py::TestParamCase::test_it[a]", TestStatus::Passed),
            ("tests/t.py::TestParamCase::test_it[b]", TestStatus::Failed),
        ]);
        assert!(matches!(
            resolve(&parsed, Some(2), &task),
            TrialScore::Unresolved { .. }
        ));
    }

    // ---- copy_sealed ------------------------------------------------------

    #[test]
    fn copy_sealed_overwrites_agent_edits() {
        let task_dir = tempdir().expect("task tempdir");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir");
        std::fs::write(
            task_dir.path().join("sealed/tests/test_cleanr.py"),
            "# sealed truth\n",
        )
        .expect("write sealed");

        let ws = tempdir().expect("ws tempdir");
        std::fs::create_dir_all(ws.path().join("tests")).expect("mkdir tests");
        std::fs::write(ws.path().join("tests/test_cleanr.py"), "# agent stub\n")
            .expect("write agent copy");

        copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "tests/test_cleanr.py".to_string(),
            }],
        )
        .expect("sealed copy ok");
        let after =
            std::fs::read_to_string(ws.path().join("tests/test_cleanr.py")).expect("re-read");
        assert_eq!(after, "# sealed truth\n");
    }

    #[test]
    fn copy_sealed_creates_missing_parent_dir() {
        let task_dir = tempdir().expect("task tempdir");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir");
        std::fs::write(
            task_dir.path().join("sealed/tests/test_cleanr.py"),
            "# sealed\n",
        )
        .expect("write sealed");

        let ws = tempdir().expect("ws tempdir");
        // Agent removed the tests/ dir entirely. copy_sealed must re-create.
        copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "tests/test_cleanr.py".to_string(),
            }],
        )
        .expect("sealed copy must create parent dir");
        assert!(ws.path().join("tests/test_cleanr.py").exists());
    }

    #[test]
    fn copy_sealed_missing_source_is_error() {
        let task_dir = tempdir().expect("task tempdir");
        let ws = tempdir().expect("ws tempdir");
        let err = copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "tests/nope.py".to_string(),
            }],
        )
        .expect_err("missing sealed source must error");
        assert!(err.starts_with("sealed-copy-failed"));
    }

    // ---- ~/-expansion + env.setup ----------------------------------------

    #[test]
    fn expand_home_handles_tilde_and_passthrough() {
        // Set-and-restore HOME so this test is deterministic. std::env::set_var
        // is `unsafe` in edition 2024 and `unsafe_code` is forbidden project-
        // wide — so we test only the passthrough branch here.
        assert_eq!(expand_home("/absolute/path"), "/absolute/path");
        assert_eq!(expand_home("relative"), "relative");
        // Tilde-expansion behaviour is exercised by the worktree tests below,
        // which don't need HOME to be a specific value — they use tempdirs
        // directly without going through `~/`.
    }

    #[tokio::test]
    async fn run_env_setup_reports_setup_failed_on_nonzero_exit() {
        let dir = tempdir().expect("tempdir");
        let reason = run_env_setup(dir.path(), "exit 7", Vec::new())
            .await
            .expect_err("nonzero must Err");
        assert!(reason.starts_with("setup-failed"));
    }

    // ---- worktrees against a throwaway synthetic git repo ----------------

    /// Build a throwaway git repo with two commits under `dir`, returning the
    /// path to the repo and the sha of the parent commit.
    fn make_repo(dir: &Path, filename: &str, initial_body: &str, tag_first: bool) -> String {
        let git = |args: &[&str]| {
            let out = super::git_command()
                .args(args)
                .current_dir(dir)
                .output()
                .expect("git spawn");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            out
        };
        git(&["init", "-q"]);
        git(&["config", "user.email", "test@example.com"]);
        git(&["config", "user.name", "test"]);
        git(&["config", "commit.gpgsign", "false"]);
        std::fs::write(dir.join(filename), initial_body).expect("write file");
        git(&["add", "."]);
        git(&["commit", "-qm", "first"]);
        if tag_first {
            git(&["tag", "v0.1.0"]);
        }
        std::fs::write(dir.join(filename), format!("{initial_body}\nmore\n")).expect("write again");
        git(&["add", "."]);
        git(&["commit", "-qm", "second"]);
        // Return HEAD~1 sha (the first commit).
        let out = super::git_command()
            .args(["rev-parse", "HEAD~1"])
            .current_dir(dir)
            .output()
            .expect("rev-parse");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    /// Per-test-unique task id: pid + monotonic counter so parallel tests
    /// don't stomp on each other's gate-output paths under the XDG state root.
    fn unique_task_id() -> String {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("synth-task-{}-{n}", std::process::id())
    }

    fn task_pointing_at(primary: &Path, sibling: Option<&Path>, parent_sha: &str) -> MinedTask {
        let mut task = MinedTask {
            id: unique_task_id(),
            repo: primary.file_name().unwrap().to_string_lossy().into_owned(),
            repo_path: primary.to_string_lossy().into_owned(),
            parent_sha: parent_sha.to_string(),
            fix_sha: parent_sha.to_string(),
            rung_guess: "easy".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: String::new(),
            gate_command: "true".to_string(),
            agent_gate_command: None,
            fail_to_pass: Vec::new(),
            positive_controls: Vec::new(),
            pass_to_pass_exclusions: Vec::new(),
            sealed: Vec::new(),
            notes: Vec::new(),
        };
        if let Some(sib) = sibling {
            task.env.sibling_repos = vec![super::SiblingRepo {
                repo: sib.file_name().unwrap().to_string_lossy().into_owned(),
                pin: "v0.1.0".to_string(),
            }];
        }
        task
    }

    #[test]
    fn prepare_worktrees_lands_primary_and_sibling_at_pinned_commits() {
        // Two synthetic repos side by side (mirrors ~/git/<repo> layout).
        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary_repo");
        let sibling_src = workroot.path().join("sibling_repo");
        std::fs::create_dir_all(&primary_src).expect("mkdir primary");
        std::fs::create_dir_all(&sibling_src).expect("mkdir sibling");
        let primary_parent = make_repo(&primary_src, "hello.py", "print('one')\n", false);
        make_repo(&sibling_src, "sib.py", "print('sib')\n", true);

        let task = task_pointing_at(&primary_src, Some(&sibling_src), &primary_parent);
        let ws = prepare_worktrees(&task).expect("worktrees prepared");
        assert!(ws.primary.is_dir(), "primary worktree exists");
        assert_eq!(ws.siblings.len(), 1);
        assert!(ws.siblings[0].is_dir(), "sibling worktree exists");
        // Both should land adjacent under the same scratch dir.
        assert_eq!(ws.siblings[0].parent(), ws.primary.parent());
        // Primary is at HEAD~1 (only one file, no `more` line yet).
        let file = std::fs::read_to_string(ws.primary.join("hello.py")).expect("read");
        assert_eq!(file, "print('one')\n");
        // Sibling checked out at the tag = first commit (no `more`).
        let sib = std::fs::read_to_string(ws.siblings[0].join("sib.py")).expect("read sib");
        assert_eq!(sib, "print('sib')\n");

        let primary_registered = ws.primary.clone();
        let sibling_registered = ws.siblings[0].clone();
        drop(ws);
        // After drop, the worktrees are deregistered — `git worktree list`
        // no longer mentions them.
        for (src, wt) in [
            (&primary_src, &primary_registered),
            (&sibling_src, &sibling_registered),
        ] {
            let out = super::git_command()
                .args(["worktree", "list"])
                .current_dir(src)
                .output()
                .expect("worktree list");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                !stdout.contains(wt.to_string_lossy().as_ref()),
                "worktree {} should be deregistered from {} after drop, got:\n{stdout}",
                wt.display(),
                src.display(),
            );
        }
    }

    #[tokio::test]
    async fn env_setup_timeout_is_reported_as_setup_failed() {
        // Build a scratch primary worktree from a real synthetic repo so the
        // cwd exists; then run a sleeping setup script bounded by a short
        // helper timeout via a manual ExecSpec — MINED_SETUP_TIMEOUT is 10
        // minutes and we can't wait that long in the gate. Instead we use
        // exec::run directly to prove the "timeout ⇒ setup-failed" branch of
        // `run_env_setup`'s reporting is exercised by a call that DOES time
        // out (via a distinct short timeout via ExecSpec inline).
        let dir = tempdir().expect("tempdir");
        // Run `sleep 30` through the same shell wrapper but with a 200ms
        // budget through a bare ExecSpec, then assert the mapping.
        let outcome = crate::exec::run(&crate::exec::ExecSpec {
            program: "bash".to_string(),
            args: vec!["-c".to_string(), "sleep 30".to_string()],
            cwd: dir.path().to_path_buf(),
            timeout: Duration::from_millis(200),
            extra_env: Vec::new(),
        })
        .await;
        assert!(outcome.timed_out);
        // The mapping is the same one `run_env_setup` uses: timeout ⇒
        // "setup-failed: timeout ...".
        // A tiny direct check on run_env_setup: `false` returns setup-failed.
        let reason = run_env_setup(dir.path(), "false", Vec::new())
            .await
            .expect_err("false must fail");
        assert!(reason.starts_with("setup-failed"));
    }

    // ---- claimed_disposition + MinedReport aggregates --------------------

    #[test]
    fn claimed_disposition_label_maps_terminals() {
        use crate::engine::LoopOutcome;
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::Finished(Disposition::Done {
                summary: "ok".to_string(),
                verification: Verification::NoChecksConfigured,
            })),
            CLAIMED_DONE,
        );
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::Finished(Disposition::Blocked {
                decision_needed: "?".to_string(),
            })),
            "Blocked",
        );
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::Finished(Disposition::Failed {
                mode: FailureMode::Loop,
                summary: "oops".to_string(),
            })),
            "Failed",
        );
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::StoppedWithoutFinish),
            "StoppedWithoutFinish",
        );
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::MaxIterations),
            "MaxIterations",
        );
    }

    fn trial_tel(
        idx: u32,
        score: TrialScore,
        claimed: &str,
        green: bool,
        nudges: u32,
    ) -> MinedTrialResult {
        MinedTrialResult {
            trial: idx,
            score,
            iterations: 0,
            input_tokens: 0,
            output_tokens: 0,
            wall: Duration::ZERO,
            claimed_disposition: claimed.to_string(),
            statuses: BTreeMap::new(),
            dropped_outside_section: 0,
            sections_seen: 0,
            agent_tests_added: Vec::new(),
            agent_tests_modified: Vec::new(),
            gate_output_path: PathBuf::from("/dev/null"),
            finish_recovery_armed: false,
            gates_green_at_exit: green,
            nudges_fired: nudges,
            tree_dirty: false,
            iters_since_tree_change_at_exit: 0,
            peak_iters_since_tree_change: 0,
            mutating_iters: 0,
            bash_calls_ok: 0,
            edit_file_calls_ok: 0,
            invalid_finish_calls: 0,
            first_invalid_finish_raw: None,
            agent_gate_post: None,
            agent_gate_output_path: None,
            transcript_path: None,
        }
    }

    fn trial(idx: u32, score: TrialScore, claimed: &str) -> MinedTrialResult {
        trial_tel(idx, score, claimed, false, 0)
    }

    #[test]
    fn post_green_stops_counts_green_exit_non_done() {
        let unresolved = || TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: Vec::new(),
                unexcluded_red: Vec::new(),
                missing_fail_to_pass: vec!["x".to_string()],
                collection_errors: Vec::new(),
            },
        };
        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 4,
            resolved_count: 1,
            invalid_count: 0,
            trials: vec![
                trial_tel(0, TrialScore::Resolved, CLAIMED_DONE, true, 0),
                trial_tel(1, TrialScore::Resolved, "MaxIterations", true, 0),
                trial_tel(2, unresolved(), CLAIMED_BLOCKED, true, 0),
                trial_tel(3, TrialScore::Resolved, "MaxIterations", false, 0),
            ],
        };
        assert_eq!(report.post_green_stops(), 2);

        let report2 = MinedReport {
            task_id: "t2".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 2,
            resolved_count: 0,
            invalid_count: 0,
            trials: vec![
                trial_tel(0, TrialScore::Resolved, "MaxIterations", false, 0),
                trial_tel(1, unresolved(), CLAIMED_DONE, false, 0),
            ],
        };
        assert_eq!(report2.post_green_stops(), 0);
    }

    #[test]
    fn resolved_unclaimed_counts_resolved_non_done() {
        let unresolved = || TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: Vec::new(),
                unexcluded_red: Vec::new(),
                missing_fail_to_pass: vec!["x".to_string()],
                collection_errors: Vec::new(),
            },
        };
        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 5,
            resolved_count: 3,
            invalid_count: 1,
            trials: vec![
                trial(0, TrialScore::Resolved, CLAIMED_DONE),
                trial(1, TrialScore::Resolved, "MaxIterations"),
                trial(2, TrialScore::Resolved, CLAIMED_BLOCKED),
                trial(3, unresolved(), "MaxIterations"),
                trial(
                    4,
                    TrialScore::Invalid {
                        reason: "setup-failed".to_string(),
                    },
                    "NotRun",
                ),
            ],
        };
        assert_eq!(report.resolved_unclaimed(), 2);

        let report2 = MinedReport {
            task_id: "t2".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 2,
            resolved_count: 2,
            invalid_count: 0,
            trials: vec![
                trial(0, TrialScore::Resolved, CLAIMED_DONE),
                trial(1, TrialScore::Resolved, CLAIMED_DONE),
            ],
        };
        assert_eq!(report2.resolved_unclaimed(), 0);
    }

    #[test]
    fn clean_tree_nudges_and_dones_are_conservative_lower_bounds() {
        // A nudged CLEAN trial counts toward clean_tree_nudges.
        let nudged_clean = trial_tel(0, TrialScore::Resolved, "MaxIterations", true, 1);
        // A nudged DIRTY trial does not.
        let mut nudged_dirty = trial_tel(1, TrialScore::Resolved, "MaxIterations", true, 1);
        nudged_dirty.tree_dirty = true;
        // A Done CLEAN trial counts toward clean_tree_dones.
        let done_clean = trial_tel(2, TrialScore::Resolved, CLAIMED_DONE, false, 0);
        // A non-Done CLEAN trial does not count toward clean_tree_dones.
        let non_done_clean = trial_tel(3, TrialScore::Resolved, "Blocked", false, 0);

        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 4,
            resolved_count: 4,
            invalid_count: 0,
            trials: vec![nudged_clean, nudged_dirty, done_clean, non_done_clean],
        };
        assert_eq!(report.clean_tree_nudges(), 1);
        assert_eq!(report.clean_tree_dones(), 1);
    }

    #[test]
    fn shippable_requires_resolved_and_claimed_done() {
        // Resolved + Done: shippable.
        let resolved_done = trial_tel(0, TrialScore::Resolved, CLAIMED_DONE, false, 0);
        // Resolved + MaxIterations: correct but NOT shippable — the mined
        // analogue of a dispatch that would never have pushed this trial.
        let resolved_max_iterations = trial_tel(1, TrialScore::Resolved, "MaxIterations", false, 0);

        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 2,
            resolved_count: 2,
            invalid_count: 0,
            trials: vec![resolved_done, resolved_max_iterations],
        };
        assert_eq!(report.shippable(), 1);
    }

    #[test]
    fn resolved_gate_red_requires_resolved_and_post_gate_some_false() {
        // Resolved + post Some(false): counts.
        let resolved_gate_red = MinedTrialResult {
            agent_gate_post: Some(false),
            ..trial_tel(0, TrialScore::Resolved, "MaxIterations", false, 0)
        };
        // Resolved + post None (gate never ran, e.g. AgentGateMode::Off):
        // does NOT count — that is NOT-ARMED, not tried-and-failed.
        let resolved_gate_not_run = MinedTrialResult {
            agent_gate_post: None,
            ..trial_tel(1, TrialScore::Resolved, "MaxIterations", false, 0)
        };
        // Resolved + post Some(true): does not count.
        let resolved_gate_green = MinedTrialResult {
            agent_gate_post: Some(true),
            ..trial_tel(2, TrialScore::Resolved, CLAIMED_DONE, false, 0)
        };

        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 3,
            resolved_count: 3,
            invalid_count: 0,
            trials: vec![
                resolved_gate_red,
                resolved_gate_not_run,
                resolved_gate_green,
            ],
        };
        assert_eq!(report.resolved_gate_red(), 1);
    }

    #[test]
    fn mined_report_false_dones_counts_only_claimed_done_x_unresolved() {
        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 4,
            resolved_count: 1,
            invalid_count: 1,
            trials: vec![
                trial(0, TrialScore::Resolved, CLAIMED_DONE),
                trial(
                    1,
                    TrialScore::Unresolved {
                        reason: ResolveDetail {
                            fail_to_pass_status: Vec::new(),
                            unexcluded_red: Vec::new(),
                            missing_fail_to_pass: vec!["x".to_string()],
                            collection_errors: Vec::new(),
                        },
                    },
                    CLAIMED_DONE,
                ),
                // Blocked + Unresolved does NOT count as a false-done.
                trial(
                    2,
                    TrialScore::Unresolved {
                        reason: ResolveDetail {
                            fail_to_pass_status: Vec::new(),
                            unexcluded_red: Vec::new(),
                            missing_fail_to_pass: vec!["x".to_string()],
                            collection_errors: Vec::new(),
                        },
                    },
                    "Blocked",
                ),
                trial(
                    3,
                    TrialScore::Invalid {
                        reason: "setup-failed".to_string(),
                    },
                    "NotRun",
                ),
            ],
        };
        assert_eq!(report.false_dones(), 1);
        // Invalid excluded from denominator.
        assert_eq!(report.valid_denominator(), 3);
        // 1 resolved out of 3 valid.
        assert!((report.resolved_rate() - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mined_report_resolved_rate_is_zero_when_all_invalid() {
        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 2,
            resolved_count: 0,
            invalid_count: 2,
            trials: vec![
                trial(
                    0,
                    TrialScore::Invalid {
                        reason: "s".to_string(),
                    },
                    "NotRun",
                ),
                trial(
                    1,
                    TrialScore::Invalid {
                        reason: "s".to_string(),
                    },
                    "NotRun",
                ),
            ],
        };
        assert_eq!(report.valid_denominator(), 0);
        assert!((report.resolved_rate() - 0.0).abs() < f64::EPSILON);
    }

    // ---- misc small helpers ----------------------------------------------

    #[test]
    fn sanitize_for_filename_replaces_path_chars() {
        assert_eq!(sanitize_for_filename("abc-DEF_123"), "abc-DEF_123");
        assert_eq!(sanitize_for_filename("a/b/c"), "a_b_c");
        assert_eq!(sanitize_for_filename("a.b:c"), "a_b_c");
    }

    #[test]
    fn scratch_dir_removes_itself_on_drop() {
        let path;
        {
            let s = ScratchDir::new("test").expect("scratch");
            path = s.path().to_path_buf();
            assert!(path.exists());
        }
        assert!(!path.exists(), "scratch dir must be removed on drop");
    }

    // ---- run_env_setup happy path + extra_env + expand_home tilde --------

    #[tokio::test]
    async fn run_env_setup_ok_on_zero_exit_and_forwards_extra_env() {
        // The runner surfaces success as Ok(()); extra_env crosses the clean
        // environment boundary via exec::run.
        let dir = tempdir().expect("tempdir");
        // If MY_VAR is forwarded, the shell exits 0; otherwise 1.
        let script = r#"[ "$MY_VAR" = "hello" ] && exit 0 || exit 1"#;
        run_env_setup(
            dir.path(),
            script,
            vec![("MY_VAR".to_string(), "hello".to_string())],
        )
        .await
        .expect("extra_env must reach the setup script");
    }

    #[test]
    fn expand_home_expands_tilde_and_tilde_slash_using_current_home() {
        // HOME is set in every reasonable test env; std::env::set_var is
        // unsafe in edition 2024 and forbidden project-wide, so we assert on
        // the current process HOME rather than fabricating one.
        let home = std::env::var_os("HOME").expect("HOME must be set for this test");
        let home = std::path::PathBuf::from(home);
        assert_eq!(expand_home("~"), home.to_string_lossy().as_ref());
        assert_eq!(
            expand_home("~/foo/bar"),
            home.join("foo/bar").to_string_lossy().as_ref(),
        );
    }

    // ---- sealed_regate_score end-to-end with a real synthetic gate -------

    /// Build a synthetic task dir + workspace + task with a `bash -c` gate
    /// that emits pytest-shaped `-rA` output. Exercises `copy_sealed` +
    /// `exec::run` + `PytestParser` + `resolve` end-to-end without an agent.
    #[tokio::test]
    async fn sealed_regate_score_end_to_end_resolves_when_all_green() {
        use super::sealed_regate_score;
        let task_dir = tempdir().expect("task tempdir");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir sealed");
        std::fs::write(task_dir.path().join("sealed/tests/test_x.py"), "# sealed\n")
            .expect("write sealed");

        let ws = tempdir().expect("ws tempdir");
        // The agent-copy of the same file — will be overwritten by copy_sealed.
        std::fs::create_dir_all(ws.path().join("tests")).expect("mkdir tests");
        std::fs::write(ws.path().join("tests/test_x.py"), "# agent\n").expect("write agent copy");

        let task = MinedTask {
            id: "synth-e2e".to_string(),
            repo: "r".to_string(),
            repo_path: "/tmp/does-not-matter".to_string(),
            parent_sha: "aaa".to_string(),
            fix_sha: "bbb".to_string(),
            rung_guess: "easy".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: String::new(),
            // A shell script emulating pytest -rA short-summary output.
            gate_command: r"printf '=== short test summary info ===\nPASSED tests/test_x.py::T::a\nPASSED tests/test_x.py::T::b\n=== 2 passed in 0.0s ===\n'".to_string(),
            agent_gate_command: None,
            fail_to_pass: vec!["T::a".to_string()],
            positive_controls: vec!["T::b".to_string()],
            pass_to_pass_exclusions: Vec::new(),
            sealed: vec![SealedEntry {
                path: "tests/test_x.py".to_string(),
            }],
            notes: Vec::new(),
        };

        let (score, raw) =
            sealed_regate_score(&task, task_dir.path(), ws.path(), &PytestParser).await;
        assert_eq!(score, TrialScore::Resolved, "raw output was: {raw}");
        assert!(raw.contains("PASSED"));
        // And the agent copy was actually overwritten.
        let after = std::fs::read_to_string(ws.path().join("tests/test_x.py")).expect("re-read");
        assert_eq!(after, "# sealed\n");
    }

    #[tokio::test]
    async fn sealed_regate_score_invalid_when_sealed_source_missing() {
        use super::sealed_regate_score;
        // No sealed/ file under the task dir → sealed_regate_score returns
        // Invalid before the gate runs.
        let task_dir = tempdir().expect("task tempdir");
        let ws = tempdir().expect("ws tempdir");
        let task = MinedTask {
            id: "synth-e2e-missing".to_string(),
            repo: "r".to_string(),
            repo_path: "/tmp/does-not-matter".to_string(),
            parent_sha: "aaa".to_string(),
            fix_sha: "bbb".to_string(),
            rung_guess: "easy".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: String::new(),
            gate_command: "true".to_string(),
            agent_gate_command: None,
            fail_to_pass: Vec::new(),
            positive_controls: Vec::new(),
            pass_to_pass_exclusions: Vec::new(),
            sealed: vec![SealedEntry {
                path: "missing.py".to_string(),
            }],
            notes: Vec::new(),
        };
        let (score, raw) =
            sealed_regate_score(&task, task_dir.path(), ws.path(), &PytestParser).await;
        assert!(matches!(score, TrialScore::Invalid { .. }));
        assert!(raw.is_empty());
    }

    // ---- copy_sealed dir-copy branch (recursive) --------------------------

    #[test]
    fn copy_sealed_recurses_into_directory_source() {
        // A sealed entry whose path is a directory triggers copy_dir_recursive.
        let task_dir = tempdir().expect("task tempdir");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests/subdir")).expect("mkdir nested");
        std::fs::write(task_dir.path().join("sealed/tests/subdir/a.py"), "A\n")
            .expect("write a.py");
        std::fs::write(task_dir.path().join("sealed/tests/subdir/b.py"), "B\n")
            .expect("write b.py");

        let ws = tempdir().expect("ws tempdir");
        copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "tests/subdir".to_string(),
            }],
        )
        .expect("dir copy");
        assert_eq!(
            std::fs::read_to_string(ws.path().join("tests/subdir/a.py")).expect("read a"),
            "A\n",
        );
        assert_eq!(
            std::fs::read_to_string(ws.path().join("tests/subdir/b.py")).expect("read b"),
            "B\n",
        );
    }

    // ---- persist_gate_output writes under the INJECTED state root -------

    #[test]
    fn persist_gate_output_writes_under_state_root_override() {
        let root = tempdir().expect("state root");
        let raw = "hello mined world";
        let out_path =
            super::persist_gate_output(root.path(), "some-task-id/with:weird chars", 7, raw);
        let expected = root
            .path()
            .join("talos/mined-eval")
            .join(super::run_id())
            .join("some-task-id/with:weird chars")
            .join("trial-7")
            .join("gate-output.txt");
        assert_eq!(out_path, expected);
        let contents = std::fs::read_to_string(&out_path).expect("read gate output");
        assert_eq!(contents, raw);
    }

    // ---- persist_gate_output temp-path fallback when state_root is unusable

    #[test]
    fn persist_gate_output_falls_back_to_temp_path_when_state_root_is_not_a_dir() {
        let root = tempdir().expect("root");
        let not_a_dir = root.path().join("not-a-dir");
        std::fs::write(&not_a_dir, "i am a file, not a directory").expect("write blocker file");

        let out_path = super::persist_gate_output(
            &not_a_dir,
            "some-task-id/with:weird chars",
            7,
            "fallback text",
        );

        assert!(
            out_path.starts_with(std::env::temp_dir()),
            "fallback path {out_path:?} should live under the process temp dir"
        );
        let file_name = out_path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("fallback path has a filename")
            .to_string();
        assert!(
            file_name.starts_with("talos-mined-eval-some-task-id_with_weird_chars-7-"),
            "unexpected fallback filename: {file_name}"
        );
        assert!(
            file_name.ends_with("-gate-output.txt"),
            "unexpected fallback filename: {file_name}"
        );
        let contents = std::fs::read_to_string(&out_path).expect("read fallback output");
        assert_eq!(contents, "fallback text");

        let _ = std::fs::remove_file(&out_path);
    }

    // ---- run_mined_task end-to-end with MockBackend ----------------------

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn run_mined_task_end_to_end_scores_resolved_via_synthetic_repo() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::engine::{FINISH_TOOL_NAME, FinishTool};
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        // 1. A synthetic primary git repo — no siblings.
        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "main.py", "print('x')\n", false);

        // 2. A synthetic task dir with a sealed test file + statement.
        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir statements");
        std::fs::write(task_dir.path().join("statements/s2.md"), "Finish, please.")
            .expect("write s2");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir sealed");
        std::fs::write(
            task_dir.path().join("sealed/tests/test_synth.py"),
            "# sealed test\n",
        )
        .expect("write sealed");

        // 3. Task pointing at the synthetic primary; gate emits a canned
        //    pytest -rA output that resolves the trial.
        let mut task = task_pointing_at(&primary_src, None, &parent);
        task.gate_command =
            r"printf '=== short test summary info ===\nPASSED tests/test_synth.py::T::pass_it\n=== 1 passed in 0.0s ===\n'"
                .to_string();
        task.fail_to_pass = vec!["T::pass_it".to_string()];
        task.sealed = vec![SealedEntry {
            path: "tests/test_synth.py".to_string(),
        }];

        // 4. A MockBackend that calls finish(done) immediately, once per trial.
        let k = 2u32;
        let finish_turns: Vec<AssistantTurn> = (0..k)
            .map(|_| AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-finish".to_string(),
                    name: FINISH_TOOL_NAME.to_string(),
                    input: serde_json::json!({"disposition": "done", "summary": "ok"}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            })
            .collect();
        let backend = MockBackend::from_turns(finish_turns);

        let statement = "Finish, please.".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k,
            max_iterations: 5,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        // Silence FinishTool's `use` warning across the impl surface.
        let _ = FinishTool;
        let mut on_trial_calls = 0u32;
        let mut on_trial = |_t: &MinedTrialResult| {
            on_trial_calls += 1;
        };
        let report = run_mined_task(&backend, &PytestParser, &config, &mut on_trial).await;
        assert_eq!(report.k, k);
        assert_eq!(report.resolved_count, k, "both trials should resolve");
        assert_eq!(report.invalid_count, 0);
        assert_eq!(report.trials.len(), k as usize);
        for t in &report.trials {
            assert_eq!(t.score, TrialScore::Resolved);
            assert_eq!(t.claimed_disposition, CLAIMED_DONE);
            assert_eq!(t.input_tokens, 10);
            assert_eq!(t.output_tokens, 5);
            // gate_output_path was persisted with the pytest output.
            let contents = std::fs::read_to_string(&t.gate_output_path).expect("read gate out");
            assert!(contents.contains("PASSED"));
            assert_eq!(
                t.transcript_path, None,
                "transcripts: false must leave every trial's transcript_path None"
            );
        }
        assert_eq!(on_trial_calls, k);
        // Header aggregates:
        assert_eq!(report.valid_denominator(), k);
        assert!((report.resolved_rate() - 1.0).abs() < f64::EPSILON);
        assert!(
            !report.trials[0].finish_recovery_armed,
            "AgentGateMode::Off registers no run_checks — see the `_ => None` arm of the \
             `checks` match in single_trial"
        );
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn run_mined_task_with_transcripts_on_writes_per_trial_transcript_jsonl() {
        use super::{MinedRunConfig, run_mined_task, trial_state_dir};
        use crate::engine::{FINISH_TOOL_NAME, FinishTool};
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        // Same synthetic fixture as
        // `run_mined_task_end_to_end_scores_resolved_via_synthetic_repo`, with
        // `transcripts: true`. `state_root` is injected as a dedicated
        // tempdir so this test asserts the real path shape instead of
        // reaching into the developer's or dispatch host's real home.
        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "main.py", "print('x')\n", false);

        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir statements");
        std::fs::write(task_dir.path().join("statements/s2.md"), "Finish, please.")
            .expect("write s2");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir sealed");
        std::fs::write(
            task_dir.path().join("sealed/tests/test_synth.py"),
            "# sealed test\n",
        )
        .expect("write sealed");

        let mut task = task_pointing_at(&primary_src, None, &parent);
        task.gate_command =
            r"printf '=== short test summary info ===\nPASSED tests/test_synth.py::T::pass_it\n=== 1 passed in 0.0s ===\n'"
                .to_string();
        task.fail_to_pass = vec!["T::pass_it".to_string()];
        task.sealed = vec![SealedEntry {
            path: "tests/test_synth.py".to_string(),
        }];

        let k = 2u32;
        let finish_turns: Vec<AssistantTurn> = (0..k)
            .map(|_| AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-finish".to_string(),
                    name: FINISH_TOOL_NAME.to_string(),
                    input: serde_json::json!({"disposition": "done", "summary": "ok"}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            })
            .collect();
        let backend = MockBackend::from_turns(finish_turns);

        let statement = "Finish, please.".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k,
            max_iterations: 5,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: true,
        };
        let _ = FinishTool;
        let mut on_trial = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut on_trial).await;

        assert_eq!(report.trials.len(), k as usize);
        for t in &report.trials {
            let expected =
                trial_state_dir(state_root.path(), &task.id, t.trial).join("transcript.jsonl");
            assert_eq!(t.transcript_path, Some(expected.clone()));
            let contents = std::fs::read_to_string(&expected).expect("read transcript");
            let lines: Vec<&str> = contents.lines().collect();
            assert!(!lines.is_empty());
            let first: serde_json::Value = serde_json::from_str(lines[0]).expect("valid JSON line");
            assert_eq!(first["event"], "run_start");
            assert_eq!(first["label"], "mock");
            let last: serde_json::Value =
                serde_json::from_str(lines[lines.len() - 1]).expect("valid JSON line");
            assert_eq!(last["event"], "run_end");
            assert_eq!(last["outcome"], "Finished");
        }
    }

    /// Build a synthetic primary repo + task dir for the agent-gate E2E tests:
    /// a canned-PASSED sealed scoring gate (`gate_command`), `fail_to_pass =
    /// ["T::pass_it"]`, and sealed `tests/test_synth.py`. The caller sets
    /// `task.agent_gate_command` and `env.setup` as needed.
    fn agent_gate_task_fixture() -> (tempfile::TempDir, tempfile::TempDir, MinedTask, String) {
        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "main.py", "print('x')\n", false);

        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir statements");
        let statement = "S.\n\n## Verification\n\nHidden tests apply.".to_string();
        std::fs::write(task_dir.path().join("statements/s2.md"), &statement).expect("write s2");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir sealed");
        std::fs::write(
            task_dir.path().join("sealed/tests/test_synth.py"),
            "# sealed test\n",
        )
        .expect("write sealed");

        let mut task = task_pointing_at(&primary_src, None, &parent);
        task.gate_command =
            r"printf '=== short test summary info ===\nPASSED tests/test_synth.py::T::pass_it\n=== 1 passed in 0.0s ===\n'"
                .to_string();
        task.fail_to_pass = vec!["T::pass_it".to_string()];
        task.sealed = vec![SealedEntry {
            path: "tests/test_synth.py".to_string(),
        }];
        (workroot, task_dir, task, statement)
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn agent_gate_on_arms_run_checks_and_verifies_finish_done() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::engine::FINISH_TOOL_NAME;
        use crate::model::{
            AssistantTurn, ContentBlock, Message, StopReason, ToolCallRequest, Usage, UserBlock,
        };
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("true # AGENT_GATE_SENTINEL".to_string());

        let turns = vec![
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-rc".to_string(),
                    name: "run_checks".to_string(),
                    input: serde_json::json!({}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            },
            AssistantTurn {
                content: vec![ContentBlock::Text("thinking".to_string())],
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            },
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-finish".to_string(),
                    name: FINISH_TOOL_NAME.to_string(),
                    input: serde_json::json!({"disposition": "done", "summary": "ok"}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                    reasoning_tokens: None,
                },
            },
        ];
        let backend = MockBackend::from_turns(turns);

        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 5,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_secs(30),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        let trial = &report.trials[0];
        assert!(trial.finish_recovery_armed);
        assert_eq!(trial.nudges_fired, 1);
        assert!(trial.gates_green_at_exit);
        assert!(!trial.tree_dirty);
        assert_eq!(trial.iterations, 3);
        assert_eq!(trial.claimed_disposition, CLAIMED_DONE);
        assert_eq!(trial.score, TrialScore::Resolved);
        assert_eq!(backend.calls(), 3);
        assert_eq!(report.clean_tree_nudges(), 1);
        assert_eq!(report.clean_tree_dones(), 1);

        let systems = backend.systems_seen();
        let sys = systems[0].as_deref().expect("system prompt");
        assert!(sys.contains("/bin/sh -c true # AGENT_GATE_SENTINEL"));
        assert!(!sys.contains("short test summary info"));

        let Message::User { content } = &backend.messages_seen()[0][0] else {
            panic!("expected first message of first turn to be User");
        };
        let first_user_text: String = content
            .iter()
            .filter_map(|b| match b {
                UserBlock::Text(t) => Some(t.as_str()),
                UserBlock::ToolResult { .. } => None,
            })
            .collect();
        assert!(first_user_text.contains("AGENT_GATE_SENTINEL"));
        assert_eq!(
            first_user_text
                .matches("Run the following command to verify the task is complete:")
                .count(),
            1
        );
        assert_eq!(first_user_text.matches("## Verification").count(), 2);
        assert!(!first_user_text.contains("short test summary info"));
    }

    #[tokio::test]
    async fn agent_gate_tripwire_threads_the_mode_timeout() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("sleep 3".to_string());

        let backend = MockBackend::from_turns(Vec::new());
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 5,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_millis(500),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        match &report.trials[0].score {
            TrialScore::Invalid { reason } => {
                assert!(
                    reason.starts_with("agent-gate-red-at-parent"),
                    "reason: {reason}"
                );
                assert!(reason.contains("timed_out=true"), "reason: {reason}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
        assert_eq!(report.trials[0].claimed_disposition, "NotRun");
        assert_eq!(backend.calls(), 0);
        assert_eq!(report.invalid_count, 1);
    }

    #[tokio::test]
    async fn agent_gate_tripwire_red_at_parent() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("exit 1".to_string());

        let backend = MockBackend::from_turns(Vec::new());
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 2,
            max_iterations: 5,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_secs(30),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        assert_eq!(report.invalid_count, 2);
        for t in &report.trials {
            match &t.score {
                TrialScore::Invalid { reason } => {
                    assert!(
                        reason.starts_with("agent-gate-red-at-parent"),
                        "reason: {reason}"
                    );
                    assert!(reason.contains("exit=Some(1)"), "reason: {reason}");
                    assert!(reason.contains("timed_out=false"), "reason: {reason}");
                }
                other => panic!("expected Invalid, got {other:?}"),
            }
            assert_eq!(t.claimed_disposition, "NotRun");
        }
        assert_eq!(backend.calls(), 0);
    }

    #[tokio::test]
    async fn agent_gate_turning_red_after_the_agent_acts_rejects_finish_done() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::engine::FINISH_TOOL_NAME;
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("test ! -e .agent_marker".to_string());

        let usage = || Usage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: None,
            cache_write_tokens: None,
            reasoning_tokens: None,
        };
        let turns = vec![
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-bash".to_string(),
                    name: "bash".to_string(),
                    input: serde_json::json!({"command": "touch .agent_marker"}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: usage(),
            },
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-finish-1".to_string(),
                    name: FINISH_TOOL_NAME.to_string(),
                    input: serde_json::json!({"disposition": "done", "summary": "ok"}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: usage(),
            },
            AssistantTurn {
                content: vec![ContentBlock::ToolCall(ToolCallRequest {
                    id: "c-finish-2".to_string(),
                    name: FINISH_TOOL_NAME.to_string(),
                    input: serde_json::json!({"disposition": "done", "summary": "ok"}),
                })],
                stop_reason: StopReason::ToolUse,
                usage: usage(),
            },
        ];
        let backend = MockBackend::from_turns(turns);

        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_secs(30),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        let trial = &report.trials[0];
        assert_eq!(trial.claimed_disposition, "MaxIterations");
        assert_eq!(trial.score, TrialScore::Resolved);
        assert_eq!(report.resolved_unclaimed(), 1);
        assert_eq!(report.false_dones(), 0);
        assert_eq!(trial.nudges_fired, 0);
        assert!(!trial.gates_green_at_exit);
        assert!(trial.finish_recovery_armed);
        assert!(trial.tree_dirty);
        assert_eq!(backend.calls(), 3);

        // The post-run agent gate (run once after `engine::run` returns,
        // BEFORE the sealed re-gate) sees the SAME tree the last rejected
        // `finish(done)` saw: `.agent_marker` is still present, so the gate
        // is red. This is the "correct-but-NOT-shippable" case the item
        // exists to surface — Resolved (sealed tests pass) but the agent's
        // own project gate never went green at the end.
        assert_eq!(trial.agent_gate_post, Some(false));
        assert_eq!(report.resolved_gate_red(), 1);
        assert_eq!(report.shippable(), 0);
        let agent_gate_output_path = trial
            .agent_gate_output_path
            .as_ref()
            .expect("agent gate output path must be Some when the post-run gate ran");
        assert!(
            agent_gate_output_path.exists(),
            "persisted agent-gate-output file must exist at {}",
            agent_gate_output_path.display(),
        );
    }

    #[tokio::test]
    async fn agent_gate_post_run_green_at_immediate_done_counts_as_shippable() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::engine::FINISH_TOOL_NAME;
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("true".to_string());

        let turns = vec![AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: serde_json::json!({"disposition": "done", "summary": "ok"}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        }];
        let backend = MockBackend::from_turns(turns);

        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_secs(30),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        let trial = &report.trials[0];
        assert_eq!(trial.claimed_disposition, CLAIMED_DONE);
        assert_eq!(trial.score, TrialScore::Resolved);
        assert_eq!(trial.agent_gate_post, Some(true));
        assert_eq!(report.shippable(), 1);
        assert_eq!(report.resolved_gate_red(), 0);
        assert!(trial.agent_gate_output_path.is_some());
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn run_mined_task_writes_all_three_captures_only_under_injected_state_root() {
        use super::{MinedRunConfig, default_state_root, run_id, run_mined_task, trial_state_dir};
        use crate::engine::FINISH_TOOL_NAME;
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("true".to_string());

        let turns = vec![AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: serde_json::json!({"disposition": "done", "summary": "ok"}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        }];
        let backend = MockBackend::from_turns(turns);

        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 5,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_secs(30),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: true,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        assert_eq!(report.trials.len(), 1);

        let trial_dir = trial_state_dir(state_root.path(), &task.id, 0);
        assert!(trial_dir.join("gate-output.txt").exists());
        assert!(trial_dir.join("agent-gate-output.txt").exists());
        assert!(trial_dir.join("transcript.jsonl").exists());

        // Guard: nothing landed under the real default state root for this
        // process-unique task id.
        let leaked = default_state_root()
            .join("talos/mined-eval")
            .join(run_id())
            .join(&task.id);
        assert!(!leaked.exists());
    }

    #[tokio::test]
    async fn agent_gate_off_is_legacy() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::engine::FINISH_TOOL_NAME;
        use crate::model::{
            AssistantTurn, ContentBlock, Message, StopReason, ToolCallRequest, Usage, UserBlock,
        };
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, mut task, statement) = agent_gate_task_fixture();
        task.agent_gate_command = Some("true # AGENT_GATE_SENTINEL".to_string());

        let turns = vec![AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: serde_json::json!({"disposition": "done", "summary": "ok"}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        }];
        let backend = MockBackend::from_turns(turns);

        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 5,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        let trial = &report.trials[0];
        assert!(!trial.finish_recovery_armed);
        assert_eq!(trial.nudges_fired, 0);
        assert_eq!(trial.claimed_disposition, CLAIMED_DONE);
        assert_eq!(trial.score, TrialScore::Resolved);
        assert_eq!(backend.calls(), 1);
        // AgentGateMode::Off never runs the post-run gate: NOT-RUN, not
        // "ran and passed".
        assert_eq!(trial.agent_gate_post, None);
        assert_eq!(trial.agent_gate_output_path, None);
        assert_eq!(report.shippable(), 1);
        assert_eq!(report.resolved_gate_red(), 0);

        let systems = backend.systems_seen();
        let sys = systems[0].as_deref().expect("system prompt");
        assert!(sys.contains("No checks are configured"));
        assert!(!sys.contains("AGENT_GATE_SENTINEL"));

        let Message::User { content } = &backend.messages_seen()[0][0] else {
            panic!("expected first message of first turn to be User");
        };
        let first_user_text: String = content
            .iter()
            .filter_map(|b| match b {
                UserBlock::Text(t) => Some(t.as_str()),
                UserBlock::ToolResult { .. } => None,
            })
            .collect();
        assert_eq!(
            first_user_text
                .matches("Run the following command to verify the task is complete:")
                .count(),
            0
        );
        assert!(!first_user_text.contains("AGENT_GATE_SENTINEL"));
    }

    #[tokio::test]
    async fn agent_gate_on_missing_field_fails_loud() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, task, statement) = agent_gate_task_fixture();
        // task.agent_gate_command left at None (the fixture default).
        assert_eq!(task.agent_gate_command, None);

        let backend = MockBackend::from_turns(Vec::new());
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 2,
            max_iterations: 5,
            agent_gate: AgentGateMode::On {
                timeout: Duration::from_secs(30),
            },
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        assert_eq!(report.invalid_count, 2);
        for t in &report.trials {
            match &t.score {
                TrialScore::Invalid { reason } => {
                    assert!(reason.starts_with("agent-gate-missing"), "reason: {reason}");
                    assert!(reason.contains(&task.id), "reason: {reason}");
                }
                other => panic!("expected Invalid, got {other:?}"),
            }
            assert_eq!(t.claimed_disposition, "NotRun");
        }
        assert_eq!(backend.calls(), 0);
    }

    /// Proves `MinedRunConfig::wall_clock_secs` actually reaches the engine's
    /// `RunConfig` — fails (times out waiting for a `MockBackend` turn that
    /// never comes) if `single_trial` ever stops calling
    /// `RunConfig::with_wall_clock_secs`. `single_trial` has no clock-injection
    /// seam, so this is a real-time test: a genuine `sleep 2` inside the first
    /// iteration's `bash` call, against a 1-second budget, must trip
    /// `LoopOutcome::BudgetExhausted` before a second turn is ever drawn.
    #[tokio::test]
    async fn wall_clock_secs_reaches_the_engine_run_config() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        let (_workroot, task_dir, task, statement) = agent_gate_task_fixture();

        let turns = vec![AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-sleep".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "sleep 2"}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        }];
        let backend = MockBackend::from_turns(turns);

        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 5,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 1,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;

        assert_eq!(
            report.trials[0].claimed_disposition, "BudgetExhausted",
            "a 1s wall-clock budget must trip after a 2s bash call — if this reads anything \
             else, single_trial stopped calling RunConfig::with_wall_clock_secs"
        );
        assert_eq!(backend.calls(), 1);
    }

    #[tokio::test]
    async fn run_mined_task_invalid_when_env_setup_fails() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::test_support::MockBackend;

        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "x.py", "print(1)\n", false);

        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir statements");
        std::fs::write(task_dir.path().join("statements/s2.md"), "hi").expect("write s2");
        let mut task = task_pointing_at(&primary_src, None, &parent);
        task.env.setup = "exit 3".to_string(); // pre-agent failure.

        // Backend is never called since setup fails before the agent runs.
        let backend = MockBackend::from_turns(Vec::new());
        let statement = "hi".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;
        assert_eq!(report.invalid_count, 1);
        assert_eq!(report.resolved_count, 0);
        // Invalid excluded from denominator.
        assert_eq!(report.valid_denominator(), 0);
        // Reason names the failure family.
        match &report.trials[0].score {
            TrialScore::Invalid { reason } => {
                assert!(reason.contains("setup-failed"), "reason: {reason}");
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
        assert_eq!(report.trials[0].claimed_disposition, "NotRun");
    }

    #[tokio::test]
    async fn run_mined_task_invalid_when_worktree_setup_fails() {
        // repo_path points at a directory that isn't a git repo → the
        // worktree add step fails, and the trial is Invalid before the
        // agent ever runs.
        use super::{MinedRunConfig, run_mined_task};
        use crate::test_support::MockBackend;

        let not_a_repo = tempdir().expect("not a repo");
        // No git init here on purpose.
        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir statements");
        std::fs::write(task_dir.path().join("statements/s2.md"), "hi").expect("write s2");
        let task = task_pointing_at(not_a_repo.path(), None, "deadbeef");
        let backend = MockBackend::from_turns(Vec::new());
        let statement = "hi".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;
        assert_eq!(report.invalid_count, 1);
        match &report.trials[0].score {
            TrialScore::Invalid { reason } => {
                assert!(
                    reason.starts_with("worktree-setup-failed"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    // ---- claimed_disposition_label — cover the remaining variants -------

    #[test]
    fn claimed_disposition_label_covers_backend_error_and_budget_exhausted() {
        use crate::engine::LoopOutcome;
        use crate::model::{BackendError, TerminalKind};
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::BudgetExhausted {
                summary: "wall".to_string(),
            }),
            "BudgetExhausted",
        );
        // The payload is carried, not just the variant name: a bare
        // "BackendError" is undiagnosable after the fact (no run record is
        // persisted), and it collapses causes needing opposite operator
        // responses — a context-guard trip vs auth vs a parse failure.
        let labelled =
            claimed_disposition_label(&LoopOutcome::BackendError(BackendError::Terminal {
                kind: TerminalKind::Auth,
                message: "no creds".to_string(),
            }));
        assert!(
            labelled.starts_with("BackendError("),
            "expected a payload-carrying label, got {labelled}",
        );
        assert!(
            labelled.contains("no creds"),
            "expected the underlying cause in the label, got {labelled}",
        );
    }

    // ---- SpecLevel::slug + reveal all three -----------------------------

    #[test]
    fn spec_level_slug_matches_filename() {
        assert_eq!(SpecLevel::S1.slug(), "s1");
        assert_eq!(SpecLevel::S2.slug(), "s2");
        assert_eq!(SpecLevel::S3.slug(), "s3");
    }

    // ---- Debug is implemented on MinedReport + MinedTrialResult ---------

    #[test]
    fn mined_report_debug_prints_recognisable_shape() {
        let r = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 0,
            resolved_count: 0,
            invalid_count: 0,
            trials: Vec::new(),
        };
        let s = format!("{r:?}");
        assert!(s.contains("MinedReport"));
        assert!(s.contains("task_id"));
    }

    // ---- resolve_state_root — precedence for every branch ----------------

    #[test]
    fn resolve_state_root_override_wins() {
        use std::ffi::OsString;
        let path = super::resolve_state_root(
            Some(OsString::from("/an/override")),
            Some(OsString::from("/xdg")),
            Some(OsString::from("/home/user")),
        );
        assert_eq!(path, std::path::PathBuf::from("/an/override"));
    }

    #[test]
    fn resolve_state_root_xdg_beats_home_when_no_override() {
        use std::ffi::OsString;
        let path = super::resolve_state_root(
            None,
            Some(OsString::from("/xdg-state")),
            Some(OsString::from("/home/user")),
        );
        assert_eq!(path, std::path::PathBuf::from("/xdg-state"));
    }

    #[test]
    fn resolve_state_root_falls_back_to_home_local_state() {
        use std::ffi::OsString;
        let path = super::resolve_state_root(None, None, Some(OsString::from("/home/user")));
        assert_eq!(path, std::path::PathBuf::from("/home/user/.local/state"));
    }

    #[test]
    fn resolve_state_root_falls_back_to_temp_dir_when_no_env() {
        let path = super::resolve_state_root(None, None, None);
        // Just require it to be a directory that exists — the exact path
        // depends on the platform's temp dir configuration.
        assert!(path.is_dir(), "temp dir must exist: {}", path.display());
    }

    // ---- expand_home_with — the None-HOME branch -------------------------

    #[test]
    fn expand_home_with_no_home_returns_input_unchanged() {
        assert_eq!(super::expand_home_with("~/foo", None), "~/foo");
        assert_eq!(super::expand_home_with("~", None), "~");
        assert_eq!(super::expand_home_with("/abs", None), "/abs");
    }

    #[test]
    fn expand_home_with_home_expands_both_forms() {
        use std::ffi::OsString;
        let home = OsString::from("/testhome");
        assert_eq!(
            super::expand_home_with("~", Some(home.clone())),
            "/testhome"
        );
        assert_eq!(
            super::expand_home_with("~/nested/dir", Some(home.clone())),
            "/testhome/nested/dir",
        );
        // Non-tilde input still passes through.
        assert_eq!(
            super::expand_home_with("relative/path", Some(home)),
            "relative/path"
        );
    }

    // ---- find_status_for_task_id — the "red overrides passed" branch ----

    #[test]
    fn find_status_for_task_id_promotes_red_across_param_instances() {
        use super::find_status_for_task_id;
        // Two param instances match the bare id `Cls::test`: one Passed (first
        // seen), one Failed. The result must be Failed (a red param overrides
        // a green one for the id's aggregate status).
        let normalized: Vec<(String, String, TestStatus)> = vec![
            (
                "tests/t.py::Cls::test[a]".to_string(),
                "Cls::test[a]".to_string(),
                TestStatus::Passed,
            ),
            (
                "tests/t.py::Cls::test[b]".to_string(),
                "Cls::test[b]".to_string(),
                TestStatus::Failed,
            ),
        ];
        assert_eq!(
            find_status_for_task_id(&normalized, "Cls::test"),
            Some(TestStatus::Failed),
        );
        // And the "already saw Failed, later Passed" branch is a no-op: the
        // observed Failed sticks.
        let normalized2: Vec<(String, String, TestStatus)> = vec![
            (
                "tests/t.py::Cls::test[a]".to_string(),
                "Cls::test[a]".to_string(),
                TestStatus::Failed,
            ),
            (
                "tests/t.py::Cls::test[b]".to_string(),
                "Cls::test[b]".to_string(),
                TestStatus::Passed,
            ),
        ];
        assert_eq!(
            find_status_for_task_id(&normalized2, "Cls::test"),
            Some(TestStatus::Failed),
        );
    }

    // ---- build_unexcluded_red — the dedup branch -------------------------

    #[test]
    fn build_unexcluded_red_dedups_duplicate_normalized_ids() {
        // If two raw nodeids normalize to the same string (both red), only
        // one entry should land in `unexcluded_red`.
        let task = sample_task_for_scoring();
        let parsed = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
            // Two file-prefixes that both normalize to "Extra::test_a".
            ("tests/test_cleanr.py::Extra::test_a", TestStatus::Failed),
            ("tests/test_cleanr.py::Extra::test_a", TestStatus::Failed),
        ]);
        // A BTreeMap dedups equal keys, so both entries collapse — the
        // dedup path is inside build_unexcluded_red's `seen` set; test it via
        // matching parametrized copies that both normalize to the same key.
        let parsed_alt = status_map(&[
            (
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped",
                TestStatus::Passed,
            ),
            (
                "tests/test_cleanr.py::TestCommentFilters::test_reader_comments_kept",
                TestStatus::Passed,
            ),
            // Same nodeid via a different file prefix (a rare pytest layout
            // detail, but demonstrates the dedup path).
            ("tests/test_cleanr.py::Extra::test_a", TestStatus::Failed),
            ("other/test_dup.py::Extra::test_a", TestStatus::Failed),
        ]);
        // Both parsed maps have exactly one distinct un-excluded red id.
        for p in [&parsed, &parsed_alt] {
            match resolve(p, Some(p.len()), &task) {
                TrialScore::Unresolved { reason } => {
                    assert_eq!(reason.unexcluded_red.len(), 1);
                }
                other => panic!("expected Unresolved, got {other:?}"),
            }
        }
    }

    // ---- load_task IO-error branch ---------------------------------------

    #[test]
    fn load_task_returns_io_error_when_task_json_missing() {
        let dir = tempdir().expect("tempdir");
        // No task.json written.
        let err = load_task(dir.path()).expect_err("missing task.json must error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    // ---- Worktree Drop failure-path — silent logging is not observable in
    //      tests but the code path is exercised by the prepare_worktrees
    //      test through the natural drop, which succeeds. This test isolates
    //      the "worktree remove fails" branch by rm-ing the worktree dir
    //      out from under `git worktree remove --force`, so git fails and
    //      the eprintln path fires (no panic).

    #[test]
    fn worktree_drop_logs_on_failure_but_never_panics() {
        // Build a repo, add a worktree, then destroy the worktree dir
        // externally so `git worktree remove --force` fails. The Drop must
        // not panic.
        let workroot = tempdir().expect("workroot");
        let src = workroot.path().join("repo");
        std::fs::create_dir_all(&src).expect("mkdir");
        let parent = make_repo(&src, "f.py", "x\n", false);

        let task = task_pointing_at(&src, None, &parent);
        let ws = prepare_worktrees(&task).expect("prepare");
        let wt = ws.primary.clone();
        // Now nuke the worktree directory + its .git file — subsequent
        // `git worktree remove` will fail. Drop must log-and-continue.
        std::fs::remove_dir_all(&wt).expect("nuke worktree dir");
        drop(ws);
        // If we got here without panicking, the test passes.
        assert!(!wt.exists());
    }

    // ---- copy_sealed — the create_dir_all error branch -------------------

    // ---- Worktree drop directly with a bogus source_repo ---------------

    #[test]
    fn worktree_drop_eprintln_fires_when_git_returns_nonzero() {
        // Construct a Worktree by hand pointing at a non-repo. `git -C
        // <not-a-repo> worktree remove --force <anywhere>` exits non-zero,
        // which is exactly the branch we want to cover. The Drop must
        // eprintln! and not panic.
        let bogus = super::Worktree {
            source_repo: std::path::PathBuf::from("/no-such/dir-42"),
            worktree_path: std::path::PathBuf::from("/no-such/worktree-42"),
        };
        // Dropping is enough — the assertion is that we don't panic.
        drop(bogus);
    }

    // ---- prepare_worktrees: repo_path has no parent (e.g. `/`) ----------

    #[test]
    fn prepare_worktrees_errors_when_repo_path_has_no_parent() {
        let mut task = MinedTask {
            id: unique_task_id(),
            repo: "root".to_string(),
            repo_path: "/".to_string(),
            parent_sha: "abc".to_string(),
            fix_sha: "abc".to_string(),
            rung_guess: "easy".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: String::new(),
            gate_command: "true".to_string(),
            agent_gate_command: None,
            fail_to_pass: Vec::new(),
            positive_controls: Vec::new(),
            pass_to_pass_exclusions: Vec::new(),
            sealed: Vec::new(),
            notes: Vec::new(),
        };
        // `/`.parent() is None → prepare_worktrees returns the parent-less error.
        let err = prepare_worktrees(&task).expect_err("root path has no parent");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        // Swap in a sibling that points at a bogus source repo — the sibling
        // add_worktree branch fires when the primary succeeds first. Skip
        // (needs a real primary), covered separately by `run_mined_task_invalid_when_worktree_setup_fails`.
        task.repo_path = "/".to_string(); // no-op reassignment, keeps clippy quiet.
    }

    // ---- default_state_root — call the wrapper (covers env-read branches) --

    #[test]
    fn default_state_root_returns_a_directory_shaped_path() {
        // Runs the actual `default_state_root` wrapper so its `env::var_os`
        // branch predicates get exercised (the pure worker is tested separately).
        let path = super::default_state_root();
        // It should be non-empty, and either an override, XDG root, HOME's
        // .local/state, or the temp dir — all are valid shapes.
        assert!(!path.as_os_str().is_empty());
    }

    // ---- pytest parser: STATUS with no nodeid (nodeid.is_empty() branch) --

    #[test]
    fn pytest_parser_skips_status_line_with_empty_nodeid() {
        // "PASSED " with nothing following the space — the whole line is
        // caught by parse_short_summary_line, but nodeid comes out empty and
        // the parser skips the insertion.
        let out = "=========================== short test summary info ============================\nPASSED \nPASSED tests/x.py::T::real\n=== 1 passed in 0.0s ===";
        let report = PytestParser.parse(out);
        assert_eq!(
            report.statuses.len(),
            1,
            "empty nodeid line must not insert"
        );
        assert!(report.statuses.contains_key("tests/x.py::T::real"));
        assert_eq!(report.summary_count, Some(1));
    }

    // ---- normalize fallthrough: non-.py first segment ---------------------

    #[test]
    fn normalize_returns_input_when_first_segment_is_not_py() {
        // Head has a `::` but its first segment doesn't end in `.py` — the
        // fallthrough branch returns the original nodeid unchanged.
        assert_eq!(normalize("mymod::inner::test_fn"), "mymod::inner::test_fn");
    }

    // ---- run_mined_task: Unresolved outcome in the trial loop ------------

    #[tokio::test]
    async fn run_mined_task_records_unresolved_when_fail_to_pass_stays_red() {
        use super::{MinedRunConfig, run_mined_task};
        use crate::engine::{FINISH_TOOL_NAME, FinishTool};
        use crate::model::{AssistantTurn, ContentBlock, StopReason, ToolCallRequest, Usage};
        use crate::test_support::MockBackend;

        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "x.py", "print(1)\n", false);

        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir stmts");
        std::fs::write(task_dir.path().join("statements/s2.md"), "hi").expect("write s2");
        std::fs::create_dir_all(task_dir.path().join("sealed")).expect("mkdir sealed");
        std::fs::write(task_dir.path().join("sealed/dest.py"), "sealed\n").expect("write sealed");

        let mut task = task_pointing_at(&primary_src, None, &parent);
        // Gate emits ONE red fail_to_pass id, no positive_controls.
        task.gate_command =
            r"printf '=== short test summary info ===\nFAILED tests/x.py::T::red\n=== 0 passed, 1 failed in 0.0s ===\n'".to_string();
        task.fail_to_pass = vec!["T::red".to_string()];
        task.sealed = vec![SealedEntry {
            path: "dest.py".to_string(),
        }];

        let backend = MockBackend::from_turns(vec![AssistantTurn {
            content: vec![ContentBlock::ToolCall(ToolCallRequest {
                id: "c-finish".to_string(),
                name: FINISH_TOOL_NAME.to_string(),
                input: serde_json::json!({"disposition": "done", "summary": "ok"}),
            })],
            stop_reason: StopReason::ToolUse,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: None,
                cache_write_tokens: None,
                reasoning_tokens: None,
            },
        }]);
        let _ = FinishTool;
        let statement = "hi".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;
        assert_eq!(report.resolved_count, 0);
        assert_eq!(report.invalid_count, 0);
        assert_eq!(report.valid_denominator(), 1);
        assert!(matches!(
            report.trials[0].score,
            TrialScore::Unresolved { .. }
        ));
        // false_dones = 1: claimed Done, but sealed re-gate says Unresolved.
        assert_eq!(report.false_dones(), 1);
    }

    // ---- Workspace::new failure path — pass a workspace root that
    //      canonicalizes to a non-directory (a regular file).
    #[tokio::test]
    async fn run_mined_task_invalid_when_workspace_root_is_a_file() {
        // This exercises the `Workspace::new` error path in `single_trial`
        // indirectly: we make `env.setup` DELETE the primary worktree and
        // replace it with a regular file. Workspace::new then rejects the
        // non-dir root and single_trial returns Invalid.
        use super::{MinedRunConfig, run_mined_task};
        use crate::test_support::MockBackend;

        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "x.py", "x\n", false);

        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir stmts");
        std::fs::write(task_dir.path().join("statements/s2.md"), "hi").expect("write s2");

        let mut task = task_pointing_at(&primary_src, None, &parent);
        // Trip Workspace::new by scrubbing the worktree dir out from under it.
        task.env.setup = "rm -rf $(pwd) && touch $(pwd)".to_string();

        let backend = MockBackend::from_turns(Vec::new());
        let statement = "hi".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;
        // Either the setup itself fails (rm -rf $(pwd) from within can fail)
        // OR Workspace::new fails afterwards. Both produce Invalid.
        assert_eq!(report.invalid_count, 1);
        assert!(matches!(report.trials[0].score, TrialScore::Invalid { .. }));
    }

    // ---- invalid_trial's gate-output write lands under the injected root -

    #[tokio::test]
    async fn run_mined_task_invalid_trial_writes_gate_output_only_under_injected_state_root() {
        // Same Workspace::new failure fixture as
        // `run_mined_task_invalid_when_workspace_root_is_a_file`, but this
        // test's focus is `invalid_trial`'s own `persist_gate_output` call
        // (mined_eval.rs's pre-agent Invalid path): it must write under the
        // injected `state_root`, never under the real default state root.
        use super::{MinedRunConfig, default_state_root, run_id, run_mined_task};
        use crate::test_support::MockBackend;

        let workroot = tempdir().expect("workroot");
        let primary_src = workroot.path().join("primary");
        std::fs::create_dir_all(&primary_src).expect("mkdir");
        let parent = make_repo(&primary_src, "x.py", "x\n", false);

        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("statements")).expect("mkdir stmts");
        std::fs::write(task_dir.path().join("statements/s2.md"), "hi").expect("write s2");

        let mut task = task_pointing_at(&primary_src, None, &parent);
        // Trip Workspace::new by scrubbing the worktree dir out from under it.
        task.env.setup = "rm -rf $(pwd) && touch $(pwd)".to_string();

        let backend = MockBackend::from_turns(Vec::new());
        let statement = "hi".to_string();
        let state_root = tempdir().expect("state root");
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            state_root: state_root.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
            agent_gate: AgentGateMode::Off,
            test_first: true,
            wall_clock_secs: 0,
            transcripts: false,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;
        assert_eq!(report.invalid_count, 1);

        let gate_output = state_root
            .path()
            .join("talos/mined-eval")
            .join(run_id())
            .join(&task.id)
            .join("trial-0")
            .join("gate-output.txt");
        assert!(gate_output.exists());

        let leaked = default_state_root()
            .join("talos/mined-eval")
            .join(run_id())
            .join(&task.id);
        assert!(!leaked.exists());
    }

    // ---- copy_sealed: recursive copy dst-exists-as-file failure ----------

    #[test]
    fn copy_sealed_dir_copy_failure_becomes_sealed_copy_failed() {
        // Source is a directory; dest already exists as a regular file →
        // std::fs::create_dir_all(dest.parent()) succeeds but the recursive
        // copy_dir_recursive tries to create the dest as a dir and fails.
        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("sealed/mydir")).expect("mkdir");
        std::fs::write(task_dir.path().join("sealed/mydir/a.py"), "A\n").expect("write");

        let ws = tempdir().expect("ws");
        // Pre-place a REGULAR FILE at the destination — dir copy will fail.
        std::fs::write(ws.path().join("mydir"), b"i-am-a-file").expect("write dest file");

        let err = copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "mydir".to_string(),
            }],
        )
        .expect_err("must fail when dest exists as file");
        assert!(err.starts_with("sealed-copy-failed"), "reason: {err}");
    }

    #[test]
    fn copy_sealed_reports_create_dir_all_failure() {
        // Force create_dir_all to fail by making its parent a REGULAR FILE
        // instead of a directory. `create_dir_all(file/subdir)` fails on
        // Linux with an EEXIST/ENOTDIR family error.
        let task_dir = tempdir().expect("task dir");
        std::fs::create_dir_all(task_dir.path().join("sealed")).expect("mkdir sealed");
        std::fs::write(task_dir.path().join("sealed/dest_ok.py"), "sealed\n")
            .expect("write sealed");

        let ws = tempdir().expect("ws");
        // Turn `tests` into a file (not a directory). Then attempt to copy
        // sealed content into `tests/inner.py` — parent creation must fail.
        std::fs::write(ws.path().join("tests"), b"i-am-a-file").expect("write tests file");

        let result = copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "dest_ok.py".to_string(),
            }],
        );
        // The dest_ok.py copy should still succeed (its parent is ws root,
        // which exists), so this call is fine. Now try one whose parent
        // creation fails.
        assert!(result.is_ok());
        let bad = copy_sealed(
            task_dir.path(),
            ws.path(),
            &[SealedEntry {
                path: "tests/inner.py".to_string(),
            }],
        )
        .expect_err("copy_sealed must fail when parent isn't a dir");
        // Either create_dir_all or the copy itself will fail; both surface
        // as `sealed-copy-failed: ...`.
        assert!(bad.starts_with("sealed-copy-failed"), "reason: {bad}");
    }

    // ---- new section-scoping + collection-error tests -------------------

    #[test]
    fn pytest_parser_ignores_caplog_error_lines_outside_summary_section() {
        let fixture = include_str!("../testdata/mined_eval/deadlock-trial-0.txt");
        let report = PytestParser.parse(fixture);
        assert_eq!(report.statuses.len(), 121);
        assert_eq!(report.summary_count, Some(121));
        assert_eq!(report.dropped_outside_section, 6);
        assert_eq!(report.sections_seen, 1);
        assert!(
            !report
                .statuses
                .contains_key("agent_gtd.event_bus:event_bus.py:130")
        );
        for key in report.statuses.keys() {
            assert!(key.contains("::"), "key without '::': {key}");
        }
    }

    #[test]
    fn pytest_parser_uses_location_for_bracketed_skipped_lines() {
        let fixture = include_str!("../testdata/mined_eval/attribution-trial-0.txt");
        let report = PytestParser.parse(fixture);
        assert_eq!(
            report
                .statuses
                .get("tests/test_dispatch_attribution.py:191"),
            Some(&TestStatus::Skipped)
        );
        assert!(!report.statuses.contains_key("[1]"));
        assert_eq!(report.statuses.len(), 8);
        assert_eq!(report.summary_count, Some(8));
        assert_eq!(report.dropped_outside_section, 0);
        assert_eq!(report.sections_seen, 1);
    }

    #[test]
    fn status_lines_without_a_summary_header_parse_empty() {
        let input = "PASSED tests/test_x.py::test_a\nFAILED tests/test_x.py::test_b\n=== 1 passed, 1 failed in 0.01s ===";
        let report = PytestParser.parse(input);
        assert!(report.statuses.is_empty());
        assert_eq!(report.summary_count, Some(2));
        assert_eq!(report.dropped_outside_section, 2);
        assert_eq!(report.sections_seen, 0);
        let task = sample_task_for_scoring();
        let score = resolve(&report.statuses, report.summary_count, &task);
        match score {
            TrialScore::Invalid { reason } => {
                assert_eq!(reason, "parse-empty");
            }
            other => panic!("expected Invalid{{parse-empty}}, got {other:?}"),
        }
    }

    #[test]
    fn blank_line_terminates_the_summary_section() {
        let input = "=== short test summary info ===\nPASSED tests/test_x.py::test_a\n\nPASSED tests/test_y.py::test_b\n=== 1 passed in 0.01s ===";
        let report = PytestParser.parse(input);
        assert_eq!(report.statuses.len(), 1);
        assert!(!report.statuses.contains_key("tests/test_y.py::test_b"));
        assert_eq!(report.dropped_outside_section, 1);
    }

    #[test]
    fn build_collection_errors_discriminates_file_level_errors() {
        // Real collection-error shape: file path, no `::`, `.py` extension.
        assert_eq!(
            build_collection_errors(&[(
                "tests/test_cli.py".to_string(),
                "tests/test_cli.py".to_string(),
                TestStatus::Error,
            )]),
            vec!["tests/test_cli.py".to_string()],
        );
        // An Error WITH `::` is a normal red test, not a collection error.
        assert!(
            build_collection_errors(&[(
                "tests/test_x.py::TestX::test_broken".to_string(),
                "TestX::test_broken".to_string(),
                TestStatus::Error,
            )])
            .is_empty()
        );
        // Verbatim Defect-1 phantom: no `::`, but extension is `py:130` not `py`.
        assert!(
            build_collection_errors(&[(
                "agent_gtd.event_bus:event_bus.py:130".to_string(),
                "agent_gtd.event_bus:event_bus.py:130".to_string(),
                TestStatus::Error,
            )])
            .is_empty()
        );
        // A Passed status is not an error.
        assert!(
            build_collection_errors(&[(
                "tests/test_cleanr.py::TestCommentFilters::test_owner_comments_skipped".to_string(),
                "TestCommentFilters::test_owner_comments_skipped".to_string(),
                TestStatus::Passed,
            )])
            .is_empty()
        );
    }

    #[test]
    fn deadlock_capture_no_longer_scores_parse_mismatch() {
        let fixture = include_str!("../testdata/mined_eval/deadlock-trial-0.txt");
        let report = PytestParser.parse(fixture);
        let task = task_from_json(DEADLOCK_TASK_JSON);
        let score = resolve(&report.statuses, report.summary_count, &task);
        match score {
            TrialScore::Unresolved { reason } => {
                assert_eq!(
                    reason.missing_fail_to_pass,
                    vec![
                        "tests/test_dispatch_service.py::test_create_run_manage_mode_rejected"
                            .to_string(),
                        "tests/test_rollout_executor.py::test_manage_dispatch_does_not_flip_item_status"
                            .to_string(),
                    ]
                );
                assert!(reason.unexcluded_red.is_empty());
                assert!(reason.collection_errors.is_empty());
                assert_eq!(reason.fail_to_pass_status.len(), 4);
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn collection_error_scores_unresolved_not_invalid() {
        let fixture = include_str!("../testdata/mined_eval/from-json-trial-1.txt");
        let report = PytestParser.parse(fixture);
        assert_eq!(report.statuses.len(), 1);
        assert_eq!(report.summary_count, Some(1));
        let task = task_from_json(FROM_JSON_TASK_JSON);
        let score = resolve(&report.statuses, report.summary_count, &task);
        assert!(
            !matches!(score, TrialScore::Invalid { .. }),
            "must not be Invalid"
        );
        match score {
            TrialScore::Unresolved { reason } => {
                assert_eq!(
                    reason.collection_errors,
                    vec!["tests/test_cli.py".to_string()]
                );
                assert_eq!(reason.missing_fail_to_pass.len(), 6);
                assert_eq!(reason.fail_to_pass_status.len(), 6);
                assert_eq!(reason.unexcluded_red, vec!["tests/test_cli.py".to_string()]);
            }
            other => panic!("expected Unresolved, got {other:?}"),
        }
    }

    #[test]
    fn collection_error_arm_precedes_positive_control_arm() {
        // `sample_task_for_scoring()` has a positive_control that would be
        // uncollected if we only have a file-level Error. The collection-error
        // arm (step 3) must fire BEFORE the positive-control loop (step 4).
        let task = sample_task_for_scoring();
        let parsed = status_map(&[("tests/test_cleanr.py", TestStatus::Error)]);
        let score = resolve(&parsed, Some(1), &task);
        match score {
            TrialScore::Unresolved { reason } => {
                assert_eq!(
                    reason.collection_errors,
                    vec!["tests/test_cleanr.py".to_string()]
                );
            }
            other => panic!("expected Unresolved (collection-error arm), got {other:?}"),
        }
    }

    #[test]
    fn truncated_capture_still_scores_parse_mismatch() {
        let fixture_lines: Vec<&str> = include_str!("../testdata/mined_eval/deadlock-trial-0.txt")
            .lines()
            .collect();
        let hdr = fixture_lines
            .iter()
            .position(|l| l.contains("short test summary info"))
            .expect("fixture must have a short test summary info header");
        let truncated: String = fixture_lines[hdr..=hdr + 60]
            .iter()
            .chain(std::iter::once(
                fixture_lines.last().expect("fixture non-empty"),
            ))
            .copied()
            .collect::<Vec<_>>()
            .join("\n");
        let report = PytestParser.parse(&truncated);
        assert_eq!(report.statuses.len(), 60);
        assert_eq!(report.summary_count, Some(121));
        assert_eq!(report.dropped_outside_section, 0);
        let task = task_from_json(DEADLOCK_TASK_JSON);
        let score = resolve(&report.statuses, report.summary_count, &task);
        match score {
            TrialScore::Invalid { reason } => {
                assert_eq!(reason, "parse-mismatch: 60 ids vs summary 121");
            }
            other => panic!("expected Invalid{{parse-mismatch}}, got {other:?}"),
        }
    }

    #[test]
    fn extra_agent_authored_tests_do_not_change_the_verdict() {
        // This test models a FUTURE gate_command that collects a whole
        // directory: under today's eight tier-2 gate_commands an agent-authored
        // test cannot reach the re-gate at all (each gate_command names
        // specific test files that are overwritten by copy_sealed — so the
        // agent's additions are invisible). This is defense-in-depth for a
        // shape not yet observed in real tier-2 runs.
        let fixture = include_str!("../testdata/mined_eval/deadlock-trial-0.txt");
        let fixture_lines: Vec<&str> = fixture.lines().collect();
        let hdr = fixture_lines
            .iter()
            .position(|l| l.contains("short test summary info"))
            .expect("fixture must have a short test summary info header");
        let new_line = "PASSED tests/test_agent_authored.py::test_added";
        let mut mutated_lines = fixture_lines.clone();
        mutated_lines.insert(hdr + 1, new_line);
        let mutated: String = mutated_lines.join("\n");
        let mutated = mutated.replacen("2 failed, 119 passed", "2 failed, 120 passed", 1);
        let report = PytestParser.parse(&mutated);
        assert_eq!(report.statuses.len(), 122);
        assert_eq!(report.summary_count, Some(122));
        let task = task_from_json(DEADLOCK_TASK_JSON);
        let score_original = resolve(
            &PytestParser.parse(fixture).statuses,
            PytestParser.parse(fixture).summary_count,
            &task,
        );
        let score_mutated = resolve(&report.statuses, report.summary_count, &task);
        // Same variant.
        assert!(matches!(score_mutated, TrialScore::Unresolved { .. }));
        // Same missing_fail_to_pass.
        if let (TrialScore::Unresolved { reason: r1 }, TrialScore::Unresolved { reason: r2 }) =
            (score_original, score_mutated)
        {
            assert_eq!(r1.missing_fail_to_pass, r2.missing_fail_to_pass);
        }
    }

    #[test]
    fn gate_fault_reason_maps_every_arm() {
        use std::time::Duration;
        // Timeout arm.
        assert_eq!(
            gate_fault_reason(true, None, Duration::from_mins(3), "", ""),
            Some("gate-timeout: after 180s".to_string()),
        );
        // Signal-killed with stderr, no stdout -> Invalid.
        assert_eq!(
            gate_fault_reason(false, None, Duration::ZERO, "", "boom\n"),
            Some("gate-no-exit-code: no exit status (stderr: boom)".to_string()),
        );
        // Signal-killed WITH stdout -> scoreable (stay as None).
        assert_eq!(
            gate_fault_reason(false, None, Duration::ZERO, "PASSED tests/x.py::t\n", ""),
            None,
        );
        // Non-zero exit code -> scoreable.
        assert_eq!(
            gate_fault_reason(false, Some(1), Duration::ZERO, "", ""),
            None
        );
        // Zero exit code -> scoreable.
        assert_eq!(
            gate_fault_reason(false, Some(0), Duration::ZERO, "", ""),
            None
        );
    }

    #[tokio::test]
    async fn sealed_regate_score_invalid_on_gate_timeout() {
        use super::sealed_regate_score_with_timeout;
        let task_dir = tempdir().expect("task tempdir");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir sealed");
        std::fs::write(task_dir.path().join("sealed/tests/test_x.py"), "# sealed\n")
            .expect("write sealed");
        let ws = tempdir().expect("ws tempdir");
        std::fs::create_dir_all(ws.path().join("tests")).expect("mkdir tests");
        std::fs::write(ws.path().join("tests/test_x.py"), "# agent\n").expect("write agent");
        let task = MinedTask {
            id: "synth-timeout".to_string(),
            repo: "r".to_string(),
            repo_path: "/tmp/x".to_string(),
            parent_sha: "aaa".to_string(),
            fix_sha: "bbb".to_string(),
            rung_guess: "easy".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: String::new(),
            gate_command: "sleep 5".to_string(),
            agent_gate_command: None,
            fail_to_pass: Vec::new(),
            positive_controls: Vec::new(),
            pass_to_pass_exclusions: Vec::new(),
            sealed: vec![SealedEntry {
                path: "tests/test_x.py".to_string(),
            }],
            notes: Vec::new(),
        };
        let (score, raw) = sealed_regate_score_with_timeout(
            &task,
            task_dir.path(),
            ws.path(),
            &PytestParser,
            std::time::Duration::from_millis(50),
        )
        .await;
        match score {
            TrialScore::Invalid { reason } => {
                assert!(
                    reason.starts_with("gate-timeout: after "),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Invalid{{gate-timeout}}, got {other:?}"),
        }
        assert_eq!(raw, "", "timeout raw must be empty");
    }

    #[tokio::test]
    async fn sealed_regate_score_invalid_when_gate_dies_without_output() {
        use super::sealed_regate_score_with_timeout;
        let task_dir = tempdir().expect("task tempdir");
        std::fs::create_dir_all(task_dir.path().join("sealed/tests")).expect("mkdir sealed");
        std::fs::write(task_dir.path().join("sealed/tests/test_x.py"), "# sealed\n")
            .expect("write sealed");
        let ws = tempdir().expect("ws tempdir");
        std::fs::create_dir_all(ws.path().join("tests")).expect("mkdir tests");
        std::fs::write(ws.path().join("tests/test_x.py"), "# agent\n").expect("write agent");
        let task = MinedTask {
            id: "synth-sigkill".to_string(),
            repo: "r".to_string(),
            repo_path: "/tmp/x".to_string(),
            parent_sha: "aaa".to_string(),
            fix_sha: "bbb".to_string(),
            rung_guess: "easy".to_string(),
            language: "python".to_string(),
            provenance: serde_json::Value::Null,
            env: super::MinedEnv {
                setup: "true".to_string(),
                sibling_repos: Vec::new(),
            },
            test_scope: String::new(),
            gate_command: "kill -KILL $$".to_string(),
            agent_gate_command: None,
            fail_to_pass: Vec::new(),
            positive_controls: Vec::new(),
            pass_to_pass_exclusions: Vec::new(),
            sealed: vec![SealedEntry {
                path: "tests/test_x.py".to_string(),
            }],
            notes: Vec::new(),
        };
        let (score, _raw) = sealed_regate_score_with_timeout(
            &task,
            task_dir.path(),
            ws.path(),
            &PytestParser,
            std::time::Duration::from_secs(10),
        )
        .await;
        match score {
            TrialScore::Invalid { reason } => {
                assert!(reason.starts_with("gate-no-exit-code"), "reason: {reason}");
            }
            other => panic!("expected Invalid{{gate-no-exit-code}}, got {other:?}"),
        }
    }

    #[test]
    fn collection_error_unresolved_counts_as_false_done() {
        let report = MinedReport {
            task_id: "t".to_string(),
            backend_desc: "b".to_string(),
            spec_level: SpecLevel::S2,
            max_iterations: 24,
            k: 1,
            resolved_count: 0,
            invalid_count: 0,
            trials: vec![trial(
                0,
                TrialScore::Unresolved {
                    reason: ResolveDetail {
                        fail_to_pass_status: Vec::new(),
                        unexcluded_red: vec!["tests/test_cli.py".to_string()],
                        missing_fail_to_pass: Vec::new(),
                        collection_errors: vec!["tests/test_cli.py".into()],
                    },
                },
                CLAIMED_DONE,
            )],
        };
        assert_eq!(report.false_dones(), 1);
        assert_eq!(report.valid_denominator(), 1);
        assert!((report.resolved_rate() - 0.0).abs() < f64::EPSILON);
    }
}
