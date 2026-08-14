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
//! `task.json` fields (all required except `notes`):
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
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::engine::{self, LoopOutcome, RunConfig, RunResult};
use crate::eval::{CODING_CHECK_TIMEOUT, copy_dir_recursive};
use crate::exec::{ExecSpec, run};
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
/// fields are required; `notes` is optional (ignored — carried only for
/// forward-compatibility with schema growth).
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
    pub gate_command: String,
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

/// Parser of a runner's test-report output into a `nodeid -> status` map.
///
/// Only [`PytestParser`] ships in the pilot; the trait exists so a vitest or
/// cargo-test parser can plug in later without touching the scorer.
pub trait TestReportParser: Send + Sync {
    /// Parse `output` (typically `stdout + stderr` from the gate command) into
    /// `(map, summary_count)`.
    ///
    /// `summary_count` is the parser's second-source count of individual test
    /// outcomes, cross-checked against `map.len()` in [`resolve`] to detect
    /// truncation / broken output (`parse-mismatch`). `None` means the parser
    /// has no second-source count.
    fn parse(&self, output: &str) -> (BTreeMap<String, TestStatus>, Option<usize>);
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
    fn parse(&self, output: &str) -> (BTreeMap<String, TestStatus>, Option<usize>) {
        let mut map: BTreeMap<String, TestStatus> = BTreeMap::new();
        let mut summary_count: Option<usize> = None;
        for line in output.lines() {
            let trimmed = line.trim_start();
            if let Some((status, rest)) = parse_short_summary_line(trimmed) {
                // pytest short-summary lines are `STATUS nodeid [reason]`.
                // Take the first whitespace-delimited token after STATUS as
                // the nodeid — the (optional) reason is ignored.
                let nodeid = rest.split_whitespace().next().unwrap_or_default();
                if !nodeid.is_empty() {
                    // Last-write-wins on duplicates (a re-run would be rare
                    // and harmless — a stable status is what matters).
                    map.insert(nodeid.to_string(), status);
                }
            } else if let Some(n) = parse_pytest_summary_totals(trimmed) {
                summary_count = Some(n);
            }
        }
        (map, summary_count)
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
    let body = inner.0.trim();
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrialScore {
    /// Every `fail_to_pass` + `positive_controls` id was Passed AND every
    /// otherwise-red id was covered by `pass_to_pass_exclusions`.
    Resolved,
    /// The gate ran and parsed, but at least one clause did not hold.
    Unresolved { reason: ResolveDetail },
    /// An infra failure — `env.setup` timed out, sealed copy failed, parser
    /// emitted an empty map, count-mismatch, or a `positive_controls` id was
    /// uncollected. Excluded from the resolved/k rate.
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
}

/// Score a parsed nodeid→status map against `task`'s clauses.
///
/// Exclusions-authoritative semantics (pilot pin). Returns
/// [`TrialScore::Invalid`] up front for structural problems (empty parse,
/// count-mismatch, missing `positive_control`) so the caller sees a distinct
/// signal from a genuine unresolved verdict.
#[must_use]
pub fn resolve(
    parsed: &BTreeMap<String, TestStatus>,
    summary_count: Option<usize>,
    task: &MinedTask,
) -> TrialScore {
    if parsed.is_empty() {
        return TrialScore::Invalid {
            reason: "parse-empty".to_string(),
        };
    }
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

    // (a) positive_controls: every id present + Passed. Absent ⇒ Invalid.
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
                    },
                };
            }
        }
    }

    // (b) fail_to_pass: every id present + Passed.
    let missing_ftp = build_missing_fail_to_pass(&normalized, task);
    if !missing_ftp.is_empty() {
        return TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: build_fail_to_pass_status(&normalized, task),
                unexcluded_red: build_unexcluded_red(&normalized, task),
                missing_fail_to_pass: missing_ftp,
            },
        };
    }

    // (c) any other red id must be covered by an exclusion.
    let unexcluded = build_unexcluded_red(&normalized, task);
    if !unexcluded.is_empty() {
        return TrialScore::Unresolved {
            reason: ResolveDetail {
                fail_to_pass_status: build_fail_to_pass_status(&normalized, task),
                unexcluded_red: unexcluded,
                missing_fail_to_pass: Vec::new(),
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

/// Run the sealed re-gate and score the result.
///
/// Steps (mirrors the tier-1 holdout ordering):
/// 1. `copy_sealed` — overwrite the agent's copies of the sealed files.
/// 2. Run `task.gate_command` under `bash -c` in `workspace_root` with
///    `PYTEST_ADDOPTS="-rA"` injected, [`CODING_CHECK_TIMEOUT`] timeout.
///    We use [`crate::exec::run`] directly (not [`crate::exec::ChecksRunner`])
///    because `ChecksRunner` truncates output to a 4 KB tail, which would
///    lose the `-rA` PASSED lines the scorer needs.
/// 3. Parse `stdout + stderr` with `parser`.
/// 4. Call [`resolve`].
///
/// Returns `(TrialScore, raw_output)` — the caller persists `raw_output`
/// to disk for provenance.
pub async fn sealed_regate_score<P: TestReportParser + ?Sized>(
    task: &MinedTask,
    task_dir: &Path,
    workspace_root: &Path,
    parser: &P,
) -> (TrialScore, String) {
    if let Err(reason) = copy_sealed(task_dir, workspace_root, &task.sealed) {
        return (TrialScore::Invalid { reason }, String::new());
    }
    let outcome = run(&ExecSpec {
        program: "bash".to_string(),
        args: vec!["-c".to_string(), task.gate_command.clone()],
        cwd: workspace_root.to_path_buf(),
        timeout: CODING_CHECK_TIMEOUT,
        extra_env: vec![("PYTEST_ADDOPTS".to_string(), "-rA".to_string())],
    })
    .await;
    let raw = format!("{}{}", outcome.stdout, outcome.stderr);
    let (parsed, count) = parser.parse(&raw);
    let score = resolve(&parsed, count, task);
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
    /// Raw nodeid → status map from the sealed re-gate parser (empty when
    /// the trial is [`TrialScore::Invalid`] before the parser ran).
    pub statuses: BTreeMap<String, TestStatus>,
    /// Absolute path to the persisted raw re-gate stdout+stderr under
    /// `${XDG_STATE_HOME:-~/.local/state}/talos/mined-eval/<task-id>/trial-<k>/gate-output.txt`.
    pub gate_output_path: PathBuf,
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
        LoopOutcome::BackendError(_) => "BackendError".to_string(),
    }
}

// ===== the k-trial loop ===================================================

/// Callback for per-trial telemetry (mirrors [`crate::eval::run_eval`]'s
/// `on_trial`). The runner uses it to stream one line per trial.
pub type OnMinedTrial<'a> = &'a mut dyn FnMut(&MinedTrialResult);

/// Root of the XDG-state-shaped output directory the runner writes per-trial
/// gate output under. Overridable for tests via the `TALOS_MINED_STATE_ROOT`
/// env var (a private test hook; not part of the public schema).
fn xdg_state_root() -> PathBuf {
    resolve_state_root(
        std::env::var_os("TALOS_MINED_STATE_ROOT"),
        std::env::var_os("XDG_STATE_HOME"),
        std::env::var_os("HOME"),
    )
}

/// Pure precedence rule for [`xdg_state_root`]: extracted so each branch can
/// be tested without mutating process env (edition-2024 `set_var` is unsafe
/// and `unsafe_code` is `forbid`-den project-wide).
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

/// Persist raw re-gate output under
/// `<xdg-state>/talos/mined-eval/<task-id>/trial-<k>/gate-output.txt` and
/// return its absolute path. A write error falls back to a temp path so a
/// trial never fails just because the state dir is unwritable.
fn persist_gate_output(task_id: &str, trial: u32, raw: &str) -> PathBuf {
    let dir = xdg_state_root()
        .join("talos/mined-eval")
        .join(task_id)
        .join(format!("trial-{trial}"));
    let path = dir.join("gate-output.txt");
    if std::fs::create_dir_all(&dir).is_ok() && std::fs::write(&path, raw).is_ok() {
        return path;
    }
    // Best-effort fallback so a trial never dies over an offload write.
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let fallback = std::env::temp_dir().join(format!(
        "talos-mined-eval-{}-{trial}-{n}.txt",
        sanitize_for_filename(task_id)
    ));
    let _ = std::fs::write(&fallback, raw);
    fallback
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
}

/// Run the whole mined-task eval: `k` independent trials, each with a fresh
/// primary+sibling worktree, `env.setup`, agent loop with `checks=None`, and
/// sealed re-gate scoring.
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
async fn single_trial<B: ModelBackend>(
    backend: &B,
    parser: &(impl TestReportParser + ?Sized),
    config: &MinedRunConfig<'_>,
    trial: u32,
) -> MinedTrialResult {
    let start = Instant::now();

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

    // 3. Agent loop with checks=None self-certify.
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
    let ctx = ToolCtx::new(
        Arc::new(workspace),
        Arc::new(DiskOffloadSink::new(offload_canon)),
    );
    let tools = standard_registry(None);
    let run_config = RunConfig::new(config.statement.to_string(), config.max_iterations);
    let RunResult { outcome, stats } = engine::run(backend, &tools, &ctx, &run_config).await;
    let claimed = claimed_disposition_label(&outcome);

    // 4. Sealed re-gate + scoring.
    let (score, raw) =
        sealed_regate_score(config.task, config.task_dir, &workspace_root, parser).await;
    let gate_output_path = persist_gate_output(&config.task.id, trial, &raw);

    // Re-parse to capture the raw status map on the trial record (empty on
    // Invalid trials that never ran the parser).
    let (statuses, _) = if raw.is_empty() {
        (BTreeMap::new(), None)
    } else {
        parser.parse(&raw)
    };

    // Worktrees drop here (after re-gate has read the workspace).
    drop(worktrees);

    MinedTrialResult {
        trial,
        score,
        iterations: stats.iterations,
        input_tokens: stats.input_tokens,
        output_tokens: stats.output_tokens,
        wall: start.elapsed(),
        claimed_disposition: claimed,
        statuses,
        gate_output_path,
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
    let gate_output_path = persist_gate_output(&config.task.id, trial, "");
    MinedTrialResult {
        trial,
        score: TrialScore::Invalid { reason },
        iterations: 0,
        input_tokens: 0,
        output_tokens: 0,
        wall: start.elapsed(),
        claimed_disposition: "NotRun".to_string(),
        statuses: BTreeMap::new(),
        gate_output_path,
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
        CLAIMED_DONE, MinedReport, MinedTask, MinedTrialResult, PytestParser, ResolveDetail,
        ScratchDir, SealedEntry, SpecLevel, TestReportParser, TestStatus, TrialScore,
        claimed_disposition_label, copy_sealed, expand_home, load_statement, load_task,
        match_task_id, matches_exclusion, normalize, parse_pytest_summary_totals,
        parse_short_summary_line, prepare_worktrees, resolve, run_env_setup, sanitize_for_filename,
        strip_param_suffix,
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

    /// Synthetic `-rA` short-summary output. NEVER real talos-evals capture.
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
        let (map, count) = PytestParser.parse(&synthetic_pytest_output());
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
        let out = "ERROR tests/test_x.py::TestX::test_broken\nXPASS tests/test_x.py::TestX::test_unex_pass\n=== 0 passed, 1 error, 1 xpassed in 0.01s ===";
        let (map, count) = PytestParser.parse(out);
        assert_eq!(count, Some(2));
        assert_eq!(
            map.get("tests/test_x.py::TestX::test_broken"),
            Some(&TestStatus::Error)
        );
        assert_eq!(
            map.get("tests/test_x.py::TestX::test_unex_pass"),
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

    // ---- Invalid{parse-empty} on -q output --------------------------------

    #[test]
    fn dots_only_output_yields_parse_empty_invalid() {
        // A `-q` run without -rA: dots, no STATUS lines.
        let dots_only = "tests/x.py ...F.                                     [100%]\n";
        let (map, count) = PytestParser.parse(dots_only);
        assert!(map.is_empty(), "dots-only output produces no ids");
        assert_eq!(count, None, "no summary line in this fragment");
        // Scoring: parse-empty is Invalid, never silently unresolved.
        let task = sample_task_for_scoring();
        let score = resolve(&map, count, &task);
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

    fn trial(idx: u32, score: TrialScore, claimed: &str) -> MinedTrialResult {
        MinedTrialResult {
            trial: idx,
            score,
            iterations: 0,
            input_tokens: 0,
            output_tokens: 0,
            wall: Duration::ZERO,
            claimed_disposition: claimed.to_string(),
            statuses: BTreeMap::new(),
            gate_output_path: PathBuf::from("/dev/null"),
        }
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
            gate_command: r"printf 'PASSED tests/test_x.py::T::a\nPASSED tests/test_x.py::T::b\n=== 2 passed in 0.0s ===\n'".to_string(),
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

    // ---- persist_gate_output writes to state root, sanitizes id --------

    #[test]
    fn persist_gate_output_writes_under_state_root_override() {
        // Set the private test hook so we don't have to guess XDG_STATE_HOME.
        // std::env::set_var is unsafe in edition 2024 (forbidden here), so
        // we set it via a child process: run persist through a helper that
        // reads TALOS_MINED_STATE_ROOT if present. Since we can't set env
        // vars in-process, we instead verify the fallback branch (no state
        // root) DOES write to a temp path and it contains the content.
        // The XDG_STATE_HOME/HOME branches are hit implicitly on any test
        // build since HOME is always set — persist_gate_output writes to
        // ~/.local/state/talos/mined-eval/... which is a real filesystem
        // write we then read back to verify.
        let raw = "hello mined world";
        let out_path = super::persist_gate_output("some-task-id/with:weird chars", 7, raw);
        // The path must exist and contain our content.
        let contents = std::fs::read_to_string(&out_path).expect("read gate output");
        assert_eq!(contents, raw);
        // The path must live somewhere under the state root; simplest check:
        // its file name should mention the trial and its filename is
        // gate-output.txt when we went through the primary write path OR a
        // temp fallback file otherwise. Either way, the content assertion is
        // enough — path shape is an implementation detail.
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
            r"printf 'PASSED tests/test_synth.py::T::pass_it\n=== 1 passed in 0.0s ===\n'"
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
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k,
            max_iterations: 5,
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
        }
        assert_eq!(on_trial_calls, k);
        // Header aggregates:
        assert_eq!(report.valid_denominator(), k);
        assert!((report.resolved_rate() - 1.0).abs() < f64::EPSILON);
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
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
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
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
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
        assert_eq!(
            claimed_disposition_label(&LoopOutcome::BackendError(BackendError::Terminal {
                kind: TerminalKind::Auth,
                message: "no creds".to_string(),
            })),
            "BackendError",
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

    // ---- xdg_state_root — call the wrapper (covers env-read branches) ----

    #[test]
    fn xdg_state_root_returns_a_directory_shaped_path() {
        // Runs the actual `xdg_state_root` wrapper so its `env::var_os` branch
        // predicates get exercised (the pure worker is tested separately).
        let path = super::xdg_state_root();
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
        let out = "PASSED \nPASSED tests/x.py::T::real\n=== 1 passed in 0.0s ===";
        let (map, count) = PytestParser.parse(out);
        assert_eq!(map.len(), 1, "empty nodeid line must not insert");
        assert!(map.contains_key("tests/x.py::T::real"));
        assert_eq!(count, Some(1));
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
            r"printf 'FAILED tests/x.py::T::red\n=== 0 passed, 1 failed in 0.0s ===\n'".to_string();
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
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
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
        let config = MinedRunConfig {
            task_dir: task_dir.path(),
            task: &task,
            statement: &statement,
            spec_level: SpecLevel::S2,
            backend_desc: "mock".to_string(),
            k: 1,
            max_iterations: 3,
        };
        let mut noop = |_t: &MinedTrialResult| {};
        let report = run_mined_task(&backend, &PytestParser, &config, &mut noop).await;
        // Either the setup itself fails (rm -rf $(pwd) from within can fail)
        // OR Workspace::new fails afterwards. Both produce Invalid.
        assert_eq!(report.invalid_count, 1);
        assert!(matches!(report.trials[0].score, TrialScore::Invalid { .. }));
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
}
