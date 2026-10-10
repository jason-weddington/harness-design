//! Tier-2 mined-task types: the talos-evals task-dir schema, task loading, statements and the tier-2 prompt, consumed by the Harbor emitter ([`crate::mined_harbor`]).
//!
//! ## Task-dir schema (talos-evals shape)
//!
//! The Harbor emitter reads a talos-evals task-family root; each task dir carries:
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
//! | `env.sibling_repos` | `Vec<{repo, pin}>` | repos bundled beside the primary at pin |
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

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ===== timeouts ==========================================================

/// Timeout for `env.setup`. A cold `uv sync` (or `pnpm install`) can exceed
/// [`crate::eval::CODING_CHECK_TIMEOUT`] (3 min), so setup gets its own wider
/// budget.
pub const MINED_SETUP_TIMEOUT: Duration = Duration::from_mins(10);

// ===== task.json ==========================================================

/// Deserialized `task.json`.
///
/// Field names are **verbatim** matches of the on-disk schema — no
/// `#[serde(rename)]`. Unknown top-level and nested keys are tolerated so the
/// schema can grow additive metadata without a harness update. All listed
/// fields are required except `notes` (optional — ignored, carried only for
/// forward-compatibility with schema growth), `provenance` (opaque) and
/// `agent_gate_command` (optional at load; see [`resolve_agent_gate_command`]).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct MinedTask {
    /// Stable task id (matches the task-dir name).
    pub id: String,
    /// Source repo short name (informational; used to resolve `env.sibling_repos`).
    pub repo: String,
    /// On-box path to the source repo. A leading `~/` is expanded via `$HOME`
    /// when the task is emitted.
    pub repo_path: String,
    /// The commit the agent starts from.
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
    /// The emitted `tests/test.sh` copies the sealed files over the agent's
    /// before the re-gate, so agent-authored edits to a sealed path cannot
    /// reach the scorer.
    pub gate_command: String,
    /// The in-run AGENT gate: the repo's real dispatch gate (incl. lint /
    /// typecheck, not just tests). Shown to the agent (appended to its
    /// prompt as a `## Verification` section), run by `run_checks`, by
    /// `finish(done)` verification, and by the pre-agent baseline tripwire.
    /// It is NEVER used for scoring — [`MinedTask::gate_command`] stays the
    /// sealed scoring judge, unchanged.
    ///
    /// Must be a CHECKER: it must not mutate the tree, and it must be green
    /// at `parent_sha` in the task env (verified by the baseline tripwire;
    /// see [`AgentGateMode::On`]).
    ///
    /// The key may be absent (it loads as `None`); the Harbor emitter then
    /// rejects the task, because every emitted task is gated.
    #[serde(default)]
    pub agent_gate_command: Option<String>,
    /// Test ids that must all be Passed for a trial to resolve.
    pub fail_to_pass: Vec<String>,
    /// Test ids that MUST remain Passed (present + green).
    ///
    /// An **absent** `positive_control` makes the trial invalid
    /// (`positive-control-uncollected`, scored by `mined_score.py`) — an
    /// environment tripwire, not a red verdict.
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
    /// Shell script bootstrapping the workspace. Runs via `bash -c` during
    /// the task image build under [`MINED_SETUP_TIMEOUT`]. A non-zero exit or
    /// timeout fails the build with `TALOS_EVAL_INVALID reason=setup-failed`,
    /// so the agent never runs.
    pub setup: String,
    /// Sibling repos to lay down alongside the primary checkout so
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
/// overrides talos's clap default of 300. The Harbor emitter formats it into
/// each task's agent-gate timeouts, and tier-2 Harbor runs pass the same
/// value as `--ak gate_timeout_secs`.
pub const DEFAULT_AGENT_GATE_TIMEOUT: Duration = Duration::from_mins(15);

/// Whether the in-run agent gate (`agent_gate_command`) is armed (`On`) or
/// not (`Off`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentGateMode {
    /// No agent gate: no Verification section in the prompt.
    Off,
    /// The agent gate is armed with timeout (the emitted agent-gate.sh and the image-build baseline tripwire use it).
    On {
        /// Timeout applied to every run of `agent_gate_command`: the
        /// baseline tripwire, `run_checks`, and `finish(done)` verification.
        timeout: Duration,
    },
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
pub fn resolve_agent_gate_command(
    task: &MinedTask,
    mode: AgentGateMode,
) -> Result<Option<&str>, String> {
    let AgentGateMode::On { .. } = mode else {
        return Ok(None);
    };
    let Some(cmd) = task.agent_gate_command.as_deref() else {
        return Err(format!(
            "task `{}`: agent_gate_command is missing (required when the agent gate is on)",
            task.id
        ));
    };
    if cmd.trim().is_empty() {
        return Err(format!(
            "task `{}`: agent_gate_command is blank (required when the agent gate is on)",
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

// ===== shared helpers =====================================================

/// A `git` [`std::process::Command`] with any inherited git-hook environment
/// SCRUBBED. When the runner (or its tests) executes under a `git commit`
/// hook, git exports `GIT_DIR`/`GIT_INDEX_FILE`/`GIT_WORK_TREE`/`GIT_PREFIX`
/// into the hook's environment; a child `git -C <other-repo>` inheriting
/// those misresolves against the WRONG repo (observed: `worktree add` dying
/// with `.git/index: Not a directory` inside a freshly-created worktree).
/// Every git invocation in the tier-2 modules must go through this constructor.
pub(crate) fn git_command() -> std::process::Command {
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

/// Expand a leading `~/` (and bare `~`) against `$HOME`. Non-tilde input is
/// returned unchanged. When `$HOME` is unset the input is returned as-is
/// (a leading `~/` will then be treated as a literal path segment by the
/// caller — better than a silent expansion to `/`).
pub(crate) fn expand_home(raw: &str) -> String {
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
pub fn tier2_task_prompt(
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

#[cfg(test)]
mod tests {
    use super::{
        AgentGateMode, DEFAULT_AGENT_GATE_TIMEOUT, MINED_SETUP_TIMEOUT, MinedTask, SealedEntry,
        SpecLevel, expand_home, load_statement, load_task, resolve_agent_gate_command,
        tier2_task_prompt,
    };
    use std::path::Path;
    use std::time::Duration;
    use tempfile::tempdir;

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
            "TestOldWireFormat::test_old_format_still_parses"
        ],
        "sealed": [
            {"path": "tests/test_cleanr.py"}
        ],
        "notes": ["mined 2026-08-13"]
    }"#;

    fn write_task(dir: &Path, body: &str) {
        std::fs::write(dir.join("task.json"), body).expect("write task.json");
    }

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

    fn on_mode() -> AgentGateMode {
        AgentGateMode::On {
            timeout: Duration::from_secs(30),
        }
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

    #[test]
    fn load_task_returns_io_error_when_task_json_missing() {
        let dir = tempdir().expect("tempdir");
        // No task.json written.
        let err = load_task(dir.path()).expect_err("missing task.json must error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn spec_level_parse_is_case_insensitive() {
        assert_eq!(SpecLevel::parse("s1"), Some(SpecLevel::S1));
        assert_eq!(SpecLevel::parse("S2"), Some(SpecLevel::S2));
        assert_eq!(SpecLevel::parse("s3"), Some(SpecLevel::S3));
        assert_eq!(SpecLevel::parse("bogus"), None);
    }

    #[test]
    fn spec_level_slug_matches_filename() {
        assert_eq!(SpecLevel::S1.slug(), "s1");
        assert_eq!(SpecLevel::S2.slug(), "s2");
        assert_eq!(SpecLevel::S3.slug(), "s3");
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

    #[test]
    fn resolve_agent_gate_command_off_ignores_the_field() {
        let mut task = sample_task();
        task.agent_gate_command = Some("x".to_string());
        assert_eq!(
            resolve_agent_gate_command(&task, AgentGateMode::Off),
            Ok(None)
        );
    }

    #[test]
    fn resolve_agent_gate_command_on_missing_is_err() {
        let task = sample_task();
        let err = resolve_agent_gate_command(&task, on_mode()).expect_err("missing must error");
        assert!(err.contains(&task.id));
        assert!(err.contains("agent_gate_command"));
        assert_eq!(
            err,
            "task `cleanr-abcdef1`: agent_gate_command is missing (required when the agent gate is on)"
        );
    }

    #[test]
    fn resolve_agent_gate_command_on_blank_is_err() {
        let mut task = sample_task();
        task.agent_gate_command = Some("   ".to_string());
        let err = resolve_agent_gate_command(&task, on_mode()).expect_err("blank must error");
        assert!(err.contains(&task.id));
        assert!(err.contains("agent_gate_command"));
        assert_eq!(
            err,
            "task `cleanr-abcdef1`: agent_gate_command is blank (required when the agent gate is on)"
        );
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

    #[test]
    fn timeouts_are_pinned() {
        assert_eq!(DEFAULT_AGENT_GATE_TIMEOUT.as_secs(), 900);
        assert_eq!(MINED_SETUP_TIMEOUT.as_secs(), 600);
    }
}
