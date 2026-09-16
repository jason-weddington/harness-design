//! Integration tests for `talos run`, spawning the compiled binary via
//! `env!("CARGO_BIN_EXE_talos")` so `cargo-llvm-cov` can collect
//! child-process coverage.
//!
//! Tests here exercise the full `main()` wiring (spec parsing, CLI errors,
//! `Workspace`+`ToolCtx`+store+`Persistence`+`run_persisted`+exit map) without any
//! live API key or real model. The deterministic `BackendError` path is the key
//! coverage driver: a refused-connection Ollama request exercises everything
//! up to and including the store write that `run_persisted` performs on every
//! terminal path.

use std::io::Write as _;
use std::process::{Command, Stdio};

/// Compiled `talos` binary path (injected by cargo at integration-test time).
const TALOS_BIN: &str = env!("CARGO_BIN_EXE_talos");

/// A minimal valid [`harness::task_spec::TaskSpec`] JSON.
fn valid_spec_json() -> &'static str {
    r#"{
        "title": "Integration test task",
        "description": "A task used by CLI integration tests.",
        "acceptance_criteria": [],
        "files_to_modify": [],
        "gate_command": ""
    }"#
}

// ============================================================================
// (a) Malformed spec → exit 1 + JSON error, no record
// ============================================================================

/// Malformed stdin spec exits 1; stderr is a one-line JSON `{"error": ...}`;
/// no store record is written (store file must not exist).
#[test]
fn malformed_spec_stdin_exits_1_with_json_error_no_record() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn talos");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"this is not valid json at all")
        .unwrap();

    let output = child.wait_with_output().expect("wait for talos");

    assert_eq!(output.status.code(), Some(1), "malformed spec must exit 1");

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let parsed: serde_json::Value = serde_json::from_str(stderr_str.trim()).unwrap_or_else(|_| {
        panic!("stderr must be valid JSON; got: {stderr_str:?}");
    });
    assert!(
        parsed.get("error").is_some(),
        "stderr JSON must have an `error` key; got: {parsed}"
    );

    // Spec parsing fails before the store is opened → the store file must
    // not exist.
    assert!(
        !store_path.exists(),
        "no store file must be written when spec parsing fails"
    );
}

// ============================================================================
// (b) No --workspace → exit 1
// ============================================================================

/// Missing `--workspace` is a clap usage error; the CLI must exit 1 (not
/// clap's default exit 2) since the locked exit-code contract has no code 2.
#[test]
fn missing_workspace_flag_exits_1() {
    let output = Command::new(TALOS_BIN)
        .args(["run"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(
        output.status.code(),
        Some(1),
        "missing --workspace must exit 1"
    );
}

// ============================================================================
// (c) Deterministic BackendError via refused-connection Ollama
// ============================================================================

/// Full wiring test without a live API key: valid spec + Ollama pointed at a
/// refused port. Asserts:
/// - exit code 1 (`BackendError` → 1 in the locked map)
/// - stdout summary is valid JSON with `outcome == "BackendError"`
/// - the `SQLite` store has a run record whose disposition is
///   `Failed { mode: TransientInfra }` (connection-refused is retryable →
///   `BackendError::Transient { kind: Network }` → `TransientInfra`)
#[tokio::test(flavor = "current_thread")]
async fn backend_error_via_refused_port_writes_store_record() {
    use harness::engine::run_id;
    use harness::run_record::{Disposition, FailureMode};
    use harness::store::{RunStore as _, SqliteRunStore};

    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();

    let task_id = "cli-test-backend-err";
    let attempt: u32 = 1;

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
            "--task-id",
            task_id,
            "--attempt",
            "1",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn talos");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(valid_spec_json().as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait for talos");

    // BackendError → exit code 1 (NOT 20 — that is task-Failed).
    assert_eq!(
        output.status.code(),
        Some(1),
        "BackendError must exit 1, not 20"
    );

    // stdout is a machine-readable JSON summary.
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let summary: serde_json::Value = serde_json::from_str(stdout_str.trim()).unwrap_or_else(|_| {
        panic!("stdout must be valid JSON summary; got: {stdout_str:?}");
    });
    assert_eq!(
        summary.get("outcome").and_then(serde_json::Value::as_str),
        Some("BackendError"),
        "summary outcome must be \"BackendError\"; got: {summary}"
    );

    // The run record must exist in the store — engine::run_persisted writes
    // a terminal checkpoint on every path including BackendError.
    let store = SqliteRunStore::open(&store_path).expect("store must be openable after run");
    let rid = run_id(task_id, attempt);
    let record = store
        .load(&rid)
        .await
        .expect("store load must not error")
        .unwrap_or_else(|| {
            panic!("run record for {rid:?} must exist in the store after BackendError")
        });

    assert!(
        matches!(
            record.disposition,
            Some(Disposition::Failed {
                mode: FailureMode::TransientInfra,
                ..
            })
        ),
        "disposition must be Failed{{TransientInfra}}; got: {:?}",
        record.disposition
    );

    // Criterion Coverage guidance is default off (no --criteria-rules flag
    // was passed) — the persisted run record's messages must not carry it.
    assert!(
        !serde_json::to_string(&record.messages)
            .unwrap()
            .contains("## Criterion Coverage"),
        "record.messages must not contain the Criterion Coverage heading by default"
    );
}

// ============================================================================
// (c2) --transcript: opt-in JSONL transcript, off by default
// ============================================================================

/// `--transcript <path>` into a not-yet-created parent directory writes the
/// exact 7-line transcript for the refused-port `BackendError` retry
/// exhaustion (default `max_retries=3`, `retry_backoff_base=500ms`).
#[tokio::test(flavor = "current_thread")]
async fn transcript_flag_writes_pinned_seven_lines_on_backend_error() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();
    // Parent (`t/`) is deliberately NOT pre-created — the writer must
    // best-effort `create_dir_all` it.
    let transcript_path = dir.path().join("t").join("run.jsonl");

    let task_id = "cli-test-transcript";
    let attempt: u32 = 1;

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
            "--task-id",
            task_id,
            "--attempt",
            "1",
            "--transcript",
            transcript_path.to_str().unwrap(),
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn talos");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(valid_spec_json().as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait for talos");
    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let summary: serde_json::Value = serde_json::from_str(stdout_str.trim())
        .unwrap_or_else(|_| panic!("stdout must be valid JSON summary; got: {stdout_str:?}"));
    let summary_outcome = summary
        .get("outcome")
        .and_then(serde_json::Value::as_str)
        .expect("summary must carry an outcome string");
    assert_eq!(summary_outcome, "BackendError");

    let contents = std::fs::read_to_string(&transcript_path).expect("transcript file must exist");
    let lines: Vec<serde_json::Value> = contents
        .lines()
        .map(|l| serde_json::from_str(l).expect("each transcript line is valid JSON"))
        .collect();
    assert_eq!(lines.len(), 7, "expected exactly 7 transcript lines");

    // Line 1: run_start.
    assert_eq!(lines[0]["event"], "run_start");
    assert_eq!(
        lines[0]["label"], "ollama:x think=unset num_ctx=unset",
        "OLLAMA_THINK/OLLAMA_NUM_CTX are unset in the child env"
    );
    assert_eq!(
        lines[0]["run_id"],
        harness::engine::run_id(task_id, attempt)
    );
    assert_eq!(lines[0]["resume"], false);
    // Criterion Coverage guidance is default off (no --criteria-rules flag).
    assert!(
        !lines[0]["messages"]
            .to_string()
            .contains("## Criterion Coverage"),
        "run_start.messages must not contain the Criterion Coverage heading by default"
    );
    assert_eq!(
        lines[0]["config"]["acceptance_audit"], false,
        "acceptance_audit is off by default (no --acceptance-audit flag)"
    );

    // Line 2: model_request (iteration 1).
    assert_eq!(lines[1]["event"], "model_request");
    assert_eq!(lines[1]["iteration"], 1);

    // Lines 3-6: 4 backend_error entries (DEFAULT_MAX_RETRIES=3 -> 4 calls).
    let expected_will_retry = [true, true, true, false];
    let expected_retry_delay_ms = [Some(500), Some(1000), Some(2000), None];
    for attempt_idx in 0..4usize {
        let line = &lines[2 + attempt_idx];
        assert_eq!(line["event"], "backend_error");
        assert_eq!(line["iteration"], 1);
        assert_eq!(line["attempt"], attempt_idx);
        assert_eq!(line["retryable"], true);
        assert_eq!(line["will_retry"], expected_will_retry[attempt_idx]);
        match expected_retry_delay_ms[attempt_idx] {
            Some(ms) => assert_eq!(line["retry_delay_ms"], ms),
            None => assert!(line["retry_delay_ms"].is_null()),
        }
    }

    // Line 7: run_end, outcome matching the stdout RunSummary.
    assert_eq!(lines[6]["event"], "run_end");
    assert_eq!(lines[6]["outcome"], summary_outcome);
}

/// Same refused-port `BackendError` retry-exhaustion shape as
/// [`transcript_flag_writes_pinned_seven_lines_on_backend_error`], with
/// `--acceptance-audit` added — the flag must reach `run_start.config` (the
/// transcript's `acceptance_audit` key flips to `true`) while the pinned
/// 7-line shape and exit code are unchanged (`BackendError` never reaches the
/// finish path, so the audit itself cannot fire here).
#[tokio::test(flavor = "current_thread")]
async fn acceptance_audit_flag_reaches_run_start_config() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();
    let transcript_path = dir.path().join("t").join("run.jsonl");

    let task_id = "cli-test-acceptance-audit";

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
            "--task-id",
            task_id,
            "--attempt",
            "1",
            "--transcript",
            transcript_path.to_str().unwrap(),
            "--acceptance-audit",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn talos");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(valid_spec_json().as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait for talos");
    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");

    let contents = std::fs::read_to_string(&transcript_path).expect("transcript file must exist");
    let lines: Vec<serde_json::Value> = contents
        .lines()
        .map(|l| serde_json::from_str(l).expect("each transcript line is valid JSON"))
        .collect();
    assert_eq!(lines[0]["event"], "run_start");
    assert_eq!(
        lines[0]["config"]["acceptance_audit"], true,
        "--acceptance-audit must reach run_start.config"
    );
}

// ============================================================================
// (c3) --criteria-rules: opt-in Criterion Coverage guidance, off by default
// ============================================================================

/// `--criteria-rules` reaches BOTH the transcript's `run_start.messages` AND
/// the always-written run-record's persisted messages — so the default
/// artifact (the `SQLite` record, written on every run whether or not
/// `--transcript` is passed) identifies a rules-on run, not just the opt-in
/// transcript.
#[tokio::test(flavor = "current_thread")]
async fn criteria_rules_flag_reaches_run_start_messages() {
    use harness::engine::run_id;
    use harness::store::{RunStore as _, SqliteRunStore};

    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();
    let transcript_path = dir.path().join("t").join("run.jsonl");

    let task_id = "cli-test-criteria-rules";
    let attempt: u32 = 1;

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
            "--task-id",
            task_id,
            "--attempt",
            "1",
            "--transcript",
            transcript_path.to_str().unwrap(),
            "--criteria-rules",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn talos");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(valid_spec_json().as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait for talos");
    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");

    let contents = std::fs::read_to_string(&transcript_path).expect("transcript file must exist");
    let lines: Vec<serde_json::Value> = contents
        .lines()
        .map(|l| serde_json::from_str(l).expect("each transcript line is valid JSON"))
        .collect();
    assert_eq!(lines[0]["event"], "run_start");
    assert!(
        lines[0]["messages"]
            .to_string()
            .contains("## Criterion Coverage"),
        "run_start.messages must contain the Criterion Coverage heading with --criteria-rules"
    );

    let store = SqliteRunStore::open(&store_path).expect("store must be openable after run");
    let rid = run_id(task_id, attempt);
    let record = store
        .load(&rid)
        .await
        .expect("store load must not error")
        .unwrap_or_else(|| panic!("run record for {rid:?} must exist in the store"));
    assert!(
        serde_json::to_string(&record.messages)
            .unwrap()
            .contains("## Criterion Coverage"),
        "the persisted run record's messages must contain the Criterion Coverage heading"
    );
}

/// Recursively check whether `root` contains any file with a `.jsonl`
/// extension — used to prove the transcript writer never fires when
/// `--transcript` is absent.
fn walk_has_jsonl(root: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if walk_has_jsonl(&path) {
                return true;
            }
        } else if path.extension().is_some_and(|e| e == "jsonl") {
            return true;
        }
    }
    false
}

/// No `--transcript` flag: with `XDG_STATE_HOME`/`HOME` pointed at a fresh
/// tempdir, a run writes zero `*.jsonl` files anywhere under it, and stderr
/// never mentions "transcript" — proving the feature is fully inert by
/// default (no env fallback exists to accidentally trip it).
#[tokio::test(flavor = "current_thread")]
async fn no_transcript_flag_writes_no_jsonl_and_is_silent() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();
    let state_home = dir.path().join("state-home");
    std::fs::create_dir_all(&state_home).unwrap();

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
            "--task-id",
            "cli-test-no-transcript",
            "--attempt",
            "1",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .env("XDG_STATE_HOME", &state_home)
        .env("HOME", dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn talos");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(valid_spec_json().as_bytes())
        .unwrap();

    let output = child.wait_with_output().expect("wait for talos");
    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");

    assert!(
        !walk_has_jsonl(dir.path()),
        "no *.jsonl file may exist anywhere under the tempdir when --transcript is absent"
    );

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr_str.contains("transcript"),
        "stderr must not mention \"transcript\" when the flag is absent; got: {stderr_str:?}"
    );
}

// ============================================================================
// (d) --help is not a usage error: plain help on stdout, exit 0
// ============================================================================

/// `--help` surfaces as `Err` from `try_parse` but must NOT take the
/// JSON-error exit-1 path — help goes to stdout plainly with exit 0.
#[test]
fn help_flag_exits_0_with_plain_help() {
    let output = Command::new(TALOS_BIN)
        .args(["run", "--help"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(output.status.code(), Some(0), "--help must exit 0");
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout_str.contains("--workspace"),
        "help text must be on stdout; got: {stdout_str:?}"
    );
    assert!(
        !stdout_str.trim_start().starts_with('{'),
        "help must be plain text, not a JSON error object"
    );
}

/// `--version` surfaces as `Err` from `try_parse` but must NOT take the
/// JSON-error exit-1 path — version goes to stdout plainly with exit 0.
///
/// It also pins the **dispatch-fleet contract** (see `crates/talos/build.rs` and
/// `scripts/publish-talos.sh`): the output is `talos <TOKEN>` where `<TOKEN>` is
/// `<semver>-g<short-sha>`, a single URL/path-safe string. `talos-update.sh`
/// extracts it via `talos --version | awk '{print $2}'` and uses it verbatim as
/// a pi-04 path component, so a token free of spaces, slashes, and plus-signs is
/// load-bearing, not cosmetic.
#[test]
fn version_flag_exits_0_with_fleet_token() {
    let output = Command::new(TALOS_BIN)
        .args(["--version"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(output.status.code(), Some(0), "--version must exit 0");
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout_str.trim_start().starts_with('{'),
        "version must be plain text, not a JSON error object"
    );

    // `awk '{print $2}'` — the exact extraction talos-update.sh uses.
    let mut fields = stdout_str.split_whitespace();
    assert_eq!(
        fields.next(),
        Some("talos"),
        "line 1 field 1 must be `talos`"
    );
    let token = fields.next().expect("a version token in field 2");

    // The token is `git describe --tags` stamped at build time (build.rs): a
    // release tag like `0.5.0`, else `<tag>-<n>-g<short-sha>`, else a bare sha
    // (no-tag fallback) or `unknown` (non-git build). Assert only the durable
    // fleet contract — a non-empty single token — not a specific version, which
    // changes every release. Path-safety is asserted below.
    assert!(!token.is_empty(), "token must be non-empty; got: {token:?}");
    // URL/path-safe: no `+`, no `/`, no whitespace (split_whitespace already
    // guarantees the last), so it drops into a pi-04 path unescaped.
    assert!(
        token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-'),
        "token must be URL/path-safe (alnum, `.`, `-` only); got: {token:?}"
    );
}

// ============================================================================
// (e) --file spec input tests
// ============================================================================

/// Valid spec via `--file` with deterministic `BackendError` via refused-port Ollama.
/// Asserts:
/// - exit code 1 (`BackendError` → 1)
/// - stdout summary is valid JSON with `outcome == "BackendError"`
/// - spec is read from file, NOT stdin (stdin set to `Stdio::null()`)
#[test]
fn valid_spec_via_file_input_backend_error_via_refused_port() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    let spec_file = dir.path().join("spec.json");
    std::fs::create_dir_all(&offload_dir).unwrap();

    // Write valid spec to a temp file
    std::fs::write(&spec_file, valid_spec_json()).expect("write spec file");

    let task_id = "file-input-test";

    let output = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--file",
            spec_file.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
            "--task-id",
            task_id,
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(
        output.status.code(),
        Some(1),
        "BackendError via --file must exit 1"
    );

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let summary: serde_json::Value = serde_json::from_str(stdout_str.trim()).unwrap_or_else(|_| {
        panic!("stdout must be valid JSON summary; got: {stdout_str:?}");
    });
    assert_eq!(
        summary.get("outcome").and_then(serde_json::Value::as_str),
        Some("BackendError"),
        "summary outcome must be \"BackendError\"; got: {summary}"
    );
}

/// Malformed JSON in `--file` exits 1; stderr is a one-line JSON `{"error": ...}`;
/// no store record is written (store file must not exist).
#[test]
fn malformed_spec_file_exits_1_with_json_error_no_record() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    let spec_file = dir.path().join("malformed.json");
    std::fs::create_dir_all(&offload_dir).unwrap();

    // Write invalid JSON to a temp file
    std::fs::write(&spec_file, "this is not valid json at all").expect("write spec file");

    let output = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--file",
            spec_file.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(
        output.status.code(),
        Some(1),
        "malformed --file must exit 1"
    );

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let parsed: serde_json::Value = serde_json::from_str(stderr_str.trim()).unwrap_or_else(|_| {
        panic!("stderr must be valid JSON; got: {stderr_str:?}");
    });
    assert!(
        parsed.get("error").is_some(),
        "stderr JSON must have an `error` key; got: {parsed}"
    );

    // Spec parsing fails before the store is opened → the store file must
    // not exist.
    assert!(
        !store_path.exists(),
        "no store file must be written when spec parsing fails"
    );
}

/// Nonexistent `--file` path exits 1; stderr is a one-line JSON `{"error": ...}`.
#[test]
fn nonexistent_file_path_exits_1_with_json_error() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let store_path = dir.path().join("run.sqlite");
    let offload_dir = dir.path().join("offload");
    let nonexistent_file = dir.path().join("does_not_exist.json");
    std::fs::create_dir_all(&offload_dir).unwrap();

    let output = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--file",
            nonexistent_file.to_str().unwrap(),
            "--run-store",
            store_path.to_str().unwrap(),
            "--offload-dir",
            offload_dir.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(
        output.status.code(),
        Some(1),
        "nonexistent --file must exit 1"
    );

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let parsed: serde_json::Value = serde_json::from_str(stderr_str.trim()).unwrap_or_else(|_| {
        panic!("stderr must be valid JSON; got: {stderr_str:?}");
    });
    assert!(
        parsed.get("error").is_some(),
        "stderr JSON must have an `error` key; got: {parsed}"
    );
}

// ============================================================================
// `talos ralph` subcommand integration tests
//
// `ralph` is a thin CLI over `harness::ralph::run_ralph`: it is NOT run-record
// persisted this cut, so these tests assert the ralph exit-code map
// (0 StopConditionMet / 20 task-side / 1 infra) and the RalphSummary JSON,
// not store records.
// ============================================================================

/// `talos ralph --help` exits 0 and prints usage text to stdout (NOT a JSON
/// error object). Mirrors `help_flag_exits_0_with_plain_help` for `run`.
#[test]
fn ralph_help_flag_exits_0_with_usage() {
    let output = Command::new(TALOS_BIN)
        .args(["ralph", "--help"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(output.status.code(), Some(0), "ralph --help must exit 0");
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout_str.contains("--stop-when") && stdout_str.contains("--objective"),
        "ralph --help must mention ralph flags; got: {stdout_str:?}"
    );
    assert!(
        !stdout_str.trim_start().starts_with('{'),
        "help must be plain text, not a JSON error object"
    );
}

/// A missing required flag (here `--objective` omitted) is a clap usage
/// error; the CLI must exit 1 (not clap's default exit 2) with a JSON
/// `{"error": ...}` on stderr.
#[test]
fn ralph_missing_required_flag_exits_1_with_json_error() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();

    let output = Command::new(TALOS_BIN)
        .args([
            "ralph",
            "--workspace",
            workspace.to_str().unwrap(),
            "--stop-when",
            "true",
            "--offload-dir",
            offload_dir.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(
        output.status.code(),
        Some(1),
        "missing --objective must exit 1"
    );

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let parsed: serde_json::Value = serde_json::from_str(stderr_str.trim()).unwrap_or_else(|_| {
        panic!("stderr must be valid JSON; got: {stderr_str:?}");
    });
    assert!(
        parsed.get("error").is_some(),
        "stderr JSON must have an `error` key; got: {parsed}"
    );
}

/// A whitespace-only `--stop-when '   '` is rejected with a JSON error and
/// exit 1 BEFORE any filesystem/backend work. An empty oracle would exit 0
/// every call and declare the objective met on iteration 1 — a false-done
/// vector — so it must be rejected up front.
#[test]
fn ralph_whitespace_stop_when_exits_1_with_json_error() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();

    let output = Command::new(TALOS_BIN)
        .args([
            "ralph",
            "--workspace",
            workspace.to_str().unwrap(),
            "--objective",
            "do something",
            "--stop-when",
            "   ",
            "--offload-dir",
            offload_dir.to_str().unwrap(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    assert_eq!(
        output.status.code(),
        Some(1),
        "whitespace-only --stop-when must exit 1"
    );

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let parsed: serde_json::Value = serde_json::from_str(stderr_str.trim()).unwrap_or_else(|_| {
        panic!("stderr must be valid JSON; got: {stderr_str:?}");
    });
    assert!(
        parsed.get("error").is_some(),
        "stderr JSON must have an `error` key; got: {parsed}"
    );

    // Rejected BEFORE any filesystem/backend work: the offload dir we passed
    // exists (we created it), but the default state dir must not have been
    // touched. The stdout must be empty (no RalphSummary printed).
    assert!(
        output.stdout.is_empty(),
        "no RalphSummary must be printed when --stop-when is rejected; got: {:?}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// Deterministic no-API-key ralph run: `git init` a temp workspace, run with
/// `TALOS_BACKEND=ollama` + `OLLAMA_MODEL=x` + `OLLAMA_BASE_URL` pointing at a
/// refused-connection port, `--stop-when 'false'`, `--max-ralph-iterations 1`.
///
/// The single inner `engine::run` hits `BackendError` (recorded on the
/// iteration, the outer loop continues), the stop-command `false` is not met
/// (exit 1, non-zero), and the single outer pass exhausts `max_outer_iterations`
/// → `RalphTerminal::MaxIterationsExhausted` → exit 20 and a `RalphSummary`
/// JSON on stdout. Mirrors `backend_error_via_refused_port_writes_store_record`
/// for the refused-connection setup.
#[test]
fn ralph_refused_ollama_exhausts_max_iterations_exit_20() {
    use std::process::Command as StdCommand;

    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path();
    let offload_dir = dir.path().join("offload");
    std::fs::create_dir_all(&offload_dir).unwrap();

    // `run_ralph` does NOT run `git init` — the workspace must already be a
    // git work tree. Initialize it (and a commit-free work tree is fine: the
    // first iteration's `git status --porcelain` is empty, so no commit is
    // attempted, and the progress signal is empty so stuck advances — but
    // with max_outer_iterations=1 the loop exhausts before stuck_k fires).
    StdCommand::new("git")
        .arg("init")
        .current_dir(workspace)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("git init must succeed");

    let output = Command::new(TALOS_BIN)
        .args([
            "ralph",
            "--workspace",
            workspace.to_str().unwrap(),
            "--objective",
            "build the thing",
            "--stop-when",
            "false",
            "--max-ralph-iterations",
            "1",
            "--offload-dir",
            offload_dir.to_str().unwrap(),
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn talos");

    // MaxIterationsExhausted → exit code 20 (NOT 1 — the inner BackendError is
    // recorded on the iteration, the outer loop continues, and the single
    // outer pass exhausts the outer cap).
    assert_eq!(
        output.status.code(),
        Some(20),
        "ralph MaxIterationsExhausted must exit 20; stderr was: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // stdout is a machine-readable RalphSummary JSON.
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let summary: serde_json::Value = serde_json::from_str(stdout_str.trim()).unwrap_or_else(|_| {
        panic!("stdout must be valid RalphSummary JSON; got: {stdout_str:?}");
    });

    // Exact field set — no run_id / record_path (ralph is not persisted).
    let obj = summary.as_object().expect("summary must be an object");
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "objective",
            "outer_iterations",
            "terminal",
            "total_inner_iterations"
        ],
        "RalphSummary must have exactly the four expected fields; got: {summary}"
    );
    assert_eq!(
        summary.get("terminal").and_then(serde_json::Value::as_str),
        Some("MaxIterationsExhausted"),
        "summary terminal must be \"MaxIterationsExhausted\"; got: {summary}"
    );
    assert_eq!(
        summary.get("objective").and_then(serde_json::Value::as_str),
        Some("build the thing")
    );
    assert_eq!(
        summary
            .get("outer_iterations")
            .and_then(serde_json::Value::as_u64),
        Some(1),
        "exactly one outer pass must have run; got: {summary}"
    );
}
