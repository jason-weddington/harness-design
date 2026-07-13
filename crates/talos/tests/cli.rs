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
