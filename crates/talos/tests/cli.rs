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
            "--state-retention-days",
            "0",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        // Isolate this run's implicit prune pass from the real
        // `$HOME/.local/state/talos` — without this, an unmodified nextest
        // run would point `remove_dir_all` at the developer's or the
        // dispatch user's real state root. `--state-retention-days 0` is a
        // second, independent belt: it disables pruning outright.
        .env("XDG_STATE_HOME", dir.path().join("state-home"))
        .env("HOME", dir.path())
        .env_remove("TALOS_STATE_RETENTION_DAYS")
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
            "--state-retention-days",
            "0",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        // Isolate the implicit prune pass from the real state root — see the
        // comment in `backend_error_via_refused_port_writes_store_record`.
        .env("XDG_STATE_HOME", dir.path().join("state-home"))
        .env("HOME", dir.path())
        .env_remove("TALOS_STATE_RETENTION_DAYS")
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
            "--state-retention-days",
            "0",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .env("XDG_STATE_HOME", &state_home)
        .env("HOME", dir.path())
        .env_remove("TALOS_STATE_RETENTION_DAYS")
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
// (c3) --transcript with no value: bare flag defaults into the state dir
// ============================================================================

/// Bare `--transcript` (no path argument) defaults to
/// `<state_home>/talos/<task-id>/transcript.jsonl`, sitting next to the run's
/// `run.sqlite`. Also pins the dispatch worker's stdout contract
/// (`agent_gtd_dispatch/talos.py::map_talos_result` reads the LAST stdout
/// line): a bare-flag run must still emit exactly one stdout line, and that
/// line must parse as a JSON object.
#[tokio::test(flavor = "current_thread")]
async fn bare_transcript_flag_defaults_into_state_dir() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let state_home = dir.path().join("state-home");
    std::fs::create_dir_all(&state_home).unwrap();

    let task_id = "cli-test-bare-transcript";

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--task-id",
            task_id,
            "--attempt",
            "1",
            "--transcript",
            "--state-retention-days",
            "0",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .env("XDG_STATE_HOME", &state_home)
        .env("HOME", dir.path())
        .env_remove("TALOS_STATE_RETENTION_DAYS")
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
    let stdout_lines: Vec<&str> = stdout_str.trim_end().lines().collect();
    assert_eq!(
        stdout_lines.len(),
        1,
        "bare --transcript must not add any stdout line beyond the single \
         RunSummary line the dispatch worker reads as the run's classification; \
         got: {stdout_str:?}"
    );
    let summary: serde_json::Value = serde_json::from_str(stdout_lines[0])
        .unwrap_or_else(|_| panic!("stdout line must be valid JSON; got: {stdout_str:?}"));
    assert!(summary.is_object(), "stdout summary must be a JSON object");

    let state_dir = state_home.join("talos").join(task_id);
    let transcript_path = state_dir.join("transcript.jsonl");
    let run_store_path = state_dir.join("run.sqlite");
    assert!(
        transcript_path.exists(),
        "bare --transcript must write transcript.jsonl into the run's state dir; \
         expected {}",
        transcript_path.display()
    );
    assert!(
        run_store_path.exists(),
        "run.sqlite must exist next to transcript.jsonl in the same state dir"
    );

    let contents = std::fs::read_to_string(&transcript_path).expect("transcript file must exist");
    let first_line = contents
        .lines()
        .next()
        .expect("transcript must have at least one line");
    let first: serde_json::Value =
        serde_json::from_str(first_line).expect("first transcript line must be valid JSON");
    assert_eq!(first["event"], "run_start");
}

/// `--transcript <path>` with an explicit value must still write exactly that
/// file, and must NOT ALSO write a `transcript.jsonl` into the run's default
/// state dir — pins that the new bare-flag default cannot silently redirect
/// an explicit caller.
#[tokio::test(flavor = "current_thread")]
async fn explicit_transcript_path_is_not_redirected_to_state_dir() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let state_home = dir.path().join("state-home");
    std::fs::create_dir_all(&state_home).unwrap();
    let custom_path = dir.path().join("custom.jsonl");

    let task_id = "cli-test-explicit-transcript";

    let mut child = Command::new(TALOS_BIN)
        .args([
            "run",
            "--workspace",
            workspace.to_str().unwrap(),
            "--task-id",
            task_id,
            "--attempt",
            "1",
            "--transcript",
            custom_path.to_str().unwrap(),
            "--state-retention-days",
            "0",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        .env_remove("OLLAMA_THINK")
        .env_remove("OLLAMA_NUM_CTX")
        .env_remove("TALOS_BEDROCK")
        .env("XDG_STATE_HOME", &state_home)
        .env("HOME", dir.path())
        .env_remove("TALOS_STATE_RETENTION_DAYS")
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
        custom_path.exists(),
        "an explicit --transcript path must be written exactly as given"
    );

    let default_transcript = state_home
        .join("talos")
        .join(task_id)
        .join("transcript.jsonl");
    assert!(
        !default_transcript.exists(),
        "an explicit --transcript path must not ALSO write transcript.jsonl \
         into the run's default state dir"
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
            "--state-retention-days",
            "0",
        ])
        .env("TALOS_BACKEND", "ollama")
        .env("OLLAMA_MODEL", "x")
        // Port 1 on loopback is reserved; connections are always refused.
        .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
        // Isolate the implicit prune pass from the real state root — see the
        // comment in `backend_error_via_refused_port_writes_store_record`.
        .env("XDG_STATE_HOME", dir.path().join("state-home"))
        .env("HOME", dir.path())
        .env_remove("TALOS_STATE_RETENTION_DAYS")
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
        // `ralph` self-touches its own `talos-ralph` state dir (AC-19); point
        // it at an isolated state root rather than the real
        // `$HOME/.local/state/talos`.
        .env("XDG_STATE_HOME", dir.path().join("state-home"))
        .env("HOME", dir.path())
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

// ============================================================================
// `--state-retention-days`: talos prunes its own XDG state dir on run start
// ============================================================================
//
// Every test here uses `build_prune_fixture`, which creates an ISOLATED
// `XDG_STATE_HOME`/`HOME` under a fresh tempdir. This is load-bearing: an
// unmodified test that spawned `talos run` without pointing these at a
// tempdir would point the implicit prune pass at the real
// `$HOME/.local/state/talos` — exactly how the leaked
// `mined-eval/synth-task-*` debris accumulated on the dispatch hosts.

/// Seconds per day, mirroring `crates/talos/src/main.rs::SECS_PER_DAY` — kept
/// local since integration tests cannot see the binary crate's private
/// constants.
const TEST_SECS_PER_DAY: u64 = 86_400;

/// Age `path`'s own mtime by `days` days: `File::open` (works on a directory
/// opened read-only on this platform) + `set_times`. No new dependency.
fn age_dir(path: &std::path::Path, days: u64) {
    let mtime =
        std::time::SystemTime::now() - std::time::Duration::from_secs(days * TEST_SECS_PER_DAY);
    std::fs::File::open(path)
        .and_then(|f| f.set_times(std::fs::FileTimes::new().set_modified(mtime)))
        .expect("set_times must succeed on a directory opened read-only");
}

/// Fixture for `--state-retention-days` integration tests: an isolated
/// `XDG_STATE_HOME`/`HOME`, a `talos_root` (`<state_home>/talos`) holding a
/// 40-day-old `stale-task` dir and a fresh `fresh-task` dir. Tests pass
/// `--task-id live-task` and deliberately omit `--run-store`/`--offload-dir`
/// so both default under `<state_home>/talos/live-task/` and the prune root
/// IS `talos_root`.
struct PruneFixture {
    _dir: tempfile::TempDir,
    home: std::path::PathBuf,
    workspace: std::path::PathBuf,
    state_home: std::path::PathBuf,
    talos_root: std::path::PathBuf,
    aged: std::path::PathBuf,
    fresh: std::path::PathBuf,
}

fn build_prune_fixture() -> PruneFixture {
    let dir = tempfile::tempdir().expect("create temp dir");
    let home = dir.path().to_path_buf();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&workspace).unwrap();
    let state_home = dir.path().join("state-home");
    let talos_root = state_home.join("talos");
    let aged = talos_root.join("stale-task");
    std::fs::create_dir_all(&aged).unwrap();
    age_dir(&aged, 40);
    let fresh = talos_root.join("fresh-task");
    std::fs::create_dir_all(&fresh).unwrap();

    PruneFixture {
        _dir: dir,
        home,
        workspace,
        state_home,
        talos_root,
        aged,
        fresh,
    }
}

/// Spawn `talos run` against `fx` with the given extra CLI args and env
/// overrides, feeding `valid_spec_json()` on stdin, and return the process
/// output.
fn spawn_prune_run(
    fx: &PruneFixture,
    extra_args: &[&str],
    env_overrides: &[(&str, Option<&str>)],
) -> std::process::Output {
    let mut cmd = Command::new(TALOS_BIN);
    cmd.args([
        "run",
        "--workspace",
        fx.workspace.to_str().unwrap(),
        "--task-id",
        "live-task",
        "--attempt",
        "1",
    ])
    .args(extra_args)
    .env("TALOS_BACKEND", "ollama")
    .env("OLLAMA_MODEL", "x")
    // Port 1 on loopback is reserved; connections are always refused.
    .env("OLLAMA_BASE_URL", "http://127.0.0.1:1")
    .env_remove("OLLAMA_THINK")
    .env_remove("OLLAMA_NUM_CTX")
    .env_remove("TALOS_BEDROCK")
    .env("XDG_STATE_HOME", &fx.state_home)
    .env("HOME", &fx.home)
    .env_remove("TALOS_STATE_RETENTION_DAYS");
    for (key, value) in env_overrides {
        match value {
            Some(v) => {
                cmd.env(key, v);
            }
            None => {
                cmd.env_remove(key);
            }
        }
    }
    let mut child = cmd
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
    child.wait_with_output().expect("wait for talos")
}

fn read_prune_report(fx: &PruneFixture) -> serde_json::Value {
    let report_path = fx.talos_root.join("prune-last.json");
    let contents =
        std::fs::read_to_string(&report_path).expect("prune-last.json must exist after a run");
    serde_json::from_str(&contents).expect("prune-last.json must be valid JSON")
}

/// `--state-retention-days 1`: the 40-day `stale-task` dir is older than the
/// 1-day window and gets removed; `fresh-task` and this run's own
/// `live-task` dir (both created "now") survive; stdout stays a single
/// `BackendError` summary line; stderr never mentions pruning;
/// `prune-last.json` records the removal.
#[test]
fn state_retention_flag_one_day_prunes_aged_leaves_fresh_and_live() {
    let fx = build_prune_fixture();
    assert!(
        fx.aged.exists(),
        "fixture invariant: aged dir must exist before spawn"
    );
    let live_task_dir = fx.talos_root.join("live-task");

    let output = spawn_prune_run(&fx, &["--state-retention-days", "1"], &[]);

    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");
    assert!(
        !fx.aged.exists(),
        "40-day dir must be pruned under 1-day retention"
    );
    assert!(fx.fresh.exists(), "fresh dir must survive");
    assert!(
        live_task_dir.exists(),
        "this run's own live-task dir must survive"
    );

    // Worker-contract safety: exactly one stdout line, parsing as JSON with
    // outcome == "BackendError" — a prune line must never become plausible
    // last-stdout/last-stderr noise on the exit-1 infra-failure path the
    // dispatch worker classifies on.
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout_str.trim().lines().count(),
        1,
        "expected exactly one stdout line; got: {stdout_str:?}"
    );
    let summary: serde_json::Value =
        serde_json::from_str(stdout_str.trim()).expect("stdout line must be valid JSON");
    assert_eq!(
        summary.get("outcome").and_then(serde_json::Value::as_str),
        Some("BackendError")
    );

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr_str.contains("prune"),
        "stderr must never mention pruning; got: {stderr_str:?}"
    );

    let report = read_prune_report(&fx);
    assert_eq!(report["removed"], 1);
    assert_eq!(report["retention_days"], 1);
    assert_eq!(report["source"], "flag");
    assert_eq!(report["disabled"], false);
    assert!(
        report["removed_names"]
            .as_array()
            .expect("removed_names must be an array")
            .iter()
            .any(|v| v == "stale-task"),
        "removed_names must contain \"stale-task\"; got: {report}"
    );
}

/// `--state-retention-days 0` disables pruning entirely: the aged dir
/// survives, and `prune-last.json` still gets written with `disabled: true`
/// and zero counts, so an operator can confirm retention is off on a host.
#[test]
fn state_retention_flag_zero_disables_pruning() {
    let fx = build_prune_fixture();
    assert!(fx.aged.exists());

    let output = spawn_prune_run(&fx, &["--state-retention-days", "0"], &[]);

    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");
    assert!(
        fx.aged.exists(),
        "disabled retention (flag=0) must not remove anything"
    );

    let report = read_prune_report(&fx);
    assert_eq!(report["disabled"], true);
    assert_eq!(report["removed"], 0);
    assert_eq!(report["examined"], 0);
    assert_eq!(report["retention_days"], 0);
    assert_eq!(report["source"], "flag");
}

/// No `--state-retention-days` flag, `TALOS_STATE_RETENTION_DAYS=0` in the
/// child env: this proves the env fallback is wired through `env_accessor`
/// in `run_cmd` itself, not only in the pure resolver's unit tests.
#[test]
fn state_retention_env_zero_disables_pruning_wired_through_run_cmd() {
    let fx = build_prune_fixture();
    assert!(fx.aged.exists());

    let output = spawn_prune_run(&fx, &[], &[("TALOS_STATE_RETENTION_DAYS", Some("0"))]);

    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");
    assert!(
        fx.aged.exists(),
        "disabled retention (env=0) must not remove anything"
    );

    let report = read_prune_report(&fx);
    assert_eq!(report["source"], "env");
    assert_eq!(report["retention_days"], 0);
}

/// No flag, no env: the compiled 30-day default applies, so the 40-day
/// `stale-task` dir is still pruned.
#[test]
fn state_retention_default_thirty_days_prunes_aged() {
    let fx = build_prune_fixture();
    assert!(fx.aged.exists());

    let output = spawn_prune_run(&fx, &[], &[]);

    assert_eq!(output.status.code(), Some(1), "BackendError must exit 1");
    assert!(
        !fx.aged.exists(),
        "40-day dir must be pruned under the 30-day default"
    );
    assert!(fx.fresh.exists());

    let report = read_prune_report(&fx);
    assert_eq!(report["source"], "default");
    assert_eq!(report["retention_days"], 30);
}

/// Silence: a run that DOES prune (removes the aged dir) produces stderr
/// byte-identical to the same run with `--state-retention-days 0` (which
/// prunes nothing) — pruning has NO stdout/stderr writer on any path.
#[test]
fn prune_pass_produces_byte_identical_stderr_whether_or_not_it_prunes() {
    let fx_pruning = build_prune_fixture();
    let fx_disabled = build_prune_fixture();

    let pruning_output = spawn_prune_run(&fx_pruning, &["--state-retention-days", "1"], &[]);
    let disabled_output = spawn_prune_run(&fx_disabled, &["--state-retention-days", "0"], &[]);

    assert!(
        !fx_pruning.aged.exists(),
        "the pruning run must actually have removed the aged dir"
    );
    assert!(
        fx_disabled.aged.exists(),
        "the disabled run must not have removed anything"
    );

    assert_eq!(
        pruning_output.stderr, disabled_output.stderr,
        "stderr must be byte-identical whether or not a prune pass actually removed anything"
    );
    for stderr in [&pruning_output.stderr, &disabled_output.stderr] {
        let text = String::from_utf8_lossy(stderr);
        assert!(
            !text.contains("prune"),
            "stderr must never mention pruning; got: {text:?}"
        );
    }
}
