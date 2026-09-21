//! Integration tests for the `somnus` CLI, spawning the compiled binary via
//! `env!("CARGO_BIN_EXE_somnus")` (the `crates/talos` precedent, so
//! `cargo-llvm-cov` collects child-process coverage). No live KB, no model,
//! no network — every HTTP shape is a wiremock on localhost, and every test
//! stops the binary before rung 1, so `ANTHROPIC_API_KEY` is a dummy and no
//! model lane is ever touched.

use std::process::{Command, Stdio};

/// Compiled `somnus` binary path (injected by cargo at integration-test time).
const SOMNUS_BIN: &str = env!("CARGO_BIN_EXE_somnus");

/// The eight env vars somnus reads (the six operator vars plus the two
/// state-dir defaults), removed from EVERY spawned child so a test never
/// inherits the runner's own environment.
const SOMNUS_ENV_VARS: [&str; 8] = [
    "SOMNUS_KB_BASE_URL",
    "SOMNUS_KB_API_KEY",
    "ANTHROPIC_API_KEY",
    "SOMNUS_TOKEN_BUDGET",
    "SOMNUS_STATE_DIR",
    "SOMNUS_DISABLED",
    "XDG_STATE_HOME",
    "HOME",
];

/// A configured child command: stdin closed, output captured, the somnus env
/// scrubbed.
fn bare_cli() -> Command {
    let mut command = Command::new(SOMNUS_BIN);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for var in SOMNUS_ENV_VARS {
        command.env_remove(var);
    }
    command
}

/// Run the binary with `args` and `env` (over the scrubbed baseline).
fn run_cli_with(args: &[&str], env: &[(String, String)]) -> (Option<i32>, String, String) {
    let mut command = bare_cli();
    command.args(args);
    for (var, value) in env {
        command.env(var, value);
    }
    let output = command.output().expect("spawn somnus");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Run the binary with `args` and no somnus env at all.
fn run_cli(args: &[&str]) -> (Option<i32>, String, String) {
    run_cli_with(args, &[])
}

/// A string-pair env slice helper for the static-var tables.
fn pairs(env: &[(&'static str, &'static str)]) -> Vec<(String, String)> {
    env.iter()
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect()
}

/// A wiremock server with a mounted `GET /api/kb/map-worklist` mock.
async fn worklist_server(body: &str, status: u16) -> wiremock::MockServer {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-worklist"))
        .respond_with(wiremock::ResponseTemplate::new(status).set_body_string(body.to_string()))
        .mount(&server)
        .await;
    server
}

/// A wiremock server answering every `GET /api/kb/map-loop-input` with
/// `status` and `body`.
async fn loop_input_server(status: u16, body: &str) -> wiremock::MockServer {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-loop-input"))
        .respond_with(wiremock::ResponseTemplate::new(status).set_body_string(body.to_string()))
        .mount(&server)
        .await;
    server
}

/// The three required vars, over a wiremock base.
fn required_env(base: &str) -> Vec<(String, String)> {
    vec![
        ("SOMNUS_KB_BASE_URL".to_string(), base.to_string()),
        ("SOMNUS_KB_API_KEY".to_string(), "kb-dummy".to_string()),
        ("ANTHROPIC_API_KEY".to_string(), "sk-dummy".to_string()),
    ]
}

// ============================================================================
// (a) `somnus --version`: the installer's fleet contract
// ============================================================================

/// `somnus --version` MUST emit the SAME output shape as `talos --version`:
/// the installer parses whitespace-separated field 2 for the version token
/// and treats absent or unparsable as the literal `none`.
#[test]
fn version_flag_exits_0_with_the_fleet_token_shape() {
    let (code, stdout, stderr) = run_cli(&["--version"]);
    assert_eq!(code, Some(0), "--version must exit 0");
    assert!(
        !stderr.contains("error"),
        "version must be plain text, not a CLI error object: {stderr}"
    );
    let trimmed = stdout.trim_end_matches('\n');
    // Exactly one line, two whitespace-separated fields.
    assert_eq!(
        trimmed.split('\n').count(),
        1,
        "version output must be exactly one line, got: {stdout:?}"
    );
    assert!(trimmed.contains(' '), "version must be `somnus <token>`");
    let mut fields = trimmed.split_whitespace();
    let program = fields.next().expect("field 1");
    let token = fields.next().expect("a version token in field 2");
    assert_eq!(program, "somnus", "field 1 must be the program name");
    assert!(
        !token.is_empty()
            && token
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')),
        "field 2 must be a URL/path-safe token ([A-Za-z0-9._-]+), got: {token}"
    );
}

// ============================================================================
// (b) argv parse: the pinned usage-error line, exit 1
// ============================================================================

/// Any clap usage error — a bare `somnus`, a bad flag, a missing `--project`
/// — exits 1 with the byte-pinned stderr line; clap's default exit 2 and
/// its `error:` rendering are overridden. `--help` follows clap's default
/// render and exits 0.
#[test]
fn every_clap_usage_error_exits_1_with_the_pinned_line() {
    let pinned = "somnus: exactly one subcommand required: nightly | run --project <ref> | backfill --project <ref>";
    for args in [
        vec![],
        vec!["--not-a-flag"],
        vec!["run"],
        vec!["run", "--project"],
        vec!["nightly", "extra"],
        vec!["bogus-subcommand"],
    ] {
        let (code, stdout, stderr) = run_cli(&args);
        assert_eq!(code, Some(1), "args {args:?}: stderr {stderr}");
        assert!(
            stderr.contains(pinned),
            "args {args:?}: pinned line missing from: {stderr}"
        );
        assert!(stdout.is_empty(), "args {args:?}: stdout was {stdout:?}");
    }
}

#[test]
fn help_follows_claps_default_render_and_exits_0() {
    let (code, stdout, _stderr) = run_cli(&["--help"]);
    assert_eq!(code, Some(0));
    assert!(stdout.contains("Usage:"));
}

// ============================================================================
// (c) the empty `--project` check, byte-for-byte preserved
// ============================================================================

/// An empty or whitespace-only `--project` value is rejected with exit 1 and
/// the pinned stderr substring — before required-env validation (spawned
/// with NO SOMNUS_* env at all).
#[test]
fn empty_project_ref_is_rejected_before_any_other_work() {
    for project_ref in ["", "   "] {
        let (code, _stdout, stderr) = run_cli(&["run", "--project", project_ref]);
        assert_eq!(
            code,
            Some(1),
            "empty --project must exit 1, stderr: {stderr}"
        );
        assert!(
            stderr.contains("somnus: --project must be a non-empty project ref"),
            "pinned rejection message missing from: {stderr}"
        );
    }
}

// ============================================================================
// (d) the kill switch
// ============================================================================

/// `SOMNUS_DISABLED=1` with NO other vars set exits 0 immediately with the
/// pinned line and nothing else — argv validation precedes the switch, so a
/// bare invocation still exits 1 even with the switch armed.
#[test]
fn the_kill_switch_exits_0_loudly_with_no_other_vars() {
    let (code, stdout, stderr) = run_cli_with(&["nightly"], &pairs(&[("SOMNUS_DISABLED", "1")]));
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(
        stderr.trim_end_matches('\n'),
        "somnus: disabled via SOMNUS_DISABLED=1; exiting before any other env read, fetch, or write"
    );
    assert!(stdout.is_empty(), "the switch must print nothing to stdout");

    // A bare invocation still exits 1: argv validation precedes the switch.
    let (code, _stdout, stderr) = run_cli_with(&[], &pairs(&[("SOMNUS_DISABLED", "1")]));
    assert_eq!(code, Some(1));
    assert!(
        stderr.contains("somnus: exactly one subcommand required"),
        "argv validation precedes the kill switch: {stderr}"
    );
}

/// `SOMNUS_DISABLED=1` with all three required vars set (base = a live
/// wiremock URI, keys = dummies) still exits 0 WITHOUT a single request.
#[tokio::test]
async fn the_kill_switch_makes_zero_requests() {
    let server = loop_input_server(200, "{}").await;
    let mut env = required_env(&server.uri());
    env.push(("SOMNUS_DISABLED".to_string(), "1".to_string()));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert_eq!(
        stderr.trim_end_matches('\n'),
        "somnus: disabled via SOMNUS_DISABLED=1; exiting before any other env read, fetch, or write"
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("captured")
            .is_empty(),
        "the switch exits before any fetch"
    );
}

/// Any value other than the exact string `1` is IGNORED: with valid env, the
/// binary proceeds past the switch to the empty-worklist exit-0 path.
#[tokio::test]
async fn a_non_one_somnus_disabled_value_proceeds_past_the_switch() {
    let server = worklist_server(r#"{"projects":[]}"#, 200).await;
    let mut env = required_env(&server.uri());
    env.push(("SOMNUS_DISABLED".to_string(), "true".to_string()));
    let state = tempfile::tempdir().expect("tempdir");
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert!(
        stderr.contains("somnus: nothing eligible tonight"),
        "the binary proceeded past the switch: {stderr}"
    );
}

// ============================================================================
// (e) required-env validation
// ============================================================================

/// A missing, empty, or whitespace-only required var exits 1 with the
/// byte-pinned `somnus: {VAR} is not set` shape, asserted per var.
#[test]
fn every_required_env_fault_exits_1_with_the_pinned_shape() {
    for var in [
        "SOMNUS_KB_BASE_URL",
        "SOMNUS_KB_API_KEY",
        "ANTHROPIC_API_KEY",
    ] {
        for raw in [None, Some(""), Some("   ")] {
            let mut env = pairs(&[
                ("SOMNUS_KB_BASE_URL", "http://kb.invalid"),
                ("SOMNUS_KB_API_KEY", "kb-dummy"),
                ("ANTHROPIC_API_KEY", "sk-dummy"),
            ]);
            if let Some(raw) = raw {
                if !raw.is_empty() && raw.trim().is_empty() {
                    // whitespace-only: replace rather than remove
                    env.retain(|(name, _)| name != var);
                } else {
                    env.retain(|(name, _)| name != var);
                }
                env.push((var.to_string(), raw.to_string()));
            } else {
                env.retain(|(name, _)| name != var);
            }
            let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
            let expected = format!("somnus: {var} is not set");
            assert_eq!(code, Some(1), "{var} with {raw:?}: {stderr}");
            assert!(
                stderr.contains(&expected),
                "{var} with {raw:?}: expected `{expected}` in {stderr}"
            );
        }
    }
}

/// A base URL ending in `/` is a configuration fault: exit 1 with the pinned
/// line, never a silently-trimmed value.
#[test]
fn a_trailing_slash_base_url_exits_1() {
    let env = pairs(&[
        ("SOMNUS_KB_BASE_URL", "http://kb.invalid/"),
        ("SOMNUS_KB_API_KEY", "kb-dummy"),
        ("ANTHROPIC_API_KEY", "sk-dummy"),
    ]);
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(1));
    assert!(
        stderr.contains("somnus: SOMNUS_KB_BASE_URL must not end in '/'"),
        "{stderr}"
    );
}

/// Any non-integer, negative, or empty `SOMNUS_TOKEN_BUDGET` exits 1 with
/// the pinned line.
#[test]
fn a_bad_token_budget_exits_1() {
    for raw in ["", "abc", "-1", "1e6"] {
        let mut env = required_env("http://kb.invalid");
        env.push(("SOMNUS_TOKEN_BUDGET".to_string(), raw.to_string()));
        let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
        assert_eq!(code, Some(1), "raw {raw:?}: {stderr}");
        assert!(
            stderr.contains("somnus: SOMNUS_TOKEN_BUDGET is not a non-negative integer"),
            "raw {raw:?}: {stderr}"
        );
    }
}

// ============================================================================
// (f) the empty night
// ============================================================================

/// A wiremock returning `{"projects":[]}` → `somnus nightly` exits 0 with
/// the pinned line, writes the invocation record with `nothing_eligible`,
/// and makes ZERO requests to `/api/kb/map-loop-input` and zero model calls.
#[tokio::test]
async fn an_empty_night_exits_0_and_writes_the_record() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-worklist"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(r#"{"projects":[]}"#))
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-loop-input"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("{}"))
        .mount(&server)
        .await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(
        code,
        Some(0),
        "nonzero must never mean nothing was eligible: {stderr}"
    );
    assert!(
        stderr.contains("somnus: nothing eligible tonight"),
        "{stderr}"
    );
    // ZERO requests to /api/kb/map-loop-input (the only request was the
    // worklist), and the Anthropic key was a dummy.
    let requests = server.received_requests().await.expect("captured");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].url.path(), "/api/kb/map-worklist");
    // The invocation record, written whatever the exit code.
    let record = std::fs::read_to_string(state.path().join("nightly-invocation.json"))
        .expect("the record is on disk");
    let value: serde_json::Value = serde_json::from_str(&record).expect("parses");
    assert_eq!(value["subcommand"], "nightly");
    assert_eq!(value["stop_reason"], "nothing_eligible");
    assert_eq!(value["exit_code"], 0);
    assert_eq!(value["armed_budget"], 550_000);
    assert_eq!(value["worklist_projects"], serde_json::json!([]));
    assert_eq!(value["units"], serde_json::json!([]));
}

// ============================================================================
// (g) the real entry point reaches the pipeline's ordinary outcomes
// ============================================================================

/// A wiremock `Respond` serving a scripted sequence of (status, body)
/// pairs, front to back, repeating the last.
#[derive(Debug)]
struct Sequenced {
    responses: std::sync::Mutex<Vec<(u16, String)>>,
}

impl Sequenced {
    fn new(responses: Vec<(u16, &str)>) -> Self {
        Self {
            responses: std::sync::Mutex::new(
                responses
                    .into_iter()
                    .map(|(status, body)| (status, body.to_string()))
                    .collect(),
            ),
        }
    }
}

impl wiremock::Respond for Sequenced {
    fn respond(&self, _request: &wiremock::Request) -> wiremock::ResponseTemplate {
        let mut responses = self.responses.lock().expect("lock");
        let (status, body) = if responses.len() > 1 {
            responses.remove(0)
        } else {
            responses[0].clone()
        };
        wiremock::ResponseTemplate::new(status).set_body_string(body)
    }
}

/// The loop-input fixture, pinned byte-for-byte in
/// `crates/somnus/fixtures/loop_input.json`.
const LOOP_INPUT_FIXTURE: &str = somnus::loop_input::FIXTURE;

/// The nightly worklist answer: one project, server-ranked first.
const WORKLIST_ONE: &str = r#"{"projects":[{"project_ref":"demo-project","mappable":true,"unpointed":4,"map_count":2,"latest_map_written_at":"2026-09-19T03:14:15Z"}]}"#;

/// Serve the worklist with one project, then the loop-input sequence:
/// FIRST GET 200 with the pinned fixture (the baseline observation
/// succeeds), SECOND GET 404 → the unit ends `UnknownProject`, exit 0.
#[tokio::test]
async fn a_nightly_unit_reaches_unknown_project_and_leaves_the_report() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-worklist"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(WORKLIST_ONE))
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-loop-input"))
        .respond_with(Sequenced::new(vec![
            (200, LOOP_INPUT_FIXTURE),
            (404, "not found"),
        ]))
        .mount(&server)
        .await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("somnus: nightly unit for demo-project ended UnknownProject"),
        "{stderr}"
    );
    // The run report, on disk under the state dir, with the PascalCase
    // outcome literal.
    let report = std::fs::read_to_string(state.path().join("demo-project/run-report.json"))
        .expect("the report is on disk");
    let value: serde_json::Value = serde_json::from_str(&report).expect("parses");
    assert_eq!(value["outcome"], "UnknownProject", "{report}");
    // The invocation record carried the unit.
    let record = std::fs::read_to_string(state.path().join("nightly-invocation.json"))
        .expect("the record is on disk");
    let value: serde_json::Value = serde_json::from_str(&record).expect("parses");
    assert_eq!(value["stop_reason"], "all_done");
    assert_eq!(value["exit_code"], 0);
    assert_eq!(value["selected_refs"], serde_json::json!(["demo-project"]));
    assert_eq!(value["units"][0]["outcome"], "UnknownProject");
}

/// Same shape over `somnus run` with the SECOND GET 409: `NotEligible`,
/// exit 0 — and NO `/api/kb/map-worklist` request (a single-project path
/// never touches the worklist).
#[tokio::test]
async fn run_reaches_not_eligible_and_never_touches_the_worklist() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-worklist"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(WORKLIST_ONE))
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-loop-input"))
        .respond_with(Sequenced::new(vec![
            (200, LOOP_INPUT_FIXTURE),
            (409, "not eligible"),
        ]))
        .mount(&server)
        .await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["run", "--project", "demo-project"], &env);
    assert_eq!(code, Some(0), "{stderr}");
    let report = std::fs::read_to_string(state.path().join("demo-project/run-report.json"))
        .expect("the report is on disk");
    let value: serde_json::Value = serde_json::from_str(&report).expect("parses");
    assert_eq!(value["outcome"], "NotEligible", "{report}");
    // The single-project path never enumerated the worklist.
    let requests = server.received_requests().await.expect("captured");
    assert!(
        requests
            .iter()
            .all(|request| request.url.path() != "/api/kb/map-worklist"),
        "no map-worklist request may exist"
    );
}

/// A wiremock answering EVERY loop-input GET with 404 → `somnus run`
/// exits 2 and the state dir carries the `Aborted` run-report whose reason
/// contains `refusing to start` (the run-start guard is preserved, and the
/// fetch runs AFTER the baseline).
#[tokio::test]
async fn an_always_404_run_exits_2_with_the_refusing_to_start_report() {
    let server = loop_input_server(404, "not found").await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["run", "--project", "demo-project"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    let report = std::fs::read_to_string(state.path().join("demo-project/run-report.json"))
        .expect("the report is on disk");
    let value: serde_json::Value = serde_json::from_str(&report).expect("parses");
    let reason = value["outcome"]["Aborted"]["reason"]
        .as_str()
        .expect("a reason");
    assert!(
        reason.contains("refusing to start"),
        "reason was {reason}: {report}"
    );
}

// ============================================================================
// (h) the worklist client faults
// ============================================================================

/// A non-401 worklist failure is a RUN fault: exit 2 with the pinned line
/// and a `fault:2` invocation record.
#[tokio::test]
async fn a_500_worklist_exits_2_and_writes_the_record() {
    let server = worklist_server("boom", 500).await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("somnus: map-worklist fetch failed: unexpected HTTP status 500"),
        "{stderr}"
    );
    let record = std::fs::read_to_string(state.path().join("nightly-invocation.json"))
        .expect("the record is on disk");
    let value: serde_json::Value = serde_json::from_str(&record).expect("parses");
    assert_eq!(value["stop_reason"], "fault:2");
    assert_eq!(value["exit_code"], 2);
}

/// `somnus backfill` arms the doubled default and drives the same pipeline:
/// an always-404 loop-input means exit 2 with the refusing-to-start report,
/// and no `/api/kb/map-worklist` request.
#[tokio::test]
async fn backfill_runs_the_pipeline_and_never_touches_the_worklist() {
    let server = loop_input_server(404, "not found").await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-worklist"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(WORKLIST_ONE))
        .mount(&server)
        .await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["backfill", "--project", "demo-project"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    let report = std::fs::read_to_string(state.path().join("demo-project/run-report.json"))
        .expect("the report is on disk");
    let value: serde_json::Value = serde_json::from_str(&report).expect("parses");
    let reason = value["outcome"]["Aborted"]["reason"]
        .as_str()
        .expect("a reason");
    assert!(
        reason.contains("refusing to start"),
        "the backfill pipeline is the same one: {report}"
    );
    let requests = server.received_requests().await.expect("captured");
    assert!(
        requests
            .iter()
            .all(|request| request.url.path() != "/api/kb/map-worklist"),
        "no map-worklist request may exist"
    );
}

/// A state dir that CANNOT be created (its parent is a file) exits 2 before
/// any fetch — the binary's state-dir creation faults, never a panic.
#[tokio::test]
async fn a_broken_state_dir_exits_2_before_any_fetch() {
    let server = loop_input_server(200, LOOP_INPUT_FIXTURE).await;
    let parent = tempfile::tempdir().expect("tempdir");
    std::fs::write(parent.path().join("not-a-dir"), "a file").expect("write");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        parent
            .path()
            .join("not-a-dir/somnus")
            .to_str()
            .expect("utf8")
            .to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("could not create the state dir"),
        "{stderr}"
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("captured")
            .is_empty(),
        "no fetch may happen before the state dir exists"
    );
}

/// An offload child that CANNOT be created (a FILE named `offload` in the
/// state dir) exits 2 before any fetch.
#[tokio::test]
async fn a_broken_offload_child_exits_2() {
    let server = loop_input_server(200, LOOP_INPUT_FIXTURE).await;
    let state = tempfile::tempdir().expect("tempdir");
    std::fs::write(state.path().join("offload"), "a file").expect("write");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("could not create the offload dir"),
        "{stderr}"
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("captured")
            .is_empty()
    );
}

/// A kb-token target that cannot be written (a DIRECTORY named `kb-token`)
/// exits 2 before any fetch.
#[tokio::test]
async fn an_unwritable_kb_token_exits_2_before_any_fetch() {
    let server = loop_input_server(200, LOOP_INPUT_FIXTURE).await;
    let state = tempfile::tempdir().expect("tempdir");
    std::fs::create_dir(state.path().join("kb-token")).expect("mkdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("could not write the kb-token file"),
        "{stderr}"
    );
    assert!(
        server
            .received_requests()
            .await
            .expect("captured")
            .is_empty()
    );
}

/// A stopping unit stops the invocation: a project whose baseline cannot be
/// observed aborts the unit, the nightly exits 2, and the record carries
/// `fault:2`.
#[tokio::test]
async fn a_stopping_unit_stops_the_nightly_and_writes_the_record() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-worklist"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_string(r#"{"projects":[{"project_ref":"demo-project","mappable":true,"unpointed":4,"map_count":2,"latest_map_written_at":null}]}"#),
        )
        .mount(&server)
        .await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api/kb/map-loop-input"))
        .respond_with(wiremock::ResponseTemplate::new(404).set_body_string("not found"))
        .mount(&server)
        .await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(2), "{stderr}");
    assert!(
        stderr.contains("somnus: nightly unit for demo-project ended Aborted"),
        "{stderr}"
    );
    let record = std::fs::read_to_string(state.path().join("nightly-invocation.json"))
        .expect("the record is on disk");
    let value: serde_json::Value = serde_json::from_str(&record).expect("parses");
    assert_eq!(value["stop_reason"], "fault:2");
    assert_eq!(value["exit_code"], 2);
    let reason = value["units"][0]["outcome"]["Aborted"]["reason"]
        .as_str()
        .expect("a reason");
    assert!(reason.contains("refusing to start"), "reason was {reason}");
}

/// A 401 worklist is a configuration fault: exit 1 with the pinned line.
#[tokio::test]
async fn a_401_worklist_exits_1_with_the_pinned_line() {
    let server = worklist_server("denied", 401).await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(1), "{stderr}");
    assert!(
        stderr.contains(
            "somnus: /api/kb/map-worklist rejected the bearer token (401); check SOMNUS_KB_API_KEY"
        ),
        "{stderr}"
    );
    // The fault exits before any unit, but the record is still written.
    let record = std::fs::read_to_string(state.path().join("nightly-invocation.json"))
        .expect("the record is on disk");
    let value: serde_json::Value = serde_json::from_str(&record).expect("parses");
    assert_eq!(value["stop_reason"], "fault:1");
    assert_eq!(value["exit_code"], 1);
}

/// The token bridge to the gate child must not outlive the process. Mode 0600
/// keeps other users out but is no defence against the owner reading the state
/// dir afterwards — which is how a token reached a transcript once. Asserted
/// on a failing run, because that is the path most likely to skip cleanup.
#[tokio::test]
async fn the_kb_token_file_does_not_survive_the_run() {
    let server = worklist_server("denied", 401).await;
    let state = tempfile::tempdir().expect("tempdir");
    let mut env = required_env(&server.uri());
    env.push((
        "SOMNUS_STATE_DIR".to_string(),
        state.path().to_str().expect("utf8").to_string(),
    ));
    let (code, _stdout, stderr) = run_cli_with(&["nightly"], &env);
    assert_eq!(code, Some(1), "{stderr}");
    // The record proves the run got past the token write and ran to the end.
    assert!(state.path().join("nightly-invocation.json").exists());
    assert!(
        !state.path().join("kb-token").exists(),
        "the kb-token file outlived the process that needed it"
    );
}
