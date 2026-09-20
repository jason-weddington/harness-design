//! Integration tests for the `somnus` CLI, spawning the compiled binary via
//! `env!("CARGO_BIN_EXE_somnus")` (the `crates/talos` precedent, so
//! `cargo-llvm-cov` collects child-process coverage). No network, no live
//! KB, no model.

use std::process::Command;

/// Compiled `somnus` binary path (injected by cargo at integration-test time).
const SOMNUS_BIN: &str = env!("CARGO_BIN_EXE_somnus");

/// Run the binary with `args`, capturing output, with no stdin.
fn run_cli(args: &[&str]) -> (Option<i32>, String, String) {
    use std::process::Stdio;

    let mut child = Command::new(SOMNUS_BIN)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn somnus");
    // Close stdin immediately: nothing in this binary reads it.
    drop(child.stdin.take());
    let output = child.wait_with_output().expect("wait for somnus");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

// ============================================================================
// (a) `somnus --version`: the installer's fleet contract
// ============================================================================

/// `somnus --version` MUST emit the SAME output shape as `talos --version`:
/// the installer parses whitespace-separated field 2 for the version token
/// and treats absent or unparsable as the literal `none`. A different shape
/// does not fail loudly; it turns every nightly run into an unconditional
/// reinstall. Field 2 must additionally be a URL/path-safe token (the
/// build.rs stamp mirrors the talos stamp: `git describe --tags --always
/// --dirty`, yielding `<semver>` or `<semver>-g<short-sha>`).
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
// (b) `somnus run --project ""`: rejected BEFORE any other work
// ============================================================================

/// An empty or whitespace-only `--project` value is rejected with exit 1 and
/// the pinned stderr substring — before any backend construction or network
/// I/O (none of which exists anywhere in the binary this cut to be ordered
/// before it).
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
// (c) the not-wired run stub: a LOUD stub, not a silent one
// ============================================================================

/// After the empty-project check, the run body is a stub that exits 1 with a
/// stderr message naming the two in-flight GTD items and the deferred
/// rung-1/rung-2 inference calls — so an accidental nightly invocation fails
/// loudly instead of burning the metered Anthropic lane or silently doing
/// nothing.
#[test]
fn run_body_stub_fails_loudly_naming_the_deferred_items() {
    let (code, _stdout, stderr) = run_cli(&["run", "--project", "any"]);
    assert_eq!(
        code,
        Some(1),
        "the not-wired stub must exit 1, stderr: {stderr}"
    );
    for substring in [
        "somnus-loop-input",
        "somnus-cluster-ledger",
        "rung-1",
        "rung-2",
    ] {
        assert!(
            stderr.contains(substring),
            "stub message must name {substring}; stderr was: {stderr}"
        );
    }
}
