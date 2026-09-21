//! The map-lint gate (leg 2) and the loop's tool registry.
//!
//! The gate is `POST /api/kb/map-lint`, shelled out as
//! `/bin/sh -c 'curl -sf …'` — the HTTP status is the verdict: `422` (or any
//! non-2xx) makes `curl -f` exit non-zero, `ChecksRunner::run` maps that to
//! `report.passed == false`, and `RunChecksTool` maps `!report.passed` to
//! `is_error`.
//!
//! The body travels as a FILE, not as stdin: `exec::run` spawns the child
//! with `.stdin(Stdio::null())` and `ChecksRunner` exposes no stdin seam, so
//! rung 3 materializes the composed body to a per-map path on disk and the
//! command posts it with `--data-binary "@{path}"`. The `@{path}` is
//! double-quoted inside the `/bin/sh -c` string, and the project ref that
//! names the parent directory is charset-validated before anything is
//! written, so no shell metacharacter can reach the child. Fail-closed by
//! construction: a missing file makes `curl` exit non-zero and the gate read
//! red.
//!
//! The bearer token is read from a file BY THE CHILD at runtime and NEVER
//! appears in argv — and therefore never in
//! `ChecksRunner::command_display()`, the rendered system prompt, the
//! transcript, or the `SQLite` run record (the leak is real:
//! `CheckCommand`'s `Display` renders program+args and flows into the system
//! prompt and the `run_start` transcript event). Because the signature takes
//! no token parameter, no token literal can ever be committed and the
//! `gitleaks` gate can never flag this crate. Feasibility is verified by the
//! harness: `exec::run` preserves exactly `TERM`, `PATH`, and `HOME` across
//! `env_clear()`, so the child's `$HOME` lookup works.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use harness::engine::{FINISH_TOOL_NAME, FinishTool};
use harness::exec::{CheckCommand, ChecksRunner};
use harness::tool::ToolRegistry;
use harness::tools::run_checks::RunChecksTool;

use crate::ops::{MapOpSink, OpTool};

/// Wall-clock bound on one map-lint invocation. The lint is a server-side
/// POST over the local host's network; thirty seconds covers a full corpus
/// pass and keeps a hung endpoint from hanging a finish.
pub const SOMNUS_GATE_TIMEOUT: Duration = Duration::from_secs(30);

/// Exit code meaning the map-lint ACCEPTED the body.
pub const GATE_PASSED: i32 = 0;
/// Exit code meaning the map-lint REFUSED the body — a real verdict.
pub const GATE_REFUSED: i32 = 1;
/// Exit code meaning the gate could not be EVALUATED at all: the token file
/// is unreadable, curl could not reach the endpoint, or the server answered
/// something that is not a lint verdict (a 401, a 500, a redirect).
pub const GATE_UNEVALUATED: i32 = 3;

/// The map-lint gate command for `kb_base` posting the composed body at
/// `body_path`, authenticating with the token read from `token_path`.
///
/// **The exit code is a verdict, not curl's opinion of the HTTP status, and
/// that distinction is the whole point of this command.** The first version
/// used `curl -s --fail-with-body`, which exits 22 for EVERY status at or
/// above 400 — so a 422 carrying the lint's findings and a 401 saying the
/// request never reached the lint were the same non-zero number. On
/// 2026-09-21 the token file went missing mid-run, all eleven gates answered
/// `{"detail":"Not authenticated"}`, and every one of them was recorded as a
/// body the lint had rejected. The gate is the one component whose entire
/// job is refusing, and it had no way to say "I could not ask".
///
/// So the script branches on the HTTP status itself:
/// [`GATE_PASSED`] for 2xx, [`GATE_REFUSED`] for 422 — the only status that
/// IS a lint verdict — and [`GATE_UNEVALUATED`] for anything else, including
/// a missing token file, which is checked first and by name so the failure
/// reads as itself rather than as an authentication error downstream of it.
///
/// The response body still reaches stdout ahead of the status line, so the
/// lint's findings land in the report excerpt and the state-dir offload
/// exactly as before.
///
/// POSIX `sh` only: `/bin/sh` is dash on the deployment hosts, so no
/// `$'...'`, no `[[`, no arrays. The token is read from the file BY THE
/// CHILD at runtime — the command string is constant per base and carries no
/// secret.
#[must_use]
pub fn map_lint_command(kb_base: &str, body_path: &Path, token_path: &Path) -> CheckCommand {
    let token = token_path.to_string_lossy();
    let body = body_path.to_string_lossy();
    let script = format!(
        concat!(
            "if [ ! -r \"{token}\" ]; then ",
            "echo \"somnus-gate: token file {token} is unreadable; the gate could not be evaluated\"; ",
            "exit {unevaluated}; fi; ",
            "out=$(curl -s -w '\\n%{{http_code}}' -X POST {base}/api/kb/map-lint ",
            "-H \"Authorization: Bearer $(cat \"{token}\")\" --data-binary \"@{body}\"); ",
            "rc=$?; ",
            "printf '%s\\n' \"$out\"; ",
            "if [ $rc -ne 0 ]; then ",
            "echo \"somnus-gate: curl exited $rc; the gate could not be evaluated\"; ",
            "exit {unevaluated}; fi; ",
            "code=$(printf '%s' \"$out\" | tail -n 1); ",
            "case \"$code\" in ",
            "2??) exit {passed} ;; ",
            "422) exit {refused} ;; ",
            "*) echo \"somnus-gate: status $code is not a lint verdict; the gate could not be evaluated\"; ",
            "exit {unevaluated} ;; ",
            "esac"
        ),
        token = token,
        body = body,
        base = kb_base,
        passed = GATE_PASSED,
        refused = GATE_REFUSED,
        unevaluated = GATE_UNEVALUATED,
    );
    CheckCommand {
        program: "/bin/sh".to_string(),
        args: vec!["-c".to_string(), script],
    }
}

/// The [`ChecksRunner`] for the map-lint gate.
///
/// The workspace root is the process working directory (falling back to `.`
/// when it cannot be read): the gate is a `curl` that ignores the working
/// directory entirely, and `ChecksRunner`'s fields are private and
/// unassertable, so this construction rule IS the pin.
#[must_use]
pub fn map_lint_runner(kb_base: &str, body_path: &Path, token_path: &Path) -> ChecksRunner {
    ChecksRunner::new(
        map_lint_command(kb_base, body_path, token_path),
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        SOMNUS_GATE_TIMEOUT,
    )
}

/// The loop's tool registry: EXACTLY six tools, no read tools, no bash, no
/// `no_change` (that op exists only as a variant of [`crate::ops::Op`]).
///
/// The harness read/mutating tools (`read_file`, `list_files`, `edit_file`,
/// `bash`) are ALL absent — the vendored spec's factor-13 rule deletes the
/// read tools from the output union: rung-1 input is injected as synthetic
/// tool-call/result events, so the model cannot wander the KB.
#[must_use]
pub fn build_registry(
    kb_base: &str,
    body_path: &Path,
    token_path: &Path,
    sink: Arc<dyn MapOpSink>,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(
        "add_pointer",
        Arc::new(OpTool::new("add_pointer", Arc::clone(&sink))),
    );
    registry.register(
        "create_map",
        Arc::new(OpTool::new("create_map", Arc::clone(&sink))),
    );
    registry.register(
        "strike_gap",
        Arc::new(OpTool::new("strike_gap", Arc::clone(&sink))),
    );
    registry.register("propose_gap", Arc::new(OpTool::new("propose_gap", sink)));

    registry.register(
        FINISH_TOOL_NAME,
        Arc::new(FinishTool { answer_mode: false }),
    );

    // Pinned workaround, tracked for migration in harness-design `49b4445e`:
    // the gate tool is registered under the LITERAL name `run_checks`
    // because the engine hardcodes the `call.name == "run_checks"` match arm
    // as the sole setter of `last_gate_green` (crates/harness/src/engine.rs).
    // An HTTP map-lint is not `run_checks` in any sense its author would
    // pick; migrate to the consumer-declared gate when that harness item
    // lands and drop the magic name.
    registry.register(
        "run_checks",
        Arc::new(RunChecksTool::new(map_lint_runner(
            kb_base, body_path, token_path,
        ))),
    );

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::SchemaOnlySink;

    // --- the gate command: token out of argv, body from a quoted file path ---

    #[test]
    fn map_lint_command_shape_pins_the_token_to_the_child_runtime() {
        let body_path = std::path::Path::new("/tmp/somnus/demo-project/kb-20001.json");
        let token_path = std::path::Path::new("/tmp/somnus/kb-token");
        let command = map_lint_command("http://kb.invalid", body_path, token_path);
        assert_eq!(command.program, "/bin/sh");
        assert_eq!(command.args[0], "-c");
        let script = &command.args[1];
        assert!(script.contains("curl -s -w '\\n%{http_code}' -X POST"));
        assert!(script.contains("http://kb.invalid/api/kb/map-lint"));
        // The token is pinned to the CHILD runtime, never argv, and read
        // from the state dir's token FILE the binary wrote.
        assert!(script.contains("Bearer $(cat"));
        assert!(script.contains("$(cat \"/tmp/somnus/kb-token\")"));
        // The body posts from a FILE: `exec::run` spawns with
        // `.stdin(Stdio::null())` and `ChecksRunner` exposes no stdin seam,
        // so the composed body must be on disk and double-quoted here.
        assert!(script.contains("--data-binary"));
        assert!(script.contains("--data-binary \"@/tmp/somnus/demo-project/kb-20001.json\""));
        // `--fail-with-body` is GONE. It exits 22 for every status at or
        // above 400, which made a 422 lint verdict and a 401 "I never
        // reached the lint" the same number — the fail-open shape that
        // recorded eleven un-asked gates as rejections.
        assert!(!script.contains("--fail-with-body"));
        assert!(!script.contains("curl -sf"));
    }

    /// The three exit codes are the contract between this script and the
    /// pipeline, so they are pinned by value: the pipeline treats anything
    /// that is not PASSED or REFUSED as an infrastructure fault.
    #[test]
    fn the_gate_exit_codes_are_pinned_and_distinct() {
        assert_eq!(GATE_PASSED, 0);
        assert_eq!(GATE_REFUSED, 1);
        assert_eq!(GATE_UNEVALUATED, 3);
    }

    /// The missing-token branch is checked FIRST and names the file, so the
    /// failure reads as itself rather than as the authentication error
    /// downstream of it.
    #[test]
    fn a_missing_token_is_named_before_curl_is_reached() {
        let command = map_lint_command(
            "http://kb.invalid",
            std::path::Path::new("/tmp/b.json"),
            std::path::Path::new("/tmp/t"),
        );
        let script = &command.args[1];
        let guard = script
            .find("[ ! -r \"/tmp/t\" ]")
            .expect("the guard is present");
        let curl = script.find("curl").expect("curl is present");
        assert!(guard < curl, "the token guard must precede the request");
        assert!(script.contains("token file /tmp/t is unreadable"));
    }

    /// Only 422 is a lint verdict. Every other status — a 401, a 500, a
    /// redirect — means the body was never judged.
    #[test]
    fn only_422_is_treated_as_a_verdict() {
        let command = map_lint_command(
            "http://kb.invalid",
            std::path::Path::new("/tmp/b.json"),
            std::path::Path::new("/tmp/t"),
        );
        let script = &command.args[1];
        assert!(script.contains("2??) exit 0 ;;"));
        assert!(script.contains("422) exit 1 ;;"));
        assert!(script.contains("is not a lint verdict"));
    }

    #[test]
    fn map_lint_runner_builds_the_pinned_command() {
        let body_path = std::path::Path::new("/tmp/somnus/demo-project/kb-20001.json");
        let runner = map_lint_runner(
            "http://kb.invalid",
            body_path,
            std::path::Path::new("/tmp/somnus/kb-token"),
        );
        assert_eq!(runner.command().program, "/bin/sh");
        assert!(runner.command().args[1].contains("http://kb.invalid/api/kb/map-lint"));
        assert!(runner.command().args[1].contains("--data-binary"));
        assert!(runner.command().args[1].contains("%{http_code}"));
    }

    // --- the registry shape: exactly six tools, closed vocabulary ---

    #[test]
    fn registry_registers_exactly_the_six_pinned_tools() {
        let registry = build_registry(
            "http://kb.invalid",
            std::path::Path::new("/tmp/somnus-body.json"),
            std::path::Path::new("/tmp/somnus/kb-token"),
            std::sync::Arc::new(SchemaOnlySink),
        );
        let mut names: Vec<String> = registry
            .list()
            .into_iter()
            .map(|schema| {
                schema["name"]
                    .as_str()
                    .expect("every schema carries a name")
                    .to_string()
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "add_pointer".to_string(),
                "create_map".to_string(),
                "finish".to_string(),
                "propose_gap".to_string(),
                "run_checks".to_string(),
                "strike_gap".to_string(),
            ]
        );
        for name in [
            "add_pointer",
            "create_map",
            "finish",
            "propose_gap",
            "run_checks",
            "strike_gap",
        ] {
            assert!(registry.get(name).is_some(), "{name} must be registered");
        }
    }

    #[test]
    fn registry_has_no_read_tools_and_no_no_change_tool() {
        let registry = build_registry(
            "http://kb.invalid",
            std::path::Path::new("/tmp/somnus-body.json"),
            std::path::Path::new("/tmp/somnus/kb-token"),
            std::sync::Arc::new(SchemaOnlySink),
        );
        for absent in ["read_file", "list_files", "edit_file", "bash", "no_change"] {
            assert!(registry.get(absent).is_none(), "{absent} must be absent");
        }
    }

    // --- the four op schemas: the pinned closed vocabulary ---

    #[test]
    fn op_tool_schemas_pin_the_closed_vocabulary_exactly() {
        let registry = build_registry(
            "http://kb.invalid",
            std::path::Path::new("/tmp/somnus-body.json"),
            std::path::Path::new("/tmp/somnus/kb-token"),
            std::sync::Arc::new(SchemaOnlySink),
        );
        let pinned = [
            ("add_pointer", vec!["map_id", "entry_id", "gloss"]),
            ("create_map", vec!["title", "orientation_prose", "pointers"]),
            ("strike_gap", vec!["map_id", "gap_text", "closing_entry_id"]),
            ("propose_gap", vec!["cluster_id", "reason"]),
        ];
        for (name, properties) in pinned {
            let tool = registry.get(name).expect("registered op tool");
            let schema = tool.schema();
            let input = &schema["input_schema"];
            assert_eq!(input["type"], "object");
            let actual: Vec<String> = input["properties"]
                .as_object()
                .expect("properties object")
                .keys()
                .cloned()
                .collect();
            let mut expected = properties.clone();
            expected.sort_unstable();
            assert_eq!(actual, expected, "{name} properties must match exactly");
            for property in properties {
                assert!(
                    input["required"]
                        .as_array()
                        .expect("required list")
                        .iter()
                        .any(|value| value == property),
                    "{name} must list {property} in required"
                );
            }
        }
    }
}
