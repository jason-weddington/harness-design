//! `edit_file` — the write tool.
//!
//! Two modes, one schema:
//! - **CREATE** (empty `old_string`): create a new file, refusing to clobber an
//!   existing one.
//! - **REPLACE** (non-empty `old_string`): replace a *unique* occurrence of
//!   `old_string` in an existing UTF-8 text file with `new_string`.
//!
//! The design is the proven agent-editing shape: exact string match, no fuzzy
//! logic, no diff/patch. Ambiguity (`old_string` matches zero times or many
//! times) and misses are surfaced as *steering errors* — actionable
//! [`ToolResult::error`] payloads that tell the model how to recover — not
//! loop-crashing exceptions.
//!
//! Writes are made atomic-for-our-threat-model by writing to a sibling temp
//! file in the same directory and renaming it over the target; see
//! [`atomic_write`] for the rationale.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tool::{Tool, ToolCtx, ToolResult};

/// Human-readable description advertised to the model in [`EditFileTool::schema`].
///
/// Kept verbose on purpose: the exact-match contract, the unique-match
/// requirement, and the empty-`old_string` convention for CREATE are the parts
/// the model most often forgets. Steering errors quote back the same wording.
const DESCRIPTION: &str = "\
Edit a text file by exact string replacement, or create a new file.\n\
\n\
`old_string` must appear in the file EXACTLY ONCE and match the file content \
byte-for-byte, including whitespace and surrounding lines — copy it verbatim \
from `read_file` output. If it matches zero times or more than once, the edit \
is refused with a steering error naming what went wrong so you can try again \
with more surrounding context.\n\
\n\
Pass an empty string for `old_string` to CREATE a new file at `path` with \
`new_string` as its contents. Refuses to overwrite an existing file; use a \
regular replacement edit for that.\n\
\n\
`path` is workspace-relative. Files are read/written as UTF-8; the tool \
refuses to edit non-text (invalid-UTF-8) files.\
";

/// The `edit_file` tool: exact-match string replacement plus a create-new-file
/// mode. See the module docs for the mode dispatch.
#[derive(Debug, Default, Clone, Copy)]
pub struct EditFileTool;

#[async_trait]
impl Tool for EditFileTool {
    // The trait fixes the return type as `&str`; returning a `&'static str`
    // here would diverge from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "edit_file"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "edit_file",
            "description": DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to the file to edit or create.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace. Must appear in the file exactly once, byte-for-byte (copy it from read_file output, including whitespace). Pass an empty string to CREATE a new file at `path`.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text. In CREATE mode this is the contents of the new file. Pass an empty string to delete the matched text.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        let (path_arg, old_string, new_string) = match parse_args(&input) {
            Ok(args) => args,
            Err(msg) => return ToolResult::error(msg),
        };

        let resolved = match ctx.workspace().resolve_write(&path_arg) {
            Ok(p) => p,
            Err(violation) => return ToolResult::error(violation.to_string()),
        };

        if old_string.is_empty() {
            create_file(&path_arg, &resolved, &new_string)
        } else {
            replace_in_file(&path_arg, &resolved, &old_string, &new_string)
        }
    }
}

/// Pull `path`, `old_string`, `new_string` out of the model-supplied JSON.
///
/// Returned errors are steering messages naming the missing / mistyped field,
/// so the model can retry with a valid shape rather than crash the loop.
fn parse_args(input: &Value) -> Result<(String, String, String), String> {
    let path = require_string(input, "path")?;
    let old_string = require_string(input, "old_string")?;
    let new_string = require_string(input, "new_string")?;
    Ok((path, old_string, new_string))
}

/// Fetch a required string field from the tool's JSON input.
///
/// A missing or non-string value is a steering error — the model reacts by
/// re-issuing the call with the correct shape.
fn require_string(input: &Value, field: &str) -> Result<String, String> {
    match input.get(field) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => Err(format!(
            "`{field}` must be a string (got `{}`)",
            short_type_name(other)
        )),
        None => Err(format!("missing required field `{field}`")),
    }
}

/// One-word JSON type name for steering messages ("string", "number", …).
fn short_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// CREATE mode: refuse to clobber, create parents, write atomically.
fn create_file(display_path: &str, resolved: &Path, new_string: &str) -> ToolResult {
    if resolved.exists() {
        return ToolResult::error(format!(
            "`{display_path}` already exists — pass a non-empty `old_string` to edit it, or delete it first via bash if you truly need to overwrite"
        ));
    }
    if let Some(parent) = resolved.parent()
        && let Err(err) = std::fs::create_dir_all(parent)
    {
        return ToolResult::error(format!(
            "failed to create parent directories for `{display_path}`: {err}"
        ));
    }
    match atomic_write(resolved, new_string.as_bytes()) {
        Ok(()) => ToolResult::ok(format!(
            "created {display_path} ({} bytes)",
            new_string.len()
        )),
        Err(err) => ToolResult::error(format!("failed to write `{display_path}`: {err}")),
    }
}

/// REPLACE mode: exactly-one-match, or a steering error explaining how to
/// recover.
fn replace_in_file(
    display_path: &str,
    resolved: &Path,
    old_string: &str,
    new_string: &str,
) -> ToolResult {
    if !resolved.exists() {
        return ToolResult::error(format!(
            "`{display_path}` does not exist — pass an empty `old_string` to create it instead"
        ));
    }
    let bytes = match std::fs::read(resolved) {
        Ok(b) => b,
        Err(err) => return ToolResult::error(format!("failed to read `{display_path}`: {err}")),
    };
    let Ok(content) = std::str::from_utf8(&bytes) else {
        return ToolResult::error(format!(
            "`{display_path}` is not valid UTF-8; edit_file refuses to edit non-text files"
        ));
    };

    // `str::matches` returns disjoint (non-overlapping) matches, which is the
    // semantics we want: a run of `aaa` with `old_string = "aa"` is *two*
    // locations from the model's point of view, and the "make it unique" hint
    // is the right recovery.
    let count = content.matches(old_string).count();
    match count {
        0 => ToolResult::error(format!(
            "old_string not found in `{display_path}` — read the file and copy the exact text, including whitespace"
        )),
        n if n > 1 => ToolResult::error(format!(
            "old_string matches {n} locations in `{display_path}` — include more surrounding context to make it unique"
        )),
        _ => {
            let updated = content.replacen(old_string, new_string, 1);
            match atomic_write(resolved, updated.as_bytes()) {
                Ok(()) => ToolResult::ok(format!("edited {display_path}")),
                Err(err) => ToolResult::error(format!("failed to write `{display_path}`: {err}")),
            }
        }
    }
}

/// Sibling counter shared across [`atomic_write`] callers so temp filenames
/// stay unique within a process even when two writes to the same directory
/// race.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `bytes` to `target` atomically-enough for our threat model.
///
/// The write goes to a sibling temp file in the same directory and is then
/// renamed over `target`. Rationale: a crash (or `kill -9`, or an OOM) mid-write
/// must not leave a truncated source file for the next `run_checks` /
/// `bash` to trip over — the harness would then chase a phantom "the
/// build broke" instead of the real regression. Same-directory rename is
/// atomic on Linux (POSIX `rename(2)` on a single filesystem), which is our
/// only deployment target (see `workspace.rs`).
///
/// The temp file's name mixes the process id with a monotonic counter so two
/// concurrent tool calls targeting the same directory pick disjoint names. A
/// leading `.` and a `.tmp` suffix keep the name obviously-not-source in case
/// of a crash between write and rename.
fn atomic_write(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let temp = temp_sibling(target);
    // Best-effort cleanup on any error path below: if we can't rename into
    // place, leaving the temp file around would litter the workspace.
    match std::fs::write(&temp, bytes) {
        Ok(()) => {}
        Err(err) => {
            let _ = std::fs::remove_file(&temp);
            return Err(err);
        }
    }
    match std::fs::rename(&temp, target) {
        Ok(()) => Ok(()),
        Err(err) => {
            let _ = std::fs::remove_file(&temp);
            Err(err)
        }
    }
}

/// Build a same-directory temp path for [`atomic_write`].
fn temp_sibling(target: &Path) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let name = target
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("edit_file");
    dir.join(format!(".{name}.edit_file.{pid}.{n}.tmp"))
}

#[cfg(test)]
mod tests {
    use super::EditFileTool;
    use crate::tool::{OffloadSink, Tool, ToolCtx};
    use crate::workspace::Workspace;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// Silent offload sink for tests — `edit_file` never offloads today, but
    /// a `ToolCtx` needs one and we don't want a real disk sink polluting the
    /// workspace under test.
    #[derive(Debug, Default)]
    struct NullSink;

    impl OffloadSink for NullSink {
        fn offload(&self, _contents: &str) -> PathBuf {
            PathBuf::from("<null>")
        }
    }

    fn ctx_for(root: &TempDir) -> ToolCtx {
        let ws = Workspace::new(root.path(), None).expect("workspace");
        ToolCtx::new(Arc::new(ws), Arc::new(NullSink))
    }

    #[test]
    fn schema_advertises_required_fields_and_key_wording() {
        let schema = EditFileTool.schema();
        assert_eq!(schema["name"], "edit_file");
        let props = &schema["input_schema"]["properties"];
        for field in ["path", "old_string", "new_string"] {
            assert!(props.get(field).is_some(), "schema missing `{field}`");
        }
        let required: Vec<&str> = schema["input_schema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(required, vec!["path", "old_string", "new_string"]);

        let desc = schema["description"].as_str().unwrap();
        assert!(desc.contains("exactly once") || desc.contains("EXACTLY ONCE"));
        assert!(
            desc.contains("empty"),
            "description mentions empty-string CREATE mode"
        );
    }

    #[tokio::test]
    async fn create_writes_new_file_including_nested_parents() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(
                json!({
                    "path": "a/b/c/new.rs",
                    "old_string": "",
                    "new_string": "fn main() {}\n",
                }),
                &ctx,
            )
            .await;

        assert!(!out.is_error, "create should succeed: {}", out.summary);
        assert!(out.summary.contains("created"));
        assert!(
            out.summary.contains("a/b/c/new.rs"),
            "summary should name the path: {}",
            out.summary
        );
        assert!(
            out.summary.contains("13 bytes"),
            "summary should include byte count: {}",
            out.summary
        );

        let written = std::fs::read_to_string(root.path().join("a/b/c/new.rs")).unwrap();
        assert_eq!(written, "fn main() {}\n");
    }

    #[tokio::test]
    async fn create_on_existing_file_is_steering_error() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        std::fs::write(root.path().join("already.txt"), "hi").unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "already.txt",
                    "old_string": "",
                    "new_string": "should not clobber",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error, "create on existing must be an error");
        assert!(out.summary.contains("already exists"));
        assert!(
            out.summary.contains("non-empty `old_string`"),
            "should steer toward replacement edit: {}",
            out.summary
        );
        // File contents untouched.
        assert_eq!(
            std::fs::read_to_string(root.path().join("already.txt")).unwrap(),
            "hi"
        );
    }

    #[tokio::test]
    async fn unique_replace_round_trips_and_leaves_rest_untouched() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        let path = root.path().join("src.rs");
        let original = "before\nlet x = 1;\nafter\n";
        std::fs::write(&path, original).unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "src.rs",
                    "old_string": "let x = 1;",
                    "new_string": "let x = 42;",
                }),
                &ctx,
            )
            .await;

        assert!(
            !out.is_error,
            "unique replace should succeed: {}",
            out.summary
        );
        assert_eq!(out.summary, "edited src.rs");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "before\nlet x = 42;\nafter\n"
        );
    }

    #[tokio::test]
    async fn empty_new_string_deletes_matched_text() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        let path = root.path().join("del.txt");
        std::fs::write(&path, "keep [DELETE ME] keep\n").unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "del.txt",
                    "old_string": "[DELETE ME] ",
                    "new_string": "",
                }),
                &ctx,
            )
            .await;

        assert!(
            !out.is_error,
            "delete-via-empty should succeed: {}",
            out.summary
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep keep\n");
    }

    #[tokio::test]
    async fn zero_match_is_steering_error_with_recovery_hint() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        std::fs::write(root.path().join("f.txt"), "actual content").unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "f.txt",
                    "old_string": "not present anywhere",
                    "new_string": "x",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("not found"));
        assert!(
            out.summary.contains("read the file"),
            "should steer toward re-reading the file: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn multi_match_is_steering_error_naming_the_count() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        std::fs::write(root.path().join("dup.txt"), "foo foo foo").unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "dup.txt",
                    "old_string": "foo",
                    "new_string": "bar",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error);
        assert!(
            out.summary.contains("3 locations"),
            "names count: {}",
            out.summary
        );
        assert!(
            out.summary.contains("surrounding context"),
            "steers toward more context: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn replace_on_missing_file_is_steering_error() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(
                json!({
                    "path": "nope.txt",
                    "old_string": "anything",
                    "new_string": "x",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("does not exist"));
        assert!(
            out.summary.contains("empty `old_string`"),
            "steers toward CREATE mode: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn path_escape_attempt_is_steering_error() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(
                json!({
                    "path": "../evil.txt",
                    "old_string": "",
                    "new_string": "pwned",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error, "escape must be rejected");
        assert!(
            out.summary.contains("../evil.txt"),
            "should quote the offending input: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn non_utf8_target_is_steering_error() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        // A stray 0xff byte is invalid UTF-8.
        std::fs::write(root.path().join("bin.dat"), [0x66, 0x6f, 0xff, 0x6f]).unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "bin.dat",
                    "old_string": "xy",
                    "new_string": "XY",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("UTF-8"));
        assert!(out.summary.contains("non-text"));
    }

    #[tokio::test]
    async fn absolute_path_is_steering_error() {
        // The workspace layer already rejects absolute paths; verify the tool
        // surfaces that error rather than letting it panic through.
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(
                json!({
                    "path": "/etc/passwd",
                    "old_string": "",
                    "new_string": "x",
                }),
                &ctx,
            )
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("/etc/passwd"));
    }

    #[tokio::test]
    async fn missing_field_is_steering_error() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(json!({ "path": "f.txt", "old_string": "" }), &ctx)
            .await;
        assert!(out.is_error);
        assert!(
            out.summary.contains("new_string"),
            "names the missing field: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn wrong_typed_field_is_steering_error() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(
                json!({
                    "path": "f.txt",
                    "old_string": 42,
                    "new_string": "x",
                }),
                &ctx,
            )
            .await;
        assert!(out.is_error);
        assert!(out.summary.contains("old_string"));
        assert!(out.summary.contains("string"));
    }

    #[test]
    fn temp_sibling_lives_in_same_directory() {
        let target = std::path::Path::new("/tmp/foo/bar.rs");
        let temp = super::temp_sibling(target);
        assert_eq!(temp.parent(), Some(std::path::Path::new("/tmp/foo")));
        let name = temp.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with(".bar.rs.edit_file."),
            "unexpected temp name: {name}"
        );
        // `.tmp` here is a literal suffix on the whole filename, not a file-type
        // extension we care to match case-insensitively — silence the clippy hint.
        assert!(
            {
                #[allow(clippy::case_sensitive_file_extension_comparisons)]
                let ok = name.ends_with(".tmp");
                ok
            },
            "temp name should end with .tmp: {name}"
        );
        // A second call picks a distinct name.
        let temp2 = super::temp_sibling(target);
        assert_ne!(temp, temp2);
    }

    #[tokio::test]
    async fn create_is_atomic_no_temp_files_left_behind() {
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        let out = EditFileTool
            .run(
                json!({
                    "path": "atomic.rs",
                    "old_string": "",
                    "new_string": "ok\n",
                }),
                &ctx,
            )
            .await;
        assert!(!out.is_error);

        // Only the target file should live in the workspace root — no `.tmp`
        // siblings from the atomic-write path.
        let entries: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![std::ffi::OsString::from("atomic.rs")]);
    }

    #[test]
    fn tool_name_is_the_dispatch_key() {
        // `name()` is what the registry indexes tools by; a drift here would
        // break dispatch silently. The schema must advertise the same name.
        assert_eq!(EditFileTool.name(), "edit_file");
        assert_eq!(EditFileTool.schema()["name"], "edit_file");
    }

    #[tokio::test]
    async fn wrong_typed_field_steering_error_names_the_json_type() {
        // `require_string` builds a human-readable steering message naming the
        // JSON type the model sent instead of a string. Every type label is a
        // contract with the model — a wrong/misleading label steers it toward
        // the wrong retry shape. Pin all of them.
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);

        // (value, expected short type name) for each non-string JSON kind.
        let cases: [(serde_json::Value, &str); 5] = [
            (json!(null), "null"),
            (json!(true), "boolean"),
            (json!(42.0), "number"),
            (json!(["a"]), "array"),
            (json!({}), "object"),
        ];
        for (value, expected_type) in cases {
            let out = EditFileTool
                .run(
                    json!({
                        "path": "f.txt",
                        "old_string": value,
                        "new_string": "x",
                    }),
                    &ctx,
                )
                .await;
            assert!(
                out.is_error,
                "non-string `old_string` ({expected_type}) must be a steering error",
            );
            assert!(
                out.summary.contains(&format!("`{expected_type}`")),
                "message must name the JSON type `{expected_type}` for {value:?}: {}",
                out.summary,
            );
            assert!(
                out.summary.contains("old_string") && out.summary.contains("string"),
                "message must name the field and the expected type: {}",
                out.summary,
            );
        }
    }

    #[tokio::test]
    async fn replace_on_a_directory_path_is_steering_error() {
        // A path whose `exists()` is true but `read()` fails (a directory) must
        // surface as a steering error, not a panic — the loop relies on every
        // tool failure being a ToolResult::error.
        let root = TempDir::new().expect("tempdir");
        let ctx = ctx_for(&root);
        std::fs::create_dir(root.path().join("adir")).unwrap();

        let out = EditFileTool
            .run(
                json!({
                    "path": "adir",
                    "old_string": "x",
                    "new_string": "y",
                }),
                &ctx,
            )
            .await;

        assert!(
            out.is_error,
            "replacing inside a directory path must be a steering error, not ok",
        );
        assert!(
            out.summary.contains("failed to read"),
            "should name the read failure: {}",
            out.summary,
        );
        assert!(out.summary.contains("adir"));
    }
}
