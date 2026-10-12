//! `read_file` — read a workspace-relative file's raw contents.
//!
//! In **build mode**, [`ReadFileTool`] returns content **without**
//! line-number prefixes. That is deliberate: [`edit_file`](crate::tools)
//! uses exact-match string replacement, and models copy `old_string`
//! verbatim out of a prior `read_file` output. A "1: foo\n2: bar\n" prefix
//! would poison the match by making the copied substring not actually exist
//! in the file — the model would have to strip prefixes itself, which is
//! exactly the kind of silent formatting drift we want to avoid at the tool
//! boundary.
//!
//! In **answer mode** there is no `edit_file` in the registry, so the same
//! name registers [`NumberedReadFileTool`] instead: each returned line is
//! prefixed with its 1-based absolute line number (`N<TAB>`, N right-aligned
//! to width 6) so agents can cite line references. The prefix is NOT part of
//! the file and must be excluded from any quoted text. Absolute offload-path
//! readbacks are returned AS-IS — the stored text is never re-prefixed.
//!
//! Optional `offset` (1-based first line) and `limit` (max lines) slice the
//! file line-wise before returning it. Absolute paths are accepted only when
//! they land inside the workspace's offload root — offloaded tool outputs are
//! advertised to the model as absolute paths and must be re-readable here.
//! Everything else routes through [`Workspace::resolve_read`], so path
//! confinement is enforced in one tested place.
//!
//! [`Workspace::resolve_read`]: crate::workspace::Workspace::resolve_read

use std::borrow::Cow;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tool::{Tool, ToolCtx, ToolResult};

/// Stable name the tool is registered and invoked under.
pub const READ_FILE_TOOL_NAME: &str = "read_file";

/// Read a file's raw contents through the [`Workspace`](crate::workspace::Workspace)
/// path-confinement seam.
///
/// See the module docs for the design rationale (why no line-number prefixes,
/// how offload paths are re-read).
#[derive(Debug, Default, Clone, Copy)]
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    // The trait fixes the return type as `&str`; a `&'static str` here would
    // diverge from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        READ_FILE_TOOL_NAME
    }

    fn schema(&self) -> Value {
        json!({
            "name": READ_FILE_TOOL_NAME,
            "description": "Read a file's contents. Output is the RAW file text — \
                            no line-number prefixes are added, so a passage copied \
                            from here matches verbatim when reused as `old_string` \
                            in `edit_file`. `path` is workspace-relative; an \
                            absolute offload path advertised by a prior truncated \
                            tool result is also readable here. Optional `offset` \
                            (1-based first line) and `limit` (max lines) slice \
                            the file line-wise before it is returned.",
            "input_schema": read_file_input_schema()
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        read_impl(input, ctx, false)
    }
}

/// The shared `input_schema` for both `read_file` variants: `path`
/// (required), optional `offset` / `limit`. Both tools register the SAME
/// `input_schema` — the only schema difference between the variants is the
/// description, and there is deliberately NO per-call numbering parameter.
fn read_file_input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to read, or an \
                                absolute offload path returned by an \
                                earlier truncated tool result."
            },
            "offset": {
                "type": "integer",
                "description": "1-based first line to return \
                                (default: 1, the start of the file)."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return \
                                (default: all remaining lines)."
            }
        },
        "required": ["path"]
    })
}

/// The answer-mode [`read_file`]: byte-identical behavior to
/// [`ReadFileTool`] EXCEPT the returned detail is line-numbered — each line
/// is prefixed with its 1-based absolute line number, right-aligned to
/// width 6, separated from the line text by one literal TAB (like
/// `cat -n`). Registered under the SAME name in
/// [`answer_registry`](crate::tools::answer_registry), whose registry has no
/// `edit_file`, so the prefix cannot poison an exact-match replacement.
///
/// Readback of an absolute offload path returns the stored text AS-IS
/// (numbered text stays numbered exactly once; non-numbered text from
/// `bash` / `list_files` / `run_checks` offloads is never prefixed),
/// because an absolute input is, by construction of
/// [`Workspace::resolve_read`](crate::workspace::Workspace::resolve_read),
/// an offload-readback path.
#[derive(Debug, Default, Clone, Copy)]
pub struct NumberedReadFileTool;

#[async_trait]
impl Tool for NumberedReadFileTool {
    // The trait fixes the return type as `&str`; a `&'static str` here would
    // diverge from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        READ_FILE_TOOL_NAME
    }

    fn schema(&self) -> Value {
        json!({
            "name": READ_FILE_TOOL_NAME,
            "description": "Read a file's contents with 1-based line numbers: \
                            each returned line is prefixed `N<TAB>` (N \
                            right-aligned to width 6, like `cat -n`). The \
                            prefix is NOT part of the file — any quote or text \
                            copied from this output must exclude the prefix. \
                            `path` is workspace-relative; an absolute offload \
                            path advertised by a prior truncated tool result is \
                            returned AS-IS — sliced by `offset`/`limit` if \
                            supplied, but never prefixed again here. Optional \
                            `offset` (1-based, the number is the file's \
                            absolute line) and `limit` (max lines) slice the \
                            file line-wise before numbering.",
            "input_schema": read_file_input_schema()
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        read_impl(input, ctx, true)
    }
}

/// Shared implementation of [`ReadFileTool`] and [`NumberedReadFileTool`].
///
/// Every behavior — parsing, path confinement, lossy decode, error steering,
/// [`DETAIL_CAP`](crate::tool::DETAIL_CAP) offloading — is identical by
/// construction. `numbered` changes only the formatting of the successful
/// detail output, and numbering is suppressed when the raw input `path` is
/// absolute (an offload readback): the stored text is returned as-is, never
/// re-prefixed.
// The `input: Value` pass-by-value mirrors the `Tool::run` signature the
// delegating `run`s call; the shared body only reads it.
#[allow(clippy::needless_pass_by_value)]
fn read_impl(input: Value, ctx: &ToolCtx, numbered: bool) -> ToolResult {
    let Some(path) = input.get("path").and_then(Value::as_str) else {
        return ToolResult::error("read_file: missing required string field `path`");
    };

    let offset = match parse_optional_u64(&input, "offset") {
        Ok(v) => v,
        Err(msg) => return ToolResult::error(msg),
    };
    let limit = match parse_optional_u64(&input, "limit") {
        Ok(v) => v,
        Err(msg) => return ToolResult::error(msg),
    };

    // Confinement first — a `PathViolation` becomes a steering error whose
    // `Display` already names the offending path and steers the model.
    let resolved = match ctx.workspace().resolve_read(path) {
        Ok(p) => p,
        Err(violation) => return ToolResult::error(violation.to_string()),
    };

    let bytes = match std::fs::read(&resolved) {
        Ok(b) => b,
        Err(err) => {
            return ToolResult::error(format!("read_file: cannot read `{path}`: {err}"));
        }
    };

    // Decode as UTF-8, replacing invalid sequences with U+FFFD. `Owned`
    // means at least one replacement happened, so we surface a lossy note
    // in the summary — the model sees it and can decide whether to trust
    // the file as text.
    let cow = String::from_utf8_lossy(&bytes);
    let was_lossy = matches!(&cow, Cow::Owned(_));
    let content = cow.into_owned();
    let total_lines = content.lines().count();

    // Numbering applies only to workspace-relative reads. An absolute input
    // is, by construction of `Workspace::resolve_read`, an offload readback,
    // whose stored text (already numbered when it came from `read_file`,
    // unnumbered when it came from `bash` / `list_files` / `run_checks`)
    // must come back exactly as stored.
    let number = numbered && !std::path::Path::new(path).is_absolute();

    // If neither slice arg was supplied, the raw variant returns the raw
    // content unchanged — this preserves any trailing newline. The numbered
    // variant enumerates `content.lines()` from 1 and joins with `\n`,
    // WITHOUT the trailing newline, because numbering is computed per line
    // (a post-hoc renumber of a joined string could not keep absolute
    // numbers).
    let has_slice = offset.is_some() || limit.is_some();
    let (detail, returned_lines) = if has_slice {
        let start = offset.unwrap_or(1);
        if start == 0 {
            return ToolResult::error("read_file: `offset` is 1-based; use offset >= 1");
        }
        let start_usize = usize::try_from(start).unwrap_or(usize::MAX);
        if total_lines > 0 && start_usize > total_lines {
            return ToolResult::error(format!(
                "read_file: offset {start} is past the end of `{path}` \
                 ({total_lines} lines)"
            ));
        }
        let skip = start_usize.saturating_sub(1);
        let take = limit.map_or(usize::MAX, |n| usize::try_from(n).unwrap_or(usize::MAX));
        let lines: Vec<&str> = content.lines().skip(skip).take(take).collect();
        let n = lines.len();
        (
            if number {
                numbered_join(skip, &lines)
            } else {
                lines.join("\n")
            },
            n,
        )
    } else if number {
        let lines: Vec<&str> = content.lines().collect();
        (numbered_join(0, &lines), total_lines)
    } else {
        (content, total_lines)
    };

    let mut summary = format!("{path}: {returned_lines} lines");
    if was_lossy {
        summary.push_str(" (non-UTF-8 bytes replaced with U+FFFD)");
    }

    // `with_detail` handles DETAIL_CAP + offload semantics uniformly, so
    // an over-cap read gets truncated inline and offloaded automatically.
    ToolResult::with_detail(summary, detail, ctx)
}

/// Prefix each of `lines` with its 1-based absolute line number — `skip + 1`
/// onward — right-aligned to width 6, followed by ONE literal TAB (U+0009)
/// and the line text without its terminator; join with `"\n"`, no trailing
/// newline.
fn numbered_join(skip: usize, lines: &[&str]) -> String {
    lines
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>6}\t{}", skip + i + 1, line))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract an optional `u64` field from a tool-call input object.
///
/// Missing and JSON `null` are treated identically (no value). Any other
/// non-integer type surfaces as a steering error the model can react to.
fn parse_optional_u64(input: &Value, field: &str) -> Result<Option<u64>, String> {
    match input.get(field).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("read_file: `{field}` must be a non-negative integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::{NumberedReadFileTool, READ_FILE_TOOL_NAME, ReadFileTool};
    use crate::tool::{DETAIL_CAP, Tool, ToolCtx};
    use crate::workspace::{DiskOffloadSink, Workspace};
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;

    /// A test harness: a real [`Workspace`] over a tempdir root and offload
    /// directory, plus a real [`DiskOffloadSink`] so oversized reads actually
    /// end up on disk and can be re-read.
    struct Harness {
        _root: TempDir,
        _offload: TempDir,
        ctx: ToolCtx,
        root_path: PathBuf,
        offload_path: PathBuf,
    }

    fn harness() -> Harness {
        let root = tempfile::tempdir().expect("root tempdir");
        let offload = tempfile::tempdir().expect("offload tempdir");
        let root_path = root.path().canonicalize().expect("canon root");
        let offload_path = offload.path().canonicalize().expect("canon offload");
        let ws = Workspace::new(&root_path, Some(offload_path.clone()))
            .expect("workspace with offload root");
        let sink = DiskOffloadSink::new(offload_path.clone());
        let ctx = ToolCtx::new(Arc::new(ws), Arc::new(sink));
        Harness {
            _root: root,
            _offload: offload,
            ctx,
            root_path,
            offload_path,
        }
    }

    #[test]
    fn tool_name_matches_constant() {
        assert_eq!(ReadFileTool.name(), READ_FILE_TOOL_NAME);
    }

    #[tokio::test]
    async fn reads_content_back_exactly() {
        let h = harness();
        std::fs::write(h.root_path.join("hello.txt"), "hello world\n").expect("write");

        let out = ReadFileTool.run(json!({"path": "hello.txt"}), &h.ctx).await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.summary, "hello.txt: 1 lines");
        // Raw content preserved — trailing newline included.
        assert_eq!(out.detail.as_deref(), Some("hello world\n"));
        assert!(out.offload_path.is_none());
    }

    #[tokio::test]
    async fn offset_and_limit_slice_one_based() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "l1\nl2\nl3\nl4\n").expect("write");

        let out = ReadFileTool
            .run(json!({"path": "f.txt", "offset": 2, "limit": 2}), &h.ctx)
            .await;

        assert!(!out.is_error);
        assert_eq!(out.summary, "f.txt: 2 lines");
        // 1-based offset: line 2 and line 3.
        assert_eq!(out.detail.as_deref(), Some("l2\nl3"));
    }

    #[tokio::test]
    async fn offset_only_returns_from_offset_to_end() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "a\nb\nc\n").expect("write");

        let out = ReadFileTool
            .run(json!({"path": "f.txt", "offset": 2}), &h.ctx)
            .await;

        assert!(!out.is_error);
        assert_eq!(out.summary, "f.txt: 2 lines");
        assert_eq!(out.detail.as_deref(), Some("b\nc"));
    }

    #[tokio::test]
    async fn limit_only_returns_from_start() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "a\nb\nc\n").expect("write");

        let out = ReadFileTool
            .run(json!({"path": "f.txt", "limit": 2}), &h.ctx)
            .await;

        assert!(!out.is_error);
        assert_eq!(out.summary, "f.txt: 2 lines");
        assert_eq!(out.detail.as_deref(), Some("a\nb"));
    }

    #[tokio::test]
    async fn null_offset_and_limit_are_treated_as_missing() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "hi").expect("write");

        let out = ReadFileTool
            .run(
                json!({"path": "f.txt", "offset": null, "limit": null}),
                &h.ctx,
            )
            .await;

        assert!(!out.is_error);
        // No slicing — raw content preserved.
        assert_eq!(out.detail.as_deref(), Some("hi"));
    }

    #[tokio::test]
    async fn offset_out_of_range_reports_line_count() {
        let h = harness();
        std::fs::write(h.root_path.join("small.txt"), "only\n").expect("write");

        let out = ReadFileTool
            .run(json!({"path": "small.txt", "offset": 10}), &h.ctx)
            .await;

        assert!(out.is_error);
        assert!(
            out.summary.contains("small.txt"),
            "names the file: {}",
            out.summary
        );
        assert!(
            out.summary.contains("1 lines"),
            "states the file's line count: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn zero_offset_is_steering_error() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "a\n").expect("write");

        let out = ReadFileTool
            .run(json!({"path": "f.txt", "offset": 0}), &h.ctx)
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("1-based"));
    }

    #[tokio::test]
    async fn missing_file_is_steering_error_naming_path() {
        let h = harness();

        let out = ReadFileTool
            .run(json!({"path": "no-such.txt"}), &h.ctx)
            .await;

        assert!(out.is_error);
        assert!(
            out.summary.contains("no-such.txt"),
            "names the path: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn parent_traversal_is_steering_error() {
        let h = harness();

        let out = ReadFileTool.run(json!({"path": "../escape"}), &h.ctx).await;

        assert!(out.is_error);
        assert!(out.summary.contains("../escape"));
        // The `PathViolation::ParentTraversal` display steers toward
        // "workspace root".
        assert!(
            out.summary.contains("workspace"),
            "steering message present: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn absolute_non_offload_path_is_steering_error() {
        let h = harness();

        let out = ReadFileTool
            .run(json!({"path": "/etc/hostname"}), &h.ctx)
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("/etc/hostname"));
        // The `PathViolation::Absolute` display steers toward "relative".
        assert!(
            out.summary.contains("relative"),
            "steering message present: {}",
            out.summary
        );
    }

    #[tokio::test]
    async fn absolute_path_under_offload_root_is_readable() {
        let h = harness();
        // Drop a real file under the offload root and read it back via an
        // absolute path — that is the offload-readback path.
        let offloaded = h.offload_path.join("offload-0000.txt");
        std::fs::write(&offloaded, "payload").expect("write offloaded file");

        let out = ReadFileTool
            .run(
                json!({"path": offloaded.to_str().expect("utf-8 path")}),
                &h.ctx,
            )
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.detail.as_deref(), Some("payload"));
    }

    #[tokio::test]
    async fn file_over_detail_cap_truncates_and_offloads() {
        let h = harness();
        let content = "a".repeat(DETAIL_CAP + 500);
        std::fs::write(h.root_path.join("big.txt"), &content).expect("write");

        let out = ReadFileTool.run(json!({"path": "big.txt"}), &h.ctx).await;

        assert!(!out.is_error);
        let detail = out.detail.expect("detail present");
        assert!(detail.contains("truncated"), "detail flagged truncated");
        let offload = out.offload_path.expect("offload path set");
        assert!(
            offload.starts_with(&h.offload_path),
            "offload path lives under the sink dir: {}",
            offload.display()
        );
        // The FULL content is readable back from the advertised offload path.
        assert_eq!(
            std::fs::read_to_string(&offload).expect("read offload"),
            content
        );
    }

    #[tokio::test]
    async fn non_utf8_bytes_are_lossy_and_noted_in_summary() {
        let h = harness();
        // 0xFF alone is not a valid UTF-8 sequence.
        std::fs::write(h.root_path.join("weird.bin"), [b'a', 0xFF, b'b']).expect("write");

        let out = ReadFileTool.run(json!({"path": "weird.bin"}), &h.ctx).await;

        assert!(!out.is_error);
        assert!(
            out.summary.contains("U+FFFD") || out.summary.contains("non-UTF"),
            "summary notes lossy decode: {}",
            out.summary
        );
        let detail = out.detail.expect("detail present");
        assert!(
            detail.contains('\u{FFFD}'),
            "detail contains replacement char"
        );
    }

    #[test]
    fn schema_is_parseable_json_and_names_all_three_properties() {
        let schema = ReadFileTool.schema();
        assert!(schema.is_object(), "schema is a JSON object");
        assert_eq!(schema["name"], READ_FILE_TOOL_NAME);

        // Round-trip through serde so we know the schema serializes cleanly.
        let text = serde_json::to_string(&schema).expect("serialize schema");
        let reparsed: serde_json::Value = serde_json::from_str(&text).expect("reparse schema");

        let props = &reparsed["input_schema"]["properties"];
        assert!(props["path"].is_object(), "path property present");
        assert!(props["offset"].is_object(), "offset property present");
        assert!(props["limit"].is_object(), "limit property present");

        let required = reparsed["input_schema"]["required"]
            .as_array()
            .expect("required is an array");
        assert!(required.iter().any(|v| v == "path"));

        // Description names both invariants the model needs to know.
        let desc = reparsed["description"]
            .as_str()
            .expect("description is a string")
            .to_lowercase();
        assert!(desc.contains("raw"), "description mentions RAW: {desc}");
        assert!(
            desc.contains("no line-number") || desc.contains("no line number"),
            "description mentions no line-number prefixes: {desc}"
        );
        assert!(
            desc.contains("offload"),
            "description mentions offload paths: {desc}"
        );
    }

    #[tokio::test]
    async fn missing_path_field_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = ReadFileTool.run(json!({}), &ctx).await;

        assert!(out.is_error);
        assert!(out.summary.contains("path"));
    }

    #[tokio::test]
    async fn non_string_path_field_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = ReadFileTool.run(json!({"path": 42}), &ctx).await;

        assert!(out.is_error);
        assert!(out.summary.contains("path"));
    }

    #[tokio::test]
    async fn non_integer_offset_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = ReadFileTool
            .run(json!({"path": "x", "offset": "two"}), &ctx)
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("offset"));
    }

    #[tokio::test]
    async fn non_integer_limit_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = ReadFileTool
            .run(json!({"path": "x", "limit": -1}), &ctx)
            .await;

        assert!(out.is_error);
        assert!(out.summary.contains("limit"));
    }

    #[tokio::test]
    async fn empty_file_reports_zero_lines() {
        let h = harness();
        std::fs::write(h.root_path.join("empty.txt"), "").expect("write");

        let out = ReadFileTool.run(json!({"path": "empty.txt"}), &h.ctx).await;

        assert!(!out.is_error);
        assert_eq!(out.summary, "empty.txt: 0 lines");
        assert_eq!(out.detail.as_deref(), Some(""));
    }

    // ------------------------------------------------------------------
    // `NumberedReadFileTool` — the answer-mode variant.
    // ------------------------------------------------------------------

    /// The 600-line × 99-'x' fixture's numbered lines for `n` in `range`.
    fn numbered_x_lines(range: std::ops::RangeInclusive<i64>) -> String {
        range
            .map(|n| format!("{:>6}\t{}", n, "x".repeat(99)))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn numbered_read_file_tool_name_matches_constant() {
        assert_eq!(NumberedReadFileTool.name(), READ_FILE_TOOL_NAME);
    }

    #[tokio::test]
    async fn numbered_whole_file_read_prefixes_each_line_with_absolute_number() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "alpha\nbeta\ngamma\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "f.txt"}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        // Width-6 right-aligned number, one TAB, line text — no trailing newline.
        assert_eq!(
            out.detail.as_deref(),
            Some("     1\talpha\n     2\tbeta\n     3\tgamma")
        );
        assert_eq!(out.summary, "f.txt: 3 lines");
    }

    #[tokio::test]
    async fn numbered_offset_and_limit_use_absolute_line_numbers() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "l1\nl2\nl3\nl4\nl5\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "f.txt", "offset": 3, "limit": 2}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.detail.as_deref(), Some("     3\tl3\n     4\tl4"));
        assert_eq!(out.summary, "f.txt: 2 lines");
    }

    #[tokio::test]
    async fn numbered_offset_only_starts_at_absolute_line() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "a\nb\nc\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "f.txt", "offset": 2}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.detail.as_deref(), Some("     2\tb\n     3\tc"));
    }

    #[tokio::test]
    async fn numbered_limit_only_starts_at_one() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "a\nb\nc\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "f.txt", "limit": 2}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.detail.as_deref(), Some("     1\ta\n     2\tb"));
    }

    #[tokio::test]
    async fn numbered_single_line_file() {
        let h = harness();
        std::fs::write(h.root_path.join("single.txt"), "only line\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "single.txt"}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.detail.as_deref(), Some("     1\tonly line"));
        assert_eq!(out.summary, "single.txt: 1 lines");
    }

    #[tokio::test]
    async fn numbered_empty_file_reports_zero_lines() {
        let h = harness();
        std::fs::write(h.root_path.join("empty.txt"), "").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "empty.txt"}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        assert_eq!(out.detail.as_deref(), Some(""));
        assert_eq!(out.summary, "empty.txt: 0 lines");
    }

    #[tokio::test]
    async fn numbered_crlf_file_yields_lines_without_carriage_return() {
        let h = harness();
        std::fs::write(
            h.root_path.join("f.txt"),
            [0x61, 0x0D, 0x0A, 0x62, 0x0D, 0x0A],
        )
        .expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "f.txt"}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        let detail = out.detail.as_deref().expect("detail present");
        assert_eq!(detail, "     1\ta\n     2\tb");
        assert!(!detail.contains('\r'), "no carriage returns: {detail:?}");
        assert_eq!(out.summary, "f.txt: 2 lines");
    }

    #[tokio::test]
    async fn numbered_offset_past_end_steering_error() {
        let h = harness();
        std::fs::write(h.root_path.join("small.txt"), "only\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "small.txt", "offset": 10}), &h.ctx)
            .await;

        assert!(out.is_error);
        assert_eq!(
            out.summary,
            "read_file: offset 10 is past the end of `small.txt` (1 lines)"
        );
    }

    #[tokio::test]
    async fn numbered_offset_zero_steering_error() {
        let h = harness();
        std::fs::write(h.root_path.join("f.txt"), "a\n").expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "f.txt", "offset": 0}), &h.ctx)
            .await;

        assert!(out.is_error);
        assert_eq!(
            out.summary,
            "read_file: `offset` is 1-based; use offset >= 1"
        );
    }

    #[tokio::test]
    async fn numbered_non_utf8_lossy_note_survives_in_summary() {
        let h = harness();
        // 0xFF alone is not a valid UTF-8 sequence.
        std::fs::write(h.root_path.join("weird.bin"), [b'a', 0xFF, b'b']).expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "weird.bin"}), &h.ctx)
            .await;

        assert!(!out.is_error);
        assert!(
            out.summary.contains("U+FFFD"),
            "summary notes lossy decode: {}",
            out.summary
        );
        assert!(
            out.detail
                .as_deref()
                .expect("detail present")
                .contains('\u{FFFD}'),
            "detail contains replacement char"
        );
    }

    #[tokio::test]
    async fn numbered_missing_path_field_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = NumberedReadFileTool.run(json!({}), &ctx).await;

        assert!(out.is_error);
        assert_eq!(
            out.summary,
            "read_file: missing required string field `path`"
        );
    }

    #[tokio::test]
    async fn numbered_non_string_path_field_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = NumberedReadFileTool.run(json!({"path": 42}), &ctx).await;

        assert!(out.is_error);
        assert_eq!(
            out.summary,
            "read_file: missing required string field `path`"
        );
    }

    #[tokio::test]
    async fn numbered_non_integer_offset_is_steering_error() {
        let ctx = ToolCtx::stub();
        let out = NumberedReadFileTool
            .run(json!({"path": "x", "offset": "two"}), &ctx)
            .await;

        assert!(out.is_error);
        assert_eq!(
            out.summary,
            "read_file: `offset` must be a non-negative integer"
        );
    }

    #[tokio::test]
    async fn numbered_file_over_detail_cap_truncates_and_offloads() {
        let h = harness();
        // 600 lines × 99 'x' + newline = 60,000 raw chars; the NUMBERED
        // whole-file detail is 64,199 chars — both over DETAIL_CAP.
        let line = "x".repeat(99);
        let raw = format!("{line}\n").repeat(600);
        std::fs::write(h.root_path.join("big2.txt"), &raw).expect("write");

        let out = NumberedReadFileTool
            .run(json!({"path": "big2.txt"}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        let detail = out.detail.expect("detail present");
        assert!(detail.contains("truncated"), "detail flagged truncated");
        let offload = out.offload_path.expect("offload path set");
        assert!(
            offload.starts_with(&h.offload_path),
            "offload path lives under the sink dir: {}",
            offload.display()
        );
        // The FULL numbered text (not the raw file, not a truncated copy)
        // is readable back from the advertised offload path.
        assert_eq!(
            std::fs::read_to_string(&offload).expect("read offload"),
            numbered_x_lines(1..=600)
        );
    }

    #[tokio::test]
    async fn numbered_offload_readback_is_not_renumbered() {
        let h = harness();
        let line = "x".repeat(99);
        let raw = format!("{line}\n").repeat(600);
        std::fs::write(h.root_path.join("big2.txt"), &raw).expect("write");

        let first = NumberedReadFileTool
            .run(json!({"path": "big2.txt"}), &h.ctx)
            .await;
        assert!(!first.is_error, "first read: {}", first.summary);
        let offload = first.offload_path.expect("over-cap read offloads");

        // Re-read the offload via its absolute path, sliced to lines 5..=7.
        let p = offload.to_str().expect("utf-8 path");
        let out = NumberedReadFileTool
            .run(json!({"path": p, "offset": 5, "limit": 3}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        // Each stored line carries its ORIGINAL absolute number exactly once —
        // no double `N\t` prefix, no renumbering from 1.
        assert_eq!(
            out.detail.as_deref(),
            Some(numbered_x_lines(5..=7).as_str())
        );
        assert_eq!(out.summary, format!("{p}: 3 lines"));
    }

    #[tokio::test]
    async fn numbered_absolute_readback_of_unnumbered_offload_is_asis() {
        let h = harness();
        // A bash/list_files/run_checks-origin offload is NOT pre-numbered.
        let offloaded = h.offload_path.join("offload-0000.txt");
        std::fs::write(&offloaded, "payload1\npayload2\n").expect("write offloaded file");

        let p = offloaded.to_str().expect("utf-8 path");
        let out = NumberedReadFileTool
            .run(json!({"path": p, "offset": 2}), &h.ctx)
            .await;

        assert!(!out.is_error, "unexpected error: {}", out.summary);
        let detail = out.detail.as_deref().expect("detail present");
        // Suppression keys on the INPUT, not on content or provenance.
        assert_eq!(detail, "payload2");
        assert!(!detail.contains('\t'), "no numbering prefix: {detail:?}");
    }

    #[test]
    fn numbered_description_states_prefix_not_in_file_and_offload_as_is() {
        let schema = NumberedReadFileTool.schema();
        assert_eq!(schema["name"], READ_FILE_TOOL_NAME);

        // input_schema byte-identical to the build-mode variant.
        assert_eq!(
            serde_json::to_string(&schema["input_schema"]).expect("serialize numbered"),
            serde_json::to_string(&ReadFileTool.schema()["input_schema"]).expect("serialize"),
            "input_schema identical"
        );

        let props = schema["input_schema"]["properties"]
            .as_object()
            .expect("properties is an object");
        let mut keys: Vec<&str> = props.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            vec!["limit", "offset", "path"],
            "exactly the three existing properties"
        );
        let required = schema["input_schema"]["required"]
            .as_array()
            .expect("required is an array");
        assert_eq!(required.len(), 1, "exactly one required property");
        assert_eq!(required[0], "path", "only `path` is required");

        // No per-call `numbered` parameter anywhere in the serialized schema.
        assert!(
            !serde_json::to_string(&schema)
                .expect("serialize schema")
                .contains("numbered"),
            "no `numbered` key in the schema"
        );

        // The description states the four invariants the model needs.
        let desc = schema["description"]
            .as_str()
            .expect("description is a string")
            .to_lowercase();
        for needle in [
            "n<tab>",
            "not part of the file",
            "exclude",
            "offset",
            "limit",
            "as-is",
        ] {
            assert!(
                desc.contains(needle),
                "description names {needle:?}: {desc}"
            );
        }
    }
}
