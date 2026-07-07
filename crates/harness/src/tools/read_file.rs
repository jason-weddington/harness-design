//! `read_file` — read a workspace-relative file's raw contents.
//!
//! Content is returned **without** line-number prefixes. That is deliberate:
//! [`edit_file`](crate::tools) uses exact-match string replacement, and models
//! copy `old_string` verbatim out of a prior `read_file` output. A "1: foo\n2:
//! bar\n" prefix would poison the match by making the copied substring not
//! actually exist in the file — the model would have to strip prefixes itself,
//! which is exactly the kind of silent formatting drift we want to avoid at
//! the tool boundary.
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
            "input_schema": {
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
            }
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
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

        // If neither slice arg was supplied, return the raw content unchanged
        // — this preserves any trailing newline. Otherwise slice line-wise
        // and rejoin with `\n`.
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
            (lines.join("\n"), n)
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
    use super::{READ_FILE_TOOL_NAME, ReadFileTool};
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
}
