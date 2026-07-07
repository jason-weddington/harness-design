//! The `list_files` tool: a bounded, deterministic directory listing over the
//! run-scoped [`Workspace`](crate::workspace::Workspace).
//!
//! It consumes the workspace seam for confinement — every model-supplied path
//! is resolved through [`Workspace::resolve_read`](crate::workspace::Workspace::resolve_read),
//! so a path that escapes the workspace becomes a *steering*
//! [`ToolResult::error`], never a panic or a host-filesystem leak.
//!
//! Design constraints (see `docs/design/01`):
//! - **Deterministic output.** Entries are sorted lexicographically per
//!   directory and rendered as workspace-relative paths, one per line, with
//!   directories suffixed `/`. Two identical calls produce byte-identical
//!   detail so the prompt cache hits.
//! - **Bounded.** The recursive walk stops at a `depth` cap (default 2, hard
//!   max 5, clamped silently).
//! - **Safe by construction.** `.git` directories are always skipped and
//!   symlinked directories are never descended (no cycle risk), because the
//!   walk inspects `symlink_metadata` and treats a symlink as a leaf.

use std::fs;
use std::path::Path;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::tool::{Tool, ToolCtx, ToolResult};

/// Default recursion depth when the caller omits `depth`.
const DEFAULT_DEPTH: u64 = 2;
/// Hard cap on recursion depth; larger requests are clamped silently.
const MAX_DEPTH: u64 = 5;

/// Lists directory contents under the workspace, recursively to a depth cap.
#[derive(Debug, Default, Clone, Copy)]
pub struct ListFilesTool;

#[async_trait]
impl Tool for ListFilesTool {
    // The trait fixes the return type as `&str`; a `&'static str` would diverge
    // from the trait signature, so the lint doesn't apply.
    #[allow(clippy::unnecessary_literal_bound)]
    fn name(&self) -> &str {
        "list_files"
    }

    fn schema(&self) -> Value {
        json!({
            "name": "list_files",
            "description": concat!(
                "List directory contents under the workspace, recursively to a depth cap. ",
                "Entries are workspace-relative, sorted lexicographically per directory, ",
                "one per line, with directories suffixed `/`. `.git` directories are skipped ",
                "and symlinked directories are not descended."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory to list. Defaults to \".\" (the workspace root).",
                        "default": "."
                    },
                    "depth": {
                        "type": "integer",
                        "description": "How many directory levels to descend. Defaults to 2; clamped to a hard maximum of 5.",
                        "default": DEFAULT_DEPTH,
                        "minimum": 1,
                        "maximum": MAX_DEPTH
                    }
                }
            }
        })
    }

    async fn run(&self, input: Value, ctx: &ToolCtx) -> ToolResult {
        let path = input.get("path").and_then(Value::as_str).unwrap_or(".");
        let depth = input
            .get("depth")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_DEPTH)
            .clamp(1, MAX_DEPTH);

        // Confinement: resolve through the workspace. An escaping path becomes a
        // steering error the model reacts to.
        let resolved = match ctx.workspace().resolve_read(path) {
            Ok(resolved) => resolved,
            Err(violation) => return ToolResult::error(violation.to_string()),
        };

        // The target must be an existing directory. `metadata` follows the
        // top-level path (a caller may legitimately list through a directory
        // symlink); a missing path or a file is a steering error.
        match fs::metadata(&resolved) {
            Ok(meta) if meta.is_dir() => {}
            Ok(_) => {
                return ToolResult::error(format!(
                    "path `{path}` is not a directory; list_files only lists directories"
                ));
            }
            Err(_) => {
                return ToolResult::error(format!(
                    "path `{path}` does not exist in the workspace; supply an existing directory"
                ));
            }
        }

        let root = ctx.workspace().root();
        let mut entries = Vec::new();
        walk(&resolved, root, depth, &mut entries);

        let listing = entries.join("\n");
        let summary = format!("{} entries under {path} (depth {depth})", entries.len());
        ToolResult::with_detail(summary, listing, ctx)
    }
}

/// Recursively collect entries of `dir` into `out`, rendered relative to
/// `root`. `remaining_depth` is the number of directory levels still allowed
/// (`>= 1` on the initial call). Entries are emitted in a deterministic
/// pre-order: each directory's children are sorted lexicographically, and a
/// subdirectory's contents follow immediately after the directory line.
///
/// `.git` directories are skipped entirely. Symlinks are treated as leaves —
/// `symlink_metadata` reports the link itself (not its target), so a symlinked
/// directory is rendered as a plain entry and never descended, avoiding cycles.
fn walk(dir: &Path, root: &Path, remaining_depth: u64, out: &mut Vec<String>) {
    let Ok(read) = fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<_> = read.filter_map(Result::ok).collect();
    children.sort_by_key(std::fs::DirEntry::file_name);

    for entry in children {
        let name = entry.file_name();
        // Always skip `.git` — its contents are never useful listing output.
        if name == ".git" {
            continue;
        }

        let path = entry.path();
        let Ok(meta) = fs::symlink_metadata(&path) else {
            continue;
        };

        // Render relative to the workspace root; the join guarantees the prefix
        // is present, so the fallback is unreachable in practice.
        let rel = path.strip_prefix(root).unwrap_or(&path);
        let rel = rel.to_string_lossy();

        // `is_dir` is false for a symlink under `symlink_metadata`, so symlinked
        // directories fall into the leaf branch: listed, but never descended.
        if meta.is_dir() {
            out.push(format!("{rel}/"));
            if remaining_depth > 1 {
                walk(&path, root, remaining_depth - 1, out);
            }
        } else {
            out.push(rel.into_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ListFilesTool;
    use crate::tool::{StubOffloadSink, Tool, ToolCtx};
    use crate::workspace::Workspace;
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use tempfile::{TempDir, tempdir};

    /// A [`ToolCtx`] rooted at `root` with the no-op offload sink.
    fn ctx_for(root: &Path) -> ToolCtx {
        let ws = Workspace::new(root, None).expect("valid workspace root");
        ToolCtx::new(Arc::new(ws), Arc::new(StubOffloadSink))
    }

    /// Build a small nested tree:
    /// ```text
    /// a.txt
    /// b/
    ///   c.txt
    ///   d/
    ///     e.txt
    /// ```
    fn nested_tree() -> TempDir {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::write(root.join("a.txt"), "a").expect("write a");
        fs::create_dir(root.join("b")).expect("mkdir b");
        fs::write(root.join("b/c.txt"), "c").expect("write c");
        fs::create_dir(root.join("b/d")).expect("mkdir d");
        fs::write(root.join("b/d/e.txt"), "e").expect("write e");
        dir
    }

    #[tokio::test]
    async fn lists_nested_entries_in_sorted_order() {
        let dir = nested_tree();
        let ctx = ctx_for(dir.path());
        let out = ListFilesTool.run(json!({"depth": 2}), &ctx).await;

        assert!(!out.is_error);
        // depth 2: children (a.txt, b/) + grandchildren (b/c.txt, b/d/), but not
        // the depth-3 b/d/e.txt. Pre-order, per-directory lexicographic sort.
        assert_eq!(out.detail.as_deref(), Some("a.txt\nb/\nb/c.txt\nb/d/"));
        assert_eq!(out.summary, "4 entries under . (depth 2)");
    }

    #[tokio::test]
    async fn depth_cap_excludes_deeper_entries() {
        let dir = nested_tree();
        let ctx = ctx_for(dir.path());
        // depth 1: only the immediate children of the root.
        let out = ListFilesTool.run(json!({"depth": 1}), &ctx).await;

        assert!(!out.is_error);
        assert_eq!(out.detail.as_deref(), Some("a.txt\nb/"));
        assert_eq!(out.summary, "2 entries under . (depth 1)");
    }

    #[tokio::test]
    async fn depth_over_max_is_clamped_silently() {
        // Six nested directory levels, each holding the next.
        let dir = tempdir().expect("tempdir");
        let deep = dir.path().join("l1/l2/l3/l4/l5/l6");
        fs::create_dir_all(&deep).expect("mkdir deep");
        fs::write(deep.join("leaf.txt"), "x").expect("write leaf");

        let ctx = ctx_for(dir.path());
        let out = ListFilesTool.run(json!({"depth": 100}), &ctx).await;

        assert!(!out.is_error);
        // Clamped to depth 5: l1..l5 present, l6 (level 6) and its leaf absent.
        let detail = out.detail.expect("detail present");
        assert!(detail.contains("l1/l2/l3/l4/l5/"), "level 5 dir present");
        assert!(
            !detail.contains("l6"),
            "level 6 must be excluded by the clamp"
        );
        assert!(
            !detail.contains("leaf.txt"),
            "the deep leaf must be excluded"
        );
        // The summary reports the clamped depth, not the requested 100.
        assert_eq!(out.summary, "5 entries under . (depth 5)");
    }

    #[tokio::test]
    async fn default_depth_is_two_when_omitted() {
        let dir = nested_tree();
        let ctx = ctx_for(dir.path());
        let out = ListFilesTool.run(json!({}), &ctx).await;

        assert!(!out.is_error);
        // Same as an explicit depth of 2.
        assert_eq!(out.detail.as_deref(), Some("a.txt\nb/\nb/c.txt\nb/d/"));
        assert_eq!(out.summary, "4 entries under . (depth 2)");
    }

    #[tokio::test]
    async fn git_directory_contents_are_excluded() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir(root.join(".git")).expect("mkdir .git");
        fs::write(root.join(".git/config"), "secret").expect("write git config");
        fs::write(root.join("real.txt"), "r").expect("write real");

        let ctx = ctx_for(root);
        let out = ListFilesTool.run(json!({"depth": 3}), &ctx).await;

        assert!(!out.is_error);
        let detail = out.detail.expect("detail present");
        assert!(detail.contains("real.txt"), "non-git entries listed");
        assert!(!detail.contains(".git"), "no .git entry at all");
        assert!(!detail.contains("config"), "no .git contents");
    }

    #[tokio::test]
    async fn symlinked_directory_is_not_descended() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir(root.join("target_dir")).expect("mkdir target");
        fs::write(root.join("target_dir/inside.txt"), "i").expect("write inside");
        std::os::unix::fs::symlink(root.join("target_dir"), root.join("link")).expect("symlink");

        let ctx = ctx_for(root);
        let out = ListFilesTool.run(json!({"depth": 3}), &ctx).await;

        assert!(!out.is_error);
        let detail = out.detail.expect("detail present");
        // The real directory is descended...
        assert!(
            detail.contains("target_dir/inside.txt"),
            "real dir descended"
        );
        // ...but the symlink is a leaf entry, never followed.
        assert!(detail.contains("link"), "symlink listed as an entry");
        assert!(
            !detail.contains("link/inside.txt"),
            "symlinked dir must not be descended"
        );
    }

    #[tokio::test]
    async fn missing_directory_is_a_steering_error() {
        let dir = tempdir().expect("tempdir");
        let ctx = ctx_for(dir.path());
        let out = ListFilesTool.run(json!({"path": "nope"}), &ctx).await;

        assert!(out.is_error);
        assert!(out.summary.contains("nope"));
        assert!(out.detail.is_none());
    }

    #[tokio::test]
    async fn file_target_is_a_steering_error() {
        let dir = tempdir().expect("tempdir");
        fs::write(dir.path().join("f.txt"), "f").expect("write file");
        let ctx = ctx_for(dir.path());
        let out = ListFilesTool.run(json!({"path": "f.txt"}), &ctx).await;

        assert!(out.is_error);
        assert!(out.summary.contains("not a directory"));
    }

    #[tokio::test]
    async fn escape_attempt_is_a_steering_error() {
        let dir = tempdir().expect("tempdir");
        let ctx = ctx_for(dir.path());
        let out = ListFilesTool.run(json!({"path": "../outside"}), &ctx).await;

        assert!(out.is_error);
        // The PathViolation Display steers toward a workspace-relative path.
        assert!(out.summary.contains("../outside"));
        assert!(out.summary.contains("workspace"));
    }

    #[tokio::test]
    async fn output_is_deterministic_across_calls() {
        let dir = nested_tree();
        let ctx = ctx_for(dir.path());

        let first = ListFilesTool.run(json!({"depth": 3}), &ctx).await;
        let second = ListFilesTool.run(json!({"depth": 3}), &ctx).await;

        assert!(!first.is_error && !second.is_error);
        assert_eq!(first.detail, second.detail, "detail is byte-equal");
        assert_eq!(first.summary, second.summary);
    }

    #[test]
    fn name_and_schema_are_well_formed() {
        let tool = ListFilesTool;
        assert_eq!(tool.name(), "list_files");
        let schema = tool.schema();
        assert_eq!(schema["name"], "list_files");
        assert_eq!(schema["input_schema"]["type"], "object");
        // Re-parse the serialized form to confirm it is valid JSON.
        let text = serde_json::to_string(&schema).expect("serialize schema");
        let reparsed: serde_json::Value = serde_json::from_str(&text).expect("reparse schema");
        assert_eq!(
            reparsed["input_schema"]["properties"]["depth"]["maximum"],
            5
        );
    }
}
