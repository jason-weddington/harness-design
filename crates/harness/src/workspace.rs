//! The run-scoped working-directory seam: a [`Workspace`] that OWNS path
//! resolution so tools never touch model-supplied path strings directly.
//!
//! Our safety model is registry-boundary + blast-radius bounds — there is no
//! separate permission layer. This module is the blast-radius bound for file
//! access: every file-touching tool asks the [`Workspace`] to turn a
//! model-supplied path string into a real, confined [`PathBuf`], and
//! confinement is enforced here, in one tested place. A resolution that would
//! escape the workspace is a [`PathViolation`] — a *steering* error the model
//! reacts to (converted to `ToolResult::error`), never a panic.
//!
//! Three pieces live here:
//! - [`Workspace`] — canonicalized root + optional offload root, with
//!   [`Workspace::resolve_write`] / [`Workspace::resolve_read`].
//! - [`PathViolation`] — the confinement-failure value type; its `Display` is a
//!   steering surface aimed at the model.
//! - [`DiskOffloadSink`] — the real [`OffloadSink`](crate::tool::OffloadSink)
//!   that persists oversized tool output to disk, closing the stub-sink TODO.
//!
//! Path semantics are Linux-only by design — deployment targets are Linux, so
//! there is no Windows path handling here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::tool::OffloadSink;

/// Placeholder path returned by [`DiskOffloadSink`] when the full output cannot
/// be persisted. Offloading is a best-effort safety net, so a failure surfaces
/// as this marker rather than an error or a panic.
const OFFLOAD_UNAVAILABLE: &str = "<offload-unavailable>";

/// Failure to construct a [`Workspace`]: a root that does not exist or is not a
/// directory. Raised only at construction time — once a `Workspace` exists its
/// roots are known-good, canonicalized directories.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceError {
    /// The workspace root does not exist or is not a directory.
    #[error("workspace root `{0}` does not exist or is not a directory")]
    RootNotADirectory(PathBuf),
    /// The offload root (when supplied) does not exist or is not a directory.
    #[error("offload root `{0}` does not exist or is not a directory")]
    OffloadRootNotADirectory(PathBuf),
}

/// A confinement failure when resolving a model-supplied path.
///
/// This is a value type whose `Display` is a *steering surface*: it names the
/// offending input and tells the model what to do instead. The contract for
/// tools is: convert a `PathViolation` into
/// `ToolResult::error(violation.to_string())` and hand it back to the model —
/// never panic, and never expose host filesystem details beyond the offending
/// input the model itself supplied.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PathViolation {
    /// An absolute path was supplied where only a workspace-relative path is
    /// allowed.
    #[error("path `{0}` is absolute; supply a path relative to the workspace root instead")]
    Absolute(String),
    /// The path escapes the workspace via `..` after lexical normalization.
    #[error(
        "path `{0}` escapes the workspace via `..`; supply a path relative to and contained within the workspace root"
    )]
    ParentTraversal(String),
    /// The path resolves outside the workspace (e.g. through a symlink pointing
    /// out of it).
    #[error(
        "path `{0}` resolves outside the workspace; tools may only touch files within the workspace root"
    )]
    EscapesRoot(String),
}

/// The run-scoped working directory. Owns path resolution and confinement.
///
/// Both roots are canonicalized at construction, so all containment checks
/// compare canonical, symlink-resolved prefixes.
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
    offload_root: Option<PathBuf>,
}

impl Workspace {
    /// Build a workspace rooted at `root`, optionally aware of an `offload_root`
    /// where oversized tool output is persisted.
    ///
    /// Both roots are canonicalized. Construction fails if `root` (or
    /// `offload_root`, when `Some`) does not exist or is not a directory.
    pub fn new(
        root: impl Into<PathBuf>,
        offload_root: Option<PathBuf>,
    ) -> Result<Self, WorkspaceError> {
        let root = root.into();
        let root = canonicalize_dir(&root).ok_or(WorkspaceError::RootNotADirectory(root))?;
        let offload_root = match offload_root {
            Some(dir) => {
                Some(canonicalize_dir(&dir).ok_or(WorkspaceError::OffloadRootNotADirectory(dir))?)
            }
            None => None,
        };
        Ok(Self { root, offload_root })
    }

    /// The canonicalized workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a model-supplied path for a tool that creates or modifies a file.
    ///
    /// Rules: reject absolute paths; reject any `..` that survives lexical
    /// normalization; join the normalized path to the workspace root; and
    /// require the deepest existing ancestor of the result to canonicalize to a
    /// location inside the root (catching symlinks that point out of the
    /// workspace while still allowing not-yet-existing paths). The offload root
    /// is never writable through this method.
    pub fn resolve_write(&self, raw: &str) -> Result<PathBuf, PathViolation> {
        self.resolve_confined(raw)
    }

    /// Resolve a model-supplied path for a tool that reads a file.
    ///
    /// Identical to [`Workspace::resolve_write`] with one addition: an absolute
    /// path is accepted *only* when its deepest existing ancestor canonicalizes
    /// inside the offload root — offloaded tool outputs are advertised to the
    /// model as absolute paths and must be re-readable. Absolute paths anywhere
    /// else are rejected.
    pub fn resolve_read(&self, raw: &str) -> Result<PathBuf, PathViolation> {
        let path = Path::new(raw);
        if path.is_absolute() {
            if let Some(offload_root) = &self.offload_root {
                let ancestor = deepest_existing(path);
                if let Ok(canon) = ancestor.canonicalize()
                    && canon.starts_with(offload_root)
                {
                    return Ok(path.to_path_buf());
                }
            }
            return Err(PathViolation::Absolute(raw.to_string()));
        }
        self.resolve_confined(raw)
    }

    /// Shared confinement rules for workspace-relative paths.
    fn resolve_confined(&self, raw: &str) -> Result<PathBuf, PathViolation> {
        if Path::new(raw).is_absolute() {
            return Err(PathViolation::Absolute(raw.to_string()));
        }

        // Lexical normalization on `/`-separated segments (Linux-only). Drops
        // empty and `.` segments; cancels a `..` against a preceding real
        // segment; keeps any `..` that cannot be cancelled so it can be
        // rejected below.
        let mut segments: Vec<&str> = Vec::new();
        for segment in raw.split('/') {
            match segment {
                "" | "." => {}
                ".." => match segments.last() {
                    Some(&last) if last != ".." => {
                        segments.pop();
                    }
                    _ => segments.push(".."),
                },
                other => segments.push(other),
            }
        }
        if segments.contains(&"..") {
            return Err(PathViolation::ParentTraversal(raw.to_string()));
        }

        let joined = self.root.join(segments.into_iter().collect::<PathBuf>());

        // Symlink containment: the deepest existing ancestor must canonicalize
        // to a location inside the root. This allows not-yet-existing paths
        // while catching a symlink inside the workspace that points outside it.
        let ancestor = deepest_existing(&joined);
        if let Ok(canon) = ancestor.canonicalize()
            && canon.starts_with(&self.root)
        {
            return Ok(joined);
        }
        Err(PathViolation::EscapesRoot(raw.to_string()))
    }
}

/// Canonicalize `path`, returning it only if it is an existing directory.
fn canonicalize_dir(path: &Path) -> Option<PathBuf> {
    let canon = path.canonicalize().ok()?;
    canon.is_dir().then_some(canon)
}

/// The deepest ancestor of `path` that exists on disk (possibly `path` itself),
/// falling back to `path` when nothing along the chain exists.
fn deepest_existing(path: &Path) -> &Path {
    path.ancestors().find(|p| p.exists()).unwrap_or(path)
}

/// The real [`OffloadSink`]: persists each oversized tool output to a fresh,
/// uniquely-named file under a run-scoped directory and returns its absolute
/// path for the model to re-read.
///
/// Honors the trait's infallible-from-caller contract: offloading is a
/// best-effort safety net, not a failure surface, so any IO error yields the
/// [`OFFLOAD_UNAVAILABLE`] placeholder path instead of panicking or erroring.
#[derive(Debug)]
pub struct DiskOffloadSink {
    dir: PathBuf,
    counter: AtomicU64,
}

impl DiskOffloadSink {
    /// Build a sink that writes offloaded output under `dir`. The directory is
    /// created on first write if it does not already exist.
    #[must_use]
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            counter: AtomicU64::new(0),
        }
    }

    /// Fallible inner write, mapped to the placeholder path by [`Self::offload`].
    fn try_offload(&self, contents: &str) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)?;
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let path = self.dir.join(format!("offload-{n:04}.txt"));
        std::fs::write(&path, contents)?;
        Ok(path)
    }
}

impl OffloadSink for DiskOffloadSink {
    fn offload(&self, contents: &str) -> PathBuf {
        self.try_offload(contents)
            .unwrap_or_else(|_| PathBuf::from(OFFLOAD_UNAVAILABLE))
    }
}

#[cfg(test)]
mod tests {
    use super::{DiskOffloadSink, OFFLOAD_UNAVAILABLE, PathViolation, Workspace, deepest_existing};
    use crate::tool::{DETAIL_CAP, OffloadSink, ToolCtx, ToolResult};
    use std::path::Path;
    use std::sync::Arc;
    use tempfile::tempdir;

    #[test]
    fn new_canonicalizes_and_rejects_nonexistent_root() {
        let dir = tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path(), None).expect("valid root");
        // Root is canonicalized (equals the canonical form of the temp dir).
        assert_eq!(ws.root(), dir.path().canonicalize().expect("canon"));

        let missing = dir.path().join("does-not-exist");
        let err = Workspace::new(&missing, None).expect_err("missing root rejected");
        assert!(matches!(err, super::WorkspaceError::RootNotADirectory(_)));
    }

    #[test]
    fn new_rejects_file_as_root() {
        let dir = tempdir().expect("tempdir");
        let file = dir.path().join("a-file.txt");
        std::fs::write(&file, "x").expect("write file");
        let err = Workspace::new(&file, None).expect_err("file root rejected");
        assert!(matches!(err, super::WorkspaceError::RootNotADirectory(_)));
    }

    #[test]
    fn new_rejects_nonexistent_offload_root() {
        let dir = tempdir().expect("tempdir");
        let missing = dir.path().join("no-offload");
        let err =
            Workspace::new(dir.path(), Some(missing)).expect_err("missing offload root rejected");
        assert!(matches!(
            err,
            super::WorkspaceError::OffloadRootNotADirectory(_)
        ));
    }

    #[test]
    fn resolve_write_accepts_relative_nested_and_not_yet_existing() {
        let dir = tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path(), None).expect("valid root");

        for raw in ["file.txt", "sub/dir/new.rs", "a/./b", "a/b/../c"] {
            let resolved = ws.resolve_write(raw).expect("relative path resolves");
            assert!(
                resolved.starts_with(ws.root()),
                "{raw} should resolve under root"
            );
        }
        // Sanity: `.`-collapsing and `..`-cancelling behave lexically.
        assert_eq!(ws.resolve_write("a/./b").unwrap(), ws.root().join("a/b"));
        assert_eq!(ws.resolve_write("a/b/../c").unwrap(), ws.root().join("a/c"));
    }

    #[test]
    fn resolve_write_rejects_absolute() {
        let dir = tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path(), None).expect("valid root");
        let err = ws
            .resolve_write("/etc/passwd")
            .expect_err("absolute rejected");
        assert!(matches!(err, PathViolation::Absolute(_)));
    }

    #[test]
    fn resolve_write_rejects_parent_traversal_including_sneaky_forms() {
        let dir = tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path(), None).expect("valid root");
        for raw in ["../escape", "a/../../b", "../../etc/passwd", ".."] {
            let err = ws
                .resolve_write(raw)
                .expect_err("parent traversal rejected");
            assert!(
                matches!(err, PathViolation::ParentTraversal(_)),
                "{raw} should be a ParentTraversal"
            );
        }
    }

    #[test]
    fn resolve_rejects_symlink_pointing_outside_workspace() {
        let root_dir = tempdir().expect("root tempdir");
        let outside_dir = tempdir().expect("outside tempdir");
        let ws = Workspace::new(root_dir.path(), None).expect("valid root");

        // A symlink INSIDE the workspace that points OUTSIDE it.
        let link = root_dir.path().join("escape-link");
        std::os::unix::fs::symlink(outside_dir.path(), &link).expect("symlink");

        let write_err = ws
            .resolve_write("escape-link/evil.txt")
            .expect_err("write through escaping symlink rejected");
        assert!(matches!(write_err, PathViolation::EscapesRoot(_)));

        let read_err = ws
            .resolve_read("escape-link/evil.txt")
            .expect_err("read through escaping symlink rejected");
        assert!(matches!(read_err, PathViolation::EscapesRoot(_)));
    }

    #[test]
    fn resolve_read_accepts_absolute_under_offload_root_only() {
        let root_dir = tempdir().expect("root tempdir");
        let offload_dir = tempdir().expect("offload tempdir");
        let offload_canon = offload_dir.path().canonicalize().expect("canon offload");
        let ws = Workspace::new(root_dir.path(), Some(offload_dir.path().to_path_buf()))
            .expect("valid roots");

        // An absolute path to a real file under the offload root is accepted.
        let offloaded = offload_canon.join("offload-0000.txt");
        std::fs::write(&offloaded, "payload").expect("write offloaded file");
        let resolved = ws
            .resolve_read(offloaded.to_str().expect("utf8"))
            .expect("offload path is readable");
        assert_eq!(resolved, offloaded);

        // An absolute path outside the offload root is rejected.
        let err = ws
            .resolve_read("/etc/hostname")
            .expect_err("absolute outside offload rejected");
        assert!(matches!(err, PathViolation::Absolute(_)));
    }

    #[test]
    fn resolve_read_rejects_absolute_when_no_offload_root() {
        let dir = tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path(), None).expect("valid root");
        let err = ws
            .resolve_read("/etc/hostname")
            .expect_err("absolute rejected without offload root");
        assert!(matches!(err, PathViolation::Absolute(_)));
    }

    #[test]
    fn resolve_read_accepts_relative_paths_like_write() {
        let dir = tempdir().expect("tempdir");
        let ws = Workspace::new(dir.path(), None).expect("valid root");
        let resolved = ws.resolve_read("sub/file.txt").expect("relative read");
        assert!(resolved.starts_with(ws.root()));
    }

    #[test]
    fn path_violation_display_names_path_and_steers() {
        let v = PathViolation::Absolute("/etc/passwd".to_string());
        let msg = v.to_string();
        assert!(msg.contains("/etc/passwd"), "names the offending path");
        assert!(msg.contains("relative"), "steers toward a relative path");

        let v = PathViolation::ParentTraversal("../x".to_string());
        assert!(v.to_string().contains("../x"));
        assert!(v.to_string().contains("workspace root"));

        let v = PathViolation::EscapesRoot("link/x".to_string());
        assert!(v.to_string().contains("link/x"));
        assert!(v.to_string().contains("outside"));
    }

    #[test]
    fn deepest_existing_falls_back_when_nothing_exists() {
        // A relative path whose components do not exist relative to the cwd
        // exhausts its ancestors and falls back to the input path.
        let p = Path::new("zzz-no-such-harness-path/child");
        let got = deepest_existing(p);
        // The empty-string ancestor does not exist, so we fall back to `p`.
        assert!(got == p || got == Path::new("zzz-no-such-harness-path"));
    }

    #[test]
    fn disk_sink_writes_readable_content_and_unique_paths() {
        let dir = tempdir().expect("tempdir");
        // Point the sink at a not-yet-existing subdir to exercise create-on-first-write.
        let sink = DiskOffloadSink::new(dir.path().join("offloads"));

        let p1 = sink.offload("first payload");
        let p2 = sink.offload("second payload");
        assert_ne!(p1, p2, "successive offloads get unique paths");
        assert_eq!(std::fs::read_to_string(&p1).unwrap(), "first payload");
        assert_eq!(std::fs::read_to_string(&p2).unwrap(), "second payload");
    }

    #[test]
    fn disk_sink_returns_placeholder_when_dir_cannot_be_created() {
        let dir = tempdir().expect("tempdir");
        // Parent is a FILE, so create_dir_all under it must fail.
        let file = dir.path().join("not-a-dir");
        std::fs::write(&file, "x").expect("write file");
        let sink = DiskOffloadSink::new(file.join("offloads"));
        assert_eq!(sink.offload("data"), Path::new(OFFLOAD_UNAVAILABLE));
    }

    #[test]
    fn disk_sink_returns_placeholder_when_write_fails() {
        let dir = tempdir().expect("tempdir");
        let sink_dir = dir.path().join("offloads");
        std::fs::create_dir_all(&sink_dir).expect("mkdir");
        // Pre-create a DIRECTORY where the first offload file would be written,
        // so the write itself fails while the dir exists.
        std::fs::create_dir(sink_dir.join("offload-0000.txt")).expect("mkdir clash");
        let sink = DiskOffloadSink::new(sink_dir);
        assert_eq!(sink.offload("data"), Path::new(OFFLOAD_UNAVAILABLE));
    }

    #[test]
    fn end_to_end_offload_over_cap_is_readable_back_through_resolve_read() {
        let root_dir = tempdir().expect("root tempdir");
        let offload_dir = tempdir().expect("offload tempdir");
        let offload_canon = offload_dir.path().canonicalize().expect("canon offload");

        let ws = Workspace::new(root_dir.path(), Some(offload_dir.path().to_path_buf()))
            .expect("valid roots");
        let sink = DiskOffloadSink::new(offload_canon);
        let ctx = ToolCtx::new(Arc::new(ws), Arc::new(sink));

        // Detail over the cap forces an offload through the real disk sink.
        let full = "z".repeat(DETAIL_CAP + 1_000);
        let result = ToolResult::with_detail("summary", full.clone(), &ctx);
        let offload_path = result.offload_path.expect("offloaded");

        // The advertised absolute path is re-readable via resolve_read...
        let resolved = ctx
            .workspace()
            .resolve_read(offload_path.to_str().expect("utf8"))
            .expect("offload path resolves for reading");
        // ...and its contents are the FULL, untruncated detail.
        assert_eq!(std::fs::read_to_string(&resolved).unwrap(), full);
    }
}
