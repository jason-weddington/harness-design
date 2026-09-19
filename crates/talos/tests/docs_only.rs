#![cfg(unix)]
//! Integration tests for `scripts/docs-only.sh` — the fail-safe POSIX `sh`
//! predicate that lets the heavy lefthook gates skip themselves on a
//! docs-only changeset (`clippy`+`test` on `pre-commit`, the four heavy
//! gates on `pre-push`).
//!
//! Every test spawns the script with [`std::process::Command`], located via
//! `CARGO_MANIFEST_DIR` so the tests work from any workspace root. The
//! classifier is pure string logic (a path is "docs" iff it begins with
//! `docs/` or ends with `.md` and contains no `/`), so the explicit-list
//! cases below need no filesystem at all; the `--staged` and `--push` modes
//! are exercised against throwaway git repos built with [`tempfile::tempdir`].

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Command, Output};

/// The predicate script, resolved from this crate's manifest dir.
const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/docs-only.sh");

/// Runs the script with `args`, optionally in `cwd`, and returns its output.
fn run_script(args: &[&str], cwd: &Path) -> Output {
    Command::new(SCRIPT)
        .args(args)
        .env("HOME", cwd)
        .current_dir(cwd)
        .output()
        .expect("spawn docs-only.sh")
}

/// Runs the script through an explicit `sh` interpreter (what lefthook's
/// `skip:` resolves to in practice) and returns its output.
fn run_via_sh(args: &[&str], cwd: &Path) -> Output {
    Command::new("sh")
        .arg(SCRIPT)
        .args(args)
        .env("HOME", cwd)
        .current_dir(cwd)
        .output()
        .expect("spawn sh docs-only.sh")
}

/// Runs `git` inside `dir` with the isolation flags every spawn here uses.
fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "init.defaultBranch=main",
        ])
        .args(args)
        .env("HOME", dir)
        .current_dir(dir)
        .status()
        .expect("spawn git");
    assert!(status.success(), "git {args:?} failed: {status:?}");
}

/// Runs the script with `--staged` semantics against `repo`.
fn run_staged(repo: &Path) -> Output {
    run_script(&["--staged"], repo)
}

/// Runs the script with `--push` semantics against `repo`.
fn run_push(repo: &Path) -> Output {
    run_script(&["--push"], repo)
}

// ============================================================================
// (a) Explicit `-- <file>...` truth table (pure classifier, no filesystem)
// ============================================================================

#[test]
fn docs_only_explicit_lists() {
    let cwd = Path::new("/");

    // Every entry docs -> exit 0.
    for args in [
        vec!["--", "docs/roadmap.md", "docs/session-summaries.md"],
        vec!["--", "README.md"],
        vec![
            "--",
            "CHANGELOG.md",
            "CLAUDE.md",
            "docs/session-summaries.md",
        ],
    ] {
        let out = run_script(&args, cwd);
        assert_eq!(
            out.status.code(),
            Some(0),
            "expected skip for {args:?}: {:?}",
            out.stderr
        );
        assert!(out.stdout.is_empty(), "stdout must stay empty: {args:?}");
    }

    // Any non-docs entry -> exit 1, fail-safe.
    for args in [
        vec!["--", "docs/roadmap.md", "crates/harness/src/engine.rs"],
        vec!["--", "crates/harness/src/engine.rs"],
        vec!["--", "crates/talos/README.md"],
        vec!["--", "./docs/x.md"],
    ] {
        let out = run_script(&args, cwd);
        assert_eq!(out.status.code(), Some(1), "expected run for {args:?}");
    }
}

#[test]
fn docs_only_list_stderr_lines() {
    let cwd = Path::new("/");

    let out = run_script(
        &["--", "docs/roadmap.md", "crates/harness/src/engine.rs"],
        cwd,
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("reason=non-docs:crates/harness/src/engine.rs"),
        "{stderr:?}"
    );

    let out = run_script(&["--", "crates/talos/README.md"], cwd);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("reason=non-docs:crates/talos/README.md"),
        "{stderr:?}"
    );

    // `--` with zero files after the separator -> empty list.
    let out = run_script(&["--"], cwd);
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("reason=empty-list"), "{stderr:?}");

    // Skip line is pinned, exactly one line, on stderr only.
    let out = run_script(&["--", "README.md"], cwd);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("docs-only: skip mode=list n=1 paths=README.md"),
        "{stderr:?}"
    );
    assert!(out.stdout.is_empty());
}

// ============================================================================
// (b) Argument validation: exactly three invocation forms, nothing else
// ============================================================================

#[test]
fn docs_only_bad_args() {
    let cwd = Path::new("/");
    for args in [
        vec![],
        vec!["--bogus"],
        vec!["--staged", "extra"],
        vec!["--push", "extra"],
        // A bare file list with no leading `--` separator is not a form.
        vec!["README.md"],
        vec!["docs/roadmap.md"],
    ] {
        let out = run_script(&args, cwd);
        assert_eq!(out.status.code(), Some(1), "expected 1 for {args:?}");
        assert!(out.stdout.is_empty(), "stdout must stay empty: {args:?}");
    }

    let out = run_script(&["--bogus"], cwd);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("reason=bad-args"), "{stderr:?}");
}

// ============================================================================
// (c) `--staged`: git-derived list, `--no-renames` regression, fail-safes
// ============================================================================

#[test]
fn docs_only_staged_mode() {
    let dir = tempfile::tempdir().expect("tempdir");
    let repo = dir.path();
    git(repo, &["init", "-b", "main"]);
    git(repo, &["config", "user.email", "t@example.com"]);
    git(repo, &["config", "user.name", "t"]);

    std::fs::create_dir_all(repo.join("src")).expect("mkdir src");
    std::fs::write(repo.join("src/a.rs"), "fn main() {}\n").expect("write src/a.rs");
    git(repo, &["add", "src/a.rs"]);
    git(repo, &["commit", "-q", "-m", "chore: seed"]);

    // Nothing staged -> empty list -> exit 1.
    let out = run_staged(repo);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("reason=empty-list"));

    // Docs-only staged changeset -> exit 0.
    std::fs::create_dir_all(repo.join("docs")).expect("mkdir docs");
    std::fs::write(repo.join("docs/x.md"), "hello\n").expect("write docs/x.md");
    git(repo, &["add", "docs/x.md"]);
    let out = run_staged(repo);
    assert_eq!(out.status.code(), Some(0), "{:?}", out.stderr);

    // Mixed changeset: any code file re-enables the gates -> exit 1.
    std::fs::write(repo.join("src/a.rs"), "fn main() { /* t */ }\n").expect("touch src/a.rs");
    git(repo, &["add", "src/a.rs"]);
    let out = run_staged(repo);
    assert_eq!(out.status.code(), Some(1), "{:?}", out.stderr);

    // Rename regression: with `--no-renames` a staged `git mv src/a.rs
    // docs/a.md` lists BOTH paths, so the deleted Rust source keeps the
    // gates running. Without the flag rename detection would collapse the
    // pair to `docs/a.md` and a source deletion would look docs-only.
    git(repo, &["reset", "-q"]);
    git(repo, &["mv", "src/a.rs", "docs/a.md"]);
    let out = run_staged(repo);
    assert_eq!(out.status.code(), Some(1), "{:?}", out.stderr);
}

#[test]
fn docs_only_git_failed_when_not_a_repo() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = run_staged(dir.path());
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("reason=git-failed"), "{stderr:?}");
}

// ============================================================================
// (d) `--push`: upstream resolution, empty range, rename regression
// ============================================================================

#[test]
fn docs_only_push_mode() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path();
    let repo = root.join("repo");
    let remote = root.join("origin.git");
    std::fs::create_dir_all(&repo).expect("mkdir repo");
    std::fs::create_dir_all(&remote).expect("mkdir remote");
    git(&remote, &["init", "--bare", "origin.git"]);
    // `git init --bare <path>` inside an empty dir nests one level down; use
    // the path git actually created for the remote URL.
    let remote = remote.join("origin.git");
    assert!(remote.is_dir(), "bare remote not created at {remote:?}");

    git(&repo, &["init", "-b", "main"]);
    git(&repo, &["config", "user.email", "t@example.com"]);
    git(&repo, &["config", "user.name", "t"]);
    std::fs::create_dir_all(repo.join("src")).expect("mkdir src");
    std::fs::write(repo.join("src/a.rs"), "fn main() {}\n").expect("write src/a.rs");
    git(&repo, &["add", "src/a.rs"]);
    git(&repo, &["commit", "-q", "-m", "chore: seed"]);

    // No upstream configured yet -> exit 1 without touching anything else.
    let out = run_push(&repo);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("reason=no-upstream"));

    // First push configures the upstream; HEAD == upstream -> empty range.
    git(
        &repo,
        &["remote", "add", "origin", remote.to_str().expect("utf8")],
    );
    git(&repo, &["push", "-q", "-u", "origin", "main"]);
    let out = run_push(&repo);
    assert_eq!(out.status.code(), Some(1), "{:?}", out.stderr);

    // Docs-only commit on top -> exit 0.
    std::fs::create_dir_all(repo.join("docs")).expect("mkdir docs");
    std::fs::write(repo.join("docs/y.md"), "notes\n").expect("write docs/y.md");
    git(&repo, &["add", "docs/y.md"]);
    git(&repo, &["commit", "-q", "-m", "docs: y"]);
    let out = run_push(&repo);
    assert_eq!(out.status.code(), Some(0), "{:?}", out.stderr);

    // Commit touching a Rust source -> exit 1.
    std::fs::write(repo.join("src/a.rs"), "fn main() { /* touch */ }\n").expect("touch src/a.rs");
    git(&repo, &["add", "src/a.rs"]);
    git(&repo, &["commit", "-q", "-m", "fix: touch"]);
    let out = run_push(&repo);
    assert_eq!(out.status.code(), Some(1), "{:?}", out.stderr);

    // Rename regression on the committed range: `--no-renames` must expose
    // the deleted `src/a.rs` alongside the new `docs/a.md`.
    git(&repo, &["mv", "src/a.rs", "docs/a.md"]);
    git(&repo, &["commit", "-q", "-m", "docs: relocate"]);
    let out = run_push(&repo);
    assert_eq!(out.status.code(), Some(1), "{:?}", out.stderr);

    // Not a git repo at all -> git fails -> exit 1.
    let plain = tempfile::tempdir_in(root).expect("plain tempdir");
    let out = run_push(plain.path());
    assert_eq!(out.status.code(), Some(1));
}

// ============================================================================
// (e) Script hygiene: executable bit, explicit interpreter
// ============================================================================

#[test]
fn docs_only_script_is_executable() {
    let meta = std::fs::metadata(SCRIPT).expect("stat docs-only.sh");
    assert_ne!(meta.permissions().mode() & 0o111, 0, "no execute bit");
}

#[test]
fn docs_only_runs_under_explicit_sh() {
    // Same interpreter the hook resolves (`/usr/bin/env sh`), invoked
    // explicitly so the test does not depend on the exec bit alone.
    let out = run_via_sh(&["--", "README.md"], Path::new("/"));
    assert_eq!(out.status.code(), Some(0), "{:?}", out.stderr);
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .contains("docs-only: skip mode=list n=1 paths=README.md")
    );
}
