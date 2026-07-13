//! Build script: stamp the git version (nearest tag + distance + sha) into
//! `talos --version`.
//!
//! The dispatch fleet publishes talos binaries to `pi-04:/srv/talos/<TOKEN>/…`
//! and reads the target back via `talos --version | awk '{print $2}'`. That
//! `<TOKEN>` must be a single URL/path-safe string. `git describe --tags` yields
//! exactly that: the release tag on a tagged commit (e.g. `0.5.0`), or
//! `<tag>-<n>-g<short-sha>` on a commit past the tag (path-safe — dots and
//! hyphens, never `+`).
//!
//! We describe the TAG rather than `CARGO_PKG_VERSION` because releases bump the
//! git tag via `cog`, NOT the crate `version` (which stays `0.1.0`). Stamping the
//! crate version reported a meaningless `0.1.0-g<sha>`; the tag reconciles the
//! binary with the CHANGELOG/tags/KB. See `scripts/publish-talos.sh`.

use std::process::Command;

fn main() {
    // `--always` falls back to a bare short-sha if no tag is reachable (fresh
    // repo); `--dirty` marks an uncommitted tree (dev builds only — publish
    // requires a clean tree, so a published token is never `-dirty`).
    let version = Command::new("git")
        .args(["describe", "--tags", "--always", "--dirty"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=TALOS_VERSION={version}");

    // Re-stamp when HEAD moves OR a tag is added (new commit / checkout / tag)
    // so the version stays fresh. Paths are relative to this crate's root
    // (crates/talos); the workspace `.git` lives two levels up. Missing in a
    // non-git build (tarball) — `describe` fails, the version falls back to
    // `unknown`, and these just no-op.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
    println!("cargo:rerun-if-changed=../../.git/refs/tags");
}
