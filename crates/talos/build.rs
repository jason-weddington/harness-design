//! Build script: stamp the git short SHA into `talos --version`.
//!
//! The dispatch fleet publishes talos binaries to `pi-04:/srv/talos/<TOKEN>/…`
//! and reads the target back via `talos --version | awk '{print $2}'`. That
//! `<TOKEN>` must be a single URL/path-safe string embedding the commit, so we
//! emit `<semver>-g<short-sha>` (hyphen `-g`, never `+`, so it is path-safe).
//! See the publish flow in `scripts/publish-talos.sh` and the fleet contract.

use std::process::Command;

fn main() {
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());

    let version = format!(
        "{}-g{}",
        std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string()),
        sha,
    );
    println!("cargo:rustc-env=TALOS_VERSION={version}");

    // Re-stamp when HEAD moves (new commit / checkout) so the SHA stays fresh.
    // Paths are relative to this crate's root (crates/talos); the workspace
    // `.git` lives two levels up. Missing in a non-git build (tarball) — the
    // SHA falls back to `unknown` and these just no-op.
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
}
