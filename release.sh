#!/usr/bin/env bash
# Manual, decoupled release at a meaningful boundary — the analog of the Python
# projects' release.sh. Derives the SemVer bump from conventional-commit history,
# tags, and pushes to origin. No crates.io publish. Deploy is a separate step.
set -euo pipefail

git diff --quiet || { echo "error: working tree is dirty"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "error: not on main"; exit 1; }

# Bump version + Cargo.toml + CHANGELOG + tag from conventional commits.
cog bump --auto

git push origin main --tags

# ./deploy.sh   # uncomment once a deploy target exists (gitignored, local)
