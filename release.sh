#!/usr/bin/env bash
# Manual, decoupled release at a meaningful boundary — the analog of the Python
# projects' release.sh. Derives the SemVer bump from conventional-commit history,
# tags, and pushes to origin. No crates.io publish. Deploy is a separate step.
set -euo pipefail

git diff --quiet || { echo "error: working tree is dirty"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "error: not on main"; exit 1; }

# Bump version + Cargo.toml + CHANGELOG + tag from conventional commits.
# This creates the `chore(version)` commit + tag, moving HEAD — so it MUST run
# before the artifact publish below (the token embeds HEAD's short SHA).
cog bump --auto

# A release ships a new artifact by definition. Publish the release commit's
# binaries to the fleet BEFORE pushing the tag, so `set -e` aborts the release
# if publish fails — origin never advertises a tag whose artifact didn't ship.
./scripts/publish-talos.sh

git push origin main --tags

# ./deploy.sh   # uncomment once a deploy target exists (gitignored, local)

# ./deploy.sh   # uncomment once a deploy target exists (gitignored, local)
