#!/usr/bin/env bash
# Manual, decoupled release at a meaningful boundary — the analog of the Python
# projects' release.sh. Derives the SemVer bump from conventional-commit history,
# tags, and pushes to origin. No crates.io publish. Deploy is a separate step.
set -euo pipefail

git diff --quiet || { echo "error: working tree is dirty"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "error: not on main"; exit 1; }

no_guard=0
for arg in "$@"; do
  case "$arg" in
    --no-guard) no_guard=1 ;;
    *) echo "usage: ./release.sh [--no-guard]" >&2; exit 2 ;;
  esac
done

# Leak scan of everything the GitHub push would publish, run before anything
# moves so a hit leaves nothing to unwind (no bump commit, tag or artifact).
# The scanner is local and gitignored: the maintainer links
# release.guard.local.sh to it, so no denylist lives in this repo. Exit 0 is
# clean, 1 is hits, 2 is a scan that could not run; anything but 0 aborts.
# --no-guard only permits a release when the guard is missing; a present guard
# always runs.
if git remote get-url github >/dev/null 2>&1; then
  if [ -x ./release.guard.local.sh ]; then
    ./release.guard.local.sh || {
      echo "error: release guard exited $?; nothing was bumped, published or pushed" >&2
      exit 1
    }
  elif [ "$no_guard" = 1 ]; then
    echo "warning: release guard SKIPPED (--no-guard): ./release.guard.local.sh is missing or not executable" >&2
  else
    echo "error: ./release.guard.local.sh is missing or not executable; link it to the leak scanner, or pass --no-guard" >&2
    exit 1
  fi
fi

# Bump version + Cargo.toml + CHANGELOG + tag from conventional commits.
# This creates the `chore(version)` commit + tag, moving HEAD — so it MUST run
# before the artifact publish below (the token embeds HEAD's short SHA).
cog bump --auto

# A release ships a new artifact by definition. Publish the release commit's
# binaries to the fleet BEFORE pushing the tag, so `set -e` aborts the release
# if publish fails — origin never advertises a tag whose artifact didn't ship.
./scripts/publish-talos.sh

git push origin main --tags
git push github main --tags

# ./deploy.sh   # uncomment once a deploy target exists (gitignored, local)
