#!/usr/bin/env bash
#
# Build + publish the macOS (Apple Silicon) talos binary next to the Linux ones.
#
# Run this ON THE MAC. It is a FOLLOWER of `publish-talos.sh`, never a leader:
# it builds exactly the commit that `pi-04:/srv/talos/latest` names, so the Mac
# binary always matches what the fleet runs, then uploads it into that same
# token directory as a sibling of the Linux arches. It never touches `latest`.
#
# Why the Mac builds its own binary: `publish-talos.sh` cross-compiles from
# Linux, and cross-compiling to macOS needs an Apple SDK plus a C cross
# toolchain (talos pulls in `ring` via rustls). The Mac already has Rust.
#
# Contract additions (the fleet contract in publish-talos.sh is unchanged):
#   - Layout: pi-04:$DIR/<TOKEN>/aarch64-apple-darwin/<bin>, one file per
#     workspace binary (the same discovery rule as publish-talos.sh: every
#     crates/<name> shipping src/main.rs, published as <name>). The directory
#     name is the Rust target triple, so it cannot collide with a `uname -m`
#     arch name, and `talos-update.sh` ignores sibling dirs it does not ask for.
#   - `latest` invariant is still "both Linux arches present"; the Mac artifact
#     is NOT part of it, so a Mac that lags never blocks the fleet.
#   - Immutable per token, like the Linux artifacts: an existing file is never
#     overwritten.
#
# Usage:
#   scripts/publish-talos-mac.sh                  # build + publish `latest`
#   scripts/publish-talos-mac.sh --token <TOKEN>  # build + publish a specific token
#   scripts/publish-talos-mac.sh --no-upload      # build + verify only
#
# Config (env, defaults target the homelab):
#   TALOS_PUBLISH_HOST     ssh target        (default: jason@pi-04)
#   TALOS_PUBLISH_DIR      remote base dir   (default: /srv/talos)
#   TALOS_MAC_TARGET_DIR   cargo target dir  (default: ~/.cache/talos-mac-target;
#                                             persistent, so rebuilds are incremental)
set -euo pipefail

PUBLISH_HOST="${TALOS_PUBLISH_HOST:-jason@pi-04}"
PUBLISH_DIR="${TALOS_PUBLISH_DIR:-/srv/talos}"
TARGET_DIR_NAME="aarch64-apple-darwin"
CARGO_TARGET="${TALOS_MAC_TARGET_DIR:-$HOME/.cache/talos-mac-target}"

TOKEN=""
UPLOAD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --token)
      [ $# -ge 2 ] || { echo "ERROR: --token requires an argument" >&2; exit 2; }
      TOKEN="$2"; shift 2 ;;
    --no-upload) UPLOAD=0; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "ERROR: unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "ERROR: this script builds the Apple Silicon binary and must run on an arm64 Mac (got $(uname -s) $(uname -m))." >&2
  exit 1
fi

cd "$(dirname "$0")/.."

# --- Resolve the token: `latest` unless pinned. This is the mutable pointer the
# fleet consumes, so it is read fresh here rather than remembered.
if [ -z "$TOKEN" ]; then
  TOKEN="$(ssh "$PUBLISH_HOST" "cat '$PUBLISH_DIR/latest'" | tr -d '[:space:]')"
  [ -n "$TOKEN" ] || { echo "ERROR: empty token from $PUBLISH_HOST:$PUBLISH_DIR/latest" >&2; exit 1; }
fi
echo "==> Token: $TOKEN"

# --- The Linux publisher must have created this token already. A Mac artifact
# for a token the fleet never received would be an orphan.
ssh "$PUBLISH_HOST" "test -d '$PUBLISH_DIR/$TOKEN'" \
  || { echo "ERROR: $PUBLISH_DIR/$TOKEN does not exist on $PUBLISH_HOST — publish-talos.sh has not published it." >&2; exit 1; }

# --- Token -> commit. `0.12.0` is a tag; `0.12.0-7-ge312f8e` is `git describe`
# output whose commit is the abbreviated SHA after `-g`.
git fetch --tags --quiet origin
if [[ "$TOKEN" =~ -g([0-9a-f]+)$ ]]; then
  REF="${BASH_REMATCH[1]}"
else
  REF="$TOKEN"
fi
COMMIT="$(git rev-parse --verify --quiet "${REF}^{commit}")" \
  || { echo "ERROR: cannot resolve '$REF' (from token '$TOKEN') to a commit in this clone." >&2; exit 1; }
echo "==> Commit: $COMMIT"

# --- Build in a detached worktree at that commit, so the tree being edited is
# never involved and the build is of exactly the published commit.
WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/talos-mac-build.XXXXXX")"
cleanup() {
  git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || rm -rf "$WORKTREE"
}
trap cleanup EXIT
git worktree add --detach --quiet "$WORKTREE" "$COMMIT"

# --- Same discovery rule as publish-talos.sh, evaluated at the PUBLISHED commit
# so the binary set matches what the Linux publish shipped for this token.
BINS=()
for dir in "$WORKTREE"/crates/*/; do
  name="$(basename "$dir")"
  [ -f "$dir/src/main.rs" ] && BINS+=("$name")
done
[ "${#BINS[@]}" -gt 0 ] || { echo "ERROR: no workspace binaries under crates/*/src/main.rs at $COMMIT" >&2; exit 1; }
echo "==> Binaries: ${BINS[*]}"

REMOTE_DIR="$PUBLISH_DIR/$TOKEN/$TARGET_DIR_NAME"
MISSING=()
for bin in "${BINS[@]}"; do
  ssh "$PUBLISH_HOST" "test -f '$REMOTE_DIR/$bin'" || MISSING+=("$bin")
done
if [ "${#MISSING[@]}" -eq 0 ]; then
  echo "==> $TOKEN/$TARGET_DIR_NAME already complete (${BINS[*]}) — nothing to do (artifacts are immutable)."
  exit 0
fi

echo "==> Building release binaries (${MISSING[*]}; $TARGET_DIR_NAME, target dir $CARGO_TARGET)…"
for bin in "${MISSING[@]}"; do
  (cd "$WORKTREE" && CARGO_TARGET_DIR="$CARGO_TARGET" cargo build --release --locked -p "$bin")
done

# --- The version must equal the token, or the binary misrepresents itself.
# Only talos is stamped by build.rs; other binaries may carry no version.
if [[ " ${MISSING[*]} " == *" talos "* ]]; then
  BUILT="$("$CARGO_TARGET/release/talos" --version | awk '{print $2}')"
  if [ "$BUILT" != "$TOKEN" ]; then
    echo "ERROR: built talos reports '$BUILT' but the token is '$TOKEN' — refusing to publish." >&2
    echo "       (Do all tags exist locally? Try: git fetch --tags origin)" >&2
    exit 1
  fi
  echo "==> Built talos $BUILT"
fi

if [ "$UPLOAD" = 0 ]; then
  echo "==> --no-upload: binaries in $CARGO_TARGET/release/ (${MISSING[*]})"
  exit 0
fi

# --- Upload each to a partial name, then rename, so a reader never sees half a file.
echo "==> Uploading to $PUBLISH_HOST:$REMOTE_DIR/…"
ssh "$PUBLISH_HOST" "mkdir -p '$REMOTE_DIR'"
for bin in "${MISSING[@]}"; do
  scp -q "$CARGO_TARGET/release/$bin" "$PUBLISH_HOST:$REMOTE_DIR/$bin.partial"
  ssh "$PUBLISH_HOST" "chmod 0755 '$REMOTE_DIR/$bin.partial' && mv -n '$REMOTE_DIR/$bin.partial' '$REMOTE_DIR/$bin' && rm -f '$REMOTE_DIR/$bin.partial'"
done

echo "==> Published $TOKEN for $TARGET_DIR_NAME: ${MISSING[*]}."
