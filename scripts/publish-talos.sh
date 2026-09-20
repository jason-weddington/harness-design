#!/usr/bin/env bash
#
# Build + publish EVERY workspace release binary to the dispatch-fleet artifact host.
#
# Every session binary in this workspace is self-hosting (we dispatch features to
# the fleet), so the binaries change every wave. Rather than each dispatch host
# compiling them itself (slow, esp. the aarch64 Pi), the fast i9 builds BOTH
# arches of EVERY binary here and publishes them to pi-04 in ONE invocation;
# each consumer's own update script then pulls the binary it cares about.
#
# Fleet contract (consumers depend on these verbatim): ONE invocation publishes
# EVERY workspace binary at ONE token, so a binary added later always arrives
# under a fresh token (a later upload at a token another binary already
# published is impossible by construction, not merely detected). Version token:
# `git describe --tags --always` — the release tag on a tagged commit (e.g.
# `0.5.0`), else `<tag>-<n>-g<short-sha>` (URL/path-safe). This is identical to
# what `crates/talos/build.rs` stamps into `talos --version` on the clean tree
# the gate enforces (build.rs adds only `--dirty`, which a clean tree never
# yields), so a consumer's `| awk '{print $2}'` still matches the published
# token. Layout: pi-04:$DIR/<TOKEN>/<arch>/<bin>  where <arch> ∈ {x86_64,
# aarch64} (matching `uname -m` on the hosts) and <bin> is the crate directory
# basename. Latest pointer: pi-04:$DIR/latest holds the current <TOKEN> (one
# line), advanced ONLY after every binary of every arch is verified present —
# the per-file check runs even on the already-published skip path.
#
# Binaries published: every `crates/<name>` shipping a `src/main.rs`, published
# under the binary name `<name>` (the pinned convention: directory basename ==
# package name == bin name). A `cargo metadata` audit right after discovery
# fails loudly if that convention ever diverges from what cargo actually
# builds, so a future crate cannot be silently under-published.
#
# Safety: runs the project gate first (a binary that fails its own gate never
# ships); artifacts are immutable per-TOKEN (an existing <TOKEN> dir is never
# overwritten); `latest` advances only once the per-file check passes.
#
# Dry-run recipe (verifies the whole publish choreography locally: no network,
# no build, no worktree writes; the staging dir lives OUTSIDE the worktree):
#   tmp=$(mktemp -d)
#   TALOS_PUBLISH_DRY_RUN=1 TALOS_PUBLISH_DRY_RUN_DIR=$tmp ./scripts/publish-talos.sh
#   TOKEN=$(git describe --tags --always)
#   test "$(cat "$tmp/latest")" = "$TOKEN"
#   test "$(cat "$tmp/$TOKEN/x86_64/<bin>")" = "dry-run placeholder"   # per binary/arch
# Re-running with the same $tmp exercises the skip path (verification still
# runs); `rm "$tmp/$TOKEN/x86_64/<bin>"` then re-running exercises the
# missing-artifact refusal (exit 1, `latest` untouched).
#
# Config (env, defaults target the homelab):
#   TALOS_PUBLISH_HOST        ssh target        (default: jason@pi-04)
#   TALOS_PUBLISH_DIR         remote base dir   (default: /srv/talos)
#   TALOS_PUBLISH_DRY_RUN     exactly `1` activates a local staging dry run
#   TALOS_PUBLISH_DRY_RUN_DIR staging root for the dry run (default: mktemp -d)
#
# Delivery is per-session (this script owns PUBLISH only): talos-update.sh
# (agent-gtd-dispatch) pulls talos onto the dispatch hosts; the KB session
# writes its own pull step reading the same `latest` over pi-04's Caddy.
set -euo pipefail

DRY_RUN=0; [ "${TALOS_PUBLISH_DRY_RUN:-}" = "1" ] && DRY_RUN=1

PUBLISH_HOST="jason@pi-04"
[ -n "${TALOS_PUBLISH_HOST-}" ] && PUBLISH_HOST="$TALOS_PUBLISH_HOST"
PUBLISH_DIR="${TALOS_PUBLISH_DIR:-/srv/talos}"
ARM_TARGET="aarch64-unknown-linux-gnu"

cd "$(dirname "$0")/.."

# --- The ONLY remote seam: every ssh/scp is routed through these two functions.
# In dry-run mode they stay on the local machine: remote_run executes the
# command as-is (every path in it is already $PUBLISH_DIR-shaped, and
# $PUBLISH_DIR has been rebound to the staging root), and remote_put ignores
# the source entirely and synthesizes a placeholder AT THE DESTINATION — never
# over the real target/ artifacts, which cargo's freshness check must keep
# trusting for the next real build.
remote_run() {
  if [ "$DRY_RUN" = 1 ]; then sh -c "$1"; else ssh "$PUBLISH_HOST" "$1"; fi
}
remote_put() {
  if [ "$DRY_RUN" = 1 ]; then
    mkdir -p "$(dirname "$2")" && printf 'dry-run placeholder\n' > "$2"
  else
    scp -q "$1" "$PUBLISH_HOST:$2"
  fi
}

# --- Dry-run staging: point every later $PUBLISH_DIR/... path at the staging
# root. No clean-tree check, no gate, no build, no network from here on.
if [ "$DRY_RUN" = 1 ]; then
  STAGING_ROOT="${TALOS_PUBLISH_DRY_RUN_DIR:-$(mktemp -d)}"
  PUBLISH_DIR="$STAGING_ROOT"
  echo "==> DRY RUN: staging root $STAGING_ROOT, gate/build/network skipped — remote 'latest' NOT advanced"
fi

# --- Discover publishable binaries: every crates/<name> shipping src/main.rs,
# published under the binary name <name> (directory basename == package name ==
# bin name). Globs sort lexicographically; the rule is pinned, not heuristic.
BINS=()
for dir in crates/*/; do
  name="$(basename "$dir")"
  if [ -f "crates/$name/src/main.rs" ]; then
    BINS+=("$name")
  fi
done
if [ "${#BINS[@]}" -eq 0 ]; then
  echo "ERROR: no workspace binaries found under crates/*/src/main.rs" >&2
  exit 1
fi
echo "==> Binaries: ${BINS[*]}"

# --- Audit the discovery rule against cargo's authority (no network, no
# build, no lockfile/target writes). Catches a future crate whose binary cargo
# sees but crates/*/src/main.rs does not (e.g. src/bin/*.rs), or a package
# name diverging from its directory.
cargo_bins="$(set +o pipefail; cargo metadata --format-version 1 --no-deps \
  | grep -oE '"kind":\["bin"\][^}]*"name":"[^"]+"' \
  | sed -E 's/.*"name":"([^"]+)".*/\1/' | sort -u | tr '\n' ' ')"
cargo_bins="${cargo_bins% }"
derived_bins="$(printf '%s\n' "${BINS[@]}" | sort -u | tr '\n' ' ')"
derived_bins="${derived_bins% }"
if [ "$derived_bins" != "$cargo_bins" ]; then
  echo "ERROR: derived binary list '$derived_bins' does not match cargo bin targets '$cargo_bins'" >&2
  exit 1
fi

if [ "$DRY_RUN" = 0 ]; then
  # --- Clean tree: the token embeds HEAD's short SHA, so a dirty tree would
  # ship binaries the token misrepresents. Refuse to publish uncommitted work.
  if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty — commit first so the version token (git SHA) is truthful." >&2
    exit 1
  fi

  # --- Gate first: a binary that can't pass its own gate never reaches the fleet.
  echo "==> Gate: fmt + clippy + nextest…"
  cargo fmt --check
  cargo clippy --workspace --all-targets -- -D warnings
  cargo nextest run --workspace

  # --- Build every binary, both arches (release): native x86_64 + cross aarch64.
  echo "==> Building release binaries (x86_64 native + $ARM_TARGET cross)…"
  for bin in "${BINS[@]}"; do
    cargo build --release -p "$bin"
    cargo build --release --target "$ARM_TARGET" -p "$bin"
  done
fi

# --- Token: identical derivation in both modes (see header for why it matches
# the build.rs stamp on a clean tree). Never derived from a binary's --version:
# the first glob-sorted crate may carry no version stamp.
TOKEN="$(git describe --tags --always)"
[ -n "$TOKEN" ] || { echo "ERROR: empty version token from git describe --tags --always" >&2; exit 1; }
echo "==> Token: $TOKEN"

# --- Immutable publish: never overwrite an existing <TOKEN> dir.
if remote_run "test -d '$PUBLISH_DIR/$TOKEN'"; then
  echo "==> $TOKEN already published — skipping upload (artifacts are immutable)."
else
  echo "==> Uploading to $PUBLISH_DIR/$TOKEN/…"
  remote_run "mkdir -p '$PUBLISH_DIR/$TOKEN/x86_64' '$PUBLISH_DIR/$TOKEN/aarch64'"
  for bin in "${BINS[@]}"; do
    remote_put "target/release/$bin" "$PUBLISH_DIR/$TOKEN/x86_64/$bin"
    remote_put "target/$ARM_TARGET/release/$bin" "$PUBLISH_DIR/$TOKEN/aarch64/$bin"
  done
fi

# --- Per-file check (defence in depth, runs on BOTH paths above): every
# published token must contain every binary of every arch, or 'latest' would
# silently keep pointing at binaries a consumer cannot find. Fail loudly BEFORE
# touching 'latest'.
for bin in "${BINS[@]}"; do
  for arch in x86_64 aarch64; do
    remote_run "test -f '$PUBLISH_DIR/$TOKEN/$arch/$bin'" \
      || { echo "ERROR: missing artifact $arch/$bin under $TOKEN — refusing to advance 'latest'." >&2; exit 1; }
    echo "==> Verified $TOKEN/$arch/$bin"
  done
done

# --- Advance 'latest' ONLY after every artifact is confirmed present.
printf '%s\n' "$TOKEN" | remote_run "cat > '$PUBLISH_DIR/latest'"

echo "==> latest → $TOKEN"
if [ "$DRY_RUN" = 1 ]; then
  echo "==> DRY RUN complete: staged $TOKEN for ${BINS[*]} — remote 'latest' NOT advanced"
else
  echo "==> Published $TOKEN to $PUBLISH_DIR: ${BINS[*]} (x86_64 + aarch64)."
fi
echo "==> macOS artifact is NOT built here. On the Mac, run: scripts/publish-talos-mac.sh"
