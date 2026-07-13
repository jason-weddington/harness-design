#!/usr/bin/env bash
#
# Build + publish talos release binaries to the dispatch-fleet artifact host.
#
# talos is self-hosting (we dispatch talos features to talos), so the binary
# changes every wave. Rather than each dispatch host compiling talos itself
# (slow, esp. the aarch64 Pi), the fast i9 builds BOTH arches here and publishes
# them to pi-04; the fleet's `talos-update.sh` then just pulls the binary.
#
# Fleet contract (talos-update.sh depends on these verbatim):
#   - Version token: `talos --version` prints `talos <git-describe>` — the
#     release tag on a tagged commit (e.g. `0.5.0`), else `<tag>-<n>-g<short-sha>`
#     (URL/path-safe; see crates/talos/build.rs). Consumer: `| awk '{print $2}'`.
#   - Layout: pi-04:$DIR/<TOKEN>/<arch>/talos  where <arch> ∈ {x86_64, aarch64}
#     (matching `uname -m` on the hosts).
#   - Latest pointer: pi-04:$DIR/latest holds the current <TOKEN> (one line),
#     advanced ONLY after both arches upload.
#
# Safety: runs the project gate first (a talos that fails its own gate never
# ships); artifacts are immutable per-TOKEN (an existing <TOKEN> dir is never
# overwritten); `latest` advances only once both arches are present.
#
# Config (env, defaults target the homelab):
#   TALOS_PUBLISH_HOST   ssh target        (default: jason@pi-04)
#   TALOS_PUBLISH_DIR    remote base dir   (default: /srv/talos)
set -euo pipefail

PUBLISH_HOST="${TALOS_PUBLISH_HOST:-jason@pi-04}"
PUBLISH_DIR="${TALOS_PUBLISH_DIR:-/srv/talos}"
ARM_TARGET="aarch64-unknown-linux-gnu"

cd "$(dirname "$0")/.."

# --- Clean tree: the token embeds HEAD's short SHA, so a dirty tree would ship
# a binary the token misrepresents. Refuse to publish uncommitted work.
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is dirty — commit first so the version token (git SHA) is truthful." >&2
  exit 1
fi

# --- Gate first: a talos that can't pass its own gate never reaches the fleet.
echo "==> Gate: fmt + clippy + nextest…"
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace

# --- Build both arches (release): native x86_64 + cross aarch64.
echo "==> Building release binaries (x86_64 native + $ARM_TARGET cross)…"
cargo build --release -p talos
cargo build --release --target "$ARM_TARGET" -p talos

X86_BIN="target/release/talos"
ARM_BIN="target/$ARM_TARGET/release/talos"

# --- Token from the freshly built native binary (single source of truth).
TOKEN="$("$X86_BIN" --version | awk '{print $2}')"
[ -n "$TOKEN" ] || { echo "ERROR: empty version token from $X86_BIN --version" >&2; exit 1; }
echo "==> Token: $TOKEN"

REMOTE="$PUBLISH_HOST:$PUBLISH_DIR"

# --- Immutable publish: never overwrite an existing <TOKEN> dir.
if ssh "$PUBLISH_HOST" "test -d '$PUBLISH_DIR/$TOKEN'"; then
  echo "==> $TOKEN already published — skipping upload (artifacts are immutable)."
else
  echo "==> Uploading both arches to $REMOTE/$TOKEN/…"
  ssh "$PUBLISH_HOST" "mkdir -p '$PUBLISH_DIR/$TOKEN/x86_64' '$PUBLISH_DIR/$TOKEN/aarch64'"
  scp -q "$X86_BIN" "$PUBLISH_HOST:$PUBLISH_DIR/$TOKEN/x86_64/talos"
  scp -q "$ARM_BIN" "$PUBLISH_HOST:$PUBLISH_DIR/$TOKEN/aarch64/talos"
fi

# --- Advance 'latest' ONLY after both arches are confirmed present.
ssh "$PUBLISH_HOST" "test -f '$PUBLISH_DIR/$TOKEN/x86_64/talos' && test -f '$PUBLISH_DIR/$TOKEN/aarch64/talos'" \
  || { echo "ERROR: both arches not present under $TOKEN — refusing to advance 'latest'." >&2; exit 1; }
printf '%s\n' "$TOKEN" | ssh "$PUBLISH_HOST" "cat > '$PUBLISH_DIR/latest'"

echo "==> latest → $TOKEN"
echo "==> Published talos $TOKEN to $REMOTE (x86_64 + aarch64)."
