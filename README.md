# harness-design

A learning project to understand agent harness design by building one in Rust —
intended to serve as a headless-dispatch build engine for Agent GTD, supporting
Anthropic API models (Haiku/Sonnet/Opus) and local Ollama models. See
[`CLAUDE.md`](./CLAUDE.md) for goals.

## Dev setup

Requires Rust (pinned in `rust-toolchain.toml`; install via
[rustup](https://rustup.rs)). Install the quality-gate tooling as prebuilt
binaries:

```bash
# cargo-binstall (one-time): https://github.com/cargo-bins/cargo-binstall
cargo binstall -y cargo-nextest cargo-llvm-cov cargo-deny cargo-machete \
  typos-cli cargo-sort cargo-release cocogitto
# lefthook + gitleaks are not on binstall — grab their GitHub release binaries
# and put them on PATH (e.g. ~/.cargo/bin).
```

Activate the git hooks (every fresh clone must do this):

```bash
lefthook install
```

## Build & test

```bash
cargo build --workspace
cargo nextest run --workspace      # fast test runner
cargo test --doc --workspace       # doctests (nextest skips these)
```

## Quality gates

Run by lefthook locally and re-run in CI (`.github/workflows/ci.yml`), which is
the real enforcement boundary since local hooks can be skipped with
`--no-verify`.

| Stage | Gates |
|---|---|
| commit-msg | conventional commits (`cog verify`) |
| pre-commit | `cargo fmt --check`, `cargo clippy -- -D warnings`, `typos`, `cargo sort --check`, `gitleaks`, `cargo nextest run` |
| pre-push | coverage `--fail-under-lines 95`, `cargo test --doc`, `cargo machete`, `cargo deny check` |
| CI (scheduled) | `cargo audit` (advisories disclosed after merge) |

What the Rust compiler enforces for free (so there's no separate gate): full
type checking, null-safety (`Option`), memory/thread safety (borrow checker),
match exhaustiveness, unused imports/variables, and `unsafe` is `forbid`-den
project-wide. The gates only add what rustc can't see.

**Coverage ratchet:** the `--fail-under-lines` literal lives in `lefthook.yml`
and `.github/workflows/ci.yml`. Bump both upward as coverage improves; never let
it regress.

## Release

Decoupled from deploy (matches the Python projects). At a meaningful boundary:

```bash
./release.sh   # cog bump --auto -> tag -> push origin main --tags
```

## Publishing talos to the dispatch fleet

talos is self-hosting — we dispatch talos features to talos — so the binary
changes every wave. Rather than each dispatch host compiling it (slow, especially
the aarch64 Pi), the fast x86_64 dev box builds **both** arches and publishes them
to the homelab artifact host (`pi-04`); the fleet's `talos-update.sh`
(in `agent-gtd-dispatch`) then just pulls the binary.

```bash
./scripts/publish-talos.sh    # gate -> build x86_64 + aarch64 -> scp to pi-04 -> advance 'latest'
```

- **Version token.** `talos --version` prints `talos <semver>-g<short-sha>`
  (stamped by `crates/talos/build.rs`). It is a single URL/path-safe string; the
  consumer reads it via `talos --version | awk '{print $2}'`.
- **pi-04 layout.** `pi-04:/srv/talos/<TOKEN>/<arch>/talos` where `<arch>` ∈
  `{x86_64, aarch64}` (matching `uname -m`), plus `pi-04:/srv/talos/latest`
  holding the current `<TOKEN>` on one line — advanced only after both arches
  upload. Artifacts are immutable per-`<TOKEN>` (a gate-failing or dirty-tree
  build never ships; an existing `<TOKEN>` dir is never overwritten).
- **One-time cross-toolchain setup** (on the x86_64 publisher):
  `rustup target add aarch64-unknown-linux-gnu` and install the linker
  `gcc-aarch64-linux-gnu` (Debian/Ubuntu). The linker is wired in
  `.cargo/config.toml`.
- **Target override.** `TALOS_PUBLISH_HOST` (default `jason@pi-04`) and
  `TALOS_PUBLISH_DIR` (default `/srv/talos`).
