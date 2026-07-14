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

## Ralph mode (`talos ralph`)

The **Ralph loop** drives an agent toward an objective by re-invoking the inner engine with a **fresh context every outer iteration** — durable state lives *outside* the context window (the code on disk, the git history, and a notes file the agent reads-then-appends), so each pass starts cold and still makes forward progress. Each iteration does exactly one unit of work; the **harness owns a git commit per iteration** (a deliberate ralph-only exception to the worker-owns-git rule). Distinct from finish-recovery (which nudges the *same* context when a gate is red) — Ralph *restarts* the context. Core: `crates/harness/src/ralph.rs`.

Two commands, deliberately **never collapsed** — get this wrong and the loop misbehaves:

- **`--gate` (inner, per-iteration):** the `run_checks` command the inner engine uses to verify a `finish(done)` claim. The harness forces the agent to loop until this is green *before* it can finish, so the tree is already green when the per-iteration commit fires.
- **`--stop-when` (outer objective oracle):** a command run via `/bin/sh -c` whose exit `0` means "objective met, stop the whole loop." This is the goal, not the per-iteration bar.

**The load-bearing gotcha (why gate ≠ stop-when):** ralph commits ONLY green `finish(done)` finishes — the harness guarantees the inner `--gate` set is green *before* it commits, so the per-iteration `git commit` (which runs the repo's pre-commit hook and is **not** `--no-verify`'d) sees a green tree. Any *non-green* inner outcome, OR a green commit whose pre-commit hook rejects it, is **reverted** to the last green commit (`git reset --hard HEAD` + `git clean -fd`, ignored files like `target/` preserved, the iteration's `PROGRESS.md` append discarded — a clean do-over) and the loop retries with a fresh context. After `--max-do-overs` (default 3) *consecutive* do-overs with no green commit between them, the loop terminates with a `DoOversExhausted` terminal (exit 20, a task-side failure like `Stuck` — not the exit-1 infra `Error`). So the pre-commit hook must be **check-only** (never a `--fix`/formatter hook that mutates — see kb-03099) and must match the inner `--gate` set. Two things must **not** be in the commit hook: (1) the `--stop-when` threshold (e.g. a coverage floor) — it's false until the objective is met, so it would fail every commit and burn do-overs; and (2) a **conventional-commit-msg** hook — ralph's commit messages are `ralph: iteration N — <objective>`, which such a hook rejects.

Example — grind an unhealthy repo up to 90% test coverage on a local Ollama model:

```bash
TALOS_BACKEND=ollama OLLAMA_MODEL=qwen3.6:35b OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_THINK=on \
talos ralph \
  --workspace /path/to/repo \
  --objective 'Raise coverage to 90%. Each iteration: run coverage, pick the single highest-value untested function, write ONE test for it, verify it passes, append a note to PROGRESS.md, then finish.' \
  --stop-when 'uv run --frozen pytest --cov=<pkg> --cov-fail-under=90 -q' \
  --gate 'uv run --frozen ruff check . && uv run --frozen ruff format --check . && uv run --frozen pytest -q' \
  --notes-file PROGRESS.md \
  --max-ralph-iterations 25
```

The workspace **must already be a git work tree** — `run_ralph` does *not* run `git init`. Backend selection reuses the same `TALOS_BACKEND` / `ANTHROPIC_*` / `OLLAMA_*` env as `talos run`. Other flags: `--inner-max-iterations` (inner cap per pass, default 500), `--stuck-k` (consecutive no-progress passes before giving up — progress = a git diff *outside* the notes file, default 3), `--max-do-overs` (consecutive non-green / rejected-green-commit do-overs before `DoOversExhausted` — each reverted to the last green commit — default 3), `--ralph-wall-clock-secs` (0 = unbounded; also `TALOS_RALPH_WALL_CLOCK_SECS`), `--stop-when-timeout-secs` / `--gate-timeout-secs` (default 300). Ralph is **not** run-record persisted this cut — it prints a `RalphSummary` JSON (objective / terminal / outer_iterations / total_inner_iterations) to stdout and exits: **0** `StopConditionMet` · **20** `Stuck` / `MaxIterationsExhausted` / `TimeBudgetExhausted` / `DoOversExhausted` · **1** `Error` (git/spawn/revert failure). Watch progress via the `ralph: iteration N` commits and the notes file, not stdout.

## Quality gates

Run by lefthook locally and re-run in CI (`.github/workflows/ci.yml`), which is
the real enforcement boundary since local hooks can be skipped with
`--no-verify`.

| Stage | Gates |
|---|---|
| commit-msg | conventional commits (`cog verify`) |
| pre-commit | `cargo fmt --check`, `cargo clippy -- -D warnings`, `typos`, `cargo sort --check`, `gitleaks`, `cargo nextest run` |
| pre-push | coverage `--fail-under-lines 98`, `cargo test --doc`, `cargo machete`, `cargo deny check` |
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
