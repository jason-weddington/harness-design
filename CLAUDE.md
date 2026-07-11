# harness-design

A learning project to understand **agent harness design** by building one in Rust.

## What this is

An agent harness — the loop that turns an LLM into an autonomous agent: prompt
assembly, tool dispatch, model I/O, conversation/state management, and the safety
rails around all of it. We build it to learn the design space hands-on, not because
an off-the-shelf harness wouldn't work.

## Goals

- **Learn agent harness design** by building a real one, in Rust.
- **Concrete use case: a build engine for Agent GTD.** This harness should be able to
  serve as another headless-dispatch build engine — the thing that picks up a groomed
  GTD task and executes it autonomously, alongside the existing Claude Code engine.
- **Model support, in order:**
  - Anthropic API — Haiku, Sonnet, Opus (claude-haiku-4-5, claude-sonnet-4-6,
    claude-opus-4-8).
  - Local **Ollama** models, for cheap/offline iteration and to learn how a harness
    abstracts over heterogeneous model backends.
- **Be the kind of codebase autonomous agents can build safely.** Strong commit-time
  and pre-push quality gates so headless agents can "run wild" without a human
  reviewing every line. (Gate stack is being researched — see below.)

## Why Rust

Part of the learning goal. Rust's compiler and type system give us a layer of
correctness/safety enforcement *for free* that Python doesn't — the borrow checker,
exhaustiveness, no-null, `Result`-based error handling. The quality-gate work is about
identifying what's genuinely additive on top of that (lint strictness, coverage,
supply-chain, secret scanning, commit hygiene) vs. what would just be ceremony.

## Knowledge Base

- **KB `project_ref`: `harness-design`** — store decisions, lessons, and conventions
  for this project under that ref. (The repo-root `.kb_project` file records this so
  the KB hook/preflight surface the right maps.)
- **Braintrust `project_ref`: `harness-design-research`** — a separate KB project
  holding ingested external sources on agent-harness design (the references cited
  in `docs/research/`, ingested via `kb_ingest_url`). Kept separate so source
  material doesn't clutter our own `harness-design` decisions/lessons. Query it
  (`kb_search`/`kb_ask` with `project_ref="harness-design-research"`) when you want
  the field's prior art on a harness question; the synthesis of it lives in
  `docs/research/00-overview.md`.
- Query the KB before guessing at architecture or conventions; capture hard-won
  lessons as you go.

## Session log (this is a learning project — document the process)

We capture *how* we work, not just what we ship. At a natural breakpoint in each
working session:

1. Log a dated `lesson_learned` entry in the KB (`project_ref: harness-design`,
   tags include `session-log`) covering that session's arc — what we did and why,
   key decisions/lessons, where we landed, and **what's next** — so it doubles as
   the handoff note for the following session.
2. Append a 3-4 sentence summary + the KB entry id to
   [`docs/session-summaries.md`](./docs/session-summaries.md) (chronological,
   newest at the bottom).
3. Bring [`docs/roadmap.md`](./docs/roadmap.md) current: rewrite its **"Where we
   are"** header to the newest shipped version, mark any milestone that shipped
   this session **✅ shipped**, and re-order/rescope what's next if the session
   changed the plan. The roadmap is the living forward view; a session that ships
   a capability but leaves the roadmap describing an older state has left the
   handoff half-done.

**When resuming a session, read the latest `docs/session-summaries.md` entry and
its linked KB entry first**, then skim `docs/roadmap.md` "Where we are" for the
current forward view. Session 1 (`kb-02851`) is the template.

## Status

Early/greenfield. The quality-gate harness is in place (commit-time + pre-push via
lefthook, mirrored in CI); `crates/harness` is a placeholder lib with one tested
function so the gates have something to enforce. **Next:** the real harness loop
(model I/O → tool dispatch → state), starting with an Anthropic provider, then Ollama.

## Layout

Cargo virtual workspace. `crates/harness` is the core library; new crates (model
providers, a CLI, the GTD build-engine adapter) get added to `members` in the root
`Cargo.toml`. Lint strictness + shared deps are centralized in `[workspace.lints]` /
`[workspace.dependencies]`.

## Build / Test

```bash
cargo build --workspace
cargo nextest run --workspace     # fast test runner (the gate)
cargo test --doc --workspace      # doctests — nextest does NOT run these
```

## Quality gates (let agents run wild)

Toolchain pinned in `rust-toolchain.toml`. Hooks orchestrated by **lefthook** —
every fresh clone must run `lefthook install`. Tools install as prebuilt binaries
via `cargo binstall` (see README); lefthook + gitleaks come from their GitHub
releases. The gate config is the source of truth: `lefthook.yml`, `deny.toml`,
`rustfmt.toml`, `cog.toml`, the `[workspace.lints]` table, and `.github/workflows/ci.yml`.

| Stage | Gates |
|---|---|
| commit-msg | conventional commits (`cog verify`) |
| pre-commit | `cargo fmt --check`, `clippy -D warnings`, `typos`, `cargo sort --check`, `gitleaks`, `cargo nextest run` |
| pre-push | coverage `--fail-under-lines 95`, `cargo test --doc`, `cargo machete`, `cargo deny check` |
| CI | re-runs all of the above + a daily scheduled `cargo audit` |

**rustc is a gate too** — type checking, null-safety, the borrow checker, match
exhaustiveness, and unused-import/variable detection are free, so there's no
mypy-equivalent gate. `unsafe` is `forbid`-den project-wide. The extra gates only
cover what the compiler can't see.

**Coverage ratchet:** the `95` lives in `lefthook.yml` AND `.github/workflows/ci.yml`.
Bump both upward as coverage improves; never regress it. Licenses are restricted to
`MIT`/`Apache-2.0` in `deny.toml` — a dep under any other license is a deliberate add.

## Release

Decoupled from deploy (matches the Python projects). At a meaningful boundary run `./release.sh`: it cuts the version (`cog bump --auto` — trust it; type commits honestly, `fix:` vs `feat:`, and don't hand-pin the version), tags, **publishes a fresh talos fleet binary** (`scripts/publish-talos.sh`, run between the bump and the push, so a release ships an artifact by definition), then `git push origin main --tags`. No crates.io publish.

## Talos fleet binary (build + push without a release)

Talos is self-hosting — the binary changes every wave — so the dispatch hosts run a *published* binary rather than compiling it themselves. A release republishes it automatically (above). To push a fresh binary **mid-work, without cutting a release** — e.g. you just merged a talos-affecting change and want the fleet on it before the next dispatch — run the two-step flow by hand:

```bash
# 1. Build both arches on this box (the i9 has the aarch64 cross-toolchain) and
#    publish to the pi-04 artifact host. Requires a CLEAN tree — the version token
#    embeds HEAD's short SHA, so commit first. Runs the gate before it ships.
./scripts/publish-talos.sh

# 2. Pull the just-published binary onto both dispatch hosts. No service restart
#    (talos is a fresh subprocess per dispatch run — the next run picks it up).
cd ~/git/agent-gtd-dispatch && ./talos-update.sh
```

`publish-talos.sh` advances `pi-04:/srv/talos/latest` to the new token only after both arches upload (artifacts are immutable per token); `talos-update.sh` reads that `latest` via pi-04's Caddy and installs on each host in `DISPATCH_HOSTS` (default `pironman01 r7-research`), skipping any host already current. The version token is `0.1.0-g<short-sha>`; to verify, both hosts' installed token should carry the same `<short-sha>` as `git rev-parse --short HEAD`.
