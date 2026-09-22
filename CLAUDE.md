# harness-design

**This started as a learning project and is now production infrastructure.** Treat it as the latter: things depend on it.

## What this is

An agent harness — the loop that turns an LLM into an autonomous agent: prompt
assembly, tool dispatch, model I/O, conversation/state management, and the safety
rails around all of it. It was built to learn the design space hands-on rather than
because an off-the-shelf harness wouldn't work, and that origin still shows in the
design records. But the learning goal is no longer what the work serves.

**What depends on this code today:**

- **`talos` is a dispatch build engine on the fleet.** Four hosts run a published
  binary that this repo cuts; GTD items on several boards route to `talos-glm`,
  `talos-glm-flash` and `talos-qwen`, and a regression here stops real work.
- **`talos` grooms this project's own backlog**, and another project's — the
  talos-flow port of groom-to-ready is the default grooming engine here (see below),
  so the harness argues the specs that then build the harness.
- **`somnus` is a nightly production loop** built on `harness` as a library: the
  personal KB's map-maintenance job, running unattended against a metered lane and
  writing to a live knowledge base. It is the first external consumer, and the
  reason several seams exist (`ChangeObserver`, the exported billed-token and
  pricing definitions, the budget predicates).

**The practical consequence:** a change here is not a study exercise whose cost is
your own time. Breaking a seam breaks a consumer that runs at 3am with nobody
watching, and "I learned something" is not a reason to ship. The quality gates,
the three-legs-of-a-Done discipline, and the run-the-binary rule exist because of
this, not in spite of it.

## Goals

- **Serve the consumers above without regressions.** That is now the first goal.
- **Concrete use case: a build engine for Agent GTD.** This harness serves as a
  headless-dispatch build engine — the thing that picks up a groomed GTD task and
  executes it autonomously, alongside the existing Claude Code engine.
- **Keep learning the design space in the open.** The design records in `docs/design/`
  and the session history are deliberate artefacts, not a by-product; see the session
  discipline below for what that means now that the project ships.
- **Model support, in order:**
  - Anthropic API — Haiku, Sonnet, Opus (claude-haiku-4-5, claude-sonnet-5,
    claude-opus-4-8).
  - Local **Ollama** models, for cheap/offline iteration and to learn how a harness
    abstracts over heterogeneous model backends.
  - **AWS Bedrock** (Converse API) — for environments where the Anthropic API isn't
    reachable (e.g. a work machine); gated by `TALOS_BEDROCK` (wins over any
    Anthropic/Ollama env), standard AWS credential chain, haiku-4-5/sonnet-5/opus-4-8 only.
- **Be the kind of codebase autonomous agents can build safely.** Strong commit-time
  and pre-push quality gates so headless agents can "run wild" without a human
  reviewing every line. (The gate stack is in place — see Quality gates below.)

## Model capability reference — SWE-bench Pro

For engine-routing decisions. Always compare within a SINGLE leaderboard — cross-harness/scaffold numbers vary widely (the same model was reported anywhere from 63% to 80% across sources), so mixed-source comparisons are meaningless.

| Model | SWE-bench Pro |
|---|---|
| Claude Opus 4.8 | 69.2% |
| Claude Sonnet 5 | 63.2% |
| GLM-5.2 | 62.1% (top open-weights) |

Source: [llm-stats.com/benchmarks/swe-bench-pro](https://llm-stats.com/benchmarks/swe-bench-pro), as of 2026-07-15. Takeaway: GLM-5.2 sits within ~1 pt of Sonnet 5 and ~7 pts of Opus 4.8 — so **model capability is not what separates the talos-glm and talos-sonnet lanes; the harness is** (talos vs. Claude Code). This is why a talos-glm miss falls back to claude-code-glm (same model, stronger harness), not to a bigger model. Refresh when the model lineup changes.

GLM-5.3 and GLM-5.3-Flash — the models behind `talos-glm` and `talos-glm-flash` since 2026-09-12 — are **not on this leaderboard yet** (re-checked 2026-09-12; the three rows above were unchanged). Until they are, don't put them in the table from another source; our own talos eval rows against the 5.2 baseline are in `kb-03220` (tier-2: both 13/24 vs 5.2's 9/24, flash with far better finish discipline).

## Evals vs. real-world usage — two instruments, one constraint

Every tier-2 task was mined from a past GTD dispatch that a Sonnet-class model completed from an adversarially groomed spec. We already know that once a groom workflow argues a spec into precision, a Sonnet- or GLM-class model one-shots it. So:

- **Tier-2 must stay strictly harder than a perfect spec.** It runs at spec level S2 (behavioural criteria, no localization, no named tests), which is what makes it discriminate between models and measure harness changes. Do not add a groomed-spec eval tier; it would measure what we already know.
- **Harness changes must not break the perfect-spec path.** A talos behaviour that helps under-specified work but costs or loops on an 18-31-criterion groomed spec is a regression, even if tier-2 improves. Guard it with tier-1's spec-shaped fixtures (TaskSpec `task.json`, rendered through the production `task_spec_prompt.md`; saturated by design, so any lost pass, false done, cap hit or iteration/token blowup is a regression), plus a few supervised real dispatches of groomed GTD items before any talos default flips.
- **Design rule:** harness features must be spec-agnostic and nearly free when the spec already did the work (e.g. a named test in the spec *is* the coverage; an audit cites existing evidence rather than re-running it). Ship behind a default-off knob; flip only after tier-2 shows benefit **and** the perfect-spec guard is clean. Decision record: `kb-03268`.

## Grooming in this project: the talos groom is the default (2026-09-19)

Groom this project's items with the **talos-flow port of groom-to-ready**, not the Claude Code `Workflow` tool, in the **arm C shape**: draft on `glm-5.3-flash`, the four critics on `glm-5.3`, synthesize on `glm-5.3-flash`. Jason's decision after the four-arm comparison (`kb-03342`): the Opus/Fable Workflow writes better specs but is unsustainable — it exhausts the weekly Anthropic plan limit in under half a week and usage credits are not worth paying — while arm C's specs built one-shot on talos-glm at ~$3 per item. All-flash is NOT acceptable (its own critics passed a spec about the wrong item); the critic stage is the safety layer and stays on full glm-5.3. This is a project-local rule — we dogfood talos with talos here — not a global one.

How to run it (from a **detached worktree** of this repo, never the checkout you are editing — the inverted tree check bounces every in-flight answer if the tree moves):

```bash
git worktree add --detach /tmp/hd-groom main
cd ~/git/talos_flow && TALOS_BACKEND=ollama OLLAMA_BASE_URL=https://ollama.com OLLAMA_THINK=high \
  uv run --frozen python -m talos_flow.workflows.groom_to_ready \
  --args args.json --workspace /tmp/hd-groom --out finals.json \
  --draft-env OLLAMA_MODEL=glm-5.3-flash:cloud --critic-env OLLAMA_MODEL=glm-5.3:cloud --synth-env OLLAMA_MODEL=glm-5.3-flash:cloud \
  --rate-in 0.15 --rate-cached 0.03 --rate-out 0.50   # flash rates; recost the critic stage at 1.40/0.26/4.40 if you want exact dollars
```

Host it in `Monitor` (the eight stdout progress lines are the events), review `finals.json` as the checkpoint, then `agent-gtd update-item <id> --from-json <spec>.json --status ready`. `args.json` is `{context, items:[{id, slug, title, seed, criticalConstraint}]}` — the same shape the JS took. Cite only paths inside this repo in `context` and seeds; the answer agents can read nothing else.

**Three rules for the worktree, and the third is the one that bites.** (1) Never point at a checkout you or anyone else may edit — a tree that moves under a run bounces every in-flight answer. (2) For a multi-repo root, every child repo must stay clean and readable for the duration; any unobservable child poisons the whole observation. (3) **Do not merge to the branch the worktree was created from while the groom is running.** The worktree stays pinned at its creation commit, so it never goes dirty and nothing fails — the guard is perfectly happy, every stage reports success, and the specs come back *correctly grounded in a tree that is already stale*. Line drift is the harmless half; the damaging half is a merge that changes the very function a spec is inserting a clause into, so the spec cannot state the one ordering decision that mattered. It is silent in both directions: the groom cannot detect it, and the specs look well-grounded because they are — against yesterday's tree.

The cheap fix is to re-groom against current `main` rather than patch line numbers; a re-groom is a couple of dollars and a wrong ordering decision is a wave. Either finish the groom before merging, or recreate the worktree at the new tip and re-run. This bit the agent_gtd session on 2026-09-20 (`kb-03432`) and this session was exposed to it the same evening — a groom ran pinned at `884ac9a` while six commits landed on main, and only survived because the batch's cross-item contracts had been injected by hand.

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

## End-of-session discipline

Three artefacts, and **they are not three copies of the same thing**. Writing them as
one narrative three times is what produced a "session" that ran for a month.

| Artefact | Audience | Tense | Answers |
|---|---|---|---|
| `docs/roadmap.md` "Where we are" | the next session | present | *what is true now* |
| `docs/session-summaries.md` | Jason, months later | past | *what changed and why* |
| KB `session-log` entry | whoever asks a question later | past | *everything else* |

**A session is one sitting.** When you stop, it ends; when work resumes, that is the
next session with the next number. There is no "Session 17, continued" — that heading
is how one entry came to span 2026-08-20 through 2026-09-19 with twelve blocks under
it, which is unreadable as history and useless as orientation.

At the end of a sitting, in this order:

1. **KB entry** — dated `lesson_learned`, `project_ref: harness-design`, tags include
   `session-log`. This is where detail belongs: the arc, the decisions and why, the
   numbers, the dead ends, what's next. Length is not a problem here. If a lesson is
   durable on its own, give it its OWN entry (`decision`, `lesson_learned`) and let
   the session log point at it — `kb-03486` and `kb-03473` are worth more standing
   alone than buried in a day's narrative.
2. **`docs/session-summaries.md`** — a `## Session N — date: title` heading and
   **3–5 sentences, ~1,200 characters at the outside**, ending with the KB id. Match
   the existing heading style; never a bare bold paragraph.
3. **`docs/roadmap.md`** — rewrite "Where we are" to the newest shipped version, mark
   shipped milestones, re-scope what's next. A session that ships a capability and
   leaves the roadmap describing an older state has left the handoff half-done.

**What belongs in the summary, now that this ships rather than teaches.** Lead with
what a consumer got or lost — a released version, a seam a consumer now depends on, a
regression that reached the fleet. Then the one thing worth remembering. That is it.

**What does NOT belong in the summary**, and this is the rule that would have kept
session 17 readable: anything that will be stale in a month. Credit balances and
spend figures, fleet binary versions, host counts, which GPU processes were stopped
to free VRAM, how many dirs a prune removed. Those are real and worth recording — in
the KB entry, where staleness is expected and dating is implicit. A summary sentence
should still be true and still be interesting in six months, or it is KB material.

**The test before you write a summary sentence:** would Jason want to read this in
six months? If it is a number that only mattered that afternoon, it fails.

**When resuming**, read the latest `docs/session-summaries.md` entry and its linked
KB entry, then `docs/roadmap.md` "Where we are". Session 20 (`kb-03504`) is the
current template for a session log; session 18 shows the summary length to aim for.

## Layout

Cargo virtual workspace. `crates/harness` is the core library (model backends,
engine loop, tools, the Ralph outer loop); `crates/talos` is the CLI / GTD
build-engine binary (`talos run`, `talos ralph`). New crates get added to `members`
in the root `Cargo.toml`. Lint strictness + shared deps are centralized in
`[workspace.lints]` / `[workspace.dependencies]`.

Current status and the forward view live in [`docs/roadmap.md`](./docs/roadmap.md)
(its "Where we are" header), not here.

## Build / Test

```bash
cargo build --workspace
cargo nextest run --workspace     # fast test runner (the gate)
cargo test --doc --workspace      # doctests — nextest does NOT run these
```

**This workspace ships BINARIES, and a green gate does not prove a binary does anything. Run it.** Before reporting that work on `talos` or `somnus` is done, invoke the actual command a user would invoke and read what comes back. `cargo run -p somnus -- run --project <ref>`, `talos run --help`, whatever the change claims to deliver.

This is not belt-and-braces. On 2026-09-20 a somnus branch merged with the full gate green — 1,189 tests, coverage over the 98% line, a grep confirming no leaked constants, a dropped-server failure test passing — and was reported complete to another session about to write a systemd unit for it. The binary was a deliberate loud stub that exited 1 without doing anything, because one contract was still unpinned. Every check performed was real and every one passed; not one of them touched whether the binary ran. The build agent had done nothing wrong — it made the gap loud on purpose. The reporting was wrong.

The general shape: **a test suite proves the library works, and only running the binary proves the product does.** Those are different claims and the gate can only make the first one. A crate can be fully wired as library code, exhaustively tested, and have a `main` that is a stub — which is often the CORRECT intermediate state, deliberately chosen so an unfinished path fails loudly instead of silently. Read `main.rs` and execute the entry point before you call it shipped.

## Quality gates (let agents run wild)

Toolchain pinned in `rust-toolchain.toml`. Hooks orchestrated by **lefthook** —
every fresh clone must run `lefthook install`. Tools install as prebuilt binaries
via `cargo binstall` (see README); lefthook + gitleaks come from their GitHub
releases. The gate config is the source of truth: `lefthook.yml`, `deny.toml`,
`rustfmt.toml`, `cog.toml`, the `[workspace.lints]` table, `scripts/docs-only.sh`, and `.github/workflows/ci.yml`.

| Stage | Gates |
|---|---|
| commit-msg | conventional commits (`cog verify`) |
| pre-commit | `cargo fmt --check`, `clippy -D warnings`, `typos`, `cargo sort --check`, `gitleaks`, `cargo nextest run` |
| pre-push | coverage `--fail-under-lines 98`, `cargo test --doc`, `cargo machete`, `cargo deny check` |
| CI | re-runs all of the above + a daily scheduled `cargo audit` |

**Docs-only skip:** `scripts/docs-only.sh` is a fail-safe predicate wired into `lefthook.yml` via `skip:` blocks: pre-commit `clippy` + `test` and all four pre-push gates (`coverage`, `doctest`, `machete`, `deny`) skip themselves when EVERY changed path is under `docs/` or is a top-level `*.md`. The skip is proof-based and fail-safe — an empty changeset, a mixed changeset, a rename out of a source directory, a missing upstream, or a broken/deleted predicate all run everything. The pre-push skip additionally requires a configured upstream, so the first `git push -u` of a new branch always runs every gate. CI always runs the full set and is the audit for a wrong local skip: if CI fails clippy/test/coverage on a commit whose local hook printed `(skip) by condition`, `scripts/docs-only.sh` has regressed — and `lefthook run pre-commit --verbose` shows the resolved path list and the `docs-only:` reason line.

**rustc is a gate too** — type checking, null-safety, the borrow checker, match
exhaustiveness, and unused-import/variable detection are free, so there's no
mypy-equivalent gate. `unsafe` is `forbid`-den project-wide. The extra gates only
cover what the compiler can't see.

**Coverage ratchet:** the `--fail-under-lines` value lives in `lefthook.yml` AND
`.github/workflows/ci.yml` (currently `98`). Bump both upward as coverage improves;
never regress it. Licenses are restricted to
`MIT`/`Apache-2.0` in `deny.toml` — a dep under any other license is a deliberate add.

## Release

Decoupled from deploy (matches the Python projects). At a meaningful boundary run `./release.sh`: it cuts the version (`cog bump --auto` — trust it; type commits honestly, `fix:` vs `feat:`, and don't hand-pin the version), tags, **publishes a fresh talos fleet binary** (`scripts/publish-talos.sh`, run between the bump and the push, so a release ships an artifact by definition), then pushes main + tags to **both** `origin` and `github`. No crates.io publish.

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

`publish-talos.sh` advances `pi-04:/srv/talos/latest` to the new token only after every binary uploads for both arches (artifacts are immutable per token, and the per-file check refuses to advance `latest` on a miss); `talos-update.sh` reads that `latest` via pi-04's Caddy and installs on each host in `DISPATCH_HOSTS`, skipping any host already current. The version token is `<semver>-g<short-sha>` (from `git describe --tags`, stamped by `crates/talos/build.rs`); to verify, **each dispatch host's** installed token should carry the same `<short-sha>` as `git rev-parse --short HEAD`, and the binary to check is the **`dispatch` user's** (`sudo -n -u dispatch talos --version`) — talos is not on root's or your own `PATH` on every host.

**`DISPATCH_HOSTS` is a hardcoded default, so a host that joins the fleet is silently outside it until someone edits the script.** That fails in the quiet direction: a delivery pass covering three of four hosts prints exactly what a complete one prints, and the missed host then runs a stale binary indefinitely. `jason-precision` joined on 2026-09-20 and this repo's own notes described the default as two hosts long after it was four. **Do not trust a remembered host list — read the current default out of `talos-update.sh`, and after any publish verify the token on each host you believe is in the fleet.** The same shape bit the unpushed-work sweep on the same day for the same reason.

**Do not read that check as fleet-wide.** `latest` is a published pointer with **independent consumers on independent schedules** — `talos-update.sh` pulls it onto `DISPATCH_HOSTS`, and other consumers (the KB hosts, once `somnus` exists) pull the same pointer from their own deploy paths with no coordination. That is deliberate: per Jason, every session owns its own release tooling, because the session with the context to know when delivery is safe is the one that should trigger it. The consequence is that a token can be live on the dispatch fleet and not yet elsewhere for an arbitrary interval, so **nothing may assume version uniformity across all consumers** — only within a consumer's own host set.

**macOS artifact.** `publish-talos.sh` ships only the two Linux arches. The Mac binary is published separately by `scripts/publish-talos-mac.sh`, run on the Mac after a Linux publish: it follows `latest` (builds that exact commit, asserts the version equals the token, uploads to `<TOKEN>/aarch64-apple-darwin/talos`, never overwrites, never touches `latest`). Interactive machines install with `~/scripts/pull_talos.sh` (installs to `~/.local/bin/talos`); the dispatch fleet still uses `talos-update.sh`. Design and rationale: `~/git/bulk-reader/docs/design.md`, "Binary distribution".
