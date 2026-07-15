# Roadmap

What's next and why, in priority order. Milestones are **capability themes**, not
date promises — a release cuts when its theme's capability is real and measurable
(the v0.1.0 rule: the first CHANGELOG line of a release should claim a capability,
not an engineering milestone). Numbers beyond the next milestone are provisional;
we re-order when we learn something.

Living document: updated at session boundaries. The per-session narrative lives in
[`session-summaries.md`](./session-summaries.md); decisions of record live in the
KB (`project_ref: harness-design`).

## Where we are — v0.8.0 + do-over fix on main (2026-07-14)

The Ralph Loop is now a complete, dogfood-proven capability. **`talos ralph` shipped (v0.8.0, `242a...` lineage from the CLI item)** — a standalone subcommand over `run_ralph` (chosen over a `--ralph-mode` flag on `run`, which is TaskSpec/disposition-shaped where ralph is objective/RalphTerminal-shaped), built one-shot by talos-glm (`kb-03106`). Then a hard dogfood: `talos ralph` on **talos-qwen** (local `qwen3.6:35b`) drove the external **dng-converter** repo (zero tests → real suite) **6% → 69% coverage** across three runs, the canonical ralph-able task "write the highest-value missing test."

That dogfood **surfaced and fixed a real orchestration bug in the loop.** Ralph committed on *any* tree change — even a non-green inner outcome — so a hard sub-task the model couldn't finish left a dirty tree, the per-iteration commit's pre-commit hook rejected it, and `RalphTerminal::Error` killed the whole loop. The **do-over fix (`242a57f`, on main, UNRELEASED — 0.8.1 pending)** makes "ralph only ever commits green" a real invariant: commit only a green `Finished(Done)`; on any non-green outcome or a hook-rejected green commit, **revert to the last green commit** (`git reset --hard HEAD` + `git clean -fd`) and retry with fresh context; a new `RalphTerminal::DoOversExhausted` (exit 20) terminates after `--max-do-overs` (default 3) consecutive do-overs, reset on every green commit; `BackendError` exempt for overnight resilience. Built by talos-glm — the harness repairing its own loop (`kb-03110`). The design heuristic for *what* to ralph is captured in `kb-03109` (a static prompt that re-binds against mutating external state).

The **GTD build-engine adapter — the milestone this project was built toward — is shipped and maturing**: the `talos-*` engine family has run as a real headless-dispatch build engine since 0.3.5, and every release since has hardened it. Talos routinely ships merged changes to its own and adjacent repos on the cheap glm lane — including, this session, the ralph CLI and the do-over fix.

**Next up:** cut **0.8.1** (the do-over fix). Then the three gaps the dogfood surfaced (GTD board): a **consecutive-`BackendError` breaker** (`6bd67e1a` — a sustained backend error churns the loop, since the exemption assumed transient; run 3 hit exactly this); **surface the `Error` terminal payload** (`57623441` — the `RalphSummary` drops which command failed; compounds `kb-03104`); and the **`num_ctx` hard-cap design question** (`a033ab51` — 32768 both truncates growing context *and* clamps large-window models; no known solution, needs a decision of record). Also queued: the **`tasks.md` backlog-executor** experiment (the mid-tier ralph-able case, now unblocked by the do-over fix) and **ralph dispatch mode** (running a ralph objective on a remote host — the always-planned step that keeps GTD complementary); plus the older queue — a distinct `Truncated` terminal (`kb-03104`), field-report worker fixes on agent-gtd-dev, runner fixture-discovery unification, and a dispatch-scale fixture tier.

## 0.3.0 — durability (persist, resume, dispose) — ✅ shipped v0.3.0

**Theme: survive the host.** The run record and `RunStore` shipped in v0.1.0 but
the loop doesn't use them yet. Wire checkpointing into the loop, unify the
loop-local `FinishDisposition` with `run_record::Disposition`, and implement the
two resume modes from the design (crash-resume; fresh-context restart). The
capability claim: *kill the harness mid-run, restart it, and the run completes* —
the deployment-agnostic promise (Pi, container, spot instance) made real.

## 0.3.5 — first dogfood (the harness builds the harness) — ✅ shipped v0.3.5

**Theme: close the loop early.** Deliberately inserted ahead of full bounded
autonomy: claim-vs-verify + the repo's own quality gates + the blunt
`max_iterations` cap are enough safety for *supervised* dispatch of small,
well-specified items. Three pieces: a `harness run` CLI binary (task spec JSON
in; run record + disposition out; exit code reflects disposition), a
task-prompt pass for the groomed-item shape (description + acceptance criteria,
not just fix-the-failing-test), and a harness engine registered in
agent-gtd-dispatch (the worker owns clone/branch/commit/push — the harness only
edits, checks, and reports). The capability claim: *the harness, running as an
Agent GTD build engine, ships a merged change to its own repo.*

Engine roster comes straight from the eval data: haiku and local qwen3.6:35b
(think=on) both clear 11/12 on exactly this task shape. Every dogfood run
generates run records + dispatch-perf-log entries under `engine: harness-*` —
the data the model-routing decision (#6) has been waiting on. First dogfood
items must match the engine's strengths: small, crisply specified, mechanically
checkable (no `search_code` yet — navigation is list+read, fine on this crate).

## 0.4.0 — bounded autonomy (finish-recovery, wall-clock budget, retry) — ✅ shipped v0.4.0

**Theme: safe to leave alone.** Design of record: [`docs/design/03-bounded-autonomy.md`](./design/03-bounded-autonomy.md).
Three items (serialized on `engine.rs`): (1) a **finish-recovery protocol** — detect
a done-but-unclaimed spin (green gates + static tree for K iters), nudge to finish or
report a one-sentence status, and on exhaustion terminate `Failed` while writing
**recovery facts** so the worker preserves the WIP branch — the harness never
fabricates `Done`, the claim moves up to the lead; (2) a **wall-clock budget** so the
harness self-terminates gracefully in the margin before the dispatch worker's hard-kill
(needs a new injectable `Clock` seam; per-process semantics); (3) **retry/backoff** on
`Transient` errors (`is_retryable` has been waiting). The capability claim: *a
pathological run terminates itself with a useful `Failed` disposition — and doesn't
throw away work it couldn't claim.*

Budgets were scoped to **wall-clock only**: token caps are inscrutable (no human-legible
right value) and cost caps have no accumulator yet (see backlog). Being designed against
real talos run data — including this wave's own finish-discipline failures.

## The GTD build-engine adapter — ✅ shipped (0.3.5 supervised → matured through 0.5.x)

**Theme: the point.** The adapter that picks up a groomed Agent GTD item, clones
the target repo, runs the loop with the project's `gate_command`, pushes a
feature branch, and comments back — the harness as a real headless-dispatch build
engine alongside Claude Code. Delivered incrementally rather than as one milestone
release: **0.3.5** shipped the supervised version (the `talos-*` engine family, the
no-MCP TaskSpec contract, the worker owning git + comment-back); subsequent releases
matured it into the unsupervised engine — finish-recovery and wall-clock bounds
(0.4.0), the stop-cold nudge (0.5.0), workspace/multi-repo dispatch and the bash tool
(0.5.1), and the ongoing field-report worker fixes on agent-gtd-dev. Talos now
routinely lands merged changes on the cheap glm lane; the remaining work is hardening
and ergonomics, not the core contract.

## Backlog (unscheduled, captured)

- **Dispatch-scale fixture tier** — to reproduce the *pass-rate* harness gap in-eval. Session 9 shipped the harness-vs-model benchmark and two "hard" fixtures (tokenbucket withheld-test + eventbus multi-file), but glm saturates them under *both* harnesses (`kb-03078`): the gap lives only at genuine dispatch scale (the 18-AC/5-file item), and a withheld-test spec precise enough to grade unambiguously is also easy to implement (precision-to-grade removes the difficulty). Reproducing the gap needs many-file, high-navigation fixtures — a real authoring effort, and the design challenge is difficulty-without-ambiguity.
- **The cost-gap finding** — Talos vs Claude Code token efficiency at equal quality: **~17× on glm** (`kb-03078`, uncached ollama endpoint) and **~8× raw / ~7.6× billed on sonnet** (`kb-03102`, both harnesses caching the real Anthropic API). A strong, cheap-to-tell result worth a blog writeup — the sharpened story is "harness overhead is real *and survives caching*, but the headline multiple is iteration-sensitive."
- **Unify runner fixture discovery** — `coding_eval` discovers all 10 fixture dirs (the 4 legacy ones without `task.json` run but without holdout, shown `-`) while `claude_code_eval` runs only the 6 with `task.json`. Either give the 4 legacy fixtures `task.json` + holdout or exclude them from `coding_eval` so the two runners cover the same set.
- **Token + cost budget caps** — deferred from 0.4.0 (which shipped wall-clock only).
  Token caps are inscrutable (no human-legible right value per task); cost caps are
  blocked on a token→price table that doesn't exist (`consumed.cost_micros` is never
  incremented). Revisit token caps only with a concrete reason; cost caps once pricing
  is wired.
- **Streaming/SSE** — cost/latency, not capability; when the live-run volume
  justifies it. (Prompt caching shipped in v0.6.0 — `98fe789`, `kb-03102`.)
- **In-run context compaction** — summarize/evict old turns as a single run
  approaches its context window, the way Claude Code auto-compacts. Talos does
  none today: it grows the conversation until the window is hit, then either
  errors (pre-flight guard, once `num_ctx` is pinned) or — the bug we just
  fixed — silently truncates. **Explicitly not planned yet.** For dispatch-size
  work a model's real window is huge (glm 1M, qwen 256k, now pinned), and the
  right lever *before* compaction is to decompose work into smaller tasks;
  Ralph's fresh-context-per-iteration is the pattern-level answer for long
  *objectives*. Revisit only if a single indivisible task genuinely overruns a
  1M window.
- **Ralph Loop — ✅ shipped (core `1b4c2bb` / 0.7.0, CLI `talos ralph` / 0.8.0, do-over fix `242a57f`), now growing.** Real and dogfood-proven (see "Where we are" for the dng-converter run). Forward directions:
  - **Ralph-ability characterization** (`kb-03109`) — the design heuristic for *which* tasks fit: a **static prompt that re-binds as external state mutates** (coverage %, a checklist, a failing-test list, a grep). Five requirements (monotone external state · pure-function-of-state prompt · cheap unambiguous stop-oracle · progress durable outside the context · units that fit one inner budget). "Write the highest-value missing test" is the canonical small-model case.
  - **`tasks.md` backlog executor (next experiment)** — the mid-tier (talos-glm) instance of the pattern: prompt = "complete the next unchecked task in `tasks.md`, mark it complete," stop-when = "all boxes checked." Turns Ralph into a generic autonomous project executor over a groomed backlog. Gated on the do-over fix (item `230f9e9b`) — that fix is the *enabling prerequisite*: without it one over-budget task corrupts the run; with it, a too-hard task gets 3 clean do-overs then loudly stops for a human. A `tasks.md`-specific v2: "mark blocked + skip to next" instead of halting the whole loop.
  - **Ralph dispatch mode** — `talos ralph` is local-only today; the always-planned next step is running a Ralph objective as a headless dispatch on a *remote* host, so the fleet (not a laptop) grinds an overnight objective. This is what keeps GTD relevant alongside `tasks.md`+Ralph: dispatch-to-remote is a must-have, and the task board is for human organization/visibility — Ralph and GTD-dispatch are complementary (GTD dispatches each item as a separate reviewed agent; Ralph grinds a whole objective in one self-restarting loop).
- **Remaining v1-design tools**: `search_code`, `comment` (the design's tools 7–8).
- **LLM-judge evidence tier** (`Evidence::Judge`) — deferred from v1 by design.
- **Model-routing policy** (open decision #6) — blocked on eval data (haiku
  floor-run, per-trial metrics).
- **OS-level sandboxing** — explicitly v2 (threat model: our own tasks on our own
  infra; blast-radius bounds + creds hygiene are the v1 answer).
