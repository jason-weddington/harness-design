# Roadmap

What's next and why, in priority order. Milestones are **capability themes**, not
date promises — a release cuts when its theme's capability is real and measurable
(the v0.1.0 rule: the first CHANGELOG line of a release should claim a capability,
not an engineering milestone). Numbers beyond the next milestone are provisional;
we re-order when we learn something.

Living document: updated at session boundaries. The per-session narrative lives in
[`session-summaries.md`](./session-summaries.md); decisions of record live in the
KB (`project_ref: harness-design`).

## Where we are — v0.4.0 (2026-07-11)

Bounded autonomy shipped — the harness is now safe to leave alone. **finish-recovery** detects a done-but-unclaimed spin (green gates + a static tree for K iterations), nudges the model to finish or give a one-sentence status, and on exhaustion terminates `Failed` while writing **recovery facts** so the worker preserves the WIP branch. The harness never fabricates a `Done` — the claim moves up to the lead, so claim-vs-verify stays inviolate while work that couldn't be claimed is still rescued. A **wall-clock budget** lets a run self-terminate gracefully in the margin before the dispatch worker's hard-kill; **bounded retry/backoff** rides transient errors. This built on 0.3.0 (durability — kill the harness mid-run, restart, the run completes) and 0.3.5 (first dogfood — the harness, running as an Agent GTD build engine named **Talos**, shipped merged changes to its own repo).

The session's load-bearing finding: **harness-vs-model, proven empirically.** Hold the model constant (glm) and swap the harness — Talos failed finish-recovery twice, `claude-code-glm` one-shot the exact same 18-AC item. When a strong model fails, suspect the harness first; `claude-code-glm` is now a proven zero-Anthropic lane for complex harness-core work while Talos matures. Talos also gained **fleet-publish**: the i9 builds both arches once and publishes to pi-04, hosts pull, retiring the compile-on-every-host tax — and `release.sh` now ships a fresh fleet artifact as part of every release (a release ships an artifact by definition).

389 tests, ~97% coverage; zero false dones across the whole eval history. **Next: 0.5.0**, the unsupervised GTD build-engine adapter (self-git, comment-back, the full engine contract) — 0.3.5 front-ran the supervised version.

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

## 0.5.0 — the GTD build-engine adapter

**Theme: the point.** The adapter that picks up a groomed Agent GTD item, clones
the target repo into the workspace, runs the loop with the project's check
command, pushes a feature branch, and comments back — the harness as a real
headless-dispatch build engine alongside Claude Code. Order matters: this lands
*after* durability and bounds because dispatch hosts restart and nobody reviews a
runaway. 0.3.5 front-runs the supervised version of this; 0.5.0 is the
unsupervised completion — self-git, comment-back, and the full engine contract.

## Backlog (unscheduled, captured)

- **Eval levers** (on the GTD board, from the saturation finding): harder task
  *shapes* — withhold-the-failing-test mode, prose-bug-report mode (the realistic
  dispatch shape); haiku floor-run for the model-routing question.
- **Token + cost budget caps** — deferred from 0.4.0 (which shipped wall-clock only).
  Token caps are inscrutable (no human-legible right value per task); cost caps are
  blocked on a token→price table that doesn't exist (`consumed.cost_micros` is never
  incremented). Revisit token caps only with a concrete reason; cost caps once pricing
  is wired.
- **Streaming/SSE + prompt caching** — cost/latency, not capability; when the
  live-run volume justifies it.
- **Remaining v1-design tools**: `search_code`, `comment` (the design's tools 7–8).
- **LLM-judge evidence tier** (`Evidence::Judge`) — deferred from v1 by design.
- **Model-routing policy** (open decision #6) — blocked on eval data (haiku
  floor-run, per-trial metrics).
- **OS-level sandboxing** — explicitly v2 (threat model: our own tasks on our own
  infra; blast-radius bounds + creds hygiene are the v1 answer).
