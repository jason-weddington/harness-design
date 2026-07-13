# Roadmap

What's next and why, in priority order. Milestones are **capability themes**, not
date promises — a release cuts when its theme's capability is real and measurable
(the v0.1.0 rule: the first CHANGELOG line of a release should claim a capability,
not an engineering milestone). Numbers beyond the next milestone are provisional;
we re-order when we learn something.

Living document: updated at session boundaries. The per-session narrative lives in
[`session-summaries.md`](./session-summaries.md); decisions of record live in the
KB (`project_ref: harness-design`).

## Where we are — v0.5.0 (2026-07-13)

The finish-discipline safety net is complete, and the harness-vs-model claim has an instrument. 0.4.0's **finish-recovery** rescued a done-but-unclaimed *spin* (model keeps acting, gates green, tree static for K iterations); v0.5.0 closes the other half — the *stop-cold* halt (a no-tool-call turn) where the model verified green then quit without claiming. The eval exposed the gap the honest way: on the 6-model matrix (`kb-03019`) finish-recovery fired **zero** rescues (it only caught the spin), a one-line instrument (`RunStats.gates_green_at_exit`) then classified **~43%** of a weak model's stops as post-green — verified work abandoned (`kb-03033`) — and the **stop-nudge extension** now nudges at the `StoppedWithoutFinish` terminal too. The harness still never fabricates a `Done`; the claim still moves up to the lead.

The session also turned the harness-vs-model anecdote into a **benchmark**: a `claude_code_eval` runner drives claude-code-glm over the same fixtures, scored by the same sealed holdout as Talos, so the two harnesses compare 1:1 on the same model. First result (`kb-03078`) flipped the expected story — both harnesses **saturate** the current fixtures (18/18, holdout 18/18, 0 false-dones), so the pass-rate gap lives only at genuine dispatch scale; but Talos is **~17× more token-efficient** than Claude Code at identical quality. The harness is the variable — on cost here, not pass rate.

~394 tests, ~97% coverage; zero false dones across the whole eval history (180+ trials). **Next: 0.6.0**, the unsupervised GTD build-engine adapter (self-git, comment-back, the full engine contract) — the milestone this project has been building toward.

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

## 0.6.0 — the GTD build-engine adapter

**Theme: the point.** The adapter that picks up a groomed Agent GTD item, clones
the target repo into the workspace, runs the loop with the project's check
command, pushes a feature branch, and comments back — the harness as a real
headless-dispatch build engine alongside Claude Code. Order matters: this lands
*after* durability and bounds because dispatch hosts restart and nobody reviews a
runaway. 0.3.5 front-runs the supervised version of this; 0.5.0 is the
unsupervised completion — self-git, comment-back, and the full engine contract.

## Backlog (unscheduled, captured)

- **Dispatch-scale fixture tier** — to reproduce the *pass-rate* harness gap in-eval. Session 9 shipped the harness-vs-model benchmark and two "hard" fixtures (tokenbucket withheld-test + eventbus multi-file), but glm saturates them under *both* harnesses (`kb-03078`): the gap lives only at genuine dispatch scale (the 18-AC/5-file item), and a withheld-test spec precise enough to grade unambiguously is also easy to implement (precision-to-grade removes the difficulty). Reproducing the gap needs many-file, high-navigation fixtures — a real authoring effort, and the design challenge is difficulty-without-ambiguity.
- **The ~17× cost finding** (`kb-03078`) — Talos vs Claude Code token efficiency at equal quality; a strong, cheap-to-tell result worth a blog writeup (post 5, or fold into the benchmark story).
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
