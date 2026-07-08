# Roadmap

What's next and why, in priority order. Milestones are **capability themes**, not
date promises — a release cuts when its theme's capability is real and measurable
(the v0.1.0 rule: the first CHANGELOG line of a release should claim a capability,
not an engineering milestone). Numbers beyond the next milestone are provisional;
we re-order when we learn something.

Living document: updated at session boundaries. The per-session narrative lives in
[`session-summaries.md`](./session-summaries.md); decisions of record live in the
KB (`project_ref: harness-design`).

## Where we are — v0.2.0 (2026-07-08)

The same loop, unchanged, completes the fixture suite on non-Anthropic models —
the `ModelBackend` anti-corruption boundary held in anger. v0.1.0 established
claim-vs-verify (`finish(done)` is a claim the harness verifies itself; a
verified `Done` carries its evidence by construction); v0.2.0 added the Ollama
backend (one adapter for localhost + ollama.com) and per-trial metrics. The
cross-backend matrix (kb-02909 v4): GLM-5.2 12/12; sonnet, haiku, and local
qwen3.6:35b (think=on) all 11/12; gpt-oss:20b 9/12 — zero false dones in 60
verified trials. The suite is pass-rate-saturated four models deep; iteration
counts still discriminate (frontier 5.00 uniform, haiku ~7.9, qwen ~9.75).
Think config is a first-class routing knob — record it on every eval row.

## 0.3.0 — durability (persist, resume, dispose)

**Theme: survive the host.** The run record and `RunStore` shipped in v0.1.0 but
the loop doesn't use them yet. Wire checkpointing into the loop, unify the
loop-local `FinishDisposition` with `run_record::Disposition`, and implement the
two resume modes from the design (crash-resume; fresh-context restart). The
capability claim: *kill the harness mid-run, restart it, and the run completes* —
the deployment-agnostic promise (Pi, container, spot instance) made real.

## 0.3.5 — first dogfood (the harness builds the harness)

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

## 0.4.0 — bounded autonomy (budgets, retry, loop detection)

**Theme: safe to leave alone.** Token/cost/wall-clock budgets enforced in the
loop; retry/backoff on `Transient` errors (`is_retryable` has been waiting);
loop/no-progress detection beyond the blunt `max_iterations`. The capability
claim: *a pathological run terminates itself with a useful `Failed` disposition
instead of burning budget* — the last prerequisite for unattended operation.

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
- **Streaming/SSE + prompt caching** — cost/latency, not capability; when the
  live-run volume justifies it.
- **Remaining v1-design tools**: `search_code`, `comment` (the design's tools 7–8).
- **LLM-judge evidence tier** (`Evidence::Judge`) — deferred from v1 by design.
- **Model-routing policy** (open decision #6) — blocked on eval data (haiku
  floor-run, per-trial metrics).
- **OS-level sandboxing** — explicitly v2 (threat model: our own tasks on our own
  infra; blast-radius bounds + creds hygiene are the v1 answer).
