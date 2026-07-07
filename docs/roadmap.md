# Roadmap

What's next and why, in priority order. Milestones are **capability themes**, not
date promises — a release cuts when its theme's capability is real and measurable
(the v0.1.0 rule: the first CHANGELOG line of a release should claim a capability,
not an engineering milestone). Numbers beyond the next milestone are provisional;
we re-order when we learn something.

Living document: updated at session boundaries. The per-session narrative lives in
[`session-summaries.md`](./session-summaries.md); decisions of record live in the
KB (`project_ref: harness-design`).

## Where we are — v0.1.0 (2026-07-07)

The harness autonomously completes a real coding task and **cannot be lied to
about "done"**: `finish(done)` is a claim the harness verifies by running the
project's checks itself, and a verified `Done` carries its evidence by
construction. Model contract (Anthropic backend), confined workspace, six tools,
templated prompts, pass^k eval with per-trial isolation, four fixture crates.
Current eval: 12/12 with sonnet — saturated at this difficulty (the red test does
the localization; see the eval-levers backlog).

## 0.2.0 — a second backend (Ollama)

**Theme: prove the abstraction.** The `ModelBackend` trait is the project's
central bet — a normalized `AssistantTurn` out, each adapter an anti-corruption
layer owning all wire translation. One backend can't validate a boundary; the
second one is the test, and Ollama is deliberately the *hard* second: local
models emit tool calls with varying fidelity, so this milestone brings the
lenient/schema-aligned parsing idea lifted from BAML (`jsonish`-style coercion +
parse-retry as steering) into the adapter where it belongs. Target models:
GLM-5.2 first, Qwen3.6 as the small-host option. The capability claim: *the same
loop, unchanged, completes the fixture suite on a local model* — and the eval
suite tells us honestly how much worse (or not) that is.

Includes from the eval backlog: per-trial metrics in `EvalReport` (iterations,
tokens, wall-clock) — pass/fail alone can't compare backends once both pass.

## 0.3.0 — durability (persist, resume, dispose)

**Theme: survive the host.** The run record and `RunStore` shipped in v0.1.0 but
the loop doesn't use them yet. Wire checkpointing into the loop, unify the
loop-local `FinishDisposition` with `run_record::Disposition`, and implement the
two resume modes from the design (crash-resume; fresh-context restart). The
capability claim: *kill the harness mid-run, restart it, and the run completes* —
the deployment-agnostic promise (Pi, container, spot instance) made real.

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
runaway.

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
