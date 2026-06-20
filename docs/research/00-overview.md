# Harness Design — Research Overview

*Synthesized 2026-06-20.*

This is the entry point to the `harness-design` research corpus: seven tracks
surveying the mid-2026 state of the art in agent-harness design, all read through
one lens — **a Rust harness built from the ground up for headless dispatch.** No
human watches a run. A groomed task goes in; the agent runs unattended (often on
an always-on Raspberry Pi), implements the change, runs the project's own quality
gates, and ends by pushing a git feature branch and commenting back on a
task-tracker item. It targets **both the Anthropic API and local Ollama models**.
Every finding below is filtered through one question: *what changes when there is
no human in the loop and the machine might reboot mid-task?*

---

## State of harness design (mid-2026)

The field has converged on a clear and repeatedly-stated thesis: **the harness,
not the model, is where reliability is built.** LangChain moved a coding agent
from 52.8% to 66.5% on Terminal Bench 2.0 by changing *only the harness*; Anthropic's
own April postmortem traced a wave of "the model got dumber" reports to three
*harness-level* bugs that passed code review, unit tests, and e2e tests. Frontier
models are capable enough for most knowledge work — the failures that remain are
overwhelmingly **context failures** (the right knowledge wasn't in scope) and
**orchestration failures** (the workflow wouldn't let the model use its
capability). The MAST taxonomy puts numbers on it: ~42% of multi-agent failures
are specification/decomposition, ~37% coordination, ~21% verification gaps — i.e.
the layers a *harness* owns, not the model's IQ.

The canonical agent is uncontroversial and small: "an LLM, a loop, and enough
tokens." The model *decides* the next step; deterministic code *executes* it,
appends the result, and decides whether to continue. The live design questions
are not about the loop's shape but about everything wrapped around it.

Three tensions structure the current literature:

1. **How open should the loop be?** Anthropic's headline counsel is *start simple,
   prefer workflows, add autonomy only when needed* — predictability first. The
   Ralph community embraces a maximally open `while true` loop with fresh context
   per iteration, leaning on filesystem state and mechanical gates. They converge
   on the *same* safety mechanisms; they disagree on how much structure to impose
   around the model. Resolution-by-phase: scripted workflow for the predictable
   outer shell, an open (hard-bounded) loop only for the inner code-edit phase.

2. **One agent or many?** Anthropic's multi-agent research system beat single-agent
   Opus by 90.2% on breadth-first research — but at ~15× the tokens, and Anthropic
   *itself* names "most coding tasks" as a poor fit for naive fan-out. Cognition's
   "Don't Build Multi-Agents" argues the opposite for build work: actions carry
   implicit interdependent decisions that conflict when parallelized (the Flappy
   Bird failure). The disagreement resolves on a **read/write axis**: fan out the
   *reading* (search, repro, docs — conflicts are cheap), serialize the *writing*
   over shared code (decisions are interdependent). Both camps agree on the boundary.

3. **Can you trust the model's "done"?** Universal consensus: **no.** A self-declared
   "done" is the documented silent-failure mode. The terminal state must be
   *mechanical* — the project's own quality gates going green — never the model's
   self-report. This is the strongest single consensus in the corpus, and it is
   *load-bearing* for us because no human is there to catch a premature "done."

The strongest area of agreement, beyond "the harness is the product": **bound
everything, outside the model.** Every source on unattended operation independently
lands on multi-layered hard limits (iterations, tokens, wall-clock, per-tool) plus
progress/loop detection — enforced in code the model cannot talk its way past.

What's *new* in mid-2026 and not yet in most agents' training priors: the Anthropic
API drifted hard (`thinking: {budget_tokens}` now 400s; reasoning depth moved to
`output_config.effort`; `output_format` → `output_config.format`); server-side
**compaction**, **context-editing**, and the **memory tool** now form a productized
long-run stack; **code execution / "code mode"** (let the agent write a program that
calls tools, instead of emitting tool-JSON) became the dominant pattern for
multi-step tool use, with 80–98% token reductions; and **durable execution**
(checkpoint-and-replay) crossed over from the database world into agents.

---

## Reading list

Annotated and prioritized. **★ = must-read** (the 3–4 that most repay the time).

**★ [Effective harnesses for long-running agents — Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)**
 — The single most on-point primary source for us. Confronts the exact problem:
 discrete sessions, no memory between them, premature-"done," over-reach. Gives the
 concrete pattern (initializer window, progress log + `passes:false` feature
 checklist + git trail, one feature per window, forced end-to-end verification).

**★ [Building Effective Agents — Anthropic](https://www.anthropic.com/engineering/building-effective-agents)**
 — The canonical workflows-vs-agents framing, the five workflow patterns
 (esp. evaluator-optimizer), and the "start simple, prefer workflows" stance that
 anchors our workflow-around-open-loop architecture.

**★ [12-Factor Agents — humanlayer](https://github.com/humanlayer/12-factor-agents)**
 — The design spine: own your control flow (Factor 8), stateless reducer (12),
 unify execution + business state (5), compact errors into context (9), HITL as a
 tool call (7). Everything about durability and resumability falls out of these.

**★ [Demystifying Evals for AI Agents — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)**
 — Why **pass^k, not pass@1** is our metric (75%/trial → 42% pass^3), how to grade
 outcomes not trajectories, isolated trials, code-graders-primary. The blueprint
 for the thin eval harness we need before trusting the build engine.

[How to Build an Agent — Thorsten Ball](https://ampcode.com/how-to-build-an-agent)
 — The minimal, concrete loop in code. Read once to internalize how little the loop
 actually is; the intelligence lives in the model, the harness supplies the loop.

[Code execution with MCP — Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)
 + [Code Mode — Cloudflare](https://blog.cloudflare.com/code-mode/)
 — The "agent writes a program to call tools" shift (98.7% / 81% token reductions).
 Cloudflare's sharper claim: LLMs are fluent in code but stutter in tool-JSON. Most
 important for us because our agent's *job is writing code*.

[Writing effective tools for AI agents — Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)
 — Tools as a "contract between deterministic systems and non-deterministic agents."
 High-leverage consolidated tools, namespacing, descriptions-as-onboarding-docs
 (moved Claude to SOTA on SWE-bench), errors as a steering surface.

[Don't Build Multi-Agents — Cognition](https://cognition.com/blog/dont-build-multi-agents)
 + [Multi-agent research system — Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
 — Read as a *pair*; they are the for/against debate on fan-out. The reconciliation
 (read-parallel, write-serial) is the whole lesson.

[Why Do Multi-Agent LLM Systems Fail? (MAST) — arXiv 2503.13657](https://arxiv.org/pdf/2503.13657)
 — The empirical grounding that most agent failures are orchestration failures, with
 a 14-mode taxonomy. The ~21% verification-gap bucket is exactly what we must own.

[Effective context engineering for AI agents — Anthropic](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
 — Context as a finite, depleting "attention budget"; just-in-time over pre-loaded;
 structured note-taking. Pairs with the Chroma context-rot study below.

[Context Rot (Chroma, via ZenML) — 18-model study](https://www.zenml.io/llmops-database/context-rot-evaluating-llm-performance-degradation-with-increasing-input-tokens)
 — The evidence that recall degrades monotonically with input length *even on
 trivial tasks*. Why context bloat is a correctness risk, not just a cost one.

[How we built Claude Code auto mode — Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
 + [Claude Code sandboxing — Anthropic](https://www.anthropic.com/engineering/claude-code-sandboxing)
 — Containment beats prompting: 93% prompt-approval fatigue, a 17% classifier
 false-negative rate, and why OS-level sandbox (filesystem + egress proxy) is the
 real boundary. The `dontAsk` permission posture is our headless-correct mode.

[April 23 postmortem — Anthropic](https://www.anthropic.com/engineering/april-23-postmortem)
 — Three harness bugs degraded quality while passing every gate. The case for gating
 *harness changes themselves* behind evals + staged rollout.

[Claude Code in CI/CD and Headless Automation — hidekazu-konishi](https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html)
 — The closest reference implementation of headless plumbing: `-p`, stream-json
 tee'd to disk, `--session-id`/`--resume`, `--permission-mode dontAsk`,
 `--max-turns`/`--max-budget-usd`, env-var auth.

[The Ralph Loop — Blake Crosley](https://blakecrosley.com/blog/ralph-agent-architecture)
 / [Thomas Wiegold](https://thomas-wiegold.com/blog/ralph-loop-how-recursive-ai-agents-work/)
 — The concrete "run a coding agent overnight" pattern, almost exactly our shape:
 fresh context per iteration, filesystem+git as memory, machine-verifiable stop hook.

[Durable Execution: The Key to Harnessing AI Agents — Inngest](https://www.inngest.com/blog/durable-execution-key-to-harnessing-ai-agents)
 + [LangGraph durable execution — Vadim](https://vadim.blog/durable-execution-agents-that-survive-failure-and-resume-where-they-left-off)
 — Checkpoint-and-replay, and the **idempotency landmine** (replay re-runs the step
 you crashed inside — sharp for `git commit`/`install`/file writes).

[Don't Break the Cache (arXiv 2601.06007)](https://arxiv.org/pdf/2601.06007)
 + [Prompt caching — Claude API docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
 — 41–80% cost / 13–31% TTFT wins, but *cache the system prompt, not the whole
 evolving transcript* (naive caching can increase latency). The Rust gotcha: random
 JSON key order silently breaks the byte-exact prefix cache.

[Claude API docs](https://docs.claude.com) — caching, [adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking),
 [effort](https://platform.claude.com/docs/en/build-with-claude/effort),
 [tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview),
 [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs),
 [compaction](https://platform.claude.com/docs/en/build-with-claude/compaction),
 [context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing),
 [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool),
 [models](https://platform.claude.com/docs/en/about-claude/models/overview)
 — The authoritative current API surface. Read these, not your training priors — the
 API drifted hard and a from-memory harness will 400 on every request.

[Ollama tool-calling](https://docs.ollama.com/capabilities/tool-calling)
 + [Ollama num_ctx](https://markaicode.com/ollama-context-length-extend/)
 — Tools port without translation (same JSON-Schema shape), but the headline gotcha:
 `num_ctx` defaults to 2048 and **silently drops oldest messages** when exceeded.

---

## Cross-cutting themes

These recurred across three or more tracks and are the load-bearing ideas:

- **The harness is the product; the model is a component.** Reliability, not raw
  capability, is the deliverable, and it is engineered in the harness (loop, gates,
  budgets, verification, observability). Appears in tracks 01, 05, 06.

- **"Done" is mechanical, never declared.** The terminal/convergence signal is the
  project's own quality gates passing against an external checklist. The model's
  "I'm done" is at most a *trigger to run the gates*. The premature-"done" pathology
  is the single most dangerous headless failure. Appears in 01, 02, 05, 06.

- **Bound everything, outside the model.** Multi-layered termination (max-iteration,
  token/cost ceiling, wall-clock breaker, per-tool caps) plus loop/no-progress
  detection — enforced in deterministic code the model can't override. Appears in
  01, 03, 04, 05, 06.

- **Externalize state; the process is disposable.** Filesystem + git + task item +
  a structured run record are the source of truth; the in-context conversation is
  scratch. This is what makes runs resumable after a Pi reboot *and* what enables
  fresh-context restarts before quality degrades. Appears in 01, 02, 05.

- **Context is a depleting resource and a correctness risk.** Context rot is real,
  monotonic, and invisible to an unattended run. Smallest-coherent-context,
  just-in-time loading, and the compaction/context-editing/memory stack are
  correctness controls, not just cost optimizations. Appears in 02, 03, 05, 07.

- **Containment beats prompting.** With no human to absorb a classifier's false
  negatives, OS-level sandboxing + a tight allowlist + least-privilege creds are
  the real safety boundary. Permission *prompts* are useless when nobody answers.
  Appears in 03, 05, 06.

- **Verification by isolation.** A fresh, read-only reviewer with no memory of the
  build's rationale structurally avoids self-review's confirmation bias — the
  in-harness stand-in for the human who reads the PR. Appears in 01, 04, 06.

- **Fan out reading, serialize writing.** The reconciliation of the multi-agent
  debate, and the rule for any parallelism we add. Appears in 03, 04.

- **Capability is heterogeneous; detect, don't assume.** Even within Anthropic,
  feature support varies by model; Ollama backfills nothing. The harness owns far
  more model-layer work for local models. Appears in 02, 03, 07.

---

## Implications for a headless-dispatch-first harness

Prioritized and opinionated. Where our headless-first stance forces a *different*
choice than an interactive assistant would make, it's called out as **[≠ interactive]**.

1. **Own a thin explicit Rust loop with a `match` over the model's action; own the
   HTTP/JSON layer too.** The switch is where budget checks, loop-detection, and
   verification gates get injected — never delegate control flow to a framework.
   Build the agent as a stateless reducer over a single serializable run record so a
   killed Pi run resumes. **Do not depend on a community Anthropic Rust SDK** — the
   API drifted hard in 2025–26 and a lagging crate 400s every request; `reqwest` +
   `serde` + an SSE parser keeps us one edit from any new field. Ollama rides the
   same transport.

2. **Make "done" the project's own gates passing — never the model's self-report —
   and bail-with-report on failure. [≠ interactive]** An interactive agent can say
   "I think that's it" and let the human judge. Ours cannot. Terminal state =
   compile + lint + type-check + tests + a runtime smoke test, run by the harness
   against an external checklist. Treat gate-tampering (weakening a test to pass) as
   a first-class detected failure. A run that can't pass the gates ends by commenting
   what it tried and why it's stuck on the task item — *not* in a false success and
   *not* in an open-ended wait.

3. **Ship multi-layered termination ON by default. [≠ interactive]** Max-iteration
   cap (3–5× expected), hard token/cost ceiling that kills the run, per-tool error
   cap (~3 consecutive), wall-clock circuit breaker, plus cheap structural
   loop-detection (repeated tool+args; diff/lint unchanged across K iters; same test
   failing K times). Every limit is a hard counter with a clean failure exit, not a
   prompt. There is no human to hit Ctrl-C on a runaway loop, so these are
   non-negotiable, not nice-to-haves.

4. **Sandbox is the safety net that replaces the absent human. [≠ interactive]**
   `dontAsk` + tight tool allowlist + OS-level sandbox (filesystem isolation +
   egress proxy), least-privilege git creds held by the supervisor and never visible
   to generated code, deny rules as the hard floor. **Never `bypassPermissions`** (it
   approves everything unlisted). Enforce hard per-run blast-radius caps (files,
   commands, packages). An interactive assistant leans on the human approving the
   risky call; we must assume the model will eventually do something harmful and
   contain it structurally. In Rust: wasmtime/WASI or a Firecracker microVM for any
   code-execution runtime.

5. **Architect as workflow-around-open-loop, not one big open loop.** Hardcode the
   predictable outer sequence (orient → edit → run gates → fix → commit → push →
   comment); reserve the bounded open loop for the inner code-edit/fix phase only.
   This honors Anthropic's "prefer workflows" guidance exactly where unattended risk
   is highest, and gives clean step boundaries for durable checkpointing.

6. **Run is a stateless reducer over a SQLite-backed record; checkpoint per step
   around side effects.** One serializable record (control-flow position, messages,
   tool results, business facts, budgets, config) keyed by a deterministic run id,
   plus an append-only stream-json event log tee'd to disk *before* parsing (audit
   trail + live-observe feed + replay source). Draw step boundaries around side
   effects; sync-checkpoint git/filesystem steps; mind the **idempotency landmine**
   (replay re-runs the step you crashed inside — prefer git ops and full-file writes,
   which are naturally re-runnable). SQLite is the right embedded store for a Pi.

7. **Default to a single-threaded build loop; fan out only reading.** One writer over
   the working tree is the spine. Read-only explorer subagents (search, repro, "find
   all callers") run in parallel and return compressed summaries — pure upside. Never
   parallel-write toward a shared contract. **Make an isolated read-only verifier
   subagent a mandatory stage** (fresh instance, no build-rationale memory) — headless,
   the harness must own MAST's ~21% verification gap a human would otherwise close.
   Synchronous wave-based fan-out is the right shape for us, since there's no
   real-time steering to lose anyway. **[≠ interactive]**

8. **Treat context as a correctness control, and run the full long-run stack.**
   Layer most-stable → least-stable with cache breakpoints at tier boundaries;
   enforce deterministic JSON serialization (`BTreeMap`/ordered structs, never
   `HashMap`) so the byte-exact prefix cache actually hits; assert
   `cache_read_input_tokens > 0` as an alarmed metric. Use server-side compaction +
   context-editing + memory for Anthropic and hand-roll the equivalents for Ollama.
   Make cross-window handoff durable and external (progress log + `passes:false`
   checklist + git trail + `init.sh`, **one feature per window**); use compaction
   only for in-window pressure relief, never as the handoff carrier. Inject a
   ~70%-budget "wrap it up" checkpoint — the harness must supply the signal a human
   otherwise gives. **[≠ interactive]**

9. **Make code execution a first-class mode; design tools and their errors as the
   only corrective channel.** Our agent's job *is* writing code, so scripting tools
   beats tool-JSON for the multi-step tasks that dominate (and is the portable
   lowest-common-denominator for weaker local models). Progressive tool disclosure
   (defer-load + a search surface). High-leverage consolidated tools (`run_tests`
   returns only failures + context), actionable steering errors, human-readable IDs,
   preview+offload+advertised-continuation for large outputs (~25K cap, persist full
   result, return a path). Description-tuning is real engineering — it moves coding
   benchmark scores.

10. **Route by capability; abstract over both backends behind one trait.** Build a
    `Capabilities` struct per backend (Anthropic from the live Models API; Ollama
    from a static registry + empirical checks) and gate every feature off it.
    Dispatch mechanical/single-step tasks to local Ollama (free, private), reserve
    frontier for multi-step/judgment work where a wrong turn is expensive and
    unwatched. Treat Ollama `num_ctx` silent truncation as ship-blocking: set it
    explicitly, token-count every prompt against it, refuse-or-prune before
    truncation. Use structured outputs (`output_config.format`) as the harness↔tracker
    disposition contract; re-validate on Ollama where adherence is best-effort.

11. **Build a thin eval harness *before* trusting the build engine, graded with
    pass^k.** 20–50 tasks mined from real dispatch failures, isolated per trial
    (clean git worktree each time), code-graders primary + calibrated LLM-judge
    secondary, chaos injection (500s, malformed output) to prove the recovery loop.
    Every failed unattended run becomes a permanent regression case — the flywheel.
    Emit OpenTelemetry GenAI-convention traces (one thread/session per run) so a
    later *interactive* session can review a failed branch. **Gate harness changes
    themselves behind the eval suite** — the April postmortem proved a harness change
    can degrade quality while passing every test.

---

## Open questions / decisions for our build

Framed as decisions we must make next.

1. **Code-execution sandbox technology.** Decide between wasmtime/WASI (lighter,
   in-process, weaker isolation for arbitrary native tooling) vs. a microVM
   (Firecracker — stronger isolation, heavier, questionable on a Pi 5) vs. a plain
   jailed working dir + tool allowlist (simplest, weakest). The build engine needs
   to run `cargo`/`npm`/`pytest`, which strains pure-WASI. *Likely: container/jail +
   allowlist on the Pi for v1; revisit if blast-radius proves insufficient.*

2. **How much code-mode vs. direct tool-calling in v1.** Code-execution wins appear
   at scale (>10 tools, multi-step); our v1 toolset may be small enough that direct
   tool-JSON is simpler. Decide the threshold and whether to defer code-mode.

3. **Subagent support in v1, or single-threaded only.** The isolated *verifier*
   subagent is high-value even single-threaded, but adds a second concurrent model
   context and orchestration surface. Decide whether the verifier ships as a true
   subagent or as a second sequential pass in the same harness.

4. **Run-record schema and checkpoint cadence.** Pin the exact serializable record
   shape, the deterministic run-id scheme, the step-boundary definition, and which
   steps get sync vs. async checkpoints. This is the foundational data model; most
   durability behavior falls out of getting it right.

5. **Cross-window handoff format.** Adopt Anthropic's progress-log + `passes:false`
   feature-checklist pattern verbatim, or design our own task-item-anchored variant
   (the GTD item already carries acceptance criteria). Decide whether the checklist
   lives on disk, in the run record, or on the tracker.

6. **Default model-routing policy.** Concrete rules for when a task goes to Ollama
   vs. Haiku vs. Sonnet vs. Opus — by task shape, prompt size vs. `num_ctx`, and
   step-count estimate. Needs a first-cut heuristic to start collecting data against.

7. **Local-model viability bar.** Empirically establish which task shapes a local
   model (Qwen 3.x class on a 16GB Pi) can actually one-shot reliably, given the
   `num_ctx`/VRAM ceiling, before we route real work to it. May be "none in v1."

8. **What "bail-with-report" looks like.** Define the terminal-escalation contract:
   the structured comment format, what partial state (if any) gets pushed as a WIP
   branch, and how the control plane distinguishes "blocked, needs a human decision"
   from "failed, retryable."

---

## Track docs

1. [The Agent Loop](01-the-agent-loop.md) — canonical loop, workflows vs. autonomous
   agents, stop/convergence conditions for unattended dispatch.
2. [Context Engineering](02-context-engineering.md) — context rot, prompt caching,
   compaction/context-editing/memory, cross-window handoff.
3. [Tools & Execution](03-tools-and-execution.md) — tool design, MCP, code-execution
   ("code mode"), large-output management, unattended tool safety.
4. [Orchestration & Subagents](04-orchestration-subagents.md) — single agent vs. many,
   the for/against fan-out debate, MAST, verification by isolation.
5. [Durability & Headless Operation](05-durability-and-headless.md) — stateless
   reducer, durable execution, the idempotency landmine, run representation.
6. [Reliability, Evals & Guardrails](06-reliability-evals-guardrails.md) —
   self-verification, error recovery, containment, pass^k evals, observability.
7. [Model-Layer Features](07-model-layer-features.md) — Anthropic API + Ollama,
   capability detection, graceful degradation, the Rust transport layer.
