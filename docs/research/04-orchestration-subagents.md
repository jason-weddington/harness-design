# Orchestration & Subagents: Single Agent vs Many

*Researched 2026-06-20.*

The central question for an agent harness: when should one agent loop be split
into many? The field has a genuine, well-articulated disagreement on this —
Anthropic publicly champions multi-agent fan-out for research, while Cognition
publicly argues against multi-agent for build tasks. Both are right, in their
domains. This doc reconstructs both positions from primary sources, reconciles
them, and translates the result for a **headless, unattended, coding/build**
harness.

---

## 1. The case FOR fan-out: Anthropic's multi-agent research system

Anthropic's [*How we built our multi-agent research
system*](https://www.anthropic.com/engineering/multi-agent-research-system)
describes the **orchestrator-worker** pattern powering Claude's Research feature:
a **lead agent** analyzes the query, develops a strategy, and **spawns subagents
that explore different aspects simultaneously**, each in its **own independent
context window**, returning **compressed findings** to the coordinator.

Why a multi-agent shape at all? Because research is *open-ended and
path-dependent*: "You can't hardcode a fixed path for exploring complex topics,
as the process is inherently dynamic." The agent must adjust strategy as findings
emerge.

The headline result: a multi-agent system with **Claude Opus 4 as lead and Claude
Sonnet 4 as subagents outperformed single-agent Opus 4 by 90.2%** on Anthropic's
internal research eval — driven by breadth-first queries that benefit from
parallel exploration.

**Why it works — the mechanism is context, not cleverness.** Token-usage analysis
found that **token usage alone explains ~80% of performance variance** on
BrowseComp (with tool calls and model choice the secondary factors, ~95%
combined). Multi-agent architectures help precisely because they **multiply
effective context**: each subagent burns its own window on a sub-question and
returns only a compressed summary. "Subagents... enable compression by operating
in parallel with their own context windows, exploring different aspects of the
question simultaneously." This is the durable insight — *subagents are a context
isolation / compression mechanism first, and a parallelism mechanism second.*

**The cost.** This performance is bought with tokens: agents use ~4× the tokens of
chat, and **multi-agent systems use ~15× the tokens of chat**. Anthropic is
explicit that this only pencils out "for tasks where the value of the task is high
enough to pay for the increased performance."

**Where Anthropic says multi-agent does NOT fit** (their own caveats, not a
critic's): domains "that require all agents to share the same context or involve
many dependencies between agents," explicitly naming **"most coding tasks"** as
having "fewer truly parallelizable tasks than research." Also: write-heavy
operations requiring shared state, and tasks needing real-time inter-agent
coordination, because "LLM agents are not yet great at coordinating and delegating
to other agents in real time."

**Coordination is fragile even when it fits.** Anthropic reports emergent
failures: "small changes to the lead agent can unpredictably change how subagents
behave." Early systems spawned too many subagents for trivial queries, duplicated
work (one subagent researched the 2021 chip shortage while another did 2025
because the brief said only "research the semiconductor shortage"), and chased
nonexistent sources. Fixes were heavy prompt engineering — giving each subagent an
**objective, output format, tool/source guidance, and clear task boundaries**, and
embedding explicit **scaling rules** (simple fact-find = 1 agent / 3–10 tool
calls; comparison = 2–4 subagents; complex research = 10+). Their lead executes
subagents **synchronously** — it waits for each wave before proceeding, which
blocks real-time steering but keeps state consistent.

---

## 2. The case AGAINST fan-out: Cognition's "Don't Build Multi-Agents"

Walden Yan's [*Don't Build Multi-Agents*](https://cognition.com/blog/dont-build-multi-agents)
(Cognition, makers of Devin) is the counterweight, written from the perspective of
**stateful, long-horizon build work**. It directly critiques OpenAI Swarm and
Microsoft AutoGen for promoting decentralized multi-agent designs. Two principles
anchor the argument:

1. **"Share context, and share full agent traces, not just individual messages."**
   It's not enough to pass a subagent the original task; production work is
   multi-turn with tool calls that accrete contextual dependencies. Passing
   summaries loses the nuance.
2. **"Actions carry implicit decisions, and conflicting decisions carry bad
   results."** When parallel agents each act on their own slice, they silently
   make incompatible assumptions.

The vivid failure case: a parent asks two subagents to build a Flappy Bird clone.
One builds a **Super Mario–style background**, the other builds a bird in a
visually incongruous style; the parent inherits two pieces that don't compose.
"Most real-world tasks have many layers of nuance that all have the potential to be
miscommunicated."

Yan's prescription:
- **Single-threaded linear agents** that carry continuous context — the principles
  hold "for free" when there's one unbroken trace.
- For tasks too long for one window, use a **dedicated context-compression model**
  that distills history into key decisions/events — but he warns this "is hard to
  get right."
- His blunt conclusion: "agents today are not quite able to engage in this style of
  long-context proactive discourse" that reliable parallel collaboration requires.

---

## 3. Reconciling the two — they're answering different questions

The disagreement is real but **largely resolves along a read/write,
parallelizable/dependent axis**. Both camps actually agree on the underlying
physics; they just sit on opposite ends of it.

| Dimension | Favors fan-out (Anthropic-style) | Favors single linear agent (Cognition-style) |
|---|---|---|
| Work shape | **Read-heavy** gather/search/compress | **Write-heavy** mutate shared artifact |
| Decomposition | Independent sub-questions | Decisions depend on each other |
| State | Each subagent's findings are self-contained | One evolving codebase/spec all parts must agree on |
| Failure of a bad split | A weaker answer | Incompatible code that doesn't compose |
| Coordination need | Low (synchronous merge of summaries) | High (every action constrains later actions) |

Crucially, **Anthropic itself names "most coding tasks" as the wrong fit for naive
fan-out** — so this is less a contradiction than two groups describing the same
boundary from opposite sides. The reconciliation:

- **Fan out the reading; serialize the writing.** Research, code search, "find all
  callers of X," reading docs, reproducing a bug across many files — parallelize
  freely; results compress to summaries and conflicts are cheap.
- **Keep the build itself single-threaded over shared state.** The actual edits to
  a codebase carry implicit, interdependent decisions (interfaces, naming,
  contracts). One writer per module is the safe rule; the moment two agents edit
  toward a shared contract in parallel, you're in Flappy Bird territory.

The 2026 industry consensus tracks this. The [Augment Code orchestration
guide](https://www.augmentcode.com/guides/multi-agent-orchestration-architecture-guide)
frames the decision test as: **"use multi-agent orchestration when tasks exceed a
single context window; stick with single agents for narrower work."** A single
agent with a concatenated toolbox is the default; you escalate to fan-out only
when context overflow or genuine independence forces it. Their recommended build
shape is a **DAG decomposed into waves** (same-level tasks run in parallel,
next wave waits), with **one-writer-per-module** and **isolated Git worktrees** to
prevent write collisions — i.e. parallelism that is engineered to *avoid* the
shared-write conflict Cognition warns about.

---

## 4. Subagents as context isolation (the part that matters even for one builder)

The most useful reframe for a *coding* harness: **subagents are valuable as a
context-hygiene tool even when the build is single-threaded.** This is the part of
the multi-agent toolkit that survives Cognition's critique, because it doesn't
require parallel writers.

Per [practical 2026 guidance on Claude Code
subagents](https://www.tembo.io/blog/claude-code-subagents), a subagent runs in its
**own context window with a custom system prompt, a scoped tool list, and
independent permissions**, and when it finishes, **only the result returns to the
parent** — "the 4,000-line stack trace, the 15 grep results, and the false leads
all stay in the subagent's transcript." Subagents have **no shared memory and no
lateral communication** by default — that isolation *is the point*.

This yields several patterns directly applicable to an unattended build engine:

- **Read-only explorer subagent** (tools: read/grep/glob only) does the messy
  exploration — "where is auth handled, what calls this function, what does the
  test harness expect" — and returns a tight summary. The builder's main context
  never fills with grep noise.
- **Reviewer / verifier subagent** as *verification-by-isolation*: a fresh
  read-only instance reviews the diff with no memory of *why* the builder made each
  choice, which structurally avoids the confirmation bias of self-review in the
  same context. This is the single most important pattern for a no-human-in-loop
  harness (see §6).
- **Tool/permission scoping by role**: read-only agents get `Read/Grep/Glob`;
  write-capable agents get `Edit/Bash` so they can reproduce-patch-retest. Where
  the tool list can't express the constraint (e.g. "read-only SQL"), a
  `PreToolUse`-style hook enforces it.

Caveats from the same sources: subagent spin-up adds **latency** (bad for quick
edits), subagents **cannot spawn their own subagents** (no deep recursion), and
**no continuous state** unless memory is explicitly enabled.

---

## 5. How fan-out actually fails: the MAST taxonomy

The strongest empirical grounding for "fan-out hurts when misapplied" is
[**MAST (Multi-Agent System Failure Taxonomy)**](https://arxiv.org/pdf/2503.13657),
a UC Berkeley study (NeurIPS 2025) that hand-annotated **1,600+ execution traces
across 7 multi-agent frameworks** (Cohen's κ = 0.88) into **14 failure modes**
under three root categories:

- **Specification & system design — ~41.8%**: task misinterpretation, ambiguous
  roles, poor decomposition, duplicate agent roles, **missing termination
  conditions**.
- **Inter-agent misalignment / coordination — ~37%**: agents talking past each
  other, conflicting actions (the Flappy Bird family of failures).
- **Verification gaps — ~21%**: weak or absent checking of the result.

Two takeaways for a harness designer. First, **most multi-agent failures are not
model-capability failures — they're orchestration failures** (bad specs, bad
coordination, missing stop/verify), exactly the layers a harness owns. Second, the
single largest bucket is *specification* — which is an argument for investing in
the brief/handoff contract before adding any parallelism.

Mitigations the field converges on (Augment Code, [Spring AI subagent
orchestration](https://spring.io/blog/2026/01/27/spring-ai-agentic-patterns-4-task-subagents/),
[beam.ai production patterns](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production)):
**schema-validated structured handoffs** (force subagents to return JSON, not
prose), **boolean exit gates** at handoffs (`tests_passed == true` before
proceeding), **hard turn/step caps** so a runaway loop bails before bankrupting the
run, a **living specification** as the external correctness anchor that survives
context resets, and **pin the cheapest competent model per role** (route on
Haiku-class, build on Sonnet-class, plan on Opus-class).

---

## 6. The unattended-coding lens — what changes with no human in the loop

For an interactive assistant, a bad fan-out is annoying but recoverable: the human
sees the Mario background and re-steers. **Our harness has no such backstop.** That
inverts several of the tradeoffs above.

- **The verifier subagent stops being optional.** In an interactive tool, the human
  *is* the verification gap that MAST's 21% bucket describes. Headless, the harness
  must own that 21% itself — and a fresh, isolated reviewer agent is the cleanest
  way to do it without confirmation bias.
- **Synchronous fan-out is a feature, not a limitation.** Anthropic notes its lead
  agent runs subagents synchronously and "can't steer in real time." For us there
  is no real-time steering anyway, so the synchronous-wave model (dispatch wave →
  wait → merge summaries → next wave) is exactly the right shape — deterministic,
  inspectable, no async state-reconciliation complexity.
- **The 15× token multiplier is a real budget line.** Fan-out must be governed by
  the harness's own cost/step limits, not by a human noticing the meter. Default to
  the cheapest shape that works and escalate to fan-out only on context overflow.

---

## Implications for our headless-dispatch harness

- **Default to a single-threaded build loop; treat fan-out as the exception, not
  the architecture.** Anthropic itself flags "most coding tasks" as the wrong fit
  for naive multi-agent, and Cognition's whole argument is that interdependent
  *write* decisions don't parallelize. One writer over the working tree is the
  spine; everything else is a context-isolation helper hanging off it.
- **Spend the fan-out budget on READING, never on parallel WRITING.** Read-only
  explorer subagents (codebase search, bug repro, "find all callers") run in
  parallel and return compressed summaries — pure upside, no shared-write conflict.
  The moment two agents would edit toward a shared contract, serialize them or use
  one-writer-per-module + isolated worktrees.
- **Make an isolated verifier subagent a mandatory pipeline stage.** Headless, the
  harness must own MAST's ~21% verification gap that a human would otherwise close.
  Run the project's own gates (build/lint/tests) AND a fresh read-only reviewer
  with no memory of the build's rationale, so it can't rationalize the diff.
- **Engineer the handoff contract before adding any parallelism.** MAST's largest
  failure bucket (~42%) is specification/decomposition. Each subagent dispatch must
  carry an explicit objective, scope boundaries, the set of tools it's given, and a
  **structured (JSON/schema-validated) return** — not prose. The brief is where
  fan-out lives or dies.
- **Bake in hard stop conditions and per-run budgets at the harness level.**
  Missing termination conditions and runaway loops are named MAST modes; with no
  human to hit Ctrl-C, turn caps, cost caps, and boolean exit gates
  (`gates_passed == true` before "ship the branch") are non-negotiable.
- **Abstract role→model binding so fan-out stays cheap.** Pin the cheapest
  competent backend per role (Ollama/Haiku-class for read-only explore and routing,
  Sonnet-class for the build, Opus-class only for planning/hard reviews). The 15×
  token cost of fan-out is only justified when the per-role model is right-sized —
  and our multi-backend abstraction is what makes that controllable.

## Sources

- [How we built our multi-agent research system — Anthropic Engineering](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Don't Build Multi-Agents — Walden Yan, Cognition](https://cognition.com/blog/dont-build-multi-agents)
- [Why Do Multi-Agent LLM Systems Fail? (MAST) — arXiv 2503.13657](https://arxiv.org/pdf/2503.13657)
- [Multi-Agent Orchestration: A Practical Architecture Without the Buzzwords — Augment Code](https://www.augmentcode.com/guides/multi-agent-orchestration-architecture-guide)
- [Claude Code Subagents: A 2026 Practical Guide — Tembo](https://www.tembo.io/blog/claude-code-subagents)
- [Spring AI Agentic Patterns (Part 4): Subagent Orchestration](https://spring.io/blog/2026/01/27/spring-ai-agentic-patterns-4-task-subagents/)
- [6 Multi-Agent Orchestration Patterns for Production (2026) — beam.ai](https://beam.ai/agentic-insights/multi-agent-orchestration-patterns-production)
- [Claude Code Subagents (2026) — Morph](https://www.morphllm.com/claude-subagents)
