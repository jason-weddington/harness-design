# Durability & Headless Operation

*Researched 2026-06-20.*

This is the core lens for the `harness-design` project: a Rust agent harness built **headless-dispatch-first**. No human watches a run. An agent is handed a groomed task, runs unattended to completion (often on an always-on Raspberry Pi), and ends by pushing a git feature branch and commenting on a task-tracker item. Every idea below is filtered through one question: *what changes when there is no human in the loop and the machine might reboot mid-task?*

---

## 1. The stateless-reducer framing (12-factor agents)

The [12-factor agents](https://github.com/humanlayer/12-factor-agents/blob/main/README.md) methodology is the canonical articulation of agent-as-reducer. The factors most load-bearing for a headless harness:

- **Factor 12 — "Make your agent a stateless reducer."** The agent is a pure function: `(state, event) → new_state`. It holds no durable state internally between calls; all state lives outside the function. This is the central insight for durability — if the agent step is a pure reducer over an externalized state object, then *any* process can re-run it, on any machine, after any crash, and get the same place. The agent process is disposable; the state is the asset.

- **Factor 5 — "Unify execution state and business state."** Don't keep a separate "agent runtime" state alongside your application's "what actually happened" state. Merge them into one serializable object: the current step, the accumulated tool outputs, the message history, the control-flow position, AND the business facts (which branch was pushed, which gate passed) all live in one record. For us, this means the run record is the single source of truth — there is no in-memory state that, if lost, can't be reconstructed from the persisted record.

- **Factor 6 — "Launch/Pause/Resume with simple APIs."** An agent should be startable, interruptible, and resumable from its last checkpoint through a small API surface, "without rebuilding execution context." Pause/resume is not a HITL nicety — it is the *same primitive* that makes crash recovery work. A run that can pause for a human can equally pause for a reboot.

- **Factor 8 — "Own your control flow."** Write your own orchestration loop rather than delegating it to a framework's hidden loop. The 12-factor argument is about avoiding lock-in; for a headless harness it's also about *auditability and control*: when no human is watching, the loop itself must own stop-detection, budget enforcement, and recovery. You cannot do that if the loop is a black box inside a dependency.

- **Factor 7 — "Contact humans with tool calls."** When the agent needs human input, it does so via a normal tool call (e.g. `request_human_input`) rather than a special blocking control path. This is the key to the HITL↔durability mapping (Section 4): a "ask the human" call and a "wait for an event" call are the same suspend primitive. *In our world there is no human to contact mid-run* — so this tool call instead becomes a **terminal escalation**: the agent posts a blocker comment on the task and exits, rather than blocking forever.

- **Factor 11 — "Trigger from anywhere."** Webhooks, cron, PR comments, task-tracker events. Headless dispatch IS this factor: the trigger is a groomed GTD item, not a chat message.

- **Factor 3 — "Own your context window."** Deliberate context engineering, not framework defaults. Critical for long unattended runs where context degrades (Section 5).

**Why it matters for us:** the stateless-reducer + unified-state pair is the design that makes everything else (crash recovery, resume after laptop-sleep, audit-after-the-fact) fall out almost for free. If we get the run-record shape right, durability is mostly bookkeeping.

---

## 2. Durable execution — guaranteeing progress despite crashes

"Durable execution" is the production database community's answer to exactly our problem, now being [rediscovered by the agent world](https://nittikkin.medium.com/agent-workflows-are-rediscovering-durable-execution-be110661ed8c). The definition, per [Inngest](https://www.inngest.com/blog/durable-execution-key-to-harnessing-ai-agents): *"code that automatically persists its state at defined checkpoints and can resume from those checkpoints after any failure."*

The shared model across the major engines:

- **Steps are the unit of durability.** Work is decomposed into steps. Each step's *result* is checkpointed to durable storage when it completes. On restart, the engine **replays** from history: every already-completed step is skipped and its recorded result is read back rather than re-executed; execution resumes at the first incomplete step. [Temporal](https://temporal.io/) describes this precisely: a worker crashing at step 5 of 10 is replaced by a new worker that "replays the event history from the beginning… every completed Activity call is skipped — instead of re-executing, the worker reads the recorded result from history," then resumes at step 5 ([AgentMarketCap analysis](https://agentmarketcap.ai/blog/2026/04/10/durable-agent-execution-production-temporal-modal-event-sourced)).

- **The financial argument is specific to agents.** Per Inngest: *"LLM calls are expensive. Re-running them on every retry doubles or triples your inference costs. Durable execution's caching behavior means you pay for each LLM call exactly once."* For us on metered Anthropic API + a slow Pi, an un-checkpointed crash that re-runs a 40-turn task is real money and real wall-clock. Checkpointing reportedly [cuts wasted processing 60%+](https://agentmarketcap.ai/blog/2026/04/10/durable-agent-execution-production-temporal-modal-event-sourced) on multi-step workflows.

- **Per-step retries with backoff.** Each step retries independently with configurable backoff, without re-running prior successful steps. [Cloudflare Workflows](https://blog.cloudflare.com/workflows-ga-production-ready-durable-execution/) exposes this as `{ retries: { limit: 3, delay: '30 seconds', backoff: 'exponential' }, timeout: '2 minutes' }` per step.

- **Exactly-once semantics via persisted step tracking.** [DBOS](https://www.dbos.dev/blog/durable-execution-crashproof-ai-agents) frames the agent failure mode bluntly: *"Automated tasks that fail and do not resume, or that resume but re-run already-completed tasks, will undermine the benefits of AI automation."* DBOS decorates functions (`@DBOS.workflow`, `@DBOS.step`) and records progress in Postgres so that "progress is never lost – even in case of failures" with "no duplicate refunds, no lost state."

### The idempotency landmine (read this twice)

The single most dangerous detail for a *coding* agent: **replay re-executes the step you were inside when you crashed.** LangGraph's docs are explicit — *"Nodes after the checkpoint re-execute, including any LLM calls, API requests, or interrupts… Every side-effect must be idempotent"* ([Vadim's analysis](https://vadim.blog/durable-execution-agents-that-survive-failure-and-resume-where-they-left-off)). Cloudflare's model is finer-grained: only completed *steps* are skipped, so the boundary you draw around a step determines what gets re-run.

For a build engine this is sharp: a step that does `git commit`, `npm install`, or a destructive file write is **not** naturally idempotent. If the harness crashes after the side-effect but before the checkpoint write, replay does it again. Our mitigations:

1. Make step boundaries align with checkpoint writes, and keep the *side-effecting* part of a step as small and idempotent as possible.
2. Prefer operations that are naturally re-runnable (git is mostly forgiving; `git commit` of an already-committed tree is a no-op; file *writes* are idempotent, file *appends* are not).
3. Where a step is genuinely non-idempotent, persist an idempotency key / "already did this" marker *before* the side effect, or treat the working tree itself as the checkpoint (the filesystem is the state — see Ralph, Section 6).

---

## 3. Checkpointing primitives — concrete designs to copy

| Engine | Checkpoint store | Pause primitive | Replay semantics | Notes for us |
|---|---|---|---|---|
| **LangGraph** | `MemorySaver` (dev), `SqliteSaver` (file), `PostgresSaver` (prod) | `interrupt()` + `Command(resume=...)` | Re-executes the node after the checkpoint; side-effects must be idempotent | `thread_id` is the primary key; deterministic IDs (`"campaign-123"`) make resume reliable ([source](https://vadim.blog/durable-execution-agents-that-survive-failure-and-resume-where-they-left-off)) |
| **Cloudflare Workflows** | Automatic, backed by Durable Objects | `step.sleep` / `step.sleepUntil` / `step.waitForEvent` | Only completed steps skipped; state returned from each step auto-persisted | Sleeping workflow "consumes nothing" — DO hibernates and is woken on wake-time ([source](https://blog.cloudflare.com/workflows-ga-production-ready-durable-execution/)) |
| **Temporal** | Event-sourced history, replayed | signals / timers / `await condition` | Full deterministic replay of event history; completed activities read from history | Battle-tested; the replay-determinism constraint is strict |
| **DBOS** | Postgres rows | (workflow-level) | Resume from last completed step; exactly-once | Embeds in your process; no separate service |

**LangGraph's durability modes** are the most directly transferable knob ([source](https://vadim.blog/durable-execution-agents-that-survive-failure-and-resume-where-they-left-off)):

- `'exit'` — persist only on completion. Cheap, but a crash loses the whole run. Fine for short tasks.
- `'async'` — write checkpoints asynchronously mid-run. Good for long runs where a little replay is acceptable.
- `'sync'` — write synchronously *before* each step. Highest durability, for high-consequence operations.

For a build engine the right default is **`sync`-style checkpointing around any step that touches git or the filesystem**, and `async` is acceptable for read-only/analysis steps. The asymmetry matters: losing an analysis step costs a re-read; losing a "branch pushed" fact costs a duplicate push.

**Key constraint on step boundaries:** put non-deterministic / side-effecting work *inside* `step.do`-equivalent boundaries so its result is captured; keep the orchestration *between* steps deterministic so replay is faithful. This is the single design rule that makes durable execution work, and it's a Rust-friendly shape: a step is an async fn returning a serializable result that the harness records.

---

## 4. HITL as suspend/resume — and what changes with no human

The elegant insight from both 12-factor (Factor 7) and the durable-execution engines: **human-in-the-loop and crash-recovery are the same primitive.** A "wait for human approval" is just a `waitForEvent` / `interrupt()` that happens to be resolved by a person instead of a timer or a system event. [Cloudflare](https://blog.cloudflare.com/workflows-ga-production-ready-durable-execution/): a sleeping/waiting workflow hibernates and consumes nothing until woken. [Inngest](https://www.inngest.com/blog/durable-execution-key-to-harnessing-ai-agents)'s `waitForEvent` "suspends execution for human-in-the-loop patterns without consuming resources." LangGraph's `interrupt()` "works identically for both clock-based and human gates."

**What changes when there is no human:** the suspend-for-human path must *not* exist as an open-ended wait, because nothing will ever resolve it. Our adaptations:

1. **No blocking HITL waits.** Any place an interactive harness would `interrupt()` for human approval becomes, in our harness, one of: (a) a **policy decision the harness makes itself** (auto-approve within a pre-granted allow-list — cf. Claude Code's `--permission-mode dontAsk`, which auto-denies anything off the allow-list rather than hanging), or (b) a **terminal escalation** — the agent posts a blocker comment on the GTD item and exits cleanly. The run ends; it does not sleep forever waiting for a human who is asleep.

2. **Pre-grant the decisions up front.** The whole point of grooming a task before dispatch is to front-load the judgment calls so the agent never needs to pause for one. Permissions, allowed tools, scope boundaries, and the definition of done are decided at dispatch time, encoded in the run config, and enforced by the harness — replacing "the human grants permission, judges completion, decides to stop" with configuration ([the Claude Code CI guidance](https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html) calls this out explicitly).

3. **Resume is for crashes, not for humans.** We still want full pause/resume — but the resolver is always a system event: the Pi rebooted and a supervisor restarts the run; the laptop woke and reattaches; a transient API 529 cleared and the retry fires. The HITL machinery is repurposed entirely toward fault-tolerance.

---

## 5. Long-horizon unattended runs — keeping the agent on track without a human

Anthropic's [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) is the most directly relevant primary source, because it confronts the exact problem: *"The core challenge of long-running agents is that they must work in discrete sessions, and each new session begins with no memory of what came before."* Their recommendations, mapped to us:

- **Two-phase harness: Initializer then Coding agent.** First context window sets up infrastructure (clone, install, init.sh, baseline test run); subsequent sessions make incremental progress against a clean baseline. We can mirror this: a dispatch begins with a deterministic setup phase before the agent loop proper.

- **Externalize the to-do as a machine-checkable artifact.** Anthropic uses a JSON feature list where every required feature starts marked "failing," with the rule *"It is unacceptable to remove or edit tests."* This is the antidote to a specific unattended failure: *"a later agent instance would look around, see that progress had been made, and declare the job done."* Without a human to say "no, you're not done," **the stop condition must be external and machine-verifiable** — not the agent's own judgment.

- **Work one item at a time.** *"This incremental approach turned out to be critical"* to stop the agent over-reaching. Smaller steps also mean finer checkpoints and cheaper replay.

- **Continuity artifacts on disk.** `claude-progress.txt` + git history + `init.sh`. New sessions begin by reading git logs and progress files "to get up to speed," then run a smoke test to "catch any undocumented bugs" before doing new work. The **filesystem + git are the durable state**, which dovetails perfectly with the stateless-reducer model — the working tree IS the externalized state object.

- **Self-verification must be forced and end-to-end.** Anthropic found agents marked features complete without validation; only when *explicitly prompted to use browser automation and test as a human would* did verification become reliable. The lesson for our build engine: **never trust the agent's self-report of "done."** Completion is defined by the project's own quality gates passing (tests, linter, type-check, build), run by the harness, not by the model claiming success.

**Context degradation is real and measurable.** The [Ralph Loop](https://blakecrosley.com/blog/ralph-agent-architecture) reports context availability dropping from 200K to ~50K usable tokens over 120 minutes of continuous operation — a strong argument for fresh-context-per-iteration over one mega-session. Anthropic's [context-engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) covers compaction strategies to extend useful horizon.

---

## 6. The Ralph Loop — a concrete unattended-overnight pattern

The [Ralph Loop](https://blakecrosley.com/blog/ralph-agent-architecture) is the most concrete published "run a coding agent overnight, unattended" architecture, and it's almost exactly our shape. Its mechanics:

- **Fresh context per iteration.** `Iteration 1 [200K] → writes code/files; Iteration 2 [200K] → reads files from disk, continues.` Each cycle gets a full context budget instead of degrading through one long session. The agent reads filesystem state, does bounded work, writes back, and **terminates** — the harness restarts it for the next iteration. This is the stateless reducer made operational: the loop *outside* the model owns continuity; the model is disposable per turn.

- **Filesystem as persistent memory.** *"Files persist across context windows. My `.claude/` directory serves as the agent's persistent memory."* State files hold recursion depth, decisions, thresholds.

- **Machine-verifiable stop hook.** A stop hook *intercepts exit attempts and blocks termination until criteria pass* — and the criteria *"must be machine-verifiable: test pass/fail, linter output, HTTP status codes, file existence checks."* This is the single most important pattern for us: **stop-detection is enforced outside the model by a deterministic gate**, preventing both premature "done" and infinite looping. Vague criteria ("write tests") produce trivial output; specific criteria ("all 81 tests pass, consensus > 70%") drive real work.

- **Failure recovery via safe-default reconstruction.** *"If state gets corrupted… the recovery pattern recreates from safe defaults rather than crashing."* Validate state files on read; reinitialize rather than propagate corruption.

- **Spawn/iteration budget with inheritance.** A hard cap on total agents (default 12) via budget inheritance, after early unbudgeted runs burned tokens "at 10x normal rate." For us this is the iteration cap + cost cap.

- **Honest fit assessment.** Strong fit: greenfield implementation with clear specs and automated verification. Weak fit: subjective quality, exploratory work. Our build-engine remit (groomed task → implement → gates → branch) is squarely in the strong-fit zone.

---

## 7. Stop, budget, and runaway control (no human = the harness must catch the wrong turn)

This is the capability an interactive human provides for free and that we must build. Synthesizing the [2026 stop-condition consensus](https://datasciencedojo.com/blog/agentic-loops-explained-from-react-to-loop-engineering-2026-guide/), [BSWEN](https://docs.bswen.com/blog/2026-03-11-prevent-ai-agent-infinite-loops/), and [Modexa](https://medium.com/@Modexa/the-agent-loop-problem-when-smart-wont-stop-ccbf8489180f): runaway agents almost always trace to four causes — *no hard stop conditions, underspecified goals, context overflow, missing cost controls.* The defense is **multi-layered termination, enforced outside the model**:

1. **Hard iteration cap.** Set to 3–5× the expected step count. Claude Code's `--max-turns` is "your first line of defense against runaway cost," exiting non-zero when hit ([source](https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html)).
2. **Token/cost budget.** A hard USD ceiling per run (Claude Code's `--max-budget-usd`; Ralph's spawn-budget inheritance). Must be designed in from the start, not bolted on.
3. **Progress / convergence detection.** *"If the agent calls the same tool with nearly the same inputs repeatedly, it's stuck; if repeated 2–3 times, stop or switch strategy."* No-progress over N steps is a stop signal.
4. **Semantic / machine-verifiable completion check.** The quality gates passing (Ralph's stop hook, Anthropic's feature list). This is the *positive* stop — the only one we trust to mean "done."
5. **Time-based circuit breaker.** Wall-clock cap, important on a slow Pi.

All five live in the harness loop, not the prompt — the model cannot be the enforcer of its own stop condition.

### Self-verification before shipping

Anthropic's security-research [GAN-style loop](https://www.epsilla.com/blogs/anthropic-harness-engineering-multi-agent-gan-architecture) is instructive: a Generator proposes, a skeptical Evaluator tries to *disprove* it, and *"every finding goes through a self-correction loop where Claude attempts to 'disprove' its own vulnerability report to filter out false positives."* For our build engine the deterministic gates are the primary Evaluator (tests/lint/build), but a cheap adversarial self-review pass before pushing the branch is a strong fit — and cheaper than a wrong branch that wastes a human review later.

---

## 8. Headless agents in CI / sandboxed runners

The CI/sandbox world has already solved much of the "run an agent unattended" plumbing; we should steal liberally.

**Claude Code headless mode** ([detailed CI guide](https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html)) is the closest reference implementation:

- `-p/--print` = batch invocation, no REPL, "no follow-up turn from a human."
- `--output-format stream-json` = newline-delimited JSON, one object per event — the natural shape for **observing, resuming, and auditing** a run after the fact. Tee it to a file *before* parsing so raw bytes survive a parse failure: `... | tee logs/run-$ts.jsonl | jq ...`.
- `--session-id <UUID>` / `--resume` / `--continue` / `--fork-session` = addressable, resumable sessions — caller-specified IDs are the deterministic-thread-id idea again.
- `--permission-mode dontAsk` = "CI-shaped: tool calls not on the allow-list are auto-denied with no prompt. This prevents hangs waiting for human approval that will never arrive." This is *the* headless-permissions stance.
- `--dangerously-skip-permissions` = only inside "a disposable, sandboxed environment where there is nothing valuable to damage."
- `CLAUDE_CODE_ENABLE_TELEMETRY=1` + OTLP = cost/usage/activity metrics for observability.
- Auth purely via env vars (`ANTHROPIC_API_KEY`, or Bedrock/Vertex via OIDC/WIF) — "no long-lived key to leak."
- Overlap prevention via `flock` mutual exclusion + idempotency.

The guide's distilled rule for what makes a task safe to run unattended is exactly our grooming bar: **idempotent, bounded, and verifiable** — *"let Claude Code make the change, then let `npm test` or the linter be the judge."*

**OpenHands** ([2026 review](https://pickuma.com/for-dev/openhands-review-open-source-autonomous-coding-agent-2026/), [self-host guide](https://www.spheron.network/blog/deploy-openhands-gpu-cloud/)) runs the agent inside a **sandboxed runtime** (browse/execute/edit), with `--headless` for CI, one-shot `-t "fix the failing tests"`, and `--resume` to pick a conversation back up — plus a REST API for programmatic/batch submission. Its 1.0 rebuilt on a model-agnostic Software Agent SDK, which is directly relevant to our **multi-backend** requirement (Anthropic API *and* local Ollama).

**Sandboxing is non-negotiable for unattended runs.** With no human to catch a destructive command, the blast radius must be contained at the OS/container level, not by the model's good judgment. On a Pi this likely means a container or at minimum a dedicated working directory + a tool allow-list, with `--dangerously-skip-permissions`-equivalent autonomy *only* inside that jail.

---

## 9. Representing a run so it can be observed, resumed, and audited

Pulling the threads together, the **run record** is the heart of the harness. It must satisfy four consumers: the loop (resume), a monitor (observe live), an auditor (reconstruct after the fact), and the dispatcher (final disposition). Design implications:

- **One unified, serializable record** (Factor 5) keyed by a deterministic run id (the thread-id pattern). It holds: current step index / control-flow position, full message history, accumulated tool results, business facts (branch name, commits, gates passed), budget counters (turns used, USD spent, wall-clock), and the config (model, allow-list, scope, definition-of-done).
- **Append-only event log** (the stream-json pattern) alongside the materialized state. The log is the audit trail and the replay source; the materialized state is the fast-resume snapshot. Event-sourced state is the [2026 production consensus](https://agentmarketcap.ai/blog/2026/04/10/durable-agent-execution-production-temporal-modal-event-sourced) for exactly this reason.
- **Checkpoint cadence tuned by consequence:** sync around side-effecting steps, async around read-only steps.
- **Resume = load record → replay/skip completed steps → continue at first incomplete step**, with idempotency guards on the boundary step.
- **Storage on a Pi:** SQLite is the obvious fit (LangGraph's `SqliteSaver`, Cloudflare's DO-over-SQLite, DBOS's relational model all validate a single-file/embedded relational checkpoint store). No external service required; survives reboots; easy to back up.

---

## Implications for our headless-dispatch harness

1. **Make the run a stateless reducer over a single SQLite-backed record, and checkpoint per step.** Adopt 12-factor's Factor 5 + 12 literally: one serializable run record (control-flow position, messages, tool results, business facts, budgets, config), keyed by a deterministic run id, written to SQLite. The agent process is disposable; the record is the asset. Resume after a Pi reboot or laptop-sleep is then "load record, skip completed steps, continue" — durability becomes bookkeeping, not heroics.

2. **Draw step boundaries around side effects and make the boundary step idempotent.** Replay re-runs the step you crashed inside (LangGraph/Cloudflare both warn on this). Use `sync` checkpointing for any step touching git or the filesystem, `async` for read-only steps. Prefer git operations (mostly re-runnable) and full-file writes (idempotent) over appends; persist an idempotency marker *before* genuinely non-reversible side effects. The working tree + git history doubles as durable state (the Ralph/Anthropic pattern).

3. **There is no HITL pause — convert every human gate into either a pre-granted policy or a terminal escalation.** Front-load all judgment at grooming time into the run config (allowed tools, scope, definition-of-done). At runtime, anything off the allow-list is auto-denied (`dontAsk` semantics), and anything that genuinely needs a human becomes a blocker comment on the GTD item + clean exit — never an open-ended wait nothing will resolve. Repurpose pause/resume entirely for crash recovery.

4. **Stop-detection lives outside the model and is the only thing we trust for "done."** Enforce five layers in the Rust loop: hard turn cap (3–5× expected), USD budget, wall-clock circuit breaker, no-progress/repeat-tool detection, and — the positive stop — the project's own quality gates passing (tests, lint, type-check, build). Never accept the model's self-report of completion; the gate is the judge. Add a cheap adversarial self-review pass before pushing the branch.

5. **Run inside a sandbox with env-var-only auth; treat the runner as disposable.** With no human to catch a destructive command, containment is the OS's job, not the model's. Dedicated working dir or container, tool allow-list, autonomy only inside the jail. Credentials via env (`ANTHROPIC_API_KEY`; OIDC where possible) — no interactive login, no long-lived keys in the repo. This also cleanly accommodates the dual Anthropic-API / Ollama backend behind one execution abstraction (the OpenHands model-agnostic-SDK lesson).

6. **Emit an append-only stream-json event log per run, teed to disk before any parsing.** It is simultaneously the audit trail, the live-observe feed for a monitor, and the replay source. Pair it with the materialized state snapshot for fast resume. This is what lets a run "the machine rebooted mid-task" be reconstructed, resumed, and reviewed after the fact — and what lets the dispatcher know exactly what shipped.

---

## Sources

- [12-factor agents — README (all 12 factors)](https://github.com/humanlayer/12-factor-agents/blob/main/README.md) — humanlayer
- [Durable Execution: The Key to Harnessing AI Agents](https://www.inngest.com/blog/durable-execution-key-to-harnessing-ai-agents) — Inngest
- [Durable Execution for Crashproof AI Agents](https://www.dbos.dev/blog/durable-execution-crashproof-ai-agents) — DBOS
- [Cloudflare Workflows is now GA: production-ready durable execution](https://blog.cloudflare.com/workflows-ga-production-ready-durable-execution/) — Cloudflare
- [Durable Execution in LangGraph: Agents That Survive Failure and Resume Where They Left Off](https://vadim.blog/durable-execution-agents-that-survive-failure-and-resume-where-they-left-off) — Vadim's blog
- [Durable Agent Execution in Production 2026: Temporal, LangGraph, and Event-Sourced State Management](https://agentmarketcap.ai/blog/2026/04/10/durable-agent-execution-production-temporal-modal-event-sourced) — AgentMarketCap
- [Agent Workflows Are Rediscovering Durable Execution](https://nittikkin.medium.com/agent-workflows-are-rediscovering-durable-execution-be110661ed8c) — Koshy / Medium
- [Temporal — Durable Execution Solutions](https://temporal.io/) — Temporal
- [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) — Anthropic
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Anthropic
- [The GAN-Style Agent Loop: Deconstructing Anthropic's Harness Architecture](https://www.epsilla.com/blogs/anthropic-harness-engineering-multi-agent-gan-architecture) — Epsilla
- [The Ralph Loop: How I Run Autonomous AI Agents Overnight](https://blakecrosley.com/blog/ralph-agent-architecture) — Blake Crosley
- [Claude Code in CI/CD and Headless Automation](https://hidekazu-konishi.com/entry/claude_code_cicd_and_headless_automation.html) — hidekazu-konishi.com
- [OpenHands Review: The Open-Source Autonomous Coding Agent in 2026](https://pickuma.com/for-dev/openhands-review-open-source-autonomous-coding-agent-2026/) — Pickuma
- [Deploy OpenHands on GPU Cloud (2026 Guide)](https://www.spheron.network/blog/deploy-openhands-gpu-cloud/) — Spheron
- [Agentic Loops: From ReAct to Loop Engineering (2026 Guide)](https://datasciencedojo.com/blog/agentic-loops-explained-from-react-to-loop-engineering-2026-guide/) — Data Science Dojo
- [How Do You Stop AI Agents From Infinite Loops?](https://docs.bswen.com/blog/2026-03-11-prevent-ai-agent-infinite-loops/) — BSWEN
- [The Agent Loop Problem: When "Smart" Won't Stop](https://medium.com/@Modexa/the-agent-loop-problem-when-smart-wont-stop-ccbf8489180f) — Modexa / Medium
- [Cloudflare Dynamic Workflows (durable execution that follows the tenant)](https://blog.cloudflare.com/dynamic-workflows/) — Cloudflare
