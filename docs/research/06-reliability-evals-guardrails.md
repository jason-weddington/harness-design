# Reliability, Evals & Guardrails for an Unattended Agent Harness

*Researched 2026-06-20.*

This track is about making an agent **trustworthy when no human is watching**. Our harness dispatches a groomed task, runs to completion on a small always-on box, then pushes a branch and comments back. Every safety net an interactive assistant gets for free — a human glancing at a diff, an "are you sure?" prompt, a person noticing the agent has gone in circles — we must build into the harness itself. The research below is organized around the four jobs the harness must own without us: **self-verification, error recovery, guardrails, and evaluation/observability.**

---

## 1. Self-verification and self-correction loops

The single highest-leverage finding across every source: **the underlying model matters less than the verification loop you wrap around it.** LangChain's coding agent jumped from 52.8% to 66.5% on Terminal Bench 2.0 by [changing nothing about the model and only changing the harness](https://addyosmani.com/blog/agent-harness-engineering/). OpenAI's framing of [harness engineering](https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents) names the same five components — sandboxed execution, state/memory, tool control, context engineering, and **feedback loops with self-correction** — and stresses that "browsers, logs, screenshots, and test runners are what let the agent observe its own work and close the self-verification loop."

Anthropic's [effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) is the most concrete primary source on how an *unattended* agent verifies itself:

- **Test like a user, not like a unit.** Their agent verified features with end-to-end browser automation (Puppeteer MCP) rather than trusting unit tests alone — observing real behavior closes the loop in a way a green test suite can fake.
- **A startup ritual that re-establishes ground truth.** Each fresh-context session runs an `init.sh` that boots the dev server and runs basic functionality checks *before* doing new work — so the agent confirms nothing regressed before it adds more.
- **Completion gates.** Features are only marked "passing" after explicit verification. A 200+ item feature list (all initially "failing") is the convergence target, and the harness instruction is blunt: *"It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality"* — i.e., the agent must not make the gate easier to pass.

The deeper principle: an agent can only self-correct on signals it can actually *see*. Our build-engine's verification loop is the project's own quality gates (compile, lint, tests, type-check) plus, where possible, a runtime smoke test — and the harness must treat "the agent edited the test to make it pass" as a failure mode to detect, not a success.

---

## 2. Error recovery patterns

[12-factor agents, Factor 9 ("Compact Errors into the Context Window")](https://github.com/humanlayer/12-factor-agents) describes the canonical self-healing loop: catch the tool exception, format the error, append it to the event thread, and loop so the model reads the error and adjusts its next attempt. This is what makes agents *durable* — they survive individual tool failures instead of crashing.

But the same factor names the failure mode this creates and the guard against it:

- **Consecutive-error counter (~3 attempts per tool).** Once exceeded, the agent must *escalate* — to a human, or by resetting/compacting context. Without this cap, error-recovery becomes an infinite, budget-draining spiral.
- Error spiraling is also contained by [Factor 8 (own your control flow)](https://github.com/humanlayer/12-factor-agents) (restructure how errors are represented), Factor 3 (prune irrelevant events from context), and Factor 10 (small focused agents — named as the *primary* prevention).

Claude Code auto mode contributes a complementary recovery pattern: [**deny-and-continue**](https://www.anthropic.com/engineering/claude-code-auto-mode). When a guardrail blocks an action, *"Claude shouldn't halt and wait for input; it should recover and try a safer approach."* The denial is fed back as context and the agent attempts an alternative path. Escalation thresholds are concrete: **3 consecutive or 20 total denials** before escalating. For us, "escalate to a human" doesn't exist mid-run — so our analog is: **deny-and-continue up to a cap, then fail the run cleanly and comment the reason back on the task**, rather than burning the entire budget thrashing.

For long runs that exceed a single context window, recovery also means **state externalization**: Anthropic's long-running harness persists a `claude-progress.txt` log plus descriptive git commits so a fresh context can reconstruct "what's done" and *revert* a bad change via git history. The git commit *is* the recoverable checkpoint. This maps directly onto our git-branch-per-task model.

---

## 3. Guardrails: the harness's standing safety net

The [qubittool Agent Harness Evaluation Guide](https://qubittool.com/blog/agent-harness-evaluation-guide) frames the harness itself as "an automated, isolated testing environment" and gives concrete guardrail targets and limits:

| Guardrail | Concrete guidance |
|---|---|
| **Step / iteration limit** | Hard `maxSteps` counter with automatic termination (their example: `maxSteps = 5`). Prevents infinite loops draining API credits. |
| **Loop rate** | Target **0%** — runaway looping is a hard failure, not a degradation. |
| **Determinism for testing** | `temperature = 0` to reduce variance when evaluating. |
| **Tool accuracy / task success** | Targets of **>95%** tool accuracy, **>90%** task success. |
| **Output validation** | Validate structured outputs; test against empty/malformed tool responses. |

### Containment beats prompting

The strongest current guidance — and a direct rebuttal to "just add a permission prompt" — is Anthropic's: **supervise what the agent is *able* to do, not what it does.** Two posts make the case quantitatively:

- [Auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode): users approve **93%** of permission prompts, so prompts breed approval fatigue and stop being a real check. Their two-stage classifier (a single-token fast filter at 8.5% false-positive, then a reasoning layer dropping that to 0.4%) replaces most prompts. Critically, it *"is not a drop-in replacement for careful human review on high-stakes infrastructure"* — and it has a **17% false-negative rate**, i.e. it misses ~1 in 6 overeager actions. For an unattended run, that residual risk must be absorbed by containment, not by a human.
- [Sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing): OS-level isolation (Linux **bubblewrap**, macOS **seatbelt**) enforces **filesystem isolation** (only approved directories, covering spawned subprocesses) and **network isolation** (egress only through an approving proxy over a unix domain socket). This *"safely reduces permission prompts by 84%"* and is what lets the agent run unattended inside known boundaries; out-of-bounds attempts trigger notification rather than silent success. Git credentials live outside the sandbox via a proxy so tokens never enter the execution environment.

### The permission evaluation model (Claude Agent SDK)

The [Agent SDK permission system](https://code.claude.com/docs/en/agent-sdk/permissions) is a clean reference architecture for a layered permission engine, evaluated in a fixed order: **hooks → deny rules → ask rules → permission mode → allow rules → `canUseTool` callback.** Key facts for a *headless* harness:

- **Deny rules win even in `bypassPermissions`.** A scoped deny like `Bash(rm *)` is enforced regardless of mode — the hard floor.
- **`dontAsk` mode is the headless-correct mode**, not `bypassPermissions`. `dontAsk` *"converts any permission prompt into a denial"*: tools pre-approved by an allowlist run; everything else is denied without a callback. The docs explicitly recommend pairing `allowedTools` with `dontAsk` for "a fixed, explicit tool surface for a headless agent." `bypassPermissions` is the opposite — it approves *everything* unlisted, and `allowed_tools` does **not** constrain it.
- **Subagents inherit `bypassPermissions`** and it can't be overridden per-subagent — a sharp footgun for any multi-agent design.

### Excessive agency (the threat model)

[OWASP LLM06:2025 (Excessive Agency)](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/) names the three axes our guardrails must bound: **excessive functionality** (tools the task doesn't need), **excessive permissions** (a read-only task wired with write credentials), and **excessive autonomy** (no verification before high-impact actions). Its mitigations are least-privilege tooling, avoiding open-ended tools (raw shell, arbitrary URL fetch), least-privilege credentials, complete mediation in downstream systems, and detective controls (log everything, rate-limit to shrink the damage window). OWASP still calls human-in-the-loop the most critical safeguard — which we structurally *cannot* provide mid-run, so we over-index on the other mitigations: tight tool surface + sandbox + least-privilege creds + a hard budget cap.

---

## 4. Evaluating the agent — turning non-determinism into repeatable tests

[Anthropic's "Demystifying Evals for AI Agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) is the canonical text and worth following closely.

**The anatomy of an eval:** a **task** (inputs + success criteria), **trials** (multiple runs to absorb non-determinism), **graders** (scoring), a **transcript** (full record), and an **outcome** (final environment state). Note the deliberate two-harness split: the *agent harness* lets the model act; the *eval harness* runs tasks end-to-end and grades. We are building the former; we need a thin version of the latter to trust it.

**Non-determinism → repeatable metrics.** You don't eliminate variance, you measure it across trials:

- **pass@k** — probability of ≥1 success in k attempts (right metric when one working solution suffices).
- **pass^k** — probability *all* k trials succeed (right metric for reliability). The math is sobering: **75% per-trial success → only ~42% pass^3.** For an unattended build engine that must be reliable, **pass^k is our metric** — a 90%-per-run agent is not a 90%-reliable agent.

**Grader types**, in order of preference: **code-based** (string match, unit tests, static analysis — fast, cheap, objective, reproducible, but brittle to valid variation) as primary; **model-based / LLM-as-judge** (natural-language rubrics) as secondary for nuance; **human** reserved for calibrating the judge. The [qubittool guide](https://qubittool.com/blog/agent-harness-evaluation-guide) contrasts brittle keyword matching against LLM-as-judge's nuance-at-a-cost. The recurring warning across Anthropic and [LangChain](https://www.langchain.com/resources/agent-observability): **LLM judges must be calibrated against human experts**, or they drift.

**Grade outcomes, not trajectories — mostly.** Anthropic: *"grade what the agent produced, not the path it took."* Pinning an exact tool sequence yields overly brittle tests because agents find valid approaches you didn't anticipate. Verify the outcome (does the code pass tests? did the refund process?), and use the transcript to *inspect* reasoning via rubric rather than asserting an exact path. The nuance from [trajectory evaluation](https://medium.com/@vinodkrane/chapter-8-agent-evaluation-for-llms-how-to-test-tools-trajectories-and-llm-as-judge-788f6f3e0d52) and LangChain: an "agent judge" *can* score the whole reasoning trajectory (catching the right answer reached by flawed reasoning, or cascading multi-turn failures), but path-convergence/efficiency is better as a *quantitative* metric than as a brittle exact-match assertion.

**What to measure beyond pass/fail:** turns, tool calls, total tokens, time-to-first-token, tokens/sec, total completion time, and — for coding — test coverage, type-check (mypy) and security (bandit) results. Latency is a first-class concern: a slow-but-correct agent is "often unusable in production."

**Sample sizes and lifecycle:** start with **20–50 tasks drawn from real failures** — early on, changes have large effect sizes so small N suffices; mature agents need larger/harder evals to detect smaller effects. Three phases: (1) collect 20–50 tasks + reference solutions + a stable isolated harness; (2) **read transcripts constantly** — a failure tells you whether the agent erred or the *grader* wrongly rejected a valid solution; (3) pair automated evals with production monitoring, run capability evals in CI/CD on every change, and graduate passing capability evals into a **regression suite**.

**The common-mistakes list is a checklist for us:** ambiguous task specs (a 0% pass rate across 100+ trials means a broken task, not an incapable agent — *"could a domain expert pass it themselves?"*); class imbalance (test "don't act" cases, not just "act" cases); **shared state between trials** (each trial must start from a clean environment or failures correlate and inflate scores); grading bugs (the Opus 4.5 example: rigid grading `96.12` vs `96.124991` plus ambiguous specs scored 42%; fixing the *graders* raised it to 95% — the agent was fine all along).

**Chaos engineering for recovery.** The qubittool guide recommends intentionally injecting faults — 500s, malformed JSON, empty tool responses — to test that the error-recovery loop (§2) actually works. This is how we get evidence our recovery code does what we think before we ship it unattended.

---

## 5. Observability / tracing of agent runs

Because no human watches the run, the **transcript is the only forensic record** — qubittool: *"Without full visibility, debugging a failed agent test is nearly impossible. Capture every prompt, tool call, and internal thought."*

Current best practice has consolidated on **OpenTelemetry GenAI semantic conventions** as the default, vendor-neutral format for agent/tool/LLM spans ([Uptrace](https://uptrace.dev/blog/opentelemetry-ai-systems), [Coralogix](https://coralogix.com/ai-blog/agentic-ai-observability/)). [LangChain's observability guidance](https://www.langchain.com/resources/agent-observability) lists what to instrument: LLM calls (model version, inputs, token usage, latency), tool invocations (selection logic, params, results, time), retrieval steps, reasoning transitions, and state reads/writes — with related traces grouped into **threads/sessions** so a whole run is one evaluable unit.

The flywheel that turns observability into reliability: production traces → build datasets from real usage → automated scoring → targeted fixes → **convert every problematic trace into a permanent regression test**. *"Regression testing locks in these gains by ensuring that once you fix a bug, it stays fixed."* For us this is exactly the Jason "build past the harness" pattern: each unattended-run failure becomes a captured eval case the harness enforces from then on.

---

## 6. From "70–80% prototype" to production reliability

The most instructive primary source on *how reliability regresses* is Anthropic's [Claude Code April postmortem](https://www.anthropic.com/engineering/april-23-postmortem). Three **harness-level** changes — not model changes — caused visible degradation: a reasoning-effort default lowered for latency; a prompt-cache bug that cleared thinking *every turn* instead of once (making the agent progressively "forgetful"); and a verbosity prompt (`≤25 words between tool calls`) that hurt coding quality. Lessons that transfer directly:

- **Harness complexity hides bugs that pass code review, unit tests, AND e2e tests.** The cache bug survived all gates and was masked by two unrelated concurrent experiments. Staggered rollouts across cohorts made the aggregate look like "broad, inconsistent degradation" with no clean root cause.
- **Model vs. harness failure is a real, diagnosable distinction.** Their Opus 4.6 review tool *missed* the bug; Opus 4.7 *found* it given proper context. The fix wasn't "blame the model" — it was broader eval suites per model, system-prompt ablations, and gradual rollouts for any intelligence-affecting change. (This is precisely Jason's two-failure-modes lens: most "the model got dumber" reports were a context/harness failure.)

Production reliability is therefore an *engineering* outcome, not a model property: pass^k-graded eval suites in CI on every harness change, transcript review, staged rollout, and regression tests minted from real failures. The [LLM Readiness Harness](https://arxiv.org/abs/2603.27355) line of work frames the same thing as structured **evaluation gates + observability instrumentation + CI integration** as the bar for production deployment.

---

## Implications for our headless-dispatch harness

- **`dontAsk` + tight allowlist + OS-level sandbox is our permission posture — never `bypassPermissions`.** With no human to absorb the auto-classifier's 17% false-negative rate, containment (bubblewrap/seatbelt-style filesystem + egress-proxy network isolation, least-privilege git creds outside the sandbox) is the real safety boundary. Deny rules are the hard floor that holds even when everything else is permissive. Build the permission engine as an ordered pipeline (hooks → deny → mode → allow → fallback-deny) mirroring the Agent SDK.

- **The verification loop *is* the product; make the project's own gates the convergence target, and detect gate-tampering.** Compile + lint + type-check + tests + a runtime smoke test, run on a startup ritual before new work and again before declaring done. Treat "the agent weakened a test to make it pass" as a first-class failure to catch — it's the headless analog of a human noticing a cheap green checkmark.

- **Every limit is a hard counter with a clean failure exit, not a prompt.** Step cap, per-tool consecutive-error cap (~3 → back off), total-denial cap (~20), wall-clock timeout, and a token/cost budget. On breach, the agent does **deny-and-continue** up to the cap, then *fails the run cleanly and comments the reason on the task* — burning the whole budget thrashing is the worst outcome for an unattended box.

- **Grade with pass^k, not pass@1 — and build a thin eval harness before trusting the build engine.** Start with 20–50 tasks mined from real dispatch failures, isolated per-trial (clean git worktree each time), code-based graders primary + a calibrated LLM-judge secondary. Inject chaos (500s, malformed tool output) to prove the recovery loop works. Every failed unattended run becomes a permanent regression case — the flywheel.

- **Emit OpenTelemetry GenAI-convention traces for every run; the transcript is the only post-mortem we get.** One thread/session per dispatched task, spanning every LLM call (model, tokens, latency), tool call (args, result, duration), and state write — across context-window boundaries. This is what lets a *later interactive* session review a failed branch, and what feeds the regression-test flywheel.

- **Externalize state to git + a progress log so a fresh context can recover and revert.** Descriptive commits are recoverable checkpoints; a progress file lets a re-spawned agent reconstruct "what's done" without the original context. This is both our long-run continuation mechanism and our rollback path when self-verification catches a bad turn. Heed the postmortem: a harness change can degrade quality while passing every test — gate harness changes themselves behind the eval suite and staged rollout.

## Sources

- [Demystifying Evals for AI Agents — Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [Effective Harnesses for Long-Running Agents — Anthropic](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- [How we built Claude Code auto mode: a safer way to skip permissions — Anthropic](https://www.anthropic.com/engineering/claude-code-auto-mode)
- [Making Claude Code more secure and autonomous with sandboxing — Anthropic](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [An Update on Recent Claude Code Quality Reports (April postmortem) — Anthropic](https://www.anthropic.com/engineering/april-23-postmortem)
- [Configure permissions — Claude Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/permissions)
- [12-Factor Agents — humanlayer (GitHub)](https://github.com/humanlayer/12-factor-agents)
- [Agent Harness Evaluation Guide — qubittool](https://qubittool.com/blog/agent-harness-evaluation-guide)
- [awesome-harness-engineering — ai-boost (GitHub)](https://github.com/ai-boost/awesome-harness-engineering)
- [AI Agent Observability: Tracing, Testing, and Improving Agents — LangChain](https://www.langchain.com/resources/agent-observability)
- [OWASP LLM06:2025 — Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)
- [Agent Harness Engineering — Addy Osmani](https://addyosmani.com/blog/agent-harness-engineering/)
- [Harness Engineering for AI Coding Agents — Augment Code](https://www.augmentcode.com/guides/harness-engineering-ai-coding-agents)
- [OpenTelemetry for AI Systems: LLM and Agent Observability — Uptrace](https://uptrace.dev/blog/opentelemetry-ai-systems)
- [Agentic AI Observability: A Practical Guide for 2026 — Coralogix](https://coralogix.com/ai-blog/agentic-ai-observability/)
- [Agent Evaluation for LLMs: Tools, Trajectories, and LLM-as-Judge — Vinod Rane (Medium)](https://medium.com/@vinodkrane/chapter-8-agent-evaluation-for-llms-how-to-test-tools-trajectories-and-llm-as-judge-788f6f3e0d52)
