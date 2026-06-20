# Tool Design & Code Execution — the agent's hands

> Researched 2026-06-20. Current state of the art in tool design, MCP, code-execution ("code mode"), large-output management, and unattended tool safety — read through the lens of a Rust harness built for headless dispatch.

This track covers how the agent *acts on the world*: the tools we hand it, how it
calls them, and how we keep an unattended run from doing damage with them. The
load-bearing question throughout: **what changes when no human is watching the tool
call before it executes?**

---

## 1. Designing good tools — the deterministic/non-deterministic contract

Anthropic's [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
is the canonical text. Its framing is the thing to internalize: a tool is a
**"contract between deterministic systems and non-deterministic agents."** Unlike a
traditional API consumed by code you wrote, a tool will be called by something that may
hallucinate arguments, misread the tool's purpose, or take a valid-but-unexpected path
to a goal. You design for that unpredictability, not against it.

### Choose high-leverage tools, not thin API wrappers

The single most repeated point: **don't expose your existing API one-endpoint-per-tool.**
Build "a few thoughtful tools targeting specific high-impact workflows"
([Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)).

- Prefer one `schedule_event` (which internally finds availability) over separate
  `list_users` + `list_events` + `create_event`. The agent shouldn't have to orchestrate
  primitive calls in context when the tool can do it in one shot.
- Consolidate multi-step operations under the hood: `search_logs` returning only relevant
  lines beats `read_logs` dumping everything; `get_customer_context` (recent transactions
  + notes in one call) beats three separate fetches.
- **Match the agent's affordances.** A computer has abundant memory; an agent has limited
  context. `search_contacts` beats `list_contacts` because listing forces the agent to
  burn tokens reading irrelevant entries one by one.

### Naming, namespacing, parameters

- **Namespacing materially affects accuracy.** Group related tools under prefixes
  (`asana_projects_search`, `asana_users_search`). Anthropic reports the choice between
  prefix- and suffix-based namespacing had "non-trivial effects on tool-use evaluations."
- **Unambiguous parameter names reduce hallucination.** Use `user_id`, not `user`.
- Clear, distinct tool names so the agent doesn't conflate two similar tools.

### Descriptions are onboarding docs, and they are high-leverage

Treat the description like you're briefing a new hire who has never seen your system:
make implicit context explicit (query-format quirks, terminology, how resources relate).
The payoff is large and empirically measured: **"even small refinements to tool
descriptions can yield dramatic improvements"** — Claude Sonnet hit state-of-the-art on
SWE-bench Verified after precise description refinements that dramatically cut error rates
([Anthropic](https://www.anthropic.com/engineering/writing-tools-for-agents)).

Anthropic's newer [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
post adds **tool-use examples**: concrete sample calls embedded in the tool definition,
capturing what JSON Schema can't (format conventions, parameter correlations, nested
structure). Internal testing moved complex-parameter-handling accuracy "from 72% to 90%."

### Return the right amount, in the right shape

- **High-signal only.** Strip low-level identifiers (`uuid`, `256px_image_url`,
  `mime_type`) the agent won't use. Resolving "arbitrary alphanumeric UUIDs to more
  semantically meaningful and interpretable language" measurably improves precision —
  return human-readable names, not opaque IDs.
- **Response-format enums.** Let the agent request `"concise"` vs `"detailed"`. Anthropic's
  concise Slack responses used **~⅓ the tokens** of detailed ones.
- **Token budgets and pagination by default.** Claude Code caps tool responses at
  **25,000 tokens** by default. Implement pagination, range selection, filtering, and
  truncation with sensible defaults.

### Error messages are a steering surface, not just a status

Don't return opaque codes. Return **"specific and actionable"** errors that nudge the
agent toward a better next move — show a correctly-formatted example, suggest a narrower
search, name the missing parameter. For an unattended agent this is doubly important:
the error message is the *only* corrective signal, since there's no human to interpret a
cryptic failure. (See §6.)

### Evaluate tools the way you'd test code — with realistic tasks

- Build a suite of **realistic, multi-call tasks** grounded in real data, not toy
  "sandbox" prompts. Strong: *"Schedule a meeting with Jane next week to discuss the Acme
  project, attach notes from our last planning meeting, and reserve a conference room."*
  Weak: *"Schedule a meeting with jane@acme.corp next week."* Strong tasks force dozens of
  tool calls and surface real failure modes.
- Pair each task with a **verifiable expected output** (exact match or LLM-judge), but
  avoid verifiers so strict they reject correct-but-differently-formatted answers.
- Collect accuracy, runtime, **tool-call count, token consumption, error rate.**
- **Let the agent improve its own tools.** Concatenate eval transcripts, paste into Claude
  Code, and have it refactor the tool implementations. Anthropic reports this beat
  expert-handwritten Slack and Asana tool suites.

---

## 2. MCP — the standard, and what it gives you

The [Model Context Protocol](https://modelcontextprotocol.io/introduction) is an open
standard ("a USB-C port for AI applications") for connecting an agent to external systems.
Architecture is **host → client → server**: the host app runs a client per connection,
and each MCP **server** exposes capabilities. Core server primitives are **tools** (model-
callable functions), **resources** (readable data/context), and **prompts** (reusable
templated workflows); clients can offer **sampling** and **roots** back. Transports are
**stdio** (local subprocess) and **streamable HTTP/SSE** (remote). It's broadly supported
(Claude, ChatGPT, VS Code, Cursor, etc.), so "build once, integrate everywhere."

For our harness the relevant takeaway is **interoperability**: speaking MCP means any of
the thousands of community servers becomes a potential tool source without bespoke
integration. But MCP's naive usage model — load every tool definition up front, route
every result back through context — is exactly what the next section attacks.

---

## 3. Code execution / "code mode" — the agent writes code that calls tools

This is the most important current shift in tool-execution design, and two independent
sources converged on it in late 2025: Anthropic's
[Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
and Cloudflare's [Code Mode](https://blog.cloudflare.com/code-mode/).

### The two problems with direct tool-calling at scale

1. **Tool-definition overload.** Loading all definitions up front scales linearly with
   tool count. Anthropic: agents wired to thousands of tools "need to process hundreds of
   thousands of tokens before reading a request." Cloudflare's example is extreme — the
   Cloudflare API has **2,500+ endpoints; exposing each as an MCP tool would cost over
   2 million tokens.**
2. **Intermediate results through context.** Every tool result transits the model's
   context. Anthropic's example: copy a meeting transcript from Google Drive to Salesforce
   and the full transcript flows through context twice — "for a 2-hour sales meeting, that
   could mean processing an additional 50,000 tokens."

### The pattern: present tools as a code API, let the agent write a program

Instead of tool-call JSON, expose the MCP servers as a typed code API (TypeScript files in
a filesystem, or generated interfaces) and give the agent a sandboxed runtime. It writes a
real program:

```typescript
const transcript = (await gdrive.getDocument({ documentId: 'abc123' })).content;
await salesforce.updateRecord({ objectType: 'SalesMeeting', recordId: '...', data: { Notes: transcript } });
```

Two mechanisms do the work:

- **Progressive disclosure.** Models are good at navigating filesystems, so the agent
  explores `./servers/` and reads only the tool files it needs (or queries a `search_tools`
  function at configurable detail: name-only → description → full schema). Anthropic reports
  one task dropping **from 150,000 to 2,000 tokens — a 98.7% reduction.**
- **Keep intermediate results out of context.** The agent filters/transforms in the
  execution environment and only `console.log`s the processed result — five rows instead
  of 10,000.

### Cloudflare's sharper claim: LLMs are *fluent in code, stutters in tool-JSON*

Cloudflare's argument is about training data: **"LLMs have seen a lot of code. They have
not seen a lot of 'tool calls.'"** Tool-call formats are synthetic constructs barely
present in pretraining, so asking a model to chain endpoints via function-call JSON is
"like putting Shakespeare through a month-long class in Mandarin and then asking him to
write a play in it." They collapse 2,500 endpoints into **two primitives — `search()` and
`execute()` — at roughly 1,000 tokens of context.** Measured savings:
[**32% fewer tokens on a single-event task, 81% fewer on a complex 31-event task**](https://workos.com/blog/cloudflare-code-mode-cuts-token-usage-by-81).

Anthropic's first-party [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use)
post productizes both halves of this:

- **Tool Search Tool** — tools marked `defer_loading: true` stay hidden until the model
  searches for them. ~72K → ~8.7K tokens for 50+ tools (**85% reduction, 95% of context
  preserved**). Opus 4 MCP accuracy 49%→74%; Opus 4.5 79.5%→88.1%. Use when definitions
  exceed ~10K tokens or there are 10+ tools; skip for small libraries (<10 tools). Shipped
  to GA in February 2026.
- **Programmatic Tool Calling** — Claude writes Python that orchestrates tools in a sandbox;
  results are processed by the code, not returned to context. **37% reduction on complex
  research (43,588→27,297 tokens)**; GAIA-style benchmarks 46.5%→51.2%. Use for large
  datasets where only aggregates matter, or 3+ dependent calls; skip for single lookups.

### Where the sources agree, and the caveats

- **Agreement:** code execution is the right tool for *many tools* and *multi-step
  chaining*. It leverages what models are good at (code) to fix what they're bad at
  (context-window management). It also keeps sensitive intermediate data out of the model
  entirely — a privacy and (for us) a context-hygiene win.
- **Caveat — it's not free.** [Simon Willison](https://simonwillison.net/2025/Nov/4/code-execution-with-mcp/)
  calls the approach "sensible" but pointed out Anthropic "outline the proposal in some
  detail but provide no code to execute on it" — *you* now own a code-execution sandbox,
  which is real operational and security surface. He frames it as evolution, not
  revolution.
- **Caveat — running model-written code is the whole ballgame for safety.** Both Anthropic
  and Cloudflare lean on sandboxing (see §5). Cloudflare runs generated TypeScript in V8
  isolates (boot in milliseconds, a few MB each, spawned per snippet) via a Dynamic Worker
  Loading API, with **no network** (`fetch`/`connect` throw), **binding-based
  authorization** (MCP servers reachable only as live JS objects in `env`, not arbitrary
  network calls), and **key containment** (credentials live with the supervisor, so "the AI
  cannot possibly write code that leaks any keys").
- **Caveat — don't over-rotate for small toolsets.** Every source agrees the wins appear at
  scale. For <10 simple tools, direct tool-calling is simpler and the overhead of a
  sandbox + codegen layer isn't justified.

---

## 4. Managing large tool outputs

Even with good tools, a coding agent will produce huge outputs (test logs, build output,
big files, `git diff`). The current convergent best practice
([Arize survey of harnesses](https://arize.com/blog/context-management-in-agent-harnesses/))
is a **memory hierarchy / demand-paging** model: "the best memory management is the kind
the program never thinks about." Active context holds previews/summaries; full content
lives on disk or in an index; the agent pages it in on request.

Concrete techniques in production harnesses:

- **Filesystem offloading.** Claude Code persists oversized tool results to disk and
  replaces them with **~2KB previews** (per-tool cap ~50,000 chars). Alyx splits large JSON
  into a compressed preview + a full server-side copy the model drills into via `jq`.
- **Truncation that *advertises the continuation*.** Don't silently drop the middle. Pi
  appends `"Showing lines 1-2000 of 50000. Use offset=2001 to continue."` — teaching the
  model that pagination exists without it having to discover the mechanism. OpenClaw uses a
  75/25 head/tail split so structure survives and incompleteness is visible.
  ([OpenHands issue #12353](https://github.com/OpenHands/OpenHands/issues/12353) notes the
  failure mode: lossy middle-truncation of >30KB outputs causes permanent information loss
  — the fix is preview + file path the agent can `cat`/`grep`.)
- **Index/search over raw dumps.** Letta parses, chunks, and embeds every file into a
  vector store with three access patterns (view / grep / semantic). Tool descriptions
  should *nudge "search instead of read."*
- **Subagent isolation.** Spawn a child agent with just the task string (Pi) or a blank
  session (OpenClaw) so a noisy exploration (e.g. reading a 10k-line log to find one error)
  never pollutes the parent's context.

---

## 5. Permissions & safety for tools — the part that matters most unattended

Interactive assistants get a free safety net: a human approves the risky tool call. A
headless agent has none. By early 2026 the industry response is **mediated execution +
out-of-process policy** rather than trusting the agent.

State of practice ([AI agent sandbox guides](https://www.firecrawl.dev/blog/ai-agent-sandbox),
[secure autonomous coding](https://lushbinary.com/blog/ai-agent-security-autonomous-coding-production-guide/)):

- **Sandbox all execution** — containers, microVMs, or cloud sandboxes (E2B, Northflank,
  Firecrawl, Modal, Docker Sandboxes, Cloudflare isolates all shipped this by early 2026).
  **Never run an agent directly on a host with production credentials.**
- **Mediated, not direct, access.** Tool calls go through a gateway/orchestrator, never
  straight to the OS. Cloudflare's binding model is the clean version: the agent can only
  reach what's been handed to it as an object; there's no ambient network or filesystem.
- **Out-of-process policy that the agent can't override.** NVIDIA NemoClaw (March 2026)
  wraps frameworks with kernel-level network allowlisting, filesystem-write restrictions,
  and config-file protection via a policy engine "that cannot be overridden by a compromised
  agent." The principle generalizes: the enforcement point must live *outside* the agent's
  reach.
- **Deterministic pre-action authorization.** An emerging line
  ([arXiv 2603.20953](https://arxiv.org/pdf/2603.20953)) argues the *decision to allow a
  tool call* should be a deterministic check evaluated **before** execution — not something
  the LLM reasons about. For unattended runs this is the right shape: a policy gate, not a
  vibe.
- **Blast-radius caps.** Rate-limit files modified, commands executed, packages installed
  per run. A runaway loop should hit a wall, not the credit-card limit.

---

## Implications for our headless-dispatch harness

- **Make code execution a first-class execution mode, not just one-tool-per-call.** Our
  agent's *job* is writing code, and the evidence (Anthropic 98.7% / 37%, Cloudflare 81%) is
  that letting it write a program to orchestrate tools beats emitting tool-JSON for any
  multi-step task — which is most of them. Design the harness so tools are also reachable as
  a typed code API the agent can script against, with **progressive disclosure** (defer
  tool-definition loading; expose a `search_tools`-style discovery surface) so we don't burn
  context loading every tool up front. Keep direct single-tool calls for the trivial cases.

- **The sandbox is not optional — it is the safety net that replaces the absent human.**
  Run on a Pi with no human approving calls, we must assume the model will eventually write
  or call something harmful. Adopt the Cloudflare model: **no ambient network/filesystem,
  binding-based access (the agent reaches only what we inject), credentials held by the
  supervisor and never visible to generated code, and an out-of-process policy gate that
  evaluates a deterministic allow/deny *before* execution.** In Rust this is a natural fit —
  wasmtime/WASI or a microVM (Firecracker) for the codegen runtime, the supervisor owning
  all secrets and tool bindings.

- **Hard blast-radius limits per run, enforced by the harness, not requested of the model.**
  Cap files touched, commands run, packages installed, wall-clock, tokens, and dollars. Tie
  these into the convergence/budget logic: hitting a cap is a *failure-to-recover* signal the
  harness owns, exactly like step limits. This is the unattended analogue of "a human would
  have stopped it three steps ago."

- **Design tools and their errors as the agent's only corrective channel.** No human will
  reinterpret a cryptic failure mid-run, so error messages must be actionable and steering
  ("search narrower," "this param is required, here's a valid example"). Favor high-leverage
  consolidated tools (`run_tests` that returns only failing cases + context, not raw output)
  over thin wrappers, and resolve IDs to human-readable names. Treat description-tuning as
  real engineering — it moved SWE-bench results, and our agent lives or dies on a coding
  benchmark.

- **Build the large-output discipline in from day one: preview + offload + advertised
  continuation.** Coding tasks generate enormous logs and diffs. Default every tool to a
  token cap (~25K like Claude Code), persist the full result to the workspace, return a
  small preview plus a path, and make truncation messages *teach* the pagination/grep
  affordance. This keeps the control loop's context lean enough to run long horizons on a
  small machine.

- **Abstract the tool layer over heterogeneous backends, and bias verbosity by model.** Since
  we target both the Anthropic API and local Ollama models, the tool/codegen abstraction must
  not assume Anthropic-only features (native tool-search, programmatic tool calling). For
  local models — weaker at tool-JSON, often strong at code — the code-execution path is
  likely *more* important, not less; lean on it as the portable lowest-common-denominator and
  use response-format enums to keep outputs concise for smaller-context local models.

## Sources

- [Writing effective tools for AI agents — Anthropic Engineering](https://www.anthropic.com/engineering/writing-tools-for-agents)
- [Code execution with MCP: building more efficient agents — Anthropic Engineering](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [Introducing advanced tool use on the Claude Developer Platform — Anthropic Engineering](https://www.anthropic.com/engineering/advanced-tool-use)
- [What is the Model Context Protocol (MCP)? — modelcontextprotocol.io](https://modelcontextprotocol.io/introduction)
- [Code Mode: the better way to use MCP — Cloudflare Blog](https://blog.cloudflare.com/code-mode/)
- [Cloudflare: Code Mode Cuts Token Usage by 81% — WorkOS](https://workos.com/blog/cloudflare-code-mode-cuts-token-usage-by-81)
- [Code execution with MCP: Building more efficient agents — Simon Willison](https://simonwillison.net/2025/Nov/4/code-execution-with-mcp/)
- [Context management in agent harnesses: memory, files, and subagents — Arize AI](https://arize.com/blog/context-management-in-agent-harnesses/)
- [Feature Request: Context Offloading for Large Tool Outputs — OpenHands issue #12353](https://github.com/OpenHands/OpenHands/issues/12353)
- [AI Agent Sandbox: How to Safely Run Autonomous Agents in 2026 — Firecrawl](https://www.firecrawl.dev/blog/ai-agent-sandbox)
- [AI Agent Security 2026: Secure Autonomous Coding Agents — Lushbinary](https://lushbinary.com/blog/ai-agent-security-autonomous-coding-production-guide/)
- [Before the Tool Call: Deterministic Pre-Action Authorization for Autonomous AI Agents — arXiv 2603.20953](https://arxiv.org/pdf/2603.20953)
