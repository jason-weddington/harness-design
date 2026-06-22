# Model-Layer Harness Features

> Researched 2026-06-20. What a headless-dispatch coding harness must do to exploit each model backend — Anthropic's API (Haiku/Sonnet/Opus) and local Ollama models — and how to abstract over both behind one Rust interface.

This track is grounded in the canonical [Claude API skill bundle](https://docs.claude.com) (the authoritative, drift-corrected reference for the current Anthropic API surface as of mid-2026), the [Ollama capability docs](https://docs.ollama.com/capabilities/tool-calling), and current frontier-vs-local comparisons. The lens throughout: **there is no human in the loop**, so every model knob is either a lever the harness pulls autonomously or a failure mode the harness must detect itself.

---

## Part 1 — Anthropic API capabilities a harness should leverage

The entire surface goes through one endpoint: `POST /v1/messages`. Tools, structured outputs, thinking, caching, and context management are all *features of that single call*, not separate APIs. A Rust harness builds one request type and toggles fields.

### 1.1 Prompt caching — the single highest-leverage feature for unattended agents

Prompt caching is a **prefix match**: the cache key is the exact bytes of the rendered prompt up to each `cache_control` breakpoint, and the render order is fixed as `tools` → `system` → `messages`. Any byte change anywhere in the prefix invalidates everything after it. ([prompt-caching reference](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))

Economics that matter for a long agentic loop: cache **reads** cost ~0.1× base input price; cache **writes** cost 1.25× (5-min TTL) or 2× (1-hour TTL). An agentic build run re-sends the entire growing transcript on every turn, so without caching the harness pays full input price on the whole history every single step. With a breakpoint on the last block of the most-recently-appended turn, each step reads the prior prefix at 0.1×. ([prompt-caching reference](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))

Knobs and constraints the harness must respect:

- **Max 4 breakpoints per request.** Place them at stability boundaries: frozen system prompt / deterministic tool list early, volatile per-turn content last.
- **Minimum cacheable prefix is model-dependent** — 4096 tokens on Opus 4.8 / 4.7 / 4.6 / Haiku 4.5; 2048 on Sonnet 4.6 / Fable 5; 1024 on Sonnet 4.5. A 3K-token prompt caches on Sonnet 4.6 but *silently won't* on Opus 4.8 (`cache_creation_input_tokens: 0`, no error). ([prompt-caching reference](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))
- **20-block lookback window.** Each breakpoint walks back at most 20 content blocks to find a prior entry. Agentic loops that emit many tool_use/tool_result pairs in one turn blow past this and silently miss — the fix is intermediate breakpoints every ~15 blocks. This is a *headless-specific* trap: an interactive user would notice cost spiking; an unattended harness must instrument it.
- **Silent invalidators** kill caching with no error: `datetime.now()` / a UUID / a per-run ID interpolated into the system prompt, non-deterministic JSON serialization (must sort keys), a tool set that varies per run. ([prompt-caching reference](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))
- **Verification is mandatory in a headless harness.** Read `usage.cache_read_input_tokens` on every response; if it stays zero across same-prefix turns, a silent invalidator is active. Note that `input_tokens` is the *uncached remainder only* — total prompt size = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.

**Cache-friendly context layering for a build engine** (apply the "render order = stability order" rule): tool definitions (sorted deterministically) and the frozen system prompt go first behind a breakpoint; the task spec / repo context that's fixed per-run goes next behind a second breakpoint; the growing turn-by-turn transcript trails after, with a rolling breakpoint on the latest turn. Never interpolate the timestamp, run ID, or task ID into the system prompt — inject them as a later message.

**Pre-warming** (`max_tokens: 0` request at run start) writes the cache before the first real turn, trading a cache-write charge now for lower first-turn latency. For a headless build engine that fires one run and walks away this is usually *not* worth it — there's no user waiting on first-token latency, and continuous turns within a run keep the cache warm on their own. Skip it. ([prompt-caching reference](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))

### 1.2 Extended thinking / reasoning — now "adaptive", and the API drifted hard

This is a documented context-failure trap. The training-data prior — `thinking: {type: "enabled", budget_tokens: N}` — is **rejected with a 400** on Fable 5 / Opus 4.8 / 4.7 and **deprecated** on Opus 4.6 / Sonnet 4.6. The current shape is `thinking: {type: "adaptive"}`: Claude decides per-request how much to think, and automatically interleaves thinking between tool calls. ([adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)) A harness written from a 2024/early-2025 memory will 400 on every request to current models.

The replacement for "how hard should it think" is the **`effort` parameter**, nested in `output_config` (not top-level): `output_config: {effort: "low"|"medium"|"high"|"xhigh"|"max"}`. Default is `high`. `xhigh` (added on Opus 4.7, between `high` and `max`) is the recommended setting for coding and agentic work — it's the Claude Code default. `max` is Opus-tier only and errors on Sonnet 4.5 / Haiku 4.5. ([effort](https://platform.claude.com/docs/en/build-with-claude/effort))

Two more thinking knobs a headless harness should set deliberately:

- **`thinking.display`** defaults to `"omitted"` on Fable 5 / Opus 4.8 / 4.7 — thinking blocks stream with *empty text*. A harness that logs reasoning for post-run debugging (the only place a headless agent's "why did it do that" lives) must set `display: "summarized"` explicitly, or the audit log is blank.
- **Task Budgets** (beta `task-budgets-2026-03-13`, Fable 5 / Opus 4.8 / 4.7): `output_config: {task_budget: {type: "tokens", total: N}}` tells the model how many tokens it has for a *whole agentic loop* — it sees a running countdown and paces itself to finish gracefully instead of being cut off. Minimum 20,000. This is distinct from `max_tokens` (an enforced per-*response* ceiling the model is unaware of). For a headless build engine, task budgets are the model-side complement to the harness's own step/cost limits: the model self-moderates *and* the harness hard-caps. ([model-migration → Task Budgets](https://platform.claude.com/docs/en/about-claude/models/migration-guide))

**Replay rule the harness must enforce:** when continuing a conversation on the same model, pass thinking blocks back *exactly as received* (including empty-text blocks — the API rejects modified blocks, not read ones). Switching models mid-conversation drops them. This matters when a harness routes different turns to different models (see §3).

### 1.3 Tool use — the core of a build engine

Tool definition is a `{name, description, input_schema}` JSON-Schema triple. The harness drives the agentic loop: send tools + messages, get back `tool_use` blocks, execute them, return `tool_result` blocks, repeat until `stop_reason == "end_turn"`. ([tool-use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview))

Features a headless coding harness should exploit:

- **Parallel tool calls (default on).** One assistant message can contain multiple `tool_use` blocks. The non-obvious rule: return **all** `tool_result` blocks in a **single** user message. Splitting them across messages silently trains the model to stop calling tools in parallel — a quiet degradation an unattended harness would never notice. For a failed tool, return `tool_result` with `is_error: true` — don't drop it.
- **Strict tool use** (`strict: true` on the tool definition, no beta): guarantees `tool_use.input` validates exactly against the schema (requires `additionalProperties: false` + `required`). For a build engine that pipes tool inputs straight into shell/file operations, this removes a class of malformed-input failures the harness would otherwise have to defend against itself.
- **`tool_choice`**: `{type: "auto"}` (default), `{type: "any"}` (must use a tool), `{type: "tool", name}` (forced), `{type: "none"}`. Add `disable_parallel_tool_use: true` to force at most one call per turn — useful when the harness must serialize file-mutating operations.
- **`pause_turn` stop reason**: server-side tools can hit an iteration limit and return `stop_reason: "pause_turn"`. The harness resumes by re-sending the assistant content — *without* adding a "Continue" message (the API detects the trailing `server_tool_use` and resumes). A headless loop must handle this or it stalls. ([tool-use-concepts](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview))
- **Tool-input JSON parsing**: Fable 5 / 4.6+ models may escape Unicode or forward-slashes differently in serialized tool input. Always parse with a real JSON parser — never raw-string-match the serialized input. A Rust harness using `serde_json` is naturally safe here, but the harness must not regex tool inputs.

**Fine-grained tool streaming** is *not* a beta feature (corrected from the older prior): set `eager_input_streaming: true` on the tool definition and call the regular streaming endpoint. This streams tool-call argument JSON as it generates, letting the harness start preparing/validating a tool call before the full block lands — marginal value for a headless engine that isn't rendering to a UI, but useful for early-cancelling obviously-wrong calls.

**Tool surface design for a headless harness** (from [agent-design](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)): bash gives breadth but hands the harness an opaque command string; promoting an action to a *dedicated tool* gives a typed, interceptable hook. For an unattended build engine, promote actions that need (a) a security gate (anything irreversible — `git push`, deletes), (b) a staleness check (an `edit` tool that rejects writes if the file changed since last read), or (c) parallel-safety classification (read-only `grep`/`glob` marked safe to run concurrently; `git push` must serialize). The harness owns these invariants because no human will catch a violation mid-run.

### 1.4 Context editing & the memory tool — surviving long unattended runs

Two distinct mechanisms (the harness must not conflate them):

- **Context editing** (beta `context-management-2025-06-27`) *clears* stale content — `context_management: {edits: [{type: "clear_tool_uses_20250919"}]}` removes old tool results (optionally their inputs too); `clear_thinking_20251015` clears thinking blocks. It prunes; it does not summarize. ([context-editing](https://platform.claude.com/docs/en/build-with-claude/context-editing))
- **Compaction** (beta `compact-2026-01-12`) *summarizes* earlier history server-side when the conversation nears the context window. Critical handling: append the **full `response.content`** (including the `compaction` block) back to messages each turn — extracting only the text silently loses the compaction state and the next request balloons. ([compaction](https://platform.claude.com/docs/en/build-with-claude/compaction))
- **Memory tool** (`memory_20250818`, client-executed): the model reads/writes a `/memories` directory the harness backs with storage. This is the *cross-run* persistence layer — for a dispatch-first harness running waves of related tasks, a memory store lets run N+1 benefit from what run N learned about the repo. The harness implements the backend and **must validate every model-supplied path** (resolve canonical, confirm within the memory root, reject `..`/symlinks). ([memory-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool))

For a long unattended build run, the recommended stack is all three: context editing to prune stale tool output within a run, compaction as a backstop near the window limit, memory for anything that should outlive the run.

### 1.5 Structured outputs — the harness's self-verification primitive

Two features:
- **JSON outputs** via `output_config: {format: {type: "json_schema", schema: {...}}}` — constrains the response to a schema. (The old top-level `output_format` param is deprecated API-wide; use `output_config.format`.)
- **Strict tool use** (`strict: true`, above) for tool-parameter validation.

Schema support is a subset of JSON Schema: types, `enum`, `const`, `anyOf`, `allOf`, `$ref`/`$def`, string formats, and `additionalProperties: false` (required on all objects). **Not** supported: recursive schemas, numeric/string-length constraints, complex array constraints — the Python/TS SDKs strip these and validate client-side; a Rust harness must do the same. ([structured-outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs))

For a headless build engine this is load-bearing: the harness can force the agent's *final disposition* into a schema — `{status: "complete"|"blocked"|"needs_decision", branch: str, gates_passed: bool, summary: str}` — and parse it reliably instead of regexing prose. That's how the unattended run reports back to the task tracker without a human interpreting free text. Note structured outputs are **incompatible with citations** (400) and with message prefilling.

### 1.6 Other knobs the harness sets per-request

- **`max_tokens`**: don't lowball — hitting the cap truncates mid-output. Default ~16K non-streaming, ~64K streaming. Fable 5 / Opus 4.6/4.7/4.8 support up to 128K but *require streaming* at that size to dodge SDK HTTP timeouts.
- **Streaming**: the SDKs refuse non-streaming requests they estimate will exceed ~10 minutes. A headless harness running high-effort/high-`max_tokens` work should default to streaming and accumulate the final message.
- **`stop_reason` handling**: a headless loop must branch on `end_turn`, `tool_use`, `pause_turn`, `max_tokens`, `refusal`, and `model_context_window_exceeded`. The last two are easy to forget and both strand an unattended run if unhandled. `stop_details` is populated *only* on `refusal` — guard before reading it.
- **Refusals & fallbacks** (Fable 5): safety classifiers can return HTTP 200 with `stop_reason: "refusal"`. A coding agent doing security-adjacent work can trip false positives. The server-side `fallbacks` parameter (beta `server-side-fallback-2026-06-01`) transparently re-serves on a fallback model in the same call — the only sanctioned target at launch is `claude-opus-4-8`. For a harness on Opus 4.8 directly this is mostly N/A, but if Fable 5 is ever in the routing table, the harness should opt in.
- **Error retries**: 408/409/429/5xx + connection errors are retried by the SDKs with backoff. A Rust harness over raw HTTP must implement this itself, reading `retry-after` on 429.

---

## Part 2 — Local models via Ollama

Ollama exposes an HTTP API at `http://localhost:11434` with two relevant endpoints: its native `/api/chat` and an OpenAI-compatible surface. For a Rust harness, both are plain HTTP/JSON.

### 2.1 Tool / function calling

Ollama supports tool calling on post-trained models (Llama 3.1+, Qwen 2.5+/3.x, Mistral Nemo, and similar). The request carries a `tools` array of `{type: "function", function: {name, description, parameters}}` using standard JSON Schema — **the same schema shape as the frontier APIs**, so tool definitions port without translation. Responses return `tool_calls` with name + parsed arguments; the agent loop (send → tool_calls → execute → append `tool` role result → resend) is structurally identical to Anthropic's. ([Ollama tool-calling](https://docs.ollama.com/capabilities/tool-calling), [DeepWiki](https://deepwiki.com/ollama/ollama/7.2-tool-calling-and-function-execution))

Ollama added **streaming tool calls** ([Ollama blog](https://ollama.com/blog/streaming-tool)): when streaming, the client must accumulate `thinking`, `content`, and `tool_calls` chunks and return the accumulated values in the follow-up request. Parallel tool calls are supported on capable models.

The gap vs frontier: tool-call *reliability* and multi-step *chain depth*. Current comparisons put frontier models (Opus-tier) at reliably handling 5+ step tool chains, while local models handle the bulk of simpler tasks well but degrade on long chains and recoverable-failure handling. ([MindStudio 2026](https://www.mindstudio.ai/blog/best-open-source-llms-agentic-coding-2026), [InsiderLLM function calling](https://insiderllm.com/guides/function-calling-local-llms/)) The gap has narrowed in 2026 — Qwen 3.x and similar have "reliable tool use" — but for an unattended build engine where a wrong turn isn't human-caught, the practical posture is: **local for simple/mechanical tasks, frontier for anything multi-step or judgment-heavy.**

### 2.2 Structured outputs

Ollama supports structured outputs by passing a JSON Schema in a `format` field on the request; it constrains generation to the schema. Quality varies by model — smaller local models adhere less reliably than frontier models, so the harness should validate the parsed output against the schema itself rather than trusting adherence. ([Instructor + Ollama guide](https://python.useinstructor.com/integrations/ollama/))

### 2.3 Context-length limits — the headline gotcha

**Ollama defaults `num_ctx` to 2048 tokens on every model, regardless of what the weights support.** This is a deliberate hardware-safety default so models boot on low-RAM machines — not a bug. The dangerous part for a headless harness: **when input exceeds `num_ctx`, Ollama silently drops the oldest messages and keeps responding** — no error, no warning. The model just "forgets" the start of the task. ([Markaicode](https://markaicode.com/ollama-context-length-extend/), [Serverman](https://www.serverman.co.uk/ai/ollama/ollama-context-window/))

For an unattended build engine this is a multiplied trap: the agent silently loses the task spec partway through a run and produces confidently-wrong work, and no human is watching. The harness **must** set `num_ctx` explicitly (per-request `options.num_ctx`, or a Modelfile) and **must** account for the cost: raising `num_ctx` grows KV-cache allocation linearly (~6 GB VRAM for a 7B model at 32K). The configured host's resources (RAM/VRAM, read from config — not a design constant) bound how large a context the harness can give a local model — which in turn bounds which tasks are even dispatchable locally. The harness should track per-model `num_ctx` and refuse to dispatch a task whose prompt would exceed it (token-count the prompt first) rather than let Ollama silently truncate.

### 2.4 What Ollama does NOT have

- No prompt caching with the read/write economics of the Anthropic API (KV-cache reuse exists at the runtime level but there's no `cache_control`/usage-reporting contract to engineer against).
- No extended-thinking/`effort` knob with the same semantics (some models emit `<think>` content; it's model-behavior, not an API lever).
- No context editing / compaction / memory-tool API features — the harness must implement transcript management entirely itself.
- No task budgets, no server-side fallbacks, no batch API.

The implication: **for local models the harness owns far more of the "model-layer" work** (context budgeting, transcript pruning, self-verification) that the Anthropic API provides as features.

### 2.5 Calling from Rust

The Anthropic API is HTTP/JSON; there is **no official Anthropic Rust SDK**. Community crates exist (`anthropic-ai-sdk`, `anthropic-sdk-rust`, `anthropic_rust`, ThreatFlux's `anthropic_rust_sdk`) claiming TS-parity, but they carry community-SDK risk: they lag the API, and the API drifted hard in 2025–26 (adaptive thinking, `effort`, the removal of `budget_tokens`/sampling params, `output_config.format`). ([crates.io anthropic-ai-sdk](https://crates.io/crates/anthropic-ai-sdk), [dimichgh/anthropic-sdk-rust](https://github.com/dimichgh/anthropic-sdk-rust)) For a harness whose whole job is to ride the current API, the pragmatic choice is to **own the HTTP/JSON layer directly** (`reqwest` + `serde` + an SSE parser for streaming), modeling exactly the request fields this doc enumerates. That keeps the harness one edit away from any new beta header or field, instead of waiting on a community crate to catch up. Ollama is likewise just HTTP/JSON over `reqwest` — same transport, different request schema.

---

## Part 3 — Abstracting over heterogeneous backends

The harness needs one internal interface (`trait ModelBackend`) with two implementations (Anthropic, Ollama) that differ enormously in capabilities. Two design pillars:

### 3.1 Capability detection, not assumption

Model capabilities are *not* uniform even within Anthropic (e.g. `effort: "max"` errors on Sonnet 4.5/Haiku; cacheable-prefix minimum varies; structured outputs are model-gated). The Anthropic **Models API** (`GET /v1/models/{id}`) returns live `max_input_tokens`, `max_tokens`, and a `capabilities` tree with `supported: true/false` leaves for thinking/adaptive/effort/structured-outputs/vision/context-management. ([models overview](https://platform.claude.com/docs/en/about-claude/models/overview)) The harness should query this at startup and gate features off it rather than hard-coding per-model assumptions.

For Ollama, there is no equivalent capability API — the harness needs its own static capability registry keyed by model name (does this model do tool calling? what `num_ctx` did we configure? does it adhere to structured-output schemas reliably?), populated from the known post-trained model list and validated empirically.

A unified `Capabilities` struct the backend trait exposes:
```
struct Capabilities {
    max_input_tokens, max_output_tokens,
    tool_calling: bool, parallel_tools: bool,
    structured_outputs: ReliabilityClass,   // guaranteed | best-effort | none
    prompt_caching: bool,                    // Anthropic yes / Ollama no
    reasoning_control: ReasoningKind,        // Effort{levels} | None
    context_managed_by: ContextOwner,        // ServerSide | HarnessSide
}
```

### 3.2 Graceful degradation — the harness backfills what the backend lacks

The design principle: **frontier-only API features must have a harness-side fallback so the same task can run on Ollama, just with more work done by the harness.**

| Capability | Anthropic | Ollama fallback (harness owns) |
|---|---|---|
| Prompt caching | `cache_control` breakpoints | none — accept full re-process cost (local is "free" anyway) |
| Reasoning depth | `effort` / adaptive thinking | prompt-level "think step by step"; no real lever |
| Context management | context editing / compaction (server) | harness-side transcript pruning + summarization |
| Cross-run memory | memory tool (model-driven) | harness-driven memory file injection |
| Structured output | guaranteed via `output_config.format` | request `format` + harness re-validates + retries on parse failure |
| Step/cost pacing | task budgets (model-aware) | harness step counter only (model unaware) |
| Convergence/stop detection | rich `stop_reason` set | minimal — harness infers from output + step budget |

The most important degradation is **context**: because Ollama silently truncates, the harness must token-count every prompt against the configured `num_ctx` and trigger *its own* pruning/summarization before the prompt exceeds it — replicating, in harness code, what the Anthropic API does as a server feature.

The second is **self-verification**: structured-output reliability drops on local models, so the harness validates the parsed disposition against its schema and re-prompts on failure, rather than trusting adherence.

### 3.3 One request model, two serializers

Both backends take JSON-Schema tools and a messages array with the same conceptual roles. The harness should carry one internal `Request` (messages, tools, the unified `Capabilities`-gated options) and have each backend serialize it to its wire format — translating the harness's neutral "reasoning effort" / "max output" / "structured format" knobs into Anthropic's `output_config.effort` / `thinking` / `output_config.format` or Ollama's `options.num_ctx` / `format`, dropping any knob the backend doesn't support. Capability detection (§3.1) decides what gets dropped vs. backfilled (§3.2).

---

## Implications for our headless-dispatch harness

- **Own the HTTP/JSON layer in Rust; don't depend on a community Anthropic SDK.** The API drifted hard in 2025–26 (adaptive thinking replaced `budget_tokens`, `effort` moved into `output_config`, `output_format` → `output_config.format`, sampling params removed on current models). A community crate that lags any of these 400s every request. `reqwest` + `serde` + an SSE parser, modeling the exact fields in §1, keeps us one edit from any new field. Ollama rides the same transport.

- **Make prompt-cache health a first-class, alarmed metric — not a hope.** A headless build run re-sends a growing transcript every turn; caching is the difference between 0.1× and full input cost on the whole history, but it fails *silently* (silent invalidators, sub-minimum prefixes, the 20-block lookback). Assert `cache_read_input_tokens > 0` after the first cached turn and surface a cost anomaly in the run report. Layer context as `tools+system` (frozen) → per-run task/repo context → rolling transcript, and never put the run ID / timestamp in the system prompt.

- **Treat `num_ctx` silent truncation as a ship-blocking hazard for the Ollama path.** Ollama defaults to 2048 tokens and *drops the oldest messages with no error* when exceeded — on an unattended run, the agent forgets the task spec and ships confident garbage. The harness must (a) set `num_ctx` explicitly per model, (b) token-count every prompt against it before dispatch, and (c) refuse-or-prune rather than let truncation happen. This backfills, in harness code, the context-management the Anthropic API provides as a feature.

- **Use structured outputs as the harness↔tracker contract, and re-validate on local backends.** Force the agent's final disposition into a JSON schema (`status`, `branch`, `gates_passed`, `summary`) via `output_config.format` (guaranteed on Opus) so the unattended run reports back parseably without a human reading prose. On Ollama, request `format` but validate the parse and retry — local adherence is best-effort, not guaranteed.

- **Pair model-aware task budgets with the harness's own hard limits.** On Opus 4.8/4.7, set `output_config.task_budget` so the model paces itself toward graceful completion, *and* enforce an independent step/cost ceiling in the harness (the model can't be trusted to self-limit, and Ollama has no budget knob at all). Set `effort: "xhigh"` for coding work, `thinking.display: "summarized"` so the post-run debug log isn't blank, and handle the full `stop_reason` set (`pause_turn`, `refusal`, `model_context_window_exceeded`) — any unhandled one strands a run with nobody watching.

- **Route by capability, and keep frontier for multi-step judgment.** Build a `Capabilities` struct per backend (Anthropic from the live Models API; Ollama from a static registry + empirical checks) and gate every feature off it. Don't assume the local model is co-located: "local model" means an Ollama-protocol endpoint at some URL (maybe localhost, maybe remote, maybe absent), not "Ollama on this box." Current evidence: local models (Qwen 3.x et al.) are reliable on mechanical/single-step tasks but degrade on 5+ step tool chains and failure recovery. Dispatch simple/mechanical build tasks to Ollama (free, private), reserve Anthropic for anything multi-step or where a wrong turn is expensive — exactly the tasks no human will catch mid-run.
