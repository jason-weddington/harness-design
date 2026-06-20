# Context Engineering for a Headless-Dispatch Harness

*Researched 2026-06-20.*

This doc surveys the mid-2026 state of the art in **context engineering** — managing the context window over long, multi-window agent runs — and translates it for a Rust harness built for **unattended headless dispatch**. The defining constraint throughout: there is no human watching mid-run, so the harness itself must own context discipline. A context window that fills up and degrades silently is, for us, a failed build with a half-implemented branch — exactly the failure mode Anthropic's long-running-agents work was written to prevent.

---

## 1. Context engineering vs. prompt engineering

Anthropic draws a sharp line. **Prompt engineering** is "methods for writing and organizing LLM instructions for optimal outcomes" — a discrete authoring task. **Context engineering** is "the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference," and it is *iterative*: it happens "each time we decide what to pass to the model" ([Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

The practical implication for a harness: prompt engineering is something you do once when you author the system prompt; context engineering is a **runtime control loop** the harness runs on every turn. For an unattended agent making hundreds of tool calls across multiple windows, the second is where the leverage is. The system prompt is fixed at dispatch time; the context *strategy* is what keeps a 3-hour run coherent.

## 2. Context is a finite, depleting resource

Two grounded constraints justify the entire discipline:

- **Context rot.** Anthropic states plainly: "as the number of tokens in the context window increases, the model's ability to accurately recall information from that context decreases." This is not a hypothetical. Chroma's *Context Rot* study tested 18 production models (GPT-4.1, Claude 4, Gemini 2.5, Qwen3) across 10K–500K-token contexts and found "model reliability decreases significantly with longer inputs, even on simple tasks like retrieval and text replication" — degradation that is **monotonic** in input length and **non-uniform** in shape ([Chroma Context Rot study, via ZenML LLMOps DB](https://www.zenml.io/llmops-database/context-rot-evaluating-llm-performance-degradation-with-increasing-input-tokens)). A striking finding: models performed *better* on randomly shuffled haystacks than on logically coherent documents, so you cannot assume "well-organized context = better recall."
- **Attention budget.** The transformer creates "n² pairwise relationships for n tokens," so as context lengthens "a model's ability to capture these pairwise relationships gets stretched thin." Anthropic frames context as having a finite "attention budget" that every token draws down, producing "a performance gradient rather than a hard cliff" ([Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

**Headless lens:** an interactive user notices when the model starts repeating itself or losing the thread and re-steers. Our agent has no such observer. The harness must treat *every avoidable token* as a quality risk, not just a cost risk — because the failure shows up not as an error but as a confidently-wrong commit. The smallest-coherent-context principle ("find the smallest possible set of high-signal tokens that maximize the likelihood of the desired outcome") is therefore a correctness control for us, not an optimization.

## 3. Layering context most-stable → least-stable (prompt-cache efficiency)

Anthropic prompt caching is **prefix-based and exact**: a cache hit requires "100% identical prompt segments… up to and including the block marked with cache control," and prefixes are built in a strict hierarchy of **tools → system → messages** ([Prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). Economics make this a first-order design concern:

- Cache **reads cost 0.1×** base input price (a 90% discount).
- Cache **writes cost 1.25×** (5-minute TTL) or **2×** (1-hour TTL).
- Up to **4 cache breakpoints** per request; the lookback window is **20 blocks** per breakpoint.
- Minimum cacheable length is model-dependent (e.g. 1,024 tokens for Opus 4.8 / Sonnet 4.6; 4,096 for Haiku 4.5).

The design rule that falls out: **order content most-stable → least-stable**, and place breakpoints at the boundaries between stability tiers. A practitioner consensus puts the optimum at "three or four breakpoints arranged from most-stable to most-volatile" — same tools, same system prompt, same early messages, with the breakpoint on "the last block that stays identical across requests" ([PromptHub](https://www.prompthub.us/blog/prompt-caching-with-openai-anthropic-and-google-models); [mager.co](https://www.mager.co/blog/2026-04-29-claude-prompt-caching/)). A single differing token anywhere in the prefix is a full cache miss at full price.

The academic evidence is now firm and **specific about what to cache**. *Don't Break the Cache* (arXiv 2601.06007) evaluated prompt caching across three providers on long-horizon agentic tasks and found **41–80% cost reduction** and **13–31% TTFT improvement** — but with a critical caveat: "caching only system prompts while excluding dynamic tool results provides more consistent benefits than naive full context caching, which can paradoxically increase latency" ([arXiv 2601.06007](https://arxiv.org/pdf/2601.06007)). Caching the whole evolving conversation can cost more than it saves.

**One serialization gotcha that bites Rust/Go/Swift:** cache keys are byte-exact, so non-deterministic JSON key ordering silently breaks the cache. The docs explicitly warn that some languages "randomize key order during JSON conversion, breaking caches" ([PromptHub](https://www.prompthub.us/blog/prompt-caching-with-openai-anthropic-and-google-models)). For our harness this is a hard requirement: **serialize tool definitions and message content with deterministic, stable key ordering** (e.g. `BTreeMap` or an explicitly-ordered struct via `serde`, never `HashMap`).

**Headless lens:** a multi-hour unattended run is dominated by re-sending the same tool defs + system prompt + accumulated history thousands of times. Cache discipline is where the bulk of the dispatch budget is won or lost, and for local Ollama models (no cache pricing) the layering still matters for the *server-side compaction* boundary discussed next.

## 4. Compaction / summarization of history

Compaction is "taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary" ([Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). The art is "selection of what to keep versus what to discard" — Anthropic's advice is to start by maximizing recall (keep everything that might matter), then tune toward precision.

As of 2026 this is a **server-side primitive**, not just a pattern you hand-roll. The `compact_20260112` edit (beta header `compact-2026-01-12`) does it for you ([Compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction)):

- **Trigger** defaults to 150,000 input tokens (minimum 50,000). When exceeded, the API generates a `<summary>` block, emits a `compaction` content block, and **automatically drops all prior message blocks** on subsequent requests — you append the response and continue from the summary.
- The default summarization prompt is tuned for continuity: *"write a summary… to provide continuity so you can continue to make progress… in a future context, where the raw history above may not be accessible… Write down anything that would be helpful, including the state, next steps, learnings."*
- **`instructions`** completely replaces that prompt (e.g. "Focus on preserving code snippets, variable names, and technical decisions"). **`pause_after_compaction: true`** returns `stop_reason: "compaction"` so you can inject preserved recent messages before continuing.
- **Caching interaction:** put a `cache_control` breakpoint at the end of the system prompt so it "remains valid and is read from cache" through a compaction event; "only the compaction summary needs to be written as a new cache entry."
- **Billing nuance:** compaction is an extra sampling step billed separately in the `usage.iterations` array; top-level `input_tokens`/`output_tokens` exclude it. Cost accounting must sum the array.

A reported result from combining memory + context editing was an **84% token reduction** on a 100-turn web-search eval and a **39%** accuracy gain on agentic search ([Anthropic Sonnet 4.5 announcement / context-editing coverage](https://www.anthropic.com/news/claude-sonnet-4-5)).

**Caveat from the long-running-agents work:** compaction alone "isn't sufficient" and "doesn't always pass perfectly clear instructions to the next agent." It is lossy by construction. ([Anthropic, Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents).) Treat compaction as a memory-pressure relief valve, **not** as the cross-window handoff mechanism (§7 covers the durable handoff).

**Ollama lens:** local models have no server-side compaction. Our harness must implement the same loop client-side: track the running token estimate, and when it crosses a model-specific threshold, summarize the transcript with the same model (or a cheaper one) and reinitialize. The Anthropic default prompt above is a ready-made template to port.

## 5. Context editing / rule-based pruning

Distinct from summarization, **context editing** *deletes* specific stale content rather than compressing everything. The `clear_tool_uses_20250919` edit (beta header `context-management-2025-06-27`) clears the oldest tool results once context crosses a threshold, replacing them with a placeholder so the model knows they existed ([Context editing docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)). Parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `trigger` | 100,000 input tokens | when clearing activates (or `{type: "tool_uses", value: N}`) |
| `keep` | 3 tool uses | most-recent tool interactions to preserve |
| `clear_at_least` | none | minimum tokens to clear per activation — "helps justify cache invalidation" |
| `exclude_tools` | none | tool names never cleared |
| `clear_tool_inputs` | `false` | also clear the tool call *parameters*, not just results |

There is also `clear_thinking_20251015` for extended-thinking blocks (must be listed first in the `edits` array; keeping thinking blocks preserves the cache, clearing them invalidates it at the clear point).

The key subtlety: **clearing invalidates the cached prefix** at the clear point. `clear_at_least` exists precisely so you clear enough to be worth the cache re-write. This is a real tension — pruning for attention budget *fights* prompt-cache hit rate. The harness has to make that trade deliberately.

**Headless lens — this is high value for a build engine.** A coding agent runs many tool calls whose results go stale fast: a `read_file` from 40 turns ago, a `grep` that's been superseded, an old test run. Those bloat context and accelerate rot, but the *fact that the action happened* often still matters. `exclude_tools` lets us keep the durable signals (e.g. the latest test-gate result) while clearing transient reads. A good default: clear old file reads and search results aggressively, exclude the quality-gate tool, and write anything load-bearing to memory *before* it's cleared (§6).

## 6. Memory tools — persistent store + retrieval across windows

The Claude **memory tool** (`memory_20250818`) is a client-side, file-based store: Claude issues `view` / `create` / `str_replace` / `insert` / `delete` / `rename` commands against a `/memories` directory, and *your application executes them* against whatever backend you choose (filesystem, DB, encrypted store) ([Memory tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)). Persistence is the point: files survive across context windows and across whole conversations, so the model can "pick up where it left off."

This is the concrete realization of Anthropic's **structured note-taking** long-horizon technique — "regularly write notes persisted to memory outside of the context window [that] get pulled back into the context window at later times" — giving "persistent memory with minimal overhead" ([Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

Two things the harness MUST own:

1. **Path-traversal protection.** The docs flag this with a warning: validate every path stays under `/memories`, canonicalize, reject `../`, `..\\`, and URL-encoded (`%2e%2e%2f`) sequences. In Rust: resolve to canonical form and assert `.starts_with(memories_root)` after `canonicalize()`. This is non-negotiable for an unattended agent that could be steered by a poisoned task description.
2. **The interruption contract.** The system prompt auto-injected with the tool says it outright: *"ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory."* That assumption is *literally true* for headless dispatch — the run can hit a budget limit or a window boundary at any point. Memory is the agent's only durable state across those resets.

**Memory + context editing/compaction work together:** when context editing is about to clear tool results, "Claude receives automatic warnings to preserve critical context to memory files before tool results are cleared." And memory "persists important information across compaction boundaries so nothing critical is lost in the summary." The trio — compaction (relieve pressure) + context editing (prune stale tool noise) + memory (durable state) — is the mid-2026 stack for long runs.

## 7. Consistent progress across multiple context windows

This is the heart of the matter for us, and Anthropic's [long-running-agents harness post](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) is the canonical text. Their findings (from building an agent that writes a whole app across many sessions):

- **The failure mode is over-reach.** The agent "tended to try to do too much at once — essentially to attempt to one-shot the app," ran out of context mid-implementation, and left "the next session to start with a feature half-implemented and undocumented." The fix: **work on one feature at a time.** "This incremental approach turned out to be critical."
- **Don't rely on compaction for handoff.** Use durable external artifacts instead: a `claude-progress.txt` log of what's been done, **descriptive git commits** as a recoverable state trail, and a **feature-list file** (JSON) with `passes: false` flags as the authoritative scope + status record.
- **A different prompt for the first window.** An *initializer* session bootstraps the memory artifacts (progress log, feature checklist, an `init.sh` to restart the dev server). Subsequent sessions begin by: `pwd` → read git log + progress file → read the feature list → pick the highest-priority not-done feature. This recovers full state "in seconds, without needing to re-explore the codebase."
- **Leave a clean state.** Each session must end in "code that would be appropriate for merging to a main branch" — no major bugs, documented. This is what makes the next window's pickup reliable.

The memory-tool docs codify this as the "multi-session software development pattern": initializer session sets up artifacts → each session reads them → end-of-session updates the progress log; **work one feature at a time, mark complete only after end-to-end verification.**

**Headless lens:** our harness IS the orchestrator of these windows. We don't get to ask a human "is this done?" The progress file + feature list + git history are the *substrate the harness reads and writes between windows* to decide whether to spawn another window, what it should pick up, and when the task is actually finished. This is where context engineering meets the harness's stop/convergence-detection responsibility (covered in the failure/verification track) — they share the same durable artifacts.

## 8. Self-verification & premature-completion (a context problem too)

A specific long-run pathology, directly from Anthropic: "after some features had already been built, a later agent instance would look around, see that progress had been made, and declare the job done." The fix is structural, not exhortative — a feature list with explicit `passes: false`, instructions to "only mark features as 'passing' after careful testing," and a rule that "it is unacceptable to remove or edit tests." Agents also initially "fail[ed] to recognize that the feature didn't work end-to-end" until "explicitly prompted to … do all testing as a human user would" ([long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).

**Headless lens:** premature "done" is the single most dangerous orchestration failure for a build engine — it ships a branch that doesn't work and reports success. The convergence signal must be **the project's own quality gates passing against a checklist**, never the model's self-assessment. Context engineering supports this by keeping the *checklist and gate results* in context (or in memory, re-read each window) as the source of truth, while the noisy exploration history is compacted/cleared away.

## 9. Context-awareness — telling the model its budget

Neither seed post gives the model an explicit live "you have N tokens left" signal, and Anthropic's own approach instead relies on **structural** discipline (one feature at a time, leave clean state) so the agent never *needs* to race a depleting budget. That said, the building blocks now exist: the token-counting endpoint supports context-management preview (`original_input_tokens` vs `input_tokens` after editing), and compaction's `pause_after_compaction` gives an explicit checkpoint. Community practice increasingly injects a budget-awareness line into the system/turn prompt so the model wraps up gracefully rather than getting truncated mid-thought.

**Headless lens:** with no human to say "wrap it up," the harness should make budget legible to the model. Two complementary moves: (a) **structurally** scope each window to one feature so budget is rarely the binding constraint; (b) when the running token estimate crosses, say, 70% of the window (the point where, per the cost-prior framing, quality degrades and inference slows), inject an explicit "you are at N% of budget — checkpoint to memory and prepare a clean handoff" instruction. This converts a silent truncation into a controlled handoff.

## 10. Just-in-time vs. pre-loaded context

The field is shifting toward **just-in-time** context: agents "maintain lightweight identifiers (file paths, stored queries, web links) and use these references to dynamically load data into context at runtime." Claude Code uses this to "perform complex data analysis over large databases… without ever loading the full data objects into context." **Hybrid** approaches "retrieve some data up front for speed" then explore further "at its discretion" ([Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). The memory tool is described as "the key primitive for just-in-time context retrieval."

The trade-off: pre-loading is faster and more deterministic; JIT is more relevant and keeps context lean but costs tool round-trips and risks the agent not fetching what it needs.

**Headless lens — strongly favor JIT for a coding agent.** A repo is far too large to pre-load and pre-loading it all would maximize context rot. Give the agent file-path/grep/read tools and a lean entrypoint (the groomed task + a repo map + the feature checklist), and let it pull code on demand. Pre-load only the few high-signal, low-volatility things: the task spec, the project's CLAUDE.md/conventions, and the quality-gate definitions. Everything else is fetched JIT and cleared when stale (§5).

---

## Implications for our headless-dispatch harness

- **Context engineering is a correctness control, not just cost.** No human catches a degraded window mid-run, so treat token bloat as a quality risk that ships wrong branches. Build the harness's per-turn loop around the smallest-coherent-context principle, and make context rot a first-class concern, not an afterthought.
- **Make the handoff durable and external; never trust compaction to carry it.** Adopt the long-running-agents pattern wholesale: an initializer window that bootstraps a progress log + feature checklist (`passes: false`) + `init.sh`, descriptive git commits as a recovery trail, and a fixed "read state → pick highest-priority not-done feature → leave clean" loop. **One feature per window.** Compaction relieves in-window pressure; the durable artifacts carry cross-window state.
- **Layer context most-stable → least-stable and enforce deterministic serialization.** Tools → system → conventions → checklist → volatile history, with cache breakpoints at the tier boundaries (Anthropic API path). In Rust, serialize tool defs/messages with stable key ordering (`BTreeMap`/ordered structs, never `HashMap`) — randomized key order silently breaks the byte-exact prefix cache. Cache the system/tools prefix separately so a compaction event doesn't invalidate it.
- **Run the full long-run stack: compaction + context-editing + memory.** Use server-side `compact_20260112` for Anthropic backends and a hand-rolled equivalent for Ollama; use `clear_tool_uses` to prune stale file-reads/searches while `exclude_tools` protects the quality-gate result; persist load-bearing state to a path-validated memory store *before* anything is cleared. Honor the "ASSUME INTERRUPTION" contract — a budget/window cutoff can land on any turn.
- **Convergence = gates pass against the checklist, never model self-report.** The premature-"done" pathology is the worst headless failure. Keep the checklist + gate results in context/memory as the source of truth; require end-to-end verification (run the project's own tests/build) before any feature flips to `passes: true`; forbid the agent from editing or deleting tests to make them green.
- **Default to just-in-time context with a lean pre-loaded core.** Pre-load only the task spec, project conventions, and gate definitions; expose file/grep/read tools for everything else. Inject an explicit budget-awareness checkpoint (~70% of window) so the agent writes to memory and hands off cleanly instead of getting truncated — the harness must supply the "wrap it up" signal a human would otherwise give.

## Sources

- Anthropic — Effective context engineering for AI agents — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic — Effective harnesses for long-running agents — https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
- Claude API Docs — Context editing — https://platform.claude.com/docs/en/build-with-claude/context-editing
- Claude API Docs — Memory tool — https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
- Claude API Docs — Compaction (compact_20260112) — https://platform.claude.com/docs/en/build-with-claude/compaction
- Claude API Docs — Prompt caching — https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks (arXiv 2601.06007) — https://arxiv.org/pdf/2601.06007
- Chroma — Context Rot: Evaluating LLM Performance Degradation with Increasing Input Tokens (via ZenML LLMOps DB) — https://www.zenml.io/llmops-database/context-rot-evaluating-llm-performance-degradation-with-increasing-input-tokens
- PromptHub — Prompt Caching with OpenAI, Anthropic, and Google Models — https://www.prompthub.us/blog/prompt-caching-with-openai-anthropic-and-google-models
- mager.co — How prompt caching actually works — https://www.mager.co/blog/2026-04-29-claude-prompt-caching/
- Anthropic — Introducing Claude Sonnet 4.5 (memory + context editing eval numbers) — https://www.anthropic.com/news/claude-sonnet-4-5
