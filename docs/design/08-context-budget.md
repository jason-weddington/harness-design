# 08 — The context budget: a resolved output cap, and compaction

Status: **DECIDED in principle (2026-09-19, Jason), design proposed here.** Two decisions are made and not reopened by this record: the per-turn output cap stops being an arbitrary constant, and talos grows a compaction mechanism so a run whose history outgrows the window compacts instead of dying. What this record adds is the shape, grounded in prior art and in measurements from our own fleet transcripts. The tiering in "The design" is a proposal; everything under "The evidence" is measured.

## Why this exists

Two dispatch runs died at exactly 32,768 output tokens, and both were recorded as something else.

`c0016303` (PhotoQueue stats sweep, 24 acceptance criteria) ran 14 iterations of orientation, one 12,278-token planning turn, five iterations executing the first criterion, then emitted a 106,478-character reasoning block on iteration 21, hit the cap mid-sentence on the words "Let me write the file.", produced no tool call, and terminated. The worker rendered that as "stalled / no convergence — consider re-decomposing". Analysis in `docs/research/09-transcript-study-two-dispatch-misses.md`, `kb-03372`.

`5bc7128b` (the groom-to-ready port to talos-flow, 35 acceptance criteria) ran seven iterations of pure reading with zero edits, then emitted a 129,141-character reasoning block on iteration 8 and died the same way. That failure was recorded in `kb-03332` as a context failure — the spec cited a source file no fleet agent could reach — which is true and is why the agent was flailing, but is not what killed the run. The terminal was the output cap. Run-log entries are append-only by convention, so that entry stands and this record carries the correction.

Both were `glm-5.3` at `think=high`. Neither was anywhere near its context window: the prompts at death were 46,004 and 22,292 tokens against a pinned `num_ctx` of 1,048,576. The second is two percent of the window.

`5211a15` (v0.11.0) now names this terminal `FailureMode::Truncated` instead of `StoppedWithoutFinish`, which fixes the diagnosis and nothing else. The run still dies, and `engine.rs:2714` returns it with `recovery_facts` deliberately `None`.

## The evidence

### There is no output cap to respect on Ollama, and a strict one on Anthropic

Measured 2026-09-19; full figures and method in `kb-03380`, and `docs/design/07-num-ctx.md` carries the correction to its own "unverified" note.

`POST https://ollama.com/api/show` returns the same shape as a local daemon and advertises **no output limit of any kind** — only `{arch}.context_length`, 1,048,576 for both `glm-5.3:cloud` and `glm-5.3-flash:cloud`. `num_predict` is a client-side ceiling honoured exactly: 50 returns `eval_count` 50 with `done_reason` `length`, and 60,000 produced a single **50,205-token turn** that stopped naturally on `done_reason` `stop`. The cloud does not clamp at 32,768. That ceiling is entirely ours.

Anthropic is the opposite and the messier case. `GET /v1/models/{id}` publishes `max_tokens` per model and exceeding it is a 400: `claude-haiku-4-5` 64,000, `claude-sonnet-5` / `claude-opus-4-8` / `claude-opus-5` 128,000 each. So one raised constant shared across backends breaks the Haiku lane specifically.

### Context pressure is real, and on a different lane from the failures

Peak prompt across every transcript on the fleet:

| Lane | peak prompt | window | used |
|---|---|---|---|
| qwen3.8:27b | 210,023 | 262,144 | 80% |
| qwen3.8:27b | 129,995 | 262,144 | 50% |
| glm-5.3-flash | 122,017 | 1,048,576 | 12% |
| glm-5.3 | 84,170 | 1,048,576 | 8% |

Both qwen runs finished successfully. We have never actually exhausted a context window. Compaction is insurance, and the lane it insures is the small-window local one, not the lane that has been failing.

The coupling matters, though: removing the output cap lets a single turn append far more than 32,768 tokens to the history, so it raises context pressure rather than lowering it. That is the argument for doing both together rather than shipping the cap change alone.

### What actually fills the context

Share of total generated-and-replayed content, measured per transcript:

| Run | tool results | reasoning |
|---|---|---|
| PhotoQueue, 24 criteria, glm-5.3 | 28.6% | 57.2% |
| groom port, 35 criteria, glm-5.3 | 22.3% | 61.9% |
| flickrasync, succeeded, flash | 48.4% | 6.1% |
| qwen3.8, peak 210K prompt | 56.0% | 42.9% |
| qwen3.8, peak 130K prompt | 65.6% | 32.9% |

Reasoning is replayed into the prompt rather than dropped. The arithmetic confirms it on PhotoQueue: iteration 15 emitted 12,278 output tokens, its two tool results were about 1,225 tokens, and iteration 16's prompt grew by 13,559.

The load-bearing consequence: **tool results and reasoning together are 85-99% of the replayed content on every lane measured.** Everything else — the system prompt at roughly 5,300 characters, the task spec at 13,000-34,000, assistant text at under 900 — is rounding error. A compaction scheme that addresses those two categories does not need to summarize anything to get most of the available relief.

## The proposed scheme, and what prior art changes about it

The starting proposal was: keep the first two turns, keep the last two turns, summarize everything in between, and keep a list of tool calls without payloads so the agent retains some history.

The shape is right and matches shipped systems. Three things need to change, two of them for correctness.

**Turn counts are the wrong unit; use tokens.** Our turns range from 21 to 32,768 output tokens, a factor of 1,500. "The last two turns" is somewhere between 200 and 65,000 tokens depending on which two. Pi keeps a token-denominated tail (`keepRecentTokens`, about 20,000) and triggers on `contextWindow − reserveTokens` with a default reserve of 16,384 (`kb-02018`). Anthropic's compaction triggers on an input-token threshold, default 150,000, minimum 50,000, and its context-editing equivalent defaults to 100,000 (`docs/research/02-context-engineering.md` §4-5).

**Never cut a tool-call/tool-result pair.** This is the correctness bug in the naive version. Pi explicitly walks boundaries to keep pairs intact. A `tool_result` whose matching `tool_use` has been summarized away is rejected outright by the Anthropic API, so a fixed turn count that happens to land mid-pair turns a compaction into a hard 400. Whatever the window rule is, the cut point must be snapped to a pair boundary.

**Do the free things before the expensive one.** Summarization needs a model call, which is a new mid-run failure mode — a summarizer that is itself truncated, or wrong, and nothing downstream can tell. Given the composition measured above, dropping old reasoning and eliding old tool-result payloads are deterministic, need no model call, and reach 85-99% of the mass. Anthropic ships these as two separate first-class strategies precisely because they are the cheap safe ones: `clear_thinking_20251015` and `clear_tool_uses_20250919`, the latter described in our own research notes as "one of the safest forms of context compaction" (`kb-01931`). Cognition's warning is the other half of the argument: history compression is "non-trivial to get right", needs domain-specific investment, and they ended up fine-tuning models for it (`kb-01964`). Anthropic's own long-running-agents work says compaction alone "isn't sufficient" and is lossy by construction (`docs/research/02-context-engineering.md` §4).

Two refinements prior art adds that the proposal did not have. Anthropic's `clear_tool_uses` takes an `exclude_tools` list, which exists so the load-bearing signal survives — for us that is the most recent `run_checks` result, the done-oracle the agent is converging on. And cleared results are replaced by a placeholder so the model knows the call happened, which is exactly the "tool calls without payloads" instinct, already validated.

## The design

**Anchors, never compacted.** The system prompt and the task message. The task message carries the acceptance criteria and runs 13,000-34,000 characters; losing it is losing the spec. This is the precise version of "keep the first two turns" — in talos the system prompt is not a message at all (`Message` has no `System` variant, `model.rs:119`), it rides `TurnRequest::system`, so the anchor set is "the system prompt plus `messages[0]`".

**Trigger, token-denominated and exact.** Every response carries its usage, so the engine knows the true prompt size of the last turn for free — no estimate needed. Compact when that value alone reaches the model's context limit times a threshold, pinned at 90%. `estimate_prompt_tokens` (`ollama.rs`) stays what it is, a `chars / 4` tripwire that under-counts and is not a sizing oracle, per design 07 option (c).

**Two corrections to this paragraph, both caught after the first implementation and both load-bearing.**

*The prompt size is not `usage.input_tokens`.* This record originally said it was. That stopped being true in `7b2c6ea`, which made `input_tokens` the **uncached remainder** — so a fully cached 200K prompt reports near zero and the trigger would never fire on exactly the long, heavily-cached runs it exists for. The raw prompt is `input_tokens + cache_read_tokens + cache_write_tokens`.

*No next-turn reserve is added, and adding one is a tautology.* This record originally said to add "a reserve for the next turn… the same number the output cap resolves to, so the two decisions share one budget". That reads well and is wrong. The derived Ollama cap **is** `limit − prompt − OUTPUT_TOKEN_MARGIN`, so adding it back to the prompt cancels the only pressure-sensitive term: `prompt + (limit − prompt − margin) = limit − margin`, a constant. The predicate collapses to `limit − 16,384 ≥ 90% of limit`, i.e. `limit ≥ 163,840` — true for both fleet windows (262,144 and 1,048,576), so compaction fired on **every pass from iteration 2 at roughly 1% window occupancy**. That is the unconditional standing policy this document explicitly rejects two sections above, it pays the re-derivation risk continuously, it rewrites the cached prefix every turn (converting cache reads at $0.26/M into uncached input at $1.40/M), and it is invisible in telemetry because `compactions ≈ iterations` reads as heavy pressure rather than as a stuck predicate. The reserve is also **redundant**: the per-iteration output cap already guarantees `prompt + output ≤ limit` on the derived lane, so overflow is the cap's job and pressure is the trigger's. Pinned by `should_compact_does_not_telescope_to_a_constant_on_derived_caps`.

**Tier 1 — drop reasoning blocks outside a retention window.** Free, deterministic, no model call. Surgically simple at our seam, because reasoning is its own variant: filter `ContentBlock::Reasoning` out of older `Message::Assistant` content (`model.rs:79`). The tool calls and their results remain, so the record of what the agent *did* is untouched; what is lost is the record of why it decided to. That distinction is the whole risk, and it is treated below rather than waved at.

**Size the retention window generously, because generosity is nearly free.** Simulated against the 164-iteration qwen run — the only run we have that approaches its window — dropping reasoning older than the window moves the peak prompt like this: keep 2 turns, 154,320 tokens; keep 3, 155,497; keep 5, 156,126; keep 10, 156,893. The whole span from 10 turns down to 2 is worth 1.2 percentage points against a 26% total saving. The curve is flat because on a long run the mass is in the old reasoning, not the recent. So the window should be set by how much history the model needs to stay coherent, not by how much context it buys, and when those two pull against each other coherence wins at almost no cost. Proposed default: 10 turns.

**Retain the tail of a dropped block rather than deleting it outright.** Cheap insurance against re-derivation: keep the last few hundred tokens of each reasoning block, which is where the conclusion lives — PhotoQueue's fatal block ended on "Let me write the file", i.e. the decision was in the final sentence and the preceding 106,000 characters were the derivation. This is a proposal, not a measured result, and the telemetry below is what would confirm or kill it.

**Do not make this unconditional.** The suggestion on the table was to drop reasoning past 2-3 turns as a standing policy every turn, independent of pressure. The data argues against paying any behavioural risk when there is no pressure to relieve: on the three short runs the same window moves the peak prompt by 0.4%, 3.0% and 3.0%, because their reasoning mass is recent or the run is too short to accumulate any. It is only the long run that sees 26%. So tier 1 stays pressure-triggered like the rest. A standing policy would also rewrite the cached prefix on every turn, which is a cost we can now actually measure since `7b2c6ea` parses `prompt_eval_cached_count`, and should be measured before anyone reaches for it.

**Tier 2 — elide tool-result payloads outside the recent window, and make the elision reversible.** Replace the `content` of an older `UserBlock::ToolResult` (`model.rs:99`) with a stub naming the tool, its arguments, and an offload path. Talos already writes oversized results to disk and tells the agent "full output at `<path>`" (`tool.rs:194-197`), and `read_file` is already permitted to read an absolute path under the offload root and nowhere else (`workspace.rs:129-143`). So the stub can point at real bytes and the agent can pull any of it back. That is strictly better than discarding payloads, and the machinery exists. Two wrinkles: `DETAIL_CAP` is 25,000 characters (`tool.rs:34`) so most results were never offloaded and must be written at compaction time; and `ToolResult` carries no structured `offload_path` on the message, only the path embedded in rendered text, so the stub has to be built from a fresh offload rather than by parsing. Exclude the most recent `run_checks` result, which always has an offload path anyway (`exec.rs:503`).

**Tier 3 — summarize the elided span**, prepended as a synthetic user message, only when tiers 1 and 2 leave the prompt over target. Anthropic's default continuity prompt is a ready-made template and our research doc already quotes it. A cheaper model may generate the summary than the one running the task (`kb-02422`), which on our fleet means flash summarizing a glm-5.3 run. Not built first: the composition data says it should rarely fire, and it is the only tier that can fail in a way nothing downstream detects.

**Intercept the error path too.** `BackendError::ContextLengthExceeded` is non-retryable and reaches `LoopOutcome::BackendError` on first occurrence (`engine.rs:2584-2618`, pinned by `context_length_exceeded_not_retried` at `engine.rs:9786`). That is the seam where a run that overruns anyway should compact and retry once rather than die. Ollama's pre-flight guard (`ollama.rs:248-252`) raises it before the request is sent; Anthropic and Bedrock only string-match a 400 after the fact (`anthropic.rs:533`, `bedrock.rs:519`).

**The output cap becomes derived, not constant.** Remove `DEFAULT_MAX_TOKENS = 32768` (`engine.rs:299`) as a default. Add a limits accessor to `ModelBackend` — today a one-method trait (`model.rs:357-363`) — as a default method so it stays object-safe, and forward it through the `Backend` dispatch enum (`main.rs:534-543`). Ollama derives the cap from the context budget, so output can never be the binding constraint before context is, and context is what compaction handles. Anthropic and Bedrock need a per-model table, since neither holds any limit today. `--max-tokens` survives as an explicit operator override that wins verbatim, exactly as `OLLAMA_NUM_CTX` does for the context window, and the resolved value and its provenance land on `BackendSettings` (`run_record.rs:69-95`) beside `num_ctx_source`. Four independent paths build `RunConfig::new` without `with_max_tokens` and silently inherit the constant today — `ralph.rs:524`, `eval.rs:417`, `mined_eval.rs:2331`, plus the CLI — so a resolver must cover all four or the eval lane and the shipped lane drift, which is the exact failure design 07 exists to prevent.

This is what makes truncation structurally impossible rather than merely rarer: the only thing that can cut a turn short is the context window, and the context window is what compaction manages.

## The re-derivation risk, and why it is acute for exactly our failing lane

The sharpest objection to tier 1: drop a reasoning block too early and the model may simply re-derive it, going into another massive reasoning turn — which is the precise failure that killed both runs this record exists for. The risk is real, and the fleet data says it is worse for us than for the harnesses that ship reasoning-clearing happily.

The reason is that our models externalize almost nothing. Total visible assistant text across an entire run: **148 characters over 21 iterations** on PhotoQueue, **202 over 8** on the groom port, **6,521 over 164** on the long qwen run — about forty characters per turn. Tool calls and their arguments persist, so the *actions* survive a drop, but the plan behind them does not, because `glm-5.3` at `think=high` writes its plan into reasoning and essentially nowhere else. Claude Code and the Anthropic API can clear thinking cheaply in part because their models narrate decisions into visible text before the thinking is gone. Ours do not.

Three mitigations, in increasing cost. Size the window generously, which the flat curve above makes nearly free. Retain each dropped block's tail so the conclusion survives even when the derivation does not. And, if telemetry shows the first two are insufficient, require externalization before dropping — the structured note-taking pattern our research notes already prescribe, where load-bearing state is written outside the context window before it is cleared; Anthropic warns the model to preserve context to memory before clearing tool results for exactly this reason.

This is the hypothesis the instrumentation exists to test, so it gets a dedicated measurement rather than a judgement call.

## Telemetry — the non-negotiable half

The instrumentation is the deliverable in the early going, because the scheme is deliberately simple and we need to see how the model behaves after a compaction. Transcripts already let us read post-compaction behaviour directly.

Counters on `RunStats` (`engine.rs:783`): number of compactions, highest tier reached, tokens reclaimed, results elided. Note `RunStats` has no `Default`, so every new field breaks roughly nine exhaustive literals across `engine.rs`, `eval.rs` and `claude_code_eval.rs`.

A `compaction` transcript event carrying the trigger numbers, the tier, the elided call ids and their offload paths, and prompt size before and after. This is required, not optional: the transcript module states a replay invariant — at every `model_request` the rebuilt history's length and block count must equal the recorded `message_count` and `block_count` (`transcript.rs:300-303`) — so a compaction that mutates `messages` silently breaks the format's own contract. Adding an event means touching `EVENT_KINDS` and its length literal (`transcript.rs:332`), the module doc, the emit site, and the two golden tests that pin exact transcript line counts (`engine.rs:10103`, `crates/talos/tests/cli.rs:273`). Worth noting while in there: `contract_violation` is already emitted (`engine.rs:2202`) and is absent from `EVENT_KINDS`, so that array is already wrong.

Two direct disorientation signals, both cheap:

- **Elided re-reads.** The agent calls `read_file` on an offload path we elided. That is the agent telling us the elision was too aggressive, and it is only measurable because tier 2 is reversible.
- **Repeated work.** The agent issues a tool call it had already issued before the compaction, detectable by hashing `(tool_name, input)`. That is the agent having forgotten what it already did.
- **Re-derivation.** Reasoning tokens in the turns immediately following a tier-1 drop, against that run's own pre-drop mean. A spike is the model re-deriving the plan we discarded, and it is the direct test of the risk above. Worth recording per turn rather than as a single ratio, since the shape matters: one large turn is a re-plan, a sustained rise is genuine disorientation.

Plus the coarse ones: iterations between compactions, iterations from the last compaction to the terminal, and the outcome distribution of runs that compacted at least once versus runs that never did.

## Default-on at 90%, with a toggle (2026-09-19, Jason)

Compaction ships **enabled by default** at a 90% window-fill threshold, with an explicit knob to turn it off. Jason's reasoning, recorded because it deliberately overrides a standing rule: *compaction is a safety net, and safety nets should be on by default.*

The standing rule it overrides is this project's "ship behind a default-off knob; flip only after tier-2 shows benefit and the perfect-spec guard is clean" (`CLAUDE.md`). That rule exists for features that change behaviour on every run — an extra prompt section, an extra model call — where default-off is how you avoid paying for something unmeasured. Compaction is not that shape. It is inert until the window is nearly full, and the measurement below shows it has never been reachable on real work at all. A safety net that is off by default is absent precisely when it is needed, and the run that needs it is an unattended dispatch with nobody watching.

The knob is still mandatory, for two reasons that have nothing to do with safety. Without a way to disable it there is no control arm, so no experiment comparing compaction against a baseline can be run. And without a way to *lower* the threshold it can never fire in a test at all. `COMPACT_THRESHOLD_PCT` is therefore configurable end-to-end (flag, then env, then the 90 default, mirroring `--state-retention-days`), where **0 disables compaction entirely** — no walk, no event, no counter.

**Vary the threshold, never the window.** Forcing compaction by shrinking `num_ctx` would be a broken experiment: the derived per-turn output cap is itself `window - prompt - margin`, so shrinking the window moves the cap too and confounds two variables. The knob exists so the window stays pinned at its production value while the trigger moves.

### Measured: the trigger has never been reachable, and the pre-fix trigger always was

Both predicates replayed over every transcript on the fleet — 54 eligible runs, 3,061 turn transitions:

| Predicate | Fired | Rate |
|---|---|---|
| Pre-fix (prompt + derived cap as reserve) | 3,061 | 100% |
| Shipped (prompt alone vs 90%) | 0 | 0% |

The highest window fill ever observed on real work is **80.1%**, on a 164-iteration qwen3.8 run at 210,023 of 262,144. Every glm run sits between 7% and 20% of its 1,048,576 window. So the shipped threshold would not have fired on anything we have ever run, which is simultaneously the evidence that the fix is correct, the evidence that compaction is insurance rather than a live need, and the reason the evals must force it.

That 80.1% figure is also the calibration argument for 90 rather than something lower: the one run that came closest to its window completed successfully without help, so a threshold below 80 would have compacted a run that did not need it.

## Compaction is Ollama-only (2026-09-19, Jason)

Compaction ships for the Ollama backend and is deliberately not implemented for Anthropic or Bedrock. The reason is economic rather than technical: running talos against the Anthropic API trades an already-paid subscription for metered per-token billing, so in practice talos is the harness we point at open-weights models. The Anthropic and Bedrock backends exist for completeness and to keep the model-layer abstraction honest, not because we run production dispatch through them.

This also disposes of the open question about whether dropping reasoning is safe under Anthropic's thinking-block replay rules. That question is now moot rather than answered, and it returns the moment anyone wants compaction on that backend.

Concretely: the compaction path is gated on the backend exposing a context limit, and Ollama is the only backend that holds one as a number (`ollama.rs:177`, resolved from `/api/show`). Anthropic and Bedrock only discover the limit reactively by string-matching a 400 (`anthropic.rs:533`, `bedrock.rs:519`), so "no advertised limit means no compaction" falls out of the design rather than needing a special case. A run on those backends behaves exactly as it does today.

The **output cap** is not scoped this way and still covers all three backends, because it is a correctness matter rather than an optimization: the Anthropic API requires `max_tokens` on every request and rejects a value above the model's published ceiling, so removing the shared constant without giving Anthropic and Bedrock a per-model value would break them outright. A small per-model table is enough there.

## Measured (2026-09-19 evening): it is correct, and the telemetry is blind to the harm it was built to find

First real runs, `glm-5.3-flash` on Ollama Cloud, window pinned at 131,072 in every arm so the derived output cap is identical and only the trigger moves.

### What works, and is now evidenced rather than argued

A 120-iteration tier-2 run at a forced 4% threshold produced **109 compactions**, every one reaching tier 2. Across all 109: **zero orphaned tool calls and zero orphaned tool results**, and the message count and block count were unchanged on every single one, so the transcript replay invariant held throughout. That is the strongest available evidence that the adjacency pairing is correct, since id-keyed pairing is precisely what would have mangled this case. 109 tool results were elided and 30,157 tokens reclaimed.

Tier 2 fires at all only because of the call-id fix. Before it, elision was dead code on Ollama.

### The re-derivation risk did not materialise

The worry was that dropping reasoning would make the model re-derive its plan. Measured on the same run: reasoning averaged 590 characters per turn before any compaction and 489 after, a ratio of **0.83**, with 56 of 93 post-compaction turns producing no reasoning at all. It went down, not up.

### But the run got materially worse, and nothing in the telemetry said so

Paired against a control on the same task at the same window with compaction disabled:

| Arm | resolved | iterations | compactions |
|---|---|---|---|
| OFF | 1/1 | 60 | 0 |
| ON, forced 4% | 0/1 | 120 (cap) | 109 |

The control solved the task in 60 iterations. The forced arm never converged. **Every disorientation counter read healthy while that happened**: zero elided re-reads, zero repeated tool calls, reasoning down rather than up.

The mechanism is legible in the tool mix. The forced arm made **one** `edit_file` call across 120 iterations against 112 `bash` calls, and 103 of the 109 elisions were bash results. On a debugging task the accumulated bash output *is* the agent's evidence; eliding it ten assistant messages later destroys its working memory of what it has already established. It then probes *differently* rather than repeating itself — which is why `compaction_repeated_calls` stayed at zero. The counters detect **repetition**; the harm was a **worse path**, and those are not the same thing.

Two consequences, both load-bearing.

**The exclusion list is too narrow.** Only the most recent `run_checks` result is protected. But agents routinely run the gate and their tests through `bash` — that is exactly what caused the finish-recovery disarm found the same evening — so the single most load-bearing evidence in a run is often sitting in a `bash` result with no protection at all.

**In-run telemetry cannot substitute for a paired arm.** Compaction's cost is an outcome-level property (did it converge, in how many iterations) and is only visible against a control. A production run has no control, so self-reported counters will always read healthy. Any future claim that compaction is harmless must come from an A/B, not from the counters.

### Scope of these claims

One trial per arm on one task, and tier-2 deltas under about three trials are noise (`kb-03240`). The forced 4% threshold is also pathological: it compacts from iteration 12 onward on essentially every turn. Production ships at 90%, and the replay over 54 fleet runs and 3,061 turn transitions says that has never once been reachable — peak fill ever observed is 80.1%. So the shipped configuration is inert, and what is measured here is the stress case, not the default.

## Out of scope

Not built here, and named so nobody re-derives them. Cross-window handoff, which our research notes are emphatic should not lean on compaction — durable artifacts (a progress file, descriptive commits, a checklist with pass flags) are the mechanism, and ralph's fresh-context restart is the pattern-level answer for long objectives. The memory tool. Server-side compaction on the Anthropic backend, which exists as a beta primitive and would give one backend a different shape from the other two. Streaming, which is a separate roadmap item; all three backends are non-streaming today and Anthropic and Ollama set no HTTP timeout at all (`anthropic.rs:84`, `ollama.rs:190`), so large non-streaming responses are slow rather than broken.

The roadmap's current entry says in-run compaction is "explicitly not planned", on the reasoning that a model's real window is huge and the right lever is decomposing work. That reasoning was sound and is now partly overtaken: the qwen lane runs at 80% of its window, and removing the output cap increases history growth. The roadmap entry is updated alongside this record.

## Open questions

Whether the trigger threshold should be per-lane. The windows differ by 4x (262,144 against 1,048,576) and so does the composition — 57-62% reasoning on the glm runs against 33-43% on qwen. The reasoning-retention window is less open than it looks, since its cost curve is flat from 2 to 10 turns on the only long run we have.

What rolling history edits do to the prompt cache. Every compaction rewrites a prefix that was previously cacheable, and caching is where the dispatch budget is won: `kb-03356` established that Ollama Cloud caches as a prefix trie in 64-token blocks, and `7b2c6ea` made cache reads visible per call. So the cost is now measurable rather than theoretical, and it should be measured on the first real compacting run rather than modelled.

Whether compacting earlier than necessary is a quality win rather than a cost. The Chroma context-rot study cited in our research notes found retrieval reliability degrades monotonically with input length, which would argue for a threshold well below the window rather than just short of it. That is measurable with the telemetry above and should not be guessed.
