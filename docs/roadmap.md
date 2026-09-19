# Roadmap

What's next and why, in priority order. Milestones are **capability themes**, not
date promises — a release cuts when its theme's capability is real and measurable
(the v0.1.0 rule: the first CHANGELOG line of a release should claim a capability,
not an engineering milestone). Numbers beyond the next milestone are provisional;
we re-order when we learn something.

Living document: updated at session boundaries. The per-session narrative lives in
[`session-summaries.md`](./session-summaries.md); decisions of record live in the
KB (`project_ref: harness-design`).

## Where we are — v0.10.0 (2026-09-13)

Talos now has **three model backends** — Anthropic, Ollama, and **AWS Bedrock** (`886aa2f`) — so it runs where the Anthropic API isn't reachable (e.g. a work machine). The Bedrock backend drives `aws-sdk-bedrockruntime`'s non-streaming Converse API over the standard AWS credential chain (no keys in code), gated by `TALOS_BEDROCK` (a non-empty value wins over any Anthropic/Ollama env), restricted to haiku-4-5/sonnet-5/opus-4-8. It is **live-verified** against real Bedrock Haiku (`gritmile-bedrock-test`). Also this release: the canonical Sonnet moved to **Sonnet 5** (`ddba2d3`).

The Bedrock dispatch was the **hardest task the project has attempted**, and it proved the core thesis. The routing question was explicitly *"is the talos harness up for it?"* — not a model-capability question, because **GLM-5.2 (62.1% SWE-bench Pro) ≈ Sonnet 5 (63.2%)** (see the CLAUDE.md reference table), with the fallback being **claude-code-glm** (same model, stronger harness), not a bigger model. **talos-glm cleared it clean** on the cheap Ollama lane (zero Anthropic spend): 293 iters, zero inline fixes, and it correctly disarmed the load-bearing licensing landmine (the AWS SDK defaults to `aws-lc-sys`/OpenSSL, against our rustls-ring posture — pinned `default-features=false` + a hand-built ring-rustls `HttpClient`, so `cargo deny` stays green with `deny.toml` untouched). The harness wrote its own AWS provider — the harness-is-the-variable thesis (`kb-03109`) validated end-to-end.

The **Ralph Loop** shipped and matured across 0.7.0–0.8.1: `talos ralph` (the CLI), and the **do-over fix** (`242a57f`, 0.8.1) that made "ralph only ever commits green" a real invariant (revert-to-green on any non-green outcome, `DoOversExhausted` after `--max-do-overs` consecutive). It's dogfood-proven: `talos ralph` on **talos-qwen** drove the external **dng-converter** repo from **6% → 69% coverage** — and that run both surfaced the do-over bug and, post-fix, delivered. The **GTD build-engine adapter** — the milestone this project was built toward — is shipped and maturing; talos routinely lands merged changes to its own repo on the cheap glm lane, including this release's Bedrock backend and the do-over fix.

Session 14 (2026-08-12, no release) put the eval suite to work as a **routing gate**: NVIDIA's launch-day nemotron-3.5-lightning scored 25/30 with the project's **first-ever false-dones** (sealed holdout caught what the visible gate scored green) and earned **no lane** (`kb-03195`, `kb-03196`) — and exposed the suite's limit: pass-rate-saturated five models deep. Session 15 (2026-08-14, no release) **built the tier-2 discriminating pipeline end-to-end**: research track 08 (SWE-bench anatomy, METR time horizon, Harness-Bench), the ratified **spec-level ladder** design (`docs/design/04`, `kb-03198` — a mined commit is a task family: S3 full-spec floor / S2 AC-only / S1 intent-only / wave composites, so discrimination comes from withholding the grooming), the new **talos-evals repo** with **8 hand-mined dual-gate-verified tasks** (probe `kb-03197`), and the **mined_eval runner** (`4d67846` — checks=None agent loop, out-of-band sealed test-id scoring, claimed-Done×Unresolved cross-tab), **smoke-proven live** (qwen @ S2 → Resolved, and the finish-discipline claim-gap already visible on trial 1).

Session 16 (2026-08-16/17, no release) ran the **first real tier-2 matrix** and spent it measuring the instrument instead of the models (`kb-03205`). The qwen row landed — **13/18 valid resolved, zero false-dones**, with the ladder genuinely discriminating (two tasks at 0/3 and 0/2 against four at 3/3; tier-1 never separated like this) — but only after the first run was voided by a `num_ctx` of 32768 that the KB had already superseded with 262144. Three fixes shipped (`74a506e`, `b556f6b`, `220406a`): the `BackendError` payload now reaches the trial line, pytest's full-width summary banner parses correctly (it had been silently disabling its own truncation cross-check on single-bucket summaries), **`num_ctx` resolves from the model's advertised context length via `/api/show`** and fails loudly when it can't, and **finish-discipline telemetry** surfaces seven `RunStats` fields per trial. That telemetry immediately refuted the lead's own hypothesis: finish-recovery isn't disarmed by `bash` noise, it is **structurally dead in tier-2** — `standard_registry(None)` means `run_checks` is never registered, so `last_gate_green` can never be true and both nudge guards are permanently false, while production talos *does* register it from `gate_command` (`kb-03201`). Tier-2 currently benchmarks a harness configuration we never ship, so its finish-discipline numbers are **not** production-relevant. Also learned: **k=3 is noisy** (13/18 vs 11/18 on identical config) — treat tier-2 deltas under ~3 trials as noise.

**Session 17 eval work (2026-08-20/21):** the tier-2 scorer fix landed (`e684080`: pytest parsing scoped to the short-summary section, and a sealed-test `ImportError` on a symbol the agent never wrote now scores `Unresolved`, not `Invalid`), and the production task prompt now **requires a failing test first** and no longer says "finish as soon as the gate passes" (`be85c75`) — which read literally told a model to finish at iteration 1 on a healthy repo. The same guidance reaches tier-2 (`5342e9a`), tier-2 measures agent test authorship per trial (`efdb1d9`, baseline 0/24 on qwen before the prompt change), and `CODING_EVAL_TEST_FIRST=0` gives tier-1 a clean A/B arm (`eee2d3f`) — **the A/B has not been run yet.**

**GLM lane cutover (2026-09-12/13):** `talos-glm` moved from glm-5.2 to **glm-5.3:cloud**, a new **`talos-glm-flash`** lane runs **glm-5.3-flash:cloud**, and `talos-sonnet` now runs **claude-sonnet-5** (it had been pinned to 4.6). Both GLM lanes pin `OLLAMA_THINK=high` — glm-5.3 defaults to max, which cost 2.3× the tokens for nothing, and flash@low produced a false done. The models are literals in the dispatch overlay, never config. Talos needed no code change; the wiring is agent-gtd-dispatch 1.21.0. Qualification (`kb-03220`): on tier-2 both 5.3 models resolve 13/24 vs 5.2's 9/24 with zero false dones, and **flash matches full 5.3 with far better finish discipline at ~half the wall and ~1/10 the price** — the candidate default for the cheap lane once dispatch data confirms it. `claude-code-glm` follows to glm-5.3 so it stays talos-glm's same-model twin. Neither model is on the SWE-bench Pro leaderboard yet.

**Tier-2 now measures the harness we ship (2026-09-13, `a0c0128` + `eb333c1`, `kb-03240`).** Every mined task carries an `agent_gate_command` — the repo's real dispatch gate (whole suite + lint/typecheck) — which tier-2 registers as `run_checks`, arming finish-recovery and claim-verifying `finish(done)` exactly like `talos run`; the sealed, file-scoped gate stays the judge. A post-run gate verdict separates *shippable* (resolved and a verified Done) from resolved-but-red. First matrix: **the iteration cap dominates** — at cap 24 models resolve ~50%, at the production cap (500) flash resolved 8/8 (re-scored) with every trial claiming a verified Done in 13-43 iterations; **finish-recovery never fired** (0 nudges in 112 trials — stalled trials sit on a red gate, not a green one), so the finish-discipline fork is closed without an engine change; the gate doubled glm-5.3's correct Done claims; **test-first** is neutral-to-positive at the production cap (the only real false done was in the off arm) and stays. Two task defects fixed (statement heading collision + a whole-suite ban; from-json-contract's unfair `unknown key` wording test, which explains most of that task's historical 0/N).

**First representative tier-2 rows (2026-09-14, cap 500, k=3, `kb-03252`).** Tier-2 history before this point is the record of how we got here, not a baseline. Re-scored for two unfair wording assertions (now pinned): glm-5.3 21/24 (3 real false dones), flash 21/24 (2), **qwen3.8:27b @128K 21/24 (2) — local and free**, qwen3.6:35b 19/24 (5). At the production cap nearly every trial ends in a verified Done, so the discriminating metric is the real false-done rate (gate-green work that fails hidden acceptance tests).

**Overnight 2026-09-15 (`kb-03264`).** Re-run with wording pinned: glm-5.3 24/24 (0 false dones), flash 22/24 (2), qwen3.8:27b at 256K 21/24 (3). Pooled over both cap-500 runs the three are one band (45 / 43 / 42 of 48). `talos-qwen` now points at qwen3.8:27b at 256K (agent-gtd-dispatch `952f2dd`, not yet deployed); the host Ollama runs an 8-bit KV cache with flash attention, and 256K fits the 5090 only with ComfyUI and race-photos stopped. Shipped: finish-disposition fix (`4c8b637`: off-enum dispositions are rejected back to the model, not coerced to Failed) and opt-in run transcripts (`d4651b6`: talos `--transcript`, `CODING_EVAL_TRANSCRIPTS`, `MINED_EVAL_TRANSCRIPTS`). The first transcript read found the dominant false-done shape is a **self-consistent misconception** (the agent's own test shares a wrong IPTC field list), which is a context failure; design doc 05 is revised accordingly.

**A′/B measured — negative (2026-09-16, `kb-03283`).** Both knobs are merged and DEFAULT OFF. On the dominant false-done shape at k=10, the criterion-coverage prompt changes nothing (probe rate 0/10 in both arms; resolution 2/10 vs 3/10 = noise), and the acceptance audit never edited anything on the trials that looked better. The tier-1 perfect-spec guard was clean. Neither default flips.

**Both knobs removed, and talos now cleans up after itself (2026-09-17, `kb-03293`).** The measured-negative knobs were reverted as dead code (`cbddb48`, ~3100 lines), keeping only an explicit test that pins the anti-loop "do not re-verify individual acceptance criteria" line. Asking where dispatch transcripts should live then exposed that **talos had no retention policy at all** — run artifacts had accumulated on the dispatch hosts since July (14 MB on pironman01, 50 MB on r7-research) — and that `mined_eval` read its state root from env deep inside the library, so unit tests wrote into the real `$HOME` (71 leaked dirs on pironman01, written by this project's own `gate_command` running nextest there). Four items shipped: state root injected as a value (`7b04ecb`), `talos run` prunes its own state dir at start (`caf7549` — `--state-retention-days` > `TALOS_STATE_RETENTION_DAYS` > 30 days, `0` disables with zero I/O, `coding-eval`/`mined-eval` subtrees descended one level since a directory's mtime refreshes when a child is added), bare `--transcript` defaulting to `<state-dir>/transcript.jsonl` (`33fee62`), and every talos dispatch passing it (agent-gtd-dispatch `28c5e6e`). Fleet on `0.10.0-19-g33fee62`. Retention lives in the binary, not a cron entry, because talos is self-hosting: it reaches the fleet through `talos-update.sh` and so cannot drift from the code that creates the mess. The groom workflow overturned two lead dispositions with code evidence — a prune line on stderr would have become the last stderr line of an exit-1 `BackendError` run, which is what the dispatch worker parses to classify a failure; and `run_ralph_cmd` only `create_dir_all`s the *offload child*, so the `talos-ralph` state dir's mtime freezes at the first-ever ralph run and would have been deleted out from under a live loop.

**Leg 3 and answer mode (2026-09-18, `kb-03311`).** A verified Done now requires an **observed tree change** (`1ec37b2`): `handle_finish_call` observes the tree at run start and at the claim and rejects an unchanged-tree `done` back to the model like a red gate — baseline-relative, because the dispatch worker stages an untracked attachments dir in the clone before the agent starts, so an absolute dirtiness test would have no-opped the check exactly where a bad push costs something. `AlreadySatisfied` is a peer disposition (reason required, checks green, exit 30); ralph counts it as a do-over. Unobservable fails open and is recorded. The principle is `kb-03300` (three legs: asserted, verified, work happened; none agent-disableable) and the agent_gtd session's claude-code contract adopted the same vocabulary. **Tier-1 trial workspaces are now git repos** (`2612751`) — they had been plain directory copies purely because committed fixtures can't carry a nested `.git`, which made the perfect-spec guard blind to the very mechanism above; guard result 30/30, 0 false dones, precondition live on every trial and never fired against real work (new baseline; earlier tier-1 rows aren't comparable). **Answer mode shipped** (`docs/design/06`; `bce155b` + `d349a61`): `talos run --mode answer --schema` returns a schema-validated JSON `result` (exit 40) instead of a diff, with the leg-3 primitive *inverted* to enforce read-only (an answer is accepted only if the tree is unchanged). Live smoke on haiku: 3 iterations, valid result, tree clean. This is talos as the sub-agent of a dynamic workflow; the Python client and the first ported workflow are next.

**Fleet published (2026-09-18, `0.10.0-27-g53b56fb` on pironman01, r7-research, r7-server):** both gates verified directly — dispatch 1.25.2 carries the exit-30/40 mapping and the guard is clean (`kb-03312`: tier-1 30/30, tier-2 23/24, precondition never fired against real work). Retention, dispatch transcripts, leg 3 and answer mode are live on the hosts but untriggered until the first real talos dispatch.

**First real work on the cheap lanes (2026-09-18/19, `kb-03318`).** The global "don't dip into Ollama credits" rule was reversed once the numbers were in front of us: Ollama Max is $300/month of non-rolling credits and we had spent $47.57 with three weeks to reset (`kb-03315`). Four backlog items were groomed in one run and dispatched as the lanes' first-ever real work — **talos-qwen 2/2 clean** (`3584df9` shared `num_ctx` resolver, 17.5 min; `d76792d` backend settings on the run record, 34.8 min) and **talos-glm-flash 2/2 clean** (`3bdc845` tier-2 tripwire never silently off, 5.0 min; `055d187` fail-safe docs-only skip for the heavy lefthook gates, 15.5 min) — zero lead fixes, identity verified from every transcript. The first run also read back everything shipped that week: retention removed 131 dirs on r7-research (50 MB → 11 MB) and 54 on pironman01, every run wrote its transcript, and every terminal record carried `change: TreeChanged`. The run record now carries think level, `num_ctx` and its source, so a row is self-describing. Docs-only commits now skip clippy/nextest/coverage/doctest/machete/deny via `scripts/docs-only.sh`, proof-based and fail-safe.

**talos-flow client shipped (2026-09-19).** Jason created the `talos_flow` repo; it was bootstrapped the same morning (uv, checkers-only hooks, `mypy --strict`, coverage 90) with a GTD project and gate, and both library items were groomed and built on Ollama Cloud lanes: `agent()` by talos-glm in 4.25 min (`1c6ace7`) and `parallel()`/`pipeline()` by talos-glm-flash in 4.75 min — 74 tests, 100% coverage, zero lead fixes. Lead-verified live through the real binary: a single `agent()` call and a two-way `parallel()` on one shared read-only checkout both returned correct schema-valid results with the tree clean, which is the first answer to design 06's shared-checkout question. Both dispatch waiters were hosted in `Monitor` and survived (2/2), retiring the foreground-wait workaround for dispatch.

**The groom workflow runs on talos, and the comparison is in (2026-09-19, afternoon; `kb-03342`).** groom-to-ready was ported to talos-flow (`312959b`) with the JS vendored as the in-repo source of truth after the first build attempt died citing a lead-machine path (`kb-03332`, and a new steering rule: a fleet spec may only cite what the clone will contain). The first real runs found a harness bug the haiku smoke test could not: Ollama Cloud flattens object tool parameters to JSON text, so `finish(answer)` never validated on glm-5.3 — fixed by parsing stringified results (`b034a63`, `kb-03340`; the rejection loops cost ~19M glm-5.3 tokens first, and a follow-up caps consecutive schema rejections, `a3612b34`). Then four groom arms on identical prompt bytes over the same three backlog items: the Opus/Fable Workflow wrote the deepest specs and alone found that `d76792d` had silently reverted the CLI num_ctx resolver (restored, `1f56186`, with `docs/design/07-num-ctx.md`); talos on glm-5.3 ($15.00) did empirical git verification inside its critic stage; flash-draft/glm-critic/flash-synth ($8.50) made the best design call on the Truncated terminal; all-flash ($1.31) produced a spec about the wrong item that its own critics passed. One arm spec per item was dispatched on talos-glm: the ralph commit-retry (`d6e93e1`), the num_ctx restore (`1f56186`) and the Truncated terminal (`5211a15`, `FailureMode::Truncated` + `--max-tokens`) all one-shot with zero lead fixes. Cost accounting (`cost.py`/`recost.py`) was itself groomed by the port and built by flash. Talos now also retries a hook-rejected ralph commit when the hook mutated the tree.

**Next up.** (1) Cost lever: find out whether Ollama Cloud prompt-caches glm and whether talos can see it; reconcile the computed USD against the Ollama usage page. (3) Adopt flash-draft / glm-critic / flash-synth as the talos groom default and keep scoring it against the Opus baseline on every groom; all-flash is not safe without a stronger critic. (4) `a3612b34` — cap consecutive `finish(answer)` schema rejections. (5) Tally spend per lane at the first Ollama reset. (6) Read production transcripts on real dispatch work; the PhotoQueue StoppedWithoutFinish transcript study already on the board is the first such example. Queued: `62d9f9b5`, `b4b0aef6`, `c8f3accf`, `6b13ac75`, `4b0094aa`, vitest parser, the `tasks.md` backlog-executor, ralph dispatch mode, Bedrock follow-ups, the four num_ctx residue items named in design 07.

## 0.3.0 — durability (persist, resume, dispose) — ✅ shipped v0.3.0

**Theme: survive the host.** The run record and `RunStore` shipped in v0.1.0 but
the loop doesn't use them yet. Wire checkpointing into the loop, unify the
loop-local `FinishDisposition` with `run_record::Disposition`, and implement the
two resume modes from the design (crash-resume; fresh-context restart). The
capability claim: *kill the harness mid-run, restart it, and the run completes* —
the deployment-agnostic promise (Pi, container, spot instance) made real.

## 0.3.5 — first dogfood (the harness builds the harness) — ✅ shipped v0.3.5

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

## 0.4.0 — bounded autonomy (finish-recovery, wall-clock budget, retry) — ✅ shipped v0.4.0

**Theme: safe to leave alone.** Design of record: [`docs/design/03-bounded-autonomy.md`](./design/03-bounded-autonomy.md).
Three items (serialized on `engine.rs`): (1) a **finish-recovery protocol** — detect
a done-but-unclaimed spin (green gates + static tree for K iters), nudge to finish or
report a one-sentence status, and on exhaustion terminate `Failed` while writing
**recovery facts** so the worker preserves the WIP branch — the harness never
fabricates `Done`, the claim moves up to the lead; (2) a **wall-clock budget** so the
harness self-terminates gracefully in the margin before the dispatch worker's hard-kill
(needs a new injectable `Clock` seam; per-process semantics); (3) **retry/backoff** on
`Transient` errors (`is_retryable` has been waiting). The capability claim: *a
pathological run terminates itself with a useful `Failed` disposition — and doesn't
throw away work it couldn't claim.*

Budgets were scoped to **wall-clock only**: token caps are inscrutable (no human-legible
right value) and cost caps have no accumulator yet (see backlog). Being designed against
real talos run data — including this wave's own finish-discipline failures.

## The GTD build-engine adapter — ✅ shipped (0.3.5 supervised → matured through 0.5.x)

**Theme: the point.** The adapter that picks up a groomed Agent GTD item, clones
the target repo, runs the loop with the project's `gate_command`, pushes a
feature branch, and comments back — the harness as a real headless-dispatch build
engine alongside Claude Code. Delivered incrementally rather than as one milestone
release: **0.3.5** shipped the supervised version (the `talos-*` engine family, the
no-MCP TaskSpec contract, the worker owning git + comment-back); subsequent releases
matured it into the unsupervised engine — finish-recovery and wall-clock bounds
(0.4.0), the stop-cold nudge (0.5.0), workspace/multi-repo dispatch and the bash tool
(0.5.1), and the ongoing field-report worker fixes on agent-gtd-dev. Talos now
routinely lands merged changes on the cheap glm lane; the remaining work is hardening
and ergonomics, not the core contract.

## Backlog (unscheduled, captured)

- **Dispatch-scale fixture tier** — to reproduce the *pass-rate* harness gap in-eval. Session 9 shipped the harness-vs-model benchmark and two "hard" fixtures (tokenbucket withheld-test + eventbus multi-file), but glm saturates them under *both* harnesses (`kb-03078`): the gap lives only at genuine dispatch scale (the 18-AC/5-file item), and a withheld-test spec precise enough to grade unambiguously is also easy to implement (precision-to-grade removes the difficulty). Reproducing the gap needs many-file, high-navigation fixtures — a real authoring effort, and the design challenge is difficulty-without-ambiguity.
- **The cost-gap finding** — Talos vs Claude Code token efficiency at equal quality: **~17× on glm** (`kb-03078`, uncached ollama endpoint) and **~8× raw / ~7.6× billed on sonnet** (`kb-03102`, both harnesses caching the real Anthropic API). A strong, cheap-to-tell result worth a blog writeup — the sharpened story is "harness overhead is real *and survives caching*, but the headline multiple is iteration-sensitive."
- **Unify runner fixture discovery** — `coding_eval` discovers all 10 fixture dirs (the 4 legacy ones without `task.json` run but without holdout, shown `-`) while `claude_code_eval` runs only the 6 with `task.json`. Either give the 4 legacy fixtures `task.json` + holdout or exclude them from `coding_eval` so the two runners cover the same set.
- **Token + cost budget caps** — deferred from 0.4.0 (which shipped wall-clock only).
  Token caps are inscrutable (no human-legible right value per task); cost caps are
  blocked on a token→price table that doesn't exist (`consumed.cost_micros` is never
  incremented). Revisit token caps only with a concrete reason; cost caps once pricing
  is wired.
- **Streaming/SSE** — cost/latency, not capability; when the live-run volume
  justifies it. (Prompt caching shipped in v0.6.0 — `98fe789`, `kb-03102`.)
- **In-run context compaction** — summarize/evict old turns as a single run
  approaches its context window, the way Claude Code auto-compacts. Talos does
  none today: it grows the conversation until the window is hit, then either
  errors (pre-flight guard, once `num_ctx` is pinned) or — the bug we just
  fixed — silently truncates. **Explicitly not planned yet.** For dispatch-size
  work a model's real window is huge (glm 1M, qwen 256k, now pinned), and the
  right lever *before* compaction is to decompose work into smaller tasks;
  Ralph's fresh-context-per-iteration is the pattern-level answer for long
  *objectives*. Revisit only if a single indivisible task genuinely overruns a
  1M window.
- **Ralph Loop — ✅ shipped (core `1b4c2bb` / 0.7.0, CLI `talos ralph` / 0.8.0, do-over fix `242a57f`), now growing.** Real and dogfood-proven (see "Where we are" for the dng-converter run). Forward directions:
  - **Ralph-ability characterization** (`kb-03109`) — the design heuristic for *which* tasks fit: a **static prompt that re-binds as external state mutates** (coverage %, a checklist, a failing-test list, a grep). Five requirements (monotone external state · pure-function-of-state prompt · cheap unambiguous stop-oracle · progress durable outside the context · units that fit one inner budget). "Write the highest-value missing test" is the canonical small-model case.
  - **`tasks.md` backlog executor (next experiment)** — the mid-tier (talos-glm) instance of the pattern: prompt = "complete the next unchecked task in `tasks.md`, mark it complete," stop-when = "all boxes checked." Turns Ralph into a generic autonomous project executor over a groomed backlog. Gated on the do-over fix (item `230f9e9b`) — that fix is the *enabling prerequisite*: without it one over-budget task corrupts the run; with it, a too-hard task gets 3 clean do-overs then loudly stops for a human. A `tasks.md`-specific v2: "mark blocked + skip to next" instead of halting the whole loop.
  - **Ralph dispatch mode** — `talos ralph` is local-only today; the always-planned next step is running a Ralph objective as a headless dispatch on a *remote* host, so the fleet (not a laptop) grinds an overnight objective. This is what keeps GTD relevant alongside `tasks.md`+Ralph: dispatch-to-remote is a must-have, and the task board is for human organization/visibility — Ralph and GTD-dispatch are complementary (GTD dispatches each item as a separate reviewed agent; Ralph grinds a whole objective in one self-restarting loop).
- **Remaining v1-design tools**: `search_code`, `comment` (the design's tools 7–8).
- **LLM-judge evidence tier** (`Evidence::Judge`) — deferred from v1 by design.
- **Model-routing policy** (open decision #6) — blocked on eval data (haiku
  floor-run, per-trial metrics).
- **OS-level sandboxing** — explicitly v2 (threat model: our own tasks on our own
  infra; blast-radius bounds + creds hygiene are the v1 answer).
