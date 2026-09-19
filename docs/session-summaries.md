# Session summaries

A running, chronological log of working sessions on harness-design — this is a
learning project, so we document the *process*, not just the output. Each entry
is a 3-4 sentence summary; the full per-session write-up (process, decisions,
handoff) lives in a dated `lesson_learned` KB entry under `project_ref:
harness-design`, linked here. Append newest at the bottom; when resuming, read
the latest entry and its linked KB entry first.

---

## Session 1 — 2026-06-20 → 06-23 — gates, research, design, foundation

Took the project from nothing to a built v1 foundation: established Rust quality
gates *before* any code (so headless agents can run safely), ran a multi-agent
research workflow into `docs/research/` plus a ~960-entry local KB braintrust
(`harness-design-research`), then locked the v1 design — the
workflow-around-open-loop shape, an 8-tool inventory (LSP deferred), and the
run-record schema with claim-vs-verify and the Done/Blocked/Failed disposition
(`docs/design/01` + `02`). Corrected two assumptions the research had baked in
(a permission/allowlist model we don't have; a Pi-specific deployment) before
grooming three GTD items and dispatching them as a headless wave (data model +
tool layer in parallel, then RunStore). All three merged clean to `main`
(`eb8a129`) — 42 tests, 97.9% coverage, zero quality misses; one build agent even
dodged a license landmine unsupervised. **Next:** the loop engine + model-backend
trait (held for an interactive session). Full write-up + handoff: **kb-02851**.

---

## Session 2 — 2026-06-23 → 06-24 — the model contract + first live loop

Designed the architecturally-live core interactively, then shipped it as an autonomous
dispatch wave. The load-bearing design call: draw the `ModelBackend` trait *high* — a
normalized `AssistantTurn` out, each adapter an anti-corruption layer owning all wire
translation, so the loop never sees provider-native shapes; role-precise content blocks
(`ContentBlock` vs `UserBlock`, `Message` an enum over role) make illegal message states
unrepresentable, and a four-variant `BackendError` classifies while the loop reacts.
Resolved open decision #2 — **direct tool-calling for v1**, code-mode deferred to re-enter
later as a registered sandboxed tool (not a rewrite), grounded in the research corpus
rather than priors (`kb-02852`) — and accepted `CDLA-Permissive-2.0` into `deny.toml` for
the rustls TLS stack (`kb-02854`). Built as a 4-item dependency wave (D model contract →
E1 Anthropic adapter ‖ E2 loop+finish → F eval harness); all merged clean to `main`, 100
tests, 98.83% coverage, with one inline `typos` fix the lead's merge-gate caught that the
E2 agent's gates missed. **The live eval is green: 5/5 pass^k against `claude-haiku-4-5`** —
the loop drives the real Anthropic API through a `finish` tool call end-to-end. Full
write-up + handoff: **kb-02855**.

---

## Session 3 — 2026-06-29 → 07-07 — the three seams, overnight waves, v0.1.0

Started from a prompt-engineering question (should we model prompts in BAML?) and ended
with a released build engine. Investigated BAML against its live repo — verdict: don't
adopt (single-shot structured-*output* layer, FFI-blob/sidecar footprint, and it replaces
exactly the layer this project exists to build) but steal its ideas: versioned templated
prompts, prompts-as-testable-fixtures, lenient parsing (parked for Ollama) (`kb-02870`).
Decided to release at a **product boundary** ("the harness autonomously completes a
coding task, verified mechanically"), not an engineering milestone. Designed three seams
interactively: the confined `Workspace` (path resolution owned in one tested place; a
clarifying question caught the offload-readback gap early), the **claim-vs-verify**
control flow (`finish(done)` is a claim; the harness runs the checks itself; rejection is
steering; `Done` carries evidence by construction), and the askama prompt layer
(compile-time-checked, versioned template files, load-bearing phrases pinned by tests).
Then three autonomous dispatch waves overnight — 5 parallel tools/prompt items, the
engine rework, the eval fixture with per-trial isolation — 7 dispatches, all clean, zero
quality misses. Morning boundary ritual: **3/3 pass^k** — haiku found and fixed the
planted bug and the harness verified `cargo test` green itself — then `./release.sh` cut
**v0.1.0** with the first CHANGELOG and pushed to GitHub. Full write-up + handoff:
**kb-02899**.

---

## Session 4 — 2026-07-07 → 07-08 — the second backend and the four-model matrix

Hardened the eval, then took the 0.2.0 Ollama milestone end-to-end in a day. Three
subtle-bug fixtures (boundary semantics, stateful omission, encoding panic) landed —
and sonnet swept them 12/12, the **saturation finding**: a red test does the
localization, so bug subtlety doesn't discriminate at frontier tier. Captured the
levers on the board, wrote `docs/roadmap.md` (capability-themed milestones), then
researched the current Ollama API against live docs: the real impedance seam is **no
tool-call IDs** (name+order matching), not the parse-from-text problem we'd predicted;
cloud and local are wire-identical, so **one adapter serves both**. One design
correction en route: the planned post-hoc truncation check would false-positive on
Ollama's KV-cache reuse — flipped to a pre-flight client-side guard. A two-item
parallel wave (OllamaBackend + per-trial RunStats/metrics, both clean) unlocked the
experiment: the same loop, byte-identical prompts, four models. **Sonnet 11/12 and
GLM-5.2 12/12 at a uniform 5.00 iterations; gpt-oss:20b (think=high) 9/12; qwen3.6:35b
7/12 — zero false dones in 36 verified trials.** Local models that completed ~nothing
under a big general harness complete 58–75% of small real tasks under this one —
directional evidence for the two-failure-modes thesis — and the opaque MaxIterations
trials became the live argument for 0.3.0 durability. 0.2.0's capability claim is met;
release ritual pending. Full write-up + handoff: **kb-02913**.

---

## Session 5 — 2026-07-08 — durability through the gate, and the gate earned it

Shipped 0.3.0 end-to-end in a day: groomed four serial items via the draft→critic
workflow (the critics caught the load-bearing seam at groom time — reconstructing an
interrupted tool call's id needs both a `call_id` on the event and a post-model-turn
checkpoint), rolled them out through a manage agent (4/4 clean per-item, 1h45m), and
ran the eval follow-ups in parallel: haiku's "floor"-run scored **11/12** (the floor is
the ceiling) and qwen3.6:35b **with think enabled** jumped 7→11/12 — the "20b > 35b"
reading was a think-config artifact, and think is now a first-class routing knob
(matrix v4: kb-02909). Decided **0.3.5 "first dogfood"** ahead of bounded autonomy:
claim-vs-verify + repo gates are enough safety for supervised dispatch, so the harness
starts building the harness after this release. Then the payoff moment:
**review-against-intent returned does-not-meet** on the "100% green" wave —
`reconcile_crash_tail` tripped over the loop's own `BudgetTick` and ignored the
`call_id` contract entirely, masked by a hand-seeded test log the engine never emits
and a single-call proof where wrong coincides with right. Remediation same session:
dispatched fix from the review findings, plus a lead catch at merge review (the entry
gate itself was still log-tail-shaped — now snapshot-shaped), doc sweep, test-strength
batch. main @ 99998ea, 299 tests, capability claim true including the two-call
discriminating case; release ritual pending. Full write-up + handoff: **kb-02931**.

---

## Session 6 — 2026-07-09/10 — the 0.3.5 epic: Talos becomes a build engine

Planned and shipped the whole first-dogfood milestone in one (overnight-extended)
session. Named the harness **Talos**; locked the no-MCP engine contract (worker
serializes the groomed item + `gate_command` into TaskSpec JSON on stdin;
disposition-mapped exit codes out, read from `LoopOutcome` so engine-broke never
collapses into task-failed; worker owns all git and comment-back — a verified
Done carries mechanical evidence, stronger than any agent self-report). Four
dispatched builds across three repos landed clean: TaskSpec + groomed-item
prompt (d9cb7cc), the `talos` CLI (0d69997 + a4c5b88 lead fix), the
`talos-{haiku,sonnet,opus,qwen,glm}` engine family in agent-gtd-dispatch
(deployed; all five advertising on both hosts), and a fully idempotent
`setup-dispatch-host.sh --with-talos` — whose groom included **read-only ssh
recon of the live hosts**, catching the two ship-blockers no repo-only groom
could see (talos absent from the sudoers NOPASSWD allowlist; `secure_path`
missing `.cargo/bin`). Verified live on both hosts: installer twice each
(second run byte-identical), sudo-boundary probes green, cold gate 22s (x86) /
115s (Pi 5) — inside the 300s ChecksRunner default. First patrol staged:
`gate_command` set, two talos-shaped items ready on `talos-haiku`.

Then, next morning with Jason watching, the patrols ran — and **the capability
claim came true**: two Talos-authored commits merged to this repo (f49dbac
`--version`, f9153cd `--file` tests, author `talos-haiku@agent-gtd-dispatch`).
The first attempt failed *instructively*: haiku finished and verified the work
by iteration 6, then spent six iterations re-verifying acceptance criteria one
at a time and hit `MaxIterations` one call short of `finish(done)` — a context
failure in our own prompt layer, diagnosed entirely from the 0.3.0 run record,
fixed with a finish-discipline line in the task template (pinned by test) plus
12→24 iteration headroom, redeployed fleet-wide via one installer re-run, and
verified Done ten minutes after failing. Patrol 2 passed first-shot and
surfaced the session's other keeper: `gate_command` (nextest-only) was weaker
than the repo's commit gate, so "verified Done" shipped lint debt — the
project's gate is now the full fmt+clippy+nextest chain, making Done mean
merge-ready. Full write-up + handoff: **kb-02956**.

---

## Session 7 — 2026-07-10 — eval hardening: the fixture ladder + sealed holdouts

The saturated eval got its teeth back. Two structural upgrades: new fixtures are
**TaskSpec-shaped** (fixture-root `task.json` routes through the production
`render_task_prompt_from_spec` path, so the eval finally measures the prompt
shape real dispatches use) and carry a **sealed `holdout/`** the agent under
eval never sees — after each trial the gate re-runs with holdout tests copied
in, making the false-done rate (self-gate green, holdout red) a first-class
metric. Four new fixtures form a graded ladder: csv-ledger (tier 2 cross-file
bugfix + distractor), walrus (tier 3 implement-to-spec), taskdeck (tier 4,
committed gate GREEN by design — the agent writes its own tests, holdout is the
truth), calc (tier 5, right-associative `^` across three coupled files). Groomed
via workflow (critics caught an exact-list test that would have reddened main on
the first fixture merge), shipped as one 5-item rollout in 32.5 min, manager
merged everything.

Then review-against-intent earned its keep a second time: **meets-with-gaps on
a 100%-green board** — csv-ledger had shipped with the answer key workspace-
visible (a `// BUG:` comment on the planted line, module docs naming the bug's
file, the distractor disclaiming itself). Build agents write code optimized for
review transparency, which is exactly backwards for adversarial eval content;
the ban is now explicit convention (kb-02965) and the spoilers are stripped
(329ad50). First data on the hardened suite (kb-02971): haiku 24/24, zero false
dones, but tier-4/5 cost ~2x the iterations and ~3x the tokens of the legacy
fixtures — discrimination lives on the cost axes until the local models weigh
in. Full write-up + handoff: **kb-02972**.

**Session 7, second sitting (same day):** The full five-model matrix landed:
glm-5.2 24/24 (most efficient — calc at 4.0 mean iterations vs haiku's 13.0),
qwen3.6:35b+think 24/24, haiku 24/24, sonnet 23/24, and gpt-oss:20b **11/24 —
the ladder's first kill**, with holdout cleanly separating capability failures
(calc/taskdeck 0/3 at the cap) from finish-discipline failures (csv-ledger:
bug fixed, holdout green, `finish` never called). Zero false dones in 120
trials; finish discipline is the universal residual failure and the prime
0.4.0 input. The copy-in-hardening patrol failed identically on talos-haiku
AND talos-sonnet (MaxIterations at ~2.3s/iteration in a 1,600-line file) —
two different models failing the same way means a harness gap, not a model
gap: the talos toolset has no search tool (captured as GTD c5836f1c); the
lead landed the item inline. A `mean_wall` column joined the eval summary.
Released as **v0.3.6** — a meaningfully more robust eval suite is the
boundary. Matrix rows: kb-02971/02973/02976; updated handoff: kb-02972.

**Session 7, third sitting:** Turned the talos-patrol post-mortem into two shipped
fixes and unblocked cheap autonomous dispatch. The copy-in patrol had failed twice
(both "haiku" and "sonnet") — but reading the run record showed grep was reachable
all along via `run_command`'s `sh -c` escape; the failure was our own task template
saying *"No search tool exists"* plus a program+args tool shape that fought the bash
training prior. Fix: reshape `run_command` → **`bash`** (single command string, honest
confinement doc) with an affirmative template (pinned test). Then the first-ever
talos-glm dispatch exposed a bigger latent bug: the dispatch-svc→agent sudo hop runs
`env_reset`, and the sudoers `env_keep` list omitted `TALOS_BACKEND`/`ANTHROPIC_MODEL`/
`OLLAMA_*` — so glm/qwen failed outright and **talos-sonnet/opus had been silently
running as haiku** (which retroactively corrected the "sonnet patrol" record). One
`env_keep` fix (live on both hosts, durable in the dispatch repo) unblocked all four
engines. Re-dispatched glm → **verified Done, 8 iterations, gate green, zero Anthropic
spend** — the harness can now build the harness on Ollama credits, protecting the
control-plane session budget. Released as **v0.3.7**. Root-cause KB: kb-02979 (sudoers),
kb-02980 (glm run), kb-02977 (corrected). Handoff: kb-02972.

---

## Session 8 — 2026-07-11 — 0.4.0 bounded-autonomy: design, groom, and the harness hits the gap it's building

Designed and groomed the **0.4.0 bounded-autonomy** wave, then tried to dogfood it
on Talos and learned exactly why 0.4.0 exists. The design (`docs/design/03`, commit
2178ce9) centers on a **finish-recovery protocol**: detect a done-but-unclaimed spin
(green gates + a static tree for K iters — high precision; red-gate spins fall to the
budget cap), nudge the model to finish or give a one-sentence status, and on N-nudge
exhaustion terminate `Failed` while writing **recovery facts** so the worker preserves
the WIP branch. The load-bearing resolution: the harness *never* fabricates `Done` —
the claim moves up to the lead, keeping claim-vs-verify inviolate while rescuing work
that would otherwise be discarded. Budgets were scoped to **wall-clock only** (Jason's
call: token caps are inscrutable, cost has no accumulator) — its point being graceful
self-termination in the margin *before* the dispatch worker's hard-kill. Plus bounded
deterministic retry/backoff on transient errors.

The `groom-to-ready` workflow earned its cost by **overturning two of my design-doc
leanings** (a new `FailureMode::FinishDiscipline` rather than reusing the
already-produced `Loop`; recovery facts on `RunRecord`, not the wired-nowhere
`DispositionReport`) and catching a **CI-invisible trap** — the nudge must be appended
to the existing user message, not pushed as a new one, or it 400s against Anthropic
while passing every MockBackend test. Three items landed ready (finish-recovery →
retry-backoff → wall-clock, serialized on `engine.rs`).

Then dispatch taught the real lesson. Rollouts can't pin a host (FR filed), so we went
direct on **talos-glm / r7-research** — and finish-recovery **failed twice**: first
`MaxIterations@24` (talos's default cap, calibrated for small dogfood items, was far
too low for a 5-file change — bumped 24→500, commit d59b1d9), then
**`StoppedWithoutFinish@56` with no WIP preserved**. That second failure *is* a live
demonstration of the two gaps 0.4.0 closes — finish-discipline and work-preservation:
Talos can't build finish-recovery because Talos doesn't *have* finish-recovery. A
harness failure, not a glm-capability one (glm is 24/24 on our hardened evals). Per
Jason's model-vs-harness frame (glm the stronger model, Claude Code the stronger
harness), we pivoted to the clean harness-isolating comparison: **claude-code-glm**
(glm held constant, harness swapped), filed for wiring on the dispatch board. The wave
is parked pending that. Full handoff: **kb-02996** (perf: kb-02987, kb-02995).

**Session 8, continued — the wave shipped, and the hypothesis proved out.** Once
claude-code-glm was wired, it **one-shot finish-recovery** (`758cf4a`, ~27 min,
verified Done) — the exact 18-AC/5-file item talos-glm had failed twice. Same model,
swap the harness, opposite outcome: **the gap was Talos-the-harness, not glm-the-model**
— an orchestration failure, not a capability one, the two-failure-mode lens confirmed
empirically. Ollama credits then ran out, so the last two items ran on claude-code-sonnet
(Claude Max): retry-backoff (`3ec0aea`) and wall-clock (`e1cd5b5`), both clean one-shots
that re-grounded correctly onto an `engine.rs` grown ~950 lines by the prior merges
(symbol-anchored dispatch notes, no re-groom). A side-quest shipped the **talos fleet-
publish** capability (`2d48267`, GTD 953fd927): version-stamped `talos --version` +
`scripts/publish-talos.sh` that builds both arches on the i9 and publishes to pi-04 —
build-once-and-pull, retiring the compile-on-every-host tax. `review-against-intent`
gated the release at **meets-with-gaps**: every load-bearing seam verified correct in
code (both recovery terminals build `RecoveryFacts` identically; three `RunConfig`
knob-sets coexist; retry can't defeat the wall-clock check; clock-only reads; schema v2;
talos exit codes) — the only gaps were the design doc over-claiming budget scope and a
stale doc comment, both fixed (`00dfa04`), two minors captured as follow-ups. Released as
**v0.4.0**. Updated handoff: **kb-02996** (perf: kb-03007 glm proof, kb-03009, kb-03010).

---

## Session 9 — 2026-07-11→13 — finish-discipline completed, and the harness-vs-model benchmark (the cost gap)

Resumed post-0.4.0 to find the fleet binaries stale, so wired `scripts/publish-talos.sh` into `release.sh` (a release ships an artifact by definition) and adopted a durable "stop hand-pinning versions — trust `cog bump --auto` + honest commit types" rule; cut **v0.4.1** to ship the current binary. Then ran the session's spine — the finish-discipline **name→measure→fix** loop: **eval matrix v5** (`kb-03019`, 6 models on 0.4.1 — five at 24/24, gpt-oss 10/24, 0 false-dones/144) revealed finish-recovery fired **zero** rescues because it only catches the green-*static* spin, not a *stop-cold* halt; a one-line instrument (`RunStats.gates_green_at_exit`, `bafb6ed`) then classified gpt-oss's stops as **~43% post-green** (`kb-03033`) — verified work abandoned; and the **stop-nudge extension** (`e6ef1c0`) closed it, finish-recovery now nudging at the `StoppedWithoutFinish` terminal too (the groom critic caught that a wrong `last_mut()` append there *silently drops* the nudge — fresh `Message::User` required). Shipped **blog post 4, "Nobody Calls Finish"** (draft), extending the Bounded-Choice Cascade to the completion boundary. Then built and ran the **harness-vs-model benchmark**: a `claude_code_eval` runner driving claude-code-glm over the same fixtures scored by a shared external holdout (`c2052a1`), plus two hard fixtures — tokenbucket + eventbus (`7690dd4`, authored by claude-code-glm, the first clean dispatch under the Anthropic usage crunch). The result (`kb-03078`) flipped the expected story: **no pass-rate gap** — both harnesses saturate (18/18, holdout 18/18, 0 false-dones), the new fixtures still too easy for glm and the gap living only at dispatch scale — but a large **cost gap**: Talos is **~17× more token-efficient** than Claude Code at identical quality. "The harness is the variable" holds — on cost here, not pass rate. Released as **v0.5.0** (which consumed the roadmap's old GTD-adapter slot; the adapter moves to 0.6.0). Full handoff: **kb-03079** (perf: kb-03016, kb-03042, kb-03076, kb-03077).

---

## Session 10 — 2026-07-13→14 — the benchmark verdict, 0.5.1, talos-glm proves judgment, and a context-failure lesson

Ran the harness-vs-model benchmark for real (talos-glm vs claude-code-glm, same fixtures, shared sealed holdout): both **saturate** (18/18, holdout 18/18, 0 false-dones) — no pass-rate gap at this fixture scale — but **Talos is ~17× more token-efficient** at identical quality (`kb-03078`). When Jason challenged whether that was a thinking-mode artifact, a direct ollama.com probe settled it: GLM-5.2 **defaults to thinking on both paths** (Anthropic `/v1/messages` and native `/api/chat`), so both ran the same effort — the ~17× is **pure input** (Claude Code re-sending its large prompt + tool schemas each turn, uncached on the ollama endpoint), not reasoning. So talos-glm is the cheaper default lane. Cut **v0.5.1**: fixed the talos version-stamp to `git describe --tags` (no more meaningless `0.1.0-g<sha>` — the crate version is decorative since we don't publish; the tag is the source of truth) and ratcheted coverage back to **98%** (`lefthook`+`ci` 95→98). That coverage work was itself the session's best dogfood: an **open-ended judgment brief** to talos-glm (decide the areas, no coverage-theater) against a coverage-98 gate — it wrote 10 meaningful tests (real error-mapping/redaction/builder assertions), hit 98.30% cleanly, no gaming, no dodges (`kb-03096`) — real judgment on the cheap lane. Enabled **workspace (multi-repo) dispatch for talos** (the guard was the only blocker, not the search tool — bash removed that; talos runs the gate via `/bin/sh -c`, so compound `cd A && x && cd ../B && y` gates work), which means talos-glm can now fix its own dispatch worker. Triaged a real talos-glm cleanr **field report** (`6555f62a`) into three **agent-gtd-dispatch worker** fixes on the agent-gtd-dev board — correcting the report's misattribution of the priority commit-after-hook bug to talos (the worker owns `git commit`), and flagging the bootstrap hazard that fixing that bug *through* the buggy worker can trip it. And the durable lesson: I twice asserted **externally-owned mutable state** (a gate_command, a spike's status) from a stale in-session snapshot — context failures, not capability — fixed with one narrow top-line rule in the global CLAUDE.md (*refetch external mutable state at point of use*), which Jason is monitoring. Full handoff: **kb-03097**.

---

## Session 11 — 2026-07-14 — checker-only gates, prompt caching shipped, the Sonnet benchmark (~8×)

A hook-vs-gate design question became a decision, then unblocked an honest Sonnet benchmark. First the decision (`kb-03099`): while triaging an in-flight worker bug (a *fixer* pre-commit hook mutates files and fails the single `git commit`, losing completed work), the principle fell out — **gates and commit hooks CHECK, agents FIX; one checker set run at both surfaces** (the git hook is Claude Code's enforcement surface *and* retry trigger; `gate_command` is talos's). The impedance was smaller than it looked: a *checker* hook still fails the commit, so CC's retry trigger survives — only the silent mutation that breaks talos's commit-once worker is removed, and caching is symmetric so nobody loses their edge. Captured, noted in the shared `headless-dispatch.md`, and filed a hook-unification follow-up on agent-gtd-dev. Then the benchmark spine: Jason asked to compare **talos-sonnet vs claude-code-sonnet with caching on**, and it turned out talos had **no request-side prompt caching at all** (only response-side usage parsing) — so we shipped it (`98fe789`, a grounded groom→glm-dispatch→merge): a static `cache_control` breakpoint on the system block (covers tools via Anthropic's tools→system→messages order) and a rolling one on the last message. Two eval-infra gaps followed (the report couldn't count cache tokens; `claude_code_eval` was glm-hardcoded), fixed in one glm dispatch (`619ef58`: `raw_in = input+cache_read+cache_write` accounting + a `CLAUDE_CODE_ENDPOINT=anthropic` mode). The result (`kb-03102`): caching confirmed live (talos fresh input ≈ 0), **raw-input ≈ 8× not the glm 17×** (billed ≈ 7.6× — caching is symmetric, so it did *not* erode the lead; the 17→8 drop is iteration-driven), no capability gap (6/6 pass + holdout, 0 false-dones). Released **v0.6.0** (two `feat:` commits → minor bump; the GTD adapter this number once tracked shipped back at 0.3.5 and has matured since). **Next:** first-class **Ralph Loop** support in talos (`--ralph-mode` + a stopping condition — the harness *restarts the agent loop with fresh context* on `finish` if the condition isn't met; distinct from finish-recovery, which nudges the *same* context on non-green gates). Full write-up + handoff: **kb-03103**.

**Session 11, continued — Ralph shipped on talos-glm, and a two-bug debugging saga (`kb-03104`).** Grooming the Ralph *core* (draft→2 critics→synth, grounded at HEAD) produced a ~19-AC spec; dispatching it to talos-glm then **stalled three times** (`StoppedWithoutFinish` at 20/26/22 iters, ~3 min each) — looking exactly like a talos-glm capability ceiling on big multi-file work. It was neither capability nor context. I chased the wrong lead first (num_ctx — talos-glm sent glm *no* context pin on the cloud URL, a real silent-truncation bug we fixed and deployed anyway), violating our own "read the record before theorizing" rule through a full build→release→deploy→re-run cycle before finally pulling the run record: **every stall was immediately preceded by a model turn that hit exactly 4096 completion tokens** — the `DEFAULT_MAX_TOKENS` cap. glm is a reasoning model; its thinking counts toward output, so on hard steps it was truncated *before emitting a tool call*, and the harness mislabeled that as `StoppedWithoutFinish`. One-line fix (`4096 → 32768`, safe below Haiku's 64K floor), fleet binary republished — and the **same item then completed on talos-glm: 116 iterations, verified Done, 428 tests green, merged** (`1b4c2bb`). The harness built its own restart loop on the cheap lane. Also this session: **v0.6.0 released** (prompt caching + benchmark v2), **github caught up** 0.3.0→0.6.0 (release.sh now pushes both remotes), and the num_ctx + max_tokens fleet fixes landed. Lessons in `kb-03104`: reasoning models need generous `max_tokens`; the disposition *lied* (truncation masked as StoppedWithoutFinish) — follow-up filed to surface a distinct `Truncated` terminal. **Next:** the `talos run --ralph-mode` CLI (thin layer over `run_ralph`).

---

## Session 12 — 2026-07-14 — the Ralph CLI shipped, then a dogfood proved the loop, surfaced its own bug, fixed it, and surfaced three more

Finished the Ralph feature and dogfooded it hard. First **`talos ralph`** shipped — a standalone subcommand over `run_ralph` (chosen over `talos run --ralph-mode`: `run` is TaskSpec/disposition-shaped, ralph is objective/RalphTerminal-shaped), built by talos-glm one-shot (33 iters, `kb-03106`), released as **v0.8.0** (fleet binary deliberately not pushed). Then the spine: bootstrapped the external **dng-converter** repo (zero tests, no gates) with a lean check-only pre-commit gate (ruff `E/F/I/W` + `format --check` + pytest — no coverage threshold, no conventional-commit-msg hook, since ralph commits are `ralph: iteration N — …`) and ran `talos ralph` on **talos-qwen** (`qwen3.6:35b`, localhost, think=on) toward 90% coverage — the canonical ralph-able task, "write the highest-value missing test." Run 1 (cap 10): **6%→43%**, ten clean iterations, healthy. Run 2 hit a real bug: qwen wrote a broken mock it couldn't green, ralph **committed the dirty tree on-change**, the pre-commit hook rejected it, and `RalphTerminal::Error` killed the whole loop — an **orchestration** bug, ralph committing non-green outcomes. That became the **do-over fix** (item 230f9e9b, `242a57f`, on main **unreleased**): commit *only* a green `Finished(Done)`; on any non-green outcome or a hook-rejected green commit, **revert to the last green commit** (`git reset --hard HEAD` + `git clean -fd`) and retry fresh; a new `RalphTerminal::DoOversExhausted` (exit 20) after `--max-do-overs` (default 3) consecutive do-overs, reset on every green commit; `BackendError` exempt. The groom-to-ready workflow ratified **three** refinements to the lead's draft (dedicated `DoOversExhausted`→exit 20 not `Error`→exit 1; `BackendError` exempt; revert discards `PROGRESS.md` too), and **talos-glm built the fix** — the harness repairing its own loop (`kb-03110`, 450 tests, 98.04%). Run 3 (post-fix, cap 100): **48%→69%** (`convert.py` 100%, `manager.py` 28%→58%, 23→43 tests) — the fix *delivered*, pushing 21 points into the hard module with no crash — but it ended on `Error`, not the clean `DoOversExhausted`: after iteration 14 it churned ~20 iterations of **sustained `BackendError`** (the ballooning test file blew the 32768 `num_ctx`), which is exempt from the do-over counter, so the loop neither progressed nor stopped cleanly until a transient git op failed. qwen's tests were genuinely good (use-site mocks, behavioral assertions, no gaming); its ceiling is the launchd/argparse tail. Captured the **ralph-ability characterization** (`kb-03109`: a static prompt that re-binds against mutating external state; five requirements; the `tasks.md` backlog-executor as the next experiment) and **three new gaps** as GTD items — consecutive-`BackendError` breaker (`6bd67e1a`), surface the `Error` terminal payload (`57623441`), and the `num_ctx` hard-cap design question (`a033ab51`). Process miss owned: I polled the **live** ralph workspace with `git`/`pytest`, racing the harness's own git ops (the observer effect that likely caused run 3's final Error) — never poll a live ralph workspace. Full handoff: **kb-03112**. **Next:** cut 0.8.1 (the do-over fix); groom the BackendError-breaker; the `tasks.md` executor experiment (now unblocked); ralph dispatch mode.

---

## Session 13 — 2026-07-15 — model-backend expansion (Sonnet 5 + AWS Bedrock), and the harness-is-the-variable thesis proven

Expanded talos from two model backends to three, and in doing so proved the project's core thesis on its hardest dispatch yet. First the small moves: renamed the canonical Sonnet to **Sonnet 5** (`ddba2d3` — `claude-sonnet-4-6`→`claude-sonnet-5` across CLAUDE.md + test/doc fixtures, leaving `docs/research/` Sonnet 4.5 citations as historical); **scrubbed CLAUDE.md** of stale content (`9560a83` — deleted the two-versions-old `## Status` section since status lives in the roadmap, fixed the coverage floor 95→98, the fleet version token to `git describe --tags`, and the Layout/Release facts); and stored a **SWE-bench Pro reference** (`10bc595` — Opus 4.8 69.2% / Sonnet 5 63.2% / GLM-5.2 62.1%, single-leaderboard so the numbers are comparable). That last one set up the session's spine: the **AWS Bedrock backend** (`886aa2f`, item `a2ea06a0`), so talos can run where the Anthropic API isn't reachable (Jason's work machine). Its `BedrockBackend` drives `aws-sdk-bedrockruntime`'s non-streaming Converse API, standard AWS credential chain (no keys in code), `TALOS_BEDROCK`-gated (wins over any Anthropic/Ollama env), 3-model restriction (haiku-4-5/sonnet-5/opus-4-8) rejected-at-construction. The groom-to-ready workflow's code-grounding critic caught the **load-bearing landmine**: the AWS SDK defaults to `aws-lc-sys` (OpenSSL-licensed, against our rustls-ring/no-OpenSSL posture and absent from `deny.toml`) — the spec pinned `default-features=false` + a hand-built ring-rustls `HttpClient`, plus sync-construction with a lazy `OnceCell` async-defer (so `backend_from_env` stays sync) and a `with_test_endpoint`+wiremock coverage seam. The routing decision was explicit and load-bearing: since **GLM-5.2 (62.1%) ≈ Sonnet 5 (63.2%)** on SWE-bench Pro, this was **not a model-capability question but a harness one** — *is the talos harness up for it?* — with the fallback being **claude-code-glm** (same model, stronger harness), not a bigger model. Answer: **talos-glm cleared it clean** — 293 iters, zero inline fixes, `cargo tree -i aws-lc-sys`→"did not match any packages", `cargo deny` green with `deny.toml` untouched, 486 tests, coverage 98.10%. The lead then ran the `#[ignore]`'d **`live_haiku_smoke` against real Bedrock** (`AWS_PROFILE=gritmile-bedrock-test`) — passed in 1.19s, a real SigV4 Converse call to Claude Haiku 4.5. The harness wrote its own AWS provider, on the cheap Ollama lane (zero Anthropic spend), with a licensing landmine correctly disarmed, live-verified — the harness-is-the-variable thesis (`kb-03109`) proven on the hardest task the project has attempted. Released as **v0.9.0** (fleet binary deliberately not shipped to dispatch hosts — other work in progress; Jason runs `talos-update.sh` later). Full handoff: **kb-03115** (perf: `kb-03114`). **Next:** verify sonnet-5/opus-4-8 live on Bedrock once provisioned; the `talos-sonnet`→sonnet-5 bump in agent-gtd-dispatch; the three Session-12 ralph gaps; the `tasks.md` executor + ralph dispatch mode; Bedrock prompt-caching/streaming follow-ups.

---

## Session 14 — 2026-08-12 — nemotron-3.5-lightning: first false-dones, no lane, and the pivot to discriminating evals

Ran NVIDIA's launch-day **nemotron-3.5-lightning** (hybrid Mamba-Transformer MoE, ~3B active/32.9B, 1M native context) through the full eval suite on the 5090 — no code shipped, but the *finding* is the artifact. The 1M window **doesn't fit on 32GB** (the wall is the ~1GiB prompt-processing compute buffer, not the KV cache, so KV-quant can't rescue it; 512K loads 100% on-GPU with flash-attn + q8-KV) and wouldn't have mattered anyway — peak context depth was ~100-130k, though the 512K window was load-bearing for eval *validity* (trials ran to 1.5M cumulative tokens; the 32k default would have silently truncated). Results (`kb-03195`): **25/30 both think=off and think=on** — but the model produced the **project's first false-dones ever** (120+ prior trials clean): tokenbucket visible-gate GREEN + Done claimed, sealed holdout RED — and **think=on doubled them (1→2) at identical pass rate**, raising confidence without correctness (opposite of the qwen +think precedent; measure think per model, never assume). Verdict: **no lane** — dominated on both axes by qwen3.6:35b (same VRAM, 24/24, zero false-dones) and talos-glm (cheap hosted, 24/24); the eval suite turned a hot new model into a routing decision in ~2 hours, and the sealed-holdout gate earned its existence by catching what production `gate_command` would have trusted (standing position: keep weak models out of the fleet rather than build a production backstop). Session log: **kb-03196**. **Next:** the suite is pass-rate-saturated 5 models deep — it kicks out weak models but can't discriminate the top tier. Pivot: research how marquee evals (SWE-bench Verified etc.) are structured + the state of harness testing, then design a discriminating eval tier (Opus vs Sonnet vs GLM).

---

## Session 15 — 2026-08-14 — the tier-2 pilot built end-to-end: spec ladder ratified, 8 tasks mined, runner shipped, smoke-proven

Turned Session 14's research into a working pipeline in one sitting. First the design closed: Jason's objection — *every mined commit was built by headless dispatch with a full groomed spec, so we know Opus/Sonnet+CC solves them* — produced the load-bearing idea (`kb-03198`): **a mined commit is a task family parameterized by spec level** (S3 full-spec calibration floor / S2 AC-only / S1 intent-only / wave composites), so discrimination comes from *withholding the grooming* — and the eval doubles as a measurement of grooming's value per engine, the routing rubric's missing axis. Then the pilot: a new **talos-evals repo** (benchmark content quarantined from dispatch agents) with **8 hand-mined, dual-gate-verified tasks** across the ladder (probe: `kb-03197`; 7 mined by parallel agents, statements lead-reviewed for fix-leakage), with mining conventions minted from real cases — empirical FAIL_TO_PASS derivation (4/8 tasks had new-but-already-passing positive controls), lockfile-pinned path-dep siblings, interface pinning for renamed public surfaces, green-at-parent PASS_TO_PASS. The **mined_eval runner** shipped (`4d67846`, groom → claude-code one-shot in 35min → two lead inline fixes: file-qualified id matching, git-hook env scrub; perf `kb-03199`), and the **smoke run went green**: qwen3.6:35b at S2 on cleanr-owner-comments → **Resolved in 100s** — with trial 1 already demonstrating the architecture (claimed=StoppedWithoutFinish yet Resolved: the scorer grades the work, the cross-tab catches the claim gap). Session log + full handoff: **kb-03200**. **Next (fresh session): the pilot matrix, cheap lanes first** (qwen/glm/haiku before Opus/Sonnet) — if the hard rung doesn't separate haiku from glm, recalibrate before buying frontier data.

---

## Session 16 — 2026-08-16/17 — the first tier-2 matrix: a voided row, the num_ctx foot gun closed, and a hypothesis killed by its own instrument

Session 15 handed off "run the pilot matrix, cheap lanes first," and the first command of the session was that run — which produced no usable data, for reasons worth more than the row. Fourteen of 24 trials died on `BackendError`, and the payload was undiagnosable because the trial line printed only the variant name (the Session-12 follow-up `57623441` biting a third time). Fixed inline, re-probed, and the cause was **`context length exceeded`**: I had run the matrix at the runner's `DEFAULT_LOCAL_NUM_CTX = 32_768` without querying the KB, where `kb-03188` already recorded **262144** as the standard and the 32k→256k fix as a measured win. A textbook **context failure** — the knowledge existed, I reasoned from the constant in front of me — compounded by escalating my own misconfiguration to Jason as a "talos has no context management" capability gap with a scope decision attached. His correction (*"we've seen this before, it must be in the KB"*) is what surfaced it. Re-run at 256k (which `ollama show` advertises and which loads **100% on GPU** at 30.9/32.6GB on the 5090 — no flash-attn or KV-quant needed, `kb-03202`): **13/18 valid resolved, zero false-dones**, and the ladder genuinely discriminating (two tasks at 0/3 and 0/2 with real `missing_ftp`, four at 3/3 — tier-1 never separated like this).

Three things shipped (`74a506e`, `b556f6b`, `220406a`; 580 tests, coverage 98 held). The lead's inline fix also corrected `parse_pytest_summary_totals`, which stripped exactly three `=` from a banner pytest pads to terminal width — gluing the padding onto the first bucket's count so it was silently dropped, under-counting multi-bucket summaries (spurious `parse-mismatch`, voiding whole tasks) and, on single-bucket summaries, returning `None` and **silently disabling the truncation cross-check exactly where it was meant to protect us**; the prior test used exactly three `=`, which is why it survived. Then a serialized two-item rollout (`69c7897a`, both claude-code-sonnet, clean, manager-merged): **num_ctx now resolves from the model's own advertised context length via `/api/show`** in one shared helper both runners use — explicit env still wins, an unresolvable model **fails loudly** naming model and endpoint (`kb-03203`) — and **finish-discipline telemetry**, seven purely-additive `RunStats` fields surfaced per trial (`kb-03204`).

The spine, though, was being wrong three times in the same shape: **a correct local code read paired with an unchecked surrounding configuration**. After num_ctx, I read `engine.rs:1358-1361` correctly (any successful `bash` — including a read-only `git diff` — latches `tree_dirty`, clears `last_gate_green`, resets the static counter) and reported finish-recovery as disarmed by bash noise. The groom workflow's **code-grounding critic overturned it**: tier-2 calls `standard_registry(None)`, so `run_checks` is never registered, `last_gate_green` can never be true, and both nudge guards are permanently false — **finish-recovery is structurally dead in tier-2**, while production talos *does* register it from `gate_command`. Tier-2 benchmarks a harness configuration we never ship. And even then I still believed bash held the static counter below K=3; the telemetry I had just shipped **refuted that too** — `peak_static` of 2/8/9/11/15 against a threshold of 3, with `bash_ok` 2-12 in the same trials. The counter was never binding; the gate guard alone was. Full correction in `kb-03201`. One more finding for every future comparison: **k=3 is noisy** — identical config gave 13/18 and 11/18 on consecutive runs, so treat tier-2 deltas under ~3 trials as nothing.

**Next:** the finish-recovery fork is a design conversation, not a dispatch — arm `run_checks` in tier-2 from the `gate_command` each task already carries (restores the production surface, but a green *parent* gate doesn't mean done there, so it risks manufacturing the false-dones this suite is best at catching), give finish-recovery a gate-independent trip (the data says it would fire), or leave tier-2 oracle-free and stop quoting its finish-discipline numbers as production-relevant. Then groom + dispatch the seeded **eval-infra fix** (`0595137d`: section-scoped pytest parsing, since caplog `ERROR` lines are still read as test ids and void `agent-gtd-rollout-deadlock` 3/3; plus Invalid-vs-Unresolved, since a sealed-test `ImportError` on a symbol the agent never wrote is the agent failing the pinned interface, not infra — scoring it `Invalid` pulls it out of the denominator and flatters weak models) **before** the glm/haiku lanes, or the same void rate bakes into three more rows. Also queued: `b4b0aef6` (talos CLI num_ctx — the shipped dispatch lane still runs qwen at 32768 while the eval lane runs it at 262144, an 8× gap between what we measure and what we ship). Session log + full handoff: **kb-03205**.

## Session 17 — 2026-08-20/21 + 2026-09-12/13: test-first prompt, glm-5.3 lanes, v0.10.0

The August sitting (never logged at the time) fixed the tier-2 scorer (section-scoped pytest parsing; an agent-caused `ImportError` scores `Unresolved`, not `Invalid`), added per-trial test-authorship telemetry, and changed the **production** task prompt to require a failing test first — and to stop saying "finish as soon as the gate passes," which on a healthy repo meant finish at iteration 1. It also routed that guidance into tier-2 and added a tier-1 A/B knob that hasn't been run yet. The September sitting paused the eval work to put **glm-5.3** on talos: qualification showed no harness change was needed and pinned think=high (5.3 at max cost 2.3× the tokens; flash at low made a false done). On tier-2 both 5.3 models beat 5.2 (13/24 vs 9/24), with **flash matching full 5.3 at far better finish discipline and ~1/10 the price**. The wiring shipped as agent-gtd-dispatch 1.21.0 — `talos-glm` → glm-5.3, new `talos-glm-flash`, `talos-sonnet` → Sonnet 5, cloud-key validity probing — after lead salvage of two claude-code-sonnet quality misses. The first patrols on both new lanes landed two ralph fixes in under 3 minutes each, with the model identity verified from the run records. The session also surfaced that the dispatch fleet still runs **talos 0.6.0-3**, so no release since 0.7.0 has reached dispatch. KB: `kb-03226` (evals `kb-03220`, cutover `kb-03222`).

**Session 17, continued (2026-09-13 afternoon):** Jason ruled that evals must measure real dispatch, so tier-2 got a production-parity agent gate — each task's real project gate as `run_checks`, finish-recovery armed, claim-verified Done — while the sealed file-scoped gate stays the judge (groomed via groom-to-ready, shipped clean by claude-code-sonnet, `a0c0128`; post-run shippability verdict `eb333c1`). The first matrix under it found the **iteration cap is the dominant variable** (≈50% at cap 24 vs 8/8 for flash at the production cap of 500), that **finish-recovery never fires** because stalled trials sit on a red gate, and that test-first is neutral-to-positive at the production cap. It also surfaced two task defects (a Verification-heading collision and an unfair `unknown key` wording test behind most of from-json-contract's historical zeros). KB: `kb-03240`, tier-1 test-first A/B `kb-03227`.

**Session 17, continued (2026-09-14):** With the tier-2 cap set to the production 500, the first representative k=3 matrix ran glm-5.3, flash, qwen3.8:27b (new generation; needs Ollama 0.34 and fits the 5090 only up to ~144K context, so it ran at 128K) and qwen3.6. The raw scores exposed two more unfair wording assertions, so every task's text assertions were audited and pinned. Re-scored, the combos cluster at 19-21/24, and the separating metric is real false dones: qwen3.8 2, flash 2, glm-5.3 3, qwen3.6 5. The local qwen3.8 matches the cloud models for free. KB: `kb-03252`.

**Session 17, overnight (2026-09-15):** Jason moved `talos-qwen` to qwen3.8:27b and asked for 256K via an 8-bit KV cache. That fits the 5090 once ComfyUI and race-photos are stopped, and a small run went 8/8. The re-run with wording pinned put glm-5.3 at 24/24, flash at 22/24 and qwen3.8 at 21/24; pooled over both cap-500 runs the three are one band. Two harness fixes shipped: off-enum finish dispositions are no longer silently coerced to Failed, and opt-in full run transcripts landed. The very first transcript read overturned the false-done diagnosis. qwen3.8 did test the "all location metadata" clause, but its test, fixture and stripper share a wrong recalled IPTC field list: a context failure, which reshaped design doc 05. A rollout-manager relaunch bug was captured (`2f5c182f`), and one more unfair test was pinned. KB: `kb-03264`.

**Session 17, A′/B experiment (2026-09-16):** Both approved false-done features shipped default-off — the opt-in Criterion Coverage prompt (`2c74407`) and the one-shot acceptance audit (`712fb21`, which stashes a verified Done so a text-only reply can never lose it). The measurement then refuted the first read of its own results: at k=3 the arms looked like wins, but `audit_changed_tree` was false on every improved trial, and a k=10 probe-rate run on the dominant shape showed the criterion rules produce a 0/10 probe rate, identical to baseline. Neither default flips; the tier-1 perfect-spec guard was clean. The likely reading is that this shape is a model-knowledge gap the eval should discriminate, not a harness defect. KB: `kb-03283`.

**Session 17, continued (2026-09-17):** The two measured-negative knobs came out (`cbddb48`, ~3100 lines — templates, flags, four env knobs, nine telemetry fields, and the `EvalOptions` refactor that existed only to carry the flag), keeping one addition: an explicit test pinning the anti-loop "do not re-verify individual acceptance criteria" line, which had only been pinned indirectly through the production golden. Then Jason asked where dispatch transcripts would be stored so they wouldn't build up forever, and the audit found something bigger — **talos had no retention policy at all**: run artifacts had accumulated since July with nothing deleting them (14 MB / 14 run dirs on pironman01, 50 MB / 38 on r7-research), and `mined_eval` resolved its state root from env deep inside the library, so unit tests wrote into the real `$HOME` (71 leaked dirs on pironman01, written by the project's own `gate_command` running nextest there). Four items shipped: the state root is now injected as a value (`7b04ecb`), `talos run` prunes its own state dir at start (`caf7549` — pure `prune_state_root` + thin caller, `--state-retention-days` > `TALOS_STATE_RETENTION_DAYS` > 30 days, `0` disables with zero I/O, aggregator subtrees descended one level because a directory's mtime refreshes when a child is added), bare `--transcript` defaults to `<state-dir>/transcript.jsonl` (`33fee62`), and every talos dispatch now passes it (agent-gtd-dispatch `28c5e6e`). Fleet republished to `0.10.0-19-g33fee62` on all three hosts. Retention lives in the binary rather than a cron entry precisely because talos is self-hosting: it reaches the fleet through `talos-update.sh` and cannot drift from the code that creates the mess.

The groom workflow earned its cost twice by **overturning lead dispositions with code evidence**. I had specified a one-line JSON prune report on stderr; the critics traced `exit_code` mapping `BackendError` to 1 with no preceding `stderr_json_error`, so on the dominant fleet infra-failure path that line would have become the last stderr line of an exit-1 run — exactly what the dispatch worker parses to classify a failure. Replaced with total silence plus a `prune-last.json` in the state root. They also found `run_ralph_cmd` only ever `create_dir_all`s the *offload child*, so the `talos-ralph` state dir's own mtime freezes at the first-ever ralph run (the real one was 65 days stale) and the first `talos run` after shipping would have deleted it out from under a live loop. Both are defects a build agent would have implemented faithfully from my spec. Verification followed the previous sitting's lesson — measure the behaviour the change should induce: the test-leak fix was confirmed by running the 126 `mined_eval` tests and seeing the directory count unchanged (959 → 959), and the deletion code was smoke-tested against a real fixture tree (aged dir removed, `mined-eval/old-run` removed individually while its equally-old parent survived, live and fresh dirs kept, a symlink to a 40-day-old directory skipped with link and target intact).

Three process findings. The rollout manager halted on `manage_relaunch_cap_exceeded` a second time, 24 s after a healthy build run started and with that run still in flight — second data point on `2f5c182f`, with the cheap fix named (don't evaluate the cap while `inFlightBuildRuns` is non-empty); the build survived the manager's death, so the lead only took over review and merge. Claude Code killed `run_in_background` tasks three times as "low on memory", including a single idle poller, and kept doing it after `drop_caches` took MemFree from 2 GB to 26.7 GB with zero pressure — the MemFree theory from 2026-09-13 is refuted, it's a false positive unrelated to real memory state. And a near-miss now codified in `~/.claude/CLAUDE.md`: reviewing a dispatched branch with `git diff main <branch>` showed 1024 deletions across 13 files and read as a severe scope violation, when the branch was merely two commits behind main — diff against the merge-base, or read the branch's own commits. **Needs Jason:** deploy agent-gtd-dispatch, or transcripts don't start recording. Session log: **kb-03293**.

**Session 17, continued (2026-09-18):** A design consult from the agent_gtd session — claude-code dispatch runs had been recorded `success` after committing nothing — turned back on us: answering from talos's code showed `RunSummary` carried no tree-changed signal, so a run that edited nothing and called `finish(done)` on an already-green repo produced a fully "verified" Done that dispatch would push; we had patched only the behavioural version, with a prompt. The principle that came out is `kb-03300` — a Done has three legs (asserted, verified, work happened) and no leg may be agent-disableable — and both harnesses now enforce the same vocabulary. Leg 3 shipped (`1ec37b2`, Opus, 52 min): `handle_finish_call` observes the tree at run start and at the claim and rejects an unchanged-tree `done` like a red gate. The groom's load-bearing catch was that the comparison must be **baseline-relative**: the dispatch worker stages an untracked attachments dir inside the clone before the agent starts, so an absolute dirtiness test would have read "changed" all run and silently no-opped the check on every attachment-carrying item. `AlreadySatisfied` is a peer disposition (exit 30); ralph counts it as a do-over rather than grinding to `Stuck`; ralph turned out to already have leg 3 (`is_green && made_changes`) — the same reasoning written in one path and never carried to its sibling, the exact shape the agent_gtd session had just fixed in their rollout manager (`kb-03296`). One lead fix at merge: a timing-flaky test that raced a zero-duration real `git` call (`tokio::time::timeout` polls the inner future before the deadline), replaced with a pure function tested against a synthetic outcome — caught only by asserting `git log -1` moved after a hook-gated commit.

Jason then asked why tier-1 workspaces had no `.git` when the tiers should differ only in difficulty. I had reported it as a designed property; it was storage format — committed fixtures can't carry a nested `.git`, `copy_dir_recursive` reproduced plain directories, and a doc comment had canonized the accident. Production always runs in a clone, so the perfect-spec guard was structurally blind to the precondition it exists to guard. Fixed (`2612751`): each trial workspace gets `git init` + `add -A` + one commit (a committed baseline is a *clean* tree like a fresh clone), best-effort, shared by both tier-1 runners — and fixed *before* the guard run, reversing my own earlier sequencing call, because a guard that can't see the mechanism is worthless for that change. Tier-1: 30/30, 18/18 holdout, 0 false dones, `no_change_rejections` 0, and `change: TreeChanged` on all 30 Dones — the first tier-1 run where the precondition was live. New baseline; earlier rows aren't comparable on anything the tree observation touches.

Then a curiosity became a feature in a day. Jason asked whether a Claude Code session could write a dynamic workflow whose sub-agents were talos. The answer is yes, and leg 3 was the missing piece: talos had one deliverable shape and a research sub-agent changes nothing. **Answer mode** (`docs/design/06`, written before grooming; `bce155b` library + `d349a61` CLI, both Opus, 18 and 26 min): `talos run --mode answer --schema` returns a schema-validated JSON `result` on `RunSummary`, exit 40, the same three legs with the payload as evidence — and the leg-3 primitive *inverted* enforces read-only, since accepting an answer requires the tree to be unchanged, so a `bash` write is caught rather than sandboxed. The orchestrator is deliberately a library (a small Python module spawning subprocesses), not an embedded runtime. `jsonschema 0.56` brought two transitive licenses outside the allow list (MIT-0, Zlib), admitted as per-crate `deny.toml` exceptions. Live smoke on haiku against a one-file repo with a wrong `add()`: exit 40, valid result, `TreeUnchanged`, 3 iterations, tree clean. Grooming lesson: two dependent items groomed in one run guess each other's symbol names; the dependent one was re-grounded against the merged code before dispatch and the build used the real names. Fleet still on `0.10.0-19-g33fee62`, publish gated on dispatch learning exits 30/40 (the agent_gtd session's release) and the tier-2 guard (6/24 at time of writing, all resolved). Session log: **kb-03311**.

**Session 17, continued (2026-09-18/19):** Jason put the Ollama usage page in front of me — Max plan, $100/month for $300/month of credits, no rate windows, no rollover, $47.57 spent with three weeks to reset — and the standing global rule flipped from "don't dip into Ollama credits" to spend aggressively with deliberate model choice (`kb-03315`): glm-5.3-flash by default for Sonnet-class groomed work ($0.15/$0.03/$0.50 per 1M in/cached/out, 1M context), full glm-5.3 when the item needs it, Anthropic lanes only after Ollama misses or when the Claude Code harness is specifically needed. I made the Ollama lanes the *default* rather than merely permitted, which is what "aggressively" means operationally, and flagged it as a one-line dial-back. The motivation was blunt: a KB search confirmed that neither `talos-glm-flash` nor `talos-qwen` had ever done real work — one two-minute patrol and eval rows between them.

So four backlog items became the lanes' first real work. One groom-to-ready run (24 agents, ~13 min) specced all four and assigned lanes; qwen ran serially (one GPU, confirmed reachable from the hosts and serving qwen3.8:27b), flash ran in parallel on two hosts. All four came back clean with zero lead fixes and green gates before push and on re-review: **qwen** — the shared `num_ctx` resolver for CLI and both eval runners (`3584df9`, 17.5 min, 83 turns) and backend settings on the run record (`d76792d`, 34.8 min, 164 turns; its spec predated its prerequisite and used the re-grounding comment's symbols correctly); **flash** — the tier-2 count-mismatch tripwire never silently off (`3bdc845`, 5.0 min, 50 turns) and the fail-safe docs-only skip for the heavy lefthook gates (`055d187`, 15.5 min, 35 turns; predicate exit semantics sanity-checked against lefthook's `skip` contract before merge). Identity was verified on every run from the transcript label and, for qwen, `ollama ps` at 100% GPU — the label-is-not-identity lesson applied. Read: qwen at roughly double a frontier model's turns for the same result at zero API cost; flash fast and clean on both shapes. Four runs is the first data, not a routing verdict.

The first of those runs was also the read-back promised the day before: r7-research's `prune-last.json` showed 604 examined and 131 removed — the July/August dirs plus ~560 leaked mined-eval test dirs — taking the state dir from 50 MB to 11 MB, pironman01's showed 54 removed, every run wrote `transcript.jsonl` beside `run.sqlite`, and every terminal record carried `change: TreeChanged`. Everything that was armed-but-untriggered on the 18th has now fired on production work. Process notes: claude-steering `1ca04bd` moves the dispatch waiter into `Monitor` because background Bash tasks are reaped on this box regardless of real memory (`kb-03238`); Jason asked me to try it next time, and every wait this session was still foreground because both flash runs finished before one could be armed. Peer messaging between sessions was observed unreliable; host state via `/info` and ssh stays the source of truth. Open: where `talos_flow` lives. Session log: **kb-03318**.

**Session 17, continued (2026-09-19, afternoon):** Jason created the `talos_flow` repo and asked to get the client moving on Ollama Cloud (the workstation GPU is waiting on a reboot). Bootstrapped it in one commit — uv, hatchling, ruff with pydocstyle, `mypy --strict`, pytest with coverage 90, checkers-only hooks (kb-03099), semantic-release, a `CLAUDE.md` that states the talos stdout/exit-code contract — renamed the starter `master` to `main`, and created the GTD project with the gate. Two items groomed in one run, with the critics reading `crates/talos/src/main.rs` and citing lines for every flag and `RunSummary` field: `agent()` on talos-glm (**4.25 min**, 55 tests, 100%) and `parallel()`/`pipeline()` on talos-glm-flash (**4.75 min**, 74 tests, 100%), zero lead fixes on either. The groom caught two things I had not specified — the schema temp file must be *closed* before the child starts, and unknown exit codes classify as engine error *before* the unparsable-summary check — and raised one open question I answered as an AC: `max_concurrency` defaults to 8 with `None` as the explicit unbounded escape, because every `agent()` is a talos subprocess and the 40-way fan-out design 06 calls affordable would otherwise be 40 processes on one host.

Verification went through the real binary, not just the fake `Runner`: `await agent(...)` on haiku parsed a correct result with `backend_settings` surfaced, and `parallel()` over two concurrent agents on **one shared read-only checkout** returned two correct schema-valid answers in 5.7 s with the tree clean — the first data point on design 06's shared-checkout question, and it says a shared checkout is fine for read-only answer agents with distinct task ids. Both dispatch waiters were hosted in `Monitor` per the updated steering (`1ca04bd`) and both survived and delivered the terminal event cleanly — 2 for 2 — so the foreground-wait workaround is retired for dispatch. Earlier in the day the global Ollama rule flipped to aggressive spend (`kb-03315`) and four harness-design items became the first real work on talos-qwen (2/2) and talos-glm-flash (2/2); today's lane tally including talos-flow is flash 3/3, qwen 2/2, full glm 1/1, all clean. The memory index was compacted into topic files. Next: port groom-to-ready as the first real workflow and get the flash-vs-Sonnet cost row. Session log: **kb-03318** (run rows kb-03324, kb-03325).

**Session 17, continued (2026-09-19, afternoon → evening):** The groom-to-ready port to talos-flow landed on the second attempt — the first died citing the JS at a path on my machine, a lead-owned context failure that is now a steering rule, a vendored `reference/` copy, and a check in the grounding critic's own prompt (claude-steering `b5691a9`). The first real glm-5.3 runs then hit a harness bug the haiku smoke test could not see: Ollama Cloud flattens object tool parameters to JSON text, so `finish(answer)` never validated; two glm agents diagnosed it from inside the run and returned `Blocked` with a decision, the engine now parses stringified results (`b034a63`, `kb-03340`), and the rejection loops cost ~19M glm-5.3 tokens before the fix. With the binary fixed, four groom arms ran on identical prompt bytes over the same three backlog items (`kb-03342`): the Opus/Fable Workflow wrote the deepest specs and alone found that `d76792d` had silently reverted the CLI num_ctx resolver; talos on glm-5.3 ($15.00) did empirical git verification in its critic stage; flash-draft/glm-critic/flash-synth ($8.50) made the best design call; all-flash ($1.31) produced a spec about the wrong item that its own critics passed. One spec per item, one per arm, all three one-shot on talos-glm with zero lead fixes: the ralph commit-retry (`d6e93e1`), the num_ctx restore with `docs/design/07-num-ctx.md` (`1f56186`), and `FailureMode::Truncated` + `--max-tokens` (`5211a15`). talos-flow also gained per-stage cost accounting, groomed by the port itself and built by flash. Lessons: the lead never edits a checkout a groom is reading; Monitor hosted 8 of 8 waits; glm-5.3 cloud is $1.40/$4.40 per 1M and talos sees zero cache reads on Ollama, so the cost lever is caching, not the model. Fleet binary republished at session end. Session log: **kb-03352** (run rows kb-03335, kb-03341, kb-03347, kb-03349, kb-03351; comparison kb-03342; bug kb-03340).
