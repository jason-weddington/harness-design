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
