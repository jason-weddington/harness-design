# 05 — Reducing false dones: per-criterion tests + a one-shot acceptance audit

Status: **MEASURED — NEGATIVE (2026-09-16).** A′ and B are built and merged, DEFAULT OFF everywhere, and neither shows a benefit on the shape it targets (`kb-03283`). Do not flip either default on this evidence. Details in the Result section at the end. Data: `kb-03252` (first representative tier-2 matrix), `kb-03240` (production-parity gate).

## The problem

Tier-2 now measures the harness we ship: each task's real project gate is the agent's `run_checks`, finish-recovery is armed, `finish(done)` is claim-verified, and the cap is the production 500. Under those conditions nearly every trial ends in a verified Done — resolved-but-unclaimed is at most one per 24 trials and no resolved trial ends with the project gate red. So the metric that separates model + harness combos is the **real false done**: a Done the project gate accepts but the hidden acceptance tests reject. That is wrong work a real dispatch would push.

The first cap-500 matrix produced 12 real false dones in 96 trials (glm-5.3 3, glm-5.3-flash 2, qwen3.8 2, qwen3.6 5), every one on a requirement the task statement spells out. They fall into three recurring shapes:

1. **Invariants — "X must not change."** rollout-deadlock AC2 says starting a wave "must leave every planned item's status untouched." Agents implement the positive behaviour (launch the manager as a property of the rollout) and never check the negative clause; `test_manage_dispatch_does_not_flip_item_status` fails. glm-5.3 missed it 3/3, qwen3.6 3/3, flash 1/3.
2. **Exhaustiveness — "all of X, not just one."** photoqueue-privacy-metadata AC1 says no readable location may remain "anywhere in the file's metadata … it is not enough to remove one representation of the location while another remains readable." Agents strip EXIF GPS and leave IPTC and XMP place names (`Columbia, Maryland, United States` survives). flash 1, qwen3.8 2, qwen3.6 1.
3. **Regressions outside the agent's attention.** qwen3.6 on cleanr-comment-filters broke two existing behaviours (`unexcluded_red=2`) while its project gate was green.

### What the transcripts showed (overnight, 2026-09-15)

The re-run with wording pinned (cap 500, k=3) had qwen3.8 at 256K with full transcripts on (`MINED_EVAL_TRANSCRIPTS=1`). All three of its false dones were on photoqueue-privacy-metadata, and the transcripts overturn my first reading of shape 2. The agent did **not** skip the "every representation" clause. It reasoned about IPTC and XMP hundreds of times and wrote a thorough test (`test_strip_location_removes_all_human_readable_location`): it builds a synthetic JPEG with location in EXIF, IPTC and XMP, then asserts that the place names are gone. But the model's recalled list of IPTC location datasets is wrong. It handles 2:90/2:91/2:92/2:99 and never 2:100/2:101 (country code/name), sometimes not 2:95 (state), and for XMP misses `Iptc4xmpCore:CountryCode`. The synthetic fixture, the stripper and the test all share that misconception, so the test passes by construction, and the sealed test with a real photo finds `{(2, 95): 'Maryland', (2, 100): 'US', (2, 101): 'United States'}` still there. The same field set recurs across all three trials, so this is a stable training-prior error, a **context failure**: the right enumeration exists in the environment (Pillow's IPTC handling, the IPTC IIM spec, any real sample photo) but the model reasoned from memory.

That adds a fourth shape and changes the recommendation:

4. **Self-consistent misconception.** The agent's test covers the clause but shares the implementation's wrong domain knowledge, so the gate is green and the test is vacuous. Across both cap-500 runs this was the single largest source of real false dones (7 of 12, all privacy-metadata).

Implications: a per-criterion-test prompt (A) and an acceptance audit (B) both rely on the agent's own understanding, so neither catches shape 4. What does is making the evidence independent of the model's memory. Two general rules can be added to the same prompt/audit surface: **derive enumerations of standard fields from an authoritative source in the environment** (library constants, installed docs, a real sample file), not recall; and **for "remove all of X" requirements, prefer an allowlist of what to keep over a denylist of what to remove.** Either one would have fixed every privacy false done we saw: an allowlist that keeps caption and keywords and drops every other IPTC/XMP location-bearing field makes the 2:100/2:101 gap impossible.

## Why the harness lets these through

The only thing standing between "I think I'm done" and an accepted Done is the project gate (`engine.rs` `handle_finish_call`: gate green → `Disposition::Done`). The gate runs the repo's existing tests plus whatever the agent wrote. The agents do write tests — nearly every false-done trial modified 1-3 test files — but those tests cover the positive path they just implemented. Nothing in the prompt or the loop makes the agent map *each* criterion, and specifically the negative and exhaustive ones, to evidence before it claims. The model has the capability (the same models resolve these tasks in other trials); the workflow never asks for the check at the moment it matters. That is an orchestration failure, not a capability gap.

## Proposal — two parts, measured separately

### A. Prompt: one test per criterion, with invariants and exhaustiveness named (cheap)

Extend the shared `test_first_approach.md` (one source for production `talos run` and tier-2) from "write a test that captures the acceptance criteria" to "write a test per criterion", with the two shapes named explicitly:

- For every criterion that says something must NOT change, be preserved, or happen only under a condition, write a test that asserts the thing stayed unchanged — not just that the new behaviour happened.
- For every criterion that says "all", "every", "any", or "not just one", enumerate the variants the task names or implies and test each of them.

Cost: more test-writing iterations (tier-1 already showed test-first costs ~1.8x tokens with no benefit on a saturated suite, `kb-03227`); the benefit case is exactly these clauses.

### B. Engine: a one-shot acceptance audit on the first Done (the real lever)

Add `RunConfig::acceptance_audit: bool` (default **off** until measured). When it is on, the **first** `finish(done)` whose gate is green is not accepted. Instead the harness returns a tool result telling the model:

> Before this is accepted: quote each acceptance criterion from the task. For each, cite the test (file::name) or command output that proves it — including every "must not / unchanged / only when" clause and every "all / every / not just one" clause. If any criterion lacks proof, keep working. When every criterion is proven, call finish(done) again.

The **second** `finish(done)` goes through normal gate verification. It is bounded (exactly one audit per run, at most one extra turn when the model really is done), deterministic (no second model, no judge prompt to tune), and fires at the only moment the model is known to believe it is done. It is the minimal form of the per-feature pass/fail checklist pattern from the research corpus (`kb-01874`) and answers `kb-02054` ("criteria must specify quality, or agents satisfy them trivially").

Telemetry to add with it: `audit_fired`, `audit_changed_tree` (the model mutated the tree after the audit, meaning the audit found a real gap), and `audit_rubber_stamped` (second Done with no tool calls in between). Without these we could not tell a working audit from theatre.

**Status (2026-09-15): B is implemented, DEFAULT OFF everywhere** — `talos run --acceptance-audit`, `CODING_EVAL_ACCEPTANCE_AUDIT=1` (tier-1), and `MINED_EVAL_ACCEPTANCE_AUDIT=1` (tier-2, requires `MINED_EVAL_AGENT_GATE` on — the audit only fires on a checks-verified `finish(done)`), plus `RunConfig::with_acceptance_audit`/`EvalOptions::acceptance_audit`/`MinedRunConfig::acceptance_audit` for programmatic callers — none of these knobs are reachable from a real GTD dispatch yet (`agent-gtd-dispatch`'s `build_talos_argv` passes neither `--transcript` nor `--acceptance-audit`; wiring that is a separate item).

The audit fires at most once per loop invocation, on the FIRST `finish(done)` whose gate is green, evaluated in order: `acceptance_audit` is on; the loop-local latch is unset (seeded from resumed history, so a crash-resume whose reconciled messages already contain the audit reply does not audit a second time — this is what makes "it happens once per run" literally true across a crash boundary); the claim is `done`; checks are configured and green; at least one iteration remains after this one (so the audit itself cannot be the thing that trips `MaxIterations`); and the wall-clock budget, checked AFTER the gate run so the gate's own duration counts, is not yet breached (so the audit itself cannot trip `BudgetExhausted`) — a hit on either of the last two conditions falls through to a normal accepted Done rather than holding it back, since firing there would be strictly worse than audit-off.

When the audit fires, every remaining tool call in that SAME assistant turn — another `finish` included — is short-circuited with a fixed "not evaluated" tool result rather than dispatched, so the workspace is provably unchanged between the audit's green gate and the model's next reply; and if that next reply produces zero tool calls (a text-only answer to the audit, the common case), the loop accepts the already-verified Done immediately rather than falling through to the stop-terminal nudge guard and `StoppedWithoutFinish` — the stashed Done carries a harness-run `Verification::Checks`, never a model self-report, so accepting it without a second gate run does not weaken the claim-vs-verify invariant.

Nine telemetry fields land on `RunStats` (and flow through to `MinedTrialResult` and the talos `--acceptance-audit` stdout summary): `audit_armed` (structurally on: audit enabled and checks configured, independent of whether it ever fires), `audit_fired`, `audit_iteration` (the 1-based iteration it fired at, 0 if never), `audit_changed_tree` (a real `edit_file`/`bash` success strictly after the fire), `audit_rubber_stamped` (the very next turn was exactly one bare `finish(done)` call), `audit_reply_text_chars` (that next turn's text length), `audit_followup_edit_file_ok` / `audit_followup_bash_ok` (splitting `audit_changed_tree` into real edits vs. re-run commands), and `audit_followup_red_done` (a post-audit `finish(done)` rejected by a red gate — the churn signal below). This status paragraph SUPERSEDES step 3 of the `Decision needed` section below: B is built default-off ahead of the A′ measurement readout, by decision, rather than gated on it.

### C. Instrumentation prerequisite: transcripts (shipped) + the agent's final diff

Update 2026-09-15: full run transcripts shipped overnight (`d4651b6`, `MINED_EVAL_TRANSCRIPTS=1`): every model turn, reasoning text, tool call input and tool result, per trial. That covers most of what this section asked for; the overnight qwen3.8 re-run has them on, so its false dones can be read turn by turn. Still worth adding: persist the agent's final `git diff` (tracked and untracked) before `copy_sealed` overwrites the sealed paths, because reconstructing a diff from edit_file/bash calls is lossy for bash-driven edits. Pure observability, no scoring change.

## Experiment

Tier-2, cap 500, agent gate on, test-first on. Arms: baseline / A / B / A+B. Models: glm-5.3-flash (the cheap default lane) and qwen3.8 (the local lane). k=5 on the three false-done-prone tasks (rollout-deadlock, privacy-metadata, comment-filters) plus k=3 on the full 8 for regressions. Tier-1 k=3 as a cost and regression check.

Success: real false dones down by at least half with resolved count not lower, and `audit_changed_tree` > 0 on the prone tasks, which shows the audit caught something rather than just adding a turn. Kill criterion: no false-done reduction, or resolution drops by more than noise (≈2-3 of 24).

## Perfect-spec constraint (Jason, 2026-09-15, `kb-03268`)

Tier-2 is deliberately harder than a groomed spec, and it must stay that way; we do not add a groomed-spec eval tier. But talos also runs production dispatches from adversarially groomed specs (18-31 criteria that pin test names, assertions and file:line locations), so A′ and B must be **spec-agnostic and nearly free when the spec already did the work**:

- A′ applies to behavioural criteria only. A test the spec names *is* the coverage: write it as specified and do not add parallel tests. Non-behavioural criteria (doc wording, grep checks, gate passes, dependency rules) are verified by running them, not by writing tests. The independence rules (enumerate from an authoritative source, allowlist for "remove all") apply only when the agent is the one deciding the list.
- B fires at most once and **cites evidence already produced in the run** (named test, command output, gate result); it never re-runs anything unless a criterion has no evidence. It must not reintroduce the per-criterion re-verification loop that made talos-haiku's first patrol hit MaxIterations; the existing "do not re-verify individual acceptance criteria" line stays, pinned by a test.
- **Regression guard:** tier-1's spec-shaped fixtures (TaskSpec `task.json` through the production template, saturated) must stay at 100% with 0 false dones and 0 MaxIterations, with mean iterations and tokens within a set margin of baseline; haiku is included because it is the model that looped. Then a few supervised real dispatches of groomed GTD items with the knobs on, compared against the perf log, before the talos default flips.

## Production path

Both parts reach `talos run` through the shared template and `RunConfig`, so what we measure is what ships. Ship B behind the default-off knob, flip the talos default after the data, then republish the fleet binary. A is a template change and would ship with the next binary.

## Risks

- The audit induces churn — the model "finds" non-gaps and edits working code, or worse, breaks a green tree. Measured by `audit_followup_edit_file_ok` (real edits) and `audit_followup_bash_ok` (post-audit command re-runs) on tasks that were already resolved, distinct from `audit_followup_red_done` (a post-audit `finish(done)` rejected by a red gate — churn that broke what was green).
- The model rubber-stamps the audit. Measured by `audit_rubber_stamped && audit_reply_text_chars == 0` — a rubber-stamped reply that also carries a written evidence list is the ideal single-turn compliant response, not theatre, so the escalation criterion is the two conditions together, not `audit_rubber_stamped` alone. If the combined signal dominates, the next step is requiring the evidence list in a structured `finish` field the harness can check for completeness against the task's AC list (production TaskSpecs have one).
- Iteration cost. At cap 500 this is budget, not failure; it will show in wall time and tokens.

## Alternatives considered

- **LLM-as-judge reviewer** (`kb-02471`): a second model reviews the diff against the criteria. More powerful, but it is a second model per run, a judge prompt to tune and evaluate, and extra cost on every run. Reserve it for if B fails.
- **Stricter statements**: we control tier-2 statements but not production task specs; the fix has to live in the harness.
- **Finish-recovery changes**: already shown not to be the lever — 0 nudges in 112 trials; stalled trials sit on a red gate (`kb-03240`).

## Decision needed

Revised after the transcripts (2026-09-15). My recommendation now is a single prompt-level change first, measured on its own, because the dominant shape is a context failure the audit cannot see:

1. **A′ (revised A):** extend `test_first_approach.md` with the per-criterion rule for invariants and exhaustiveness, plus the two independence rules: derive enumerations of standard fields from an authoritative source in the environment rather than memory, and prefer allowlists for "remove all of X". One template, both surfaces (production `talos run` and tier-2), no engine change.
2. Measure A′ against baseline on flash + qwen3.8, k=5 on the three false-done-prone tasks plus k=3 on the full 8.
3. Build B (the acceptance audit) only if the invariant shape (rollout-deadlock) survives A′. C is mostly delivered by transcripts; the final-diff capture is a small add-on.

Alternatively approve A′+B together if you would rather spend one overnight run than two. Either way this is your call, since it changes the production prompt.

## Result (2026-09-16): both knobs measured, neither earns a default flip

Decisive run: photoqueue-privacy-metadata only (the dominant false-done shape), qwen3.8:27b at 256K, cap 500, **k=10 per arm**, transcripts on. Primary metric is the **probe rate** — did the agent derive the IPTC field list from an authoritative source (`PIL.IptcImagePlugin.getiptcinfo`, a real file) instead of memory — because that is the behaviour A′ is supposed to induce and it varies far less than resolution.

| arm | resolved | false dones | probe rate | mentions 2:100/2:101 | allowlist language |
|---|---|---|---|---|---|
| baseline | 2/10 | 8 | 0/10 | 0/10 | 0/10 |
| A′ (criteria rules on) | 3/10 | 7 | 0/10 | 0/10 | 0/10 |

**A′ produced no behavioural change on its target shape.** The rule sits in the prompt and the agent does not act on it; 2/10 → 3/10 is noise. **B is equally unproven:** on every trial where an arm looked better (qwen tier-2 24/24 vs 21/24 baseline; haiku tier-1 false dones 2 → 0) `audit_changed_tree` was false — the audit fired, the model wrote a 700-3000 character evidence list, and then made no edits, so it cannot have fixed the work. The tier-1 perfect-spec guard was clean (no pass-rate or cap-hit regression; flash paid 30-40% more tokens for nothing on saturated fixtures), so the features are harmless, just not useful yet.

Process note, the same shape as `kb-03201`: the first read of these arms called the mechanism confirmed from ONE k=3 transcript where the agent happened to probe PIL. The telemetry (`audit_changed_tree`) and the k=10 behavioural metric both refuted it. For a prompt or loop change, measure the behaviour the change should induce before any outcome claim.

Open direction: this shape may be a model-knowledge (context) gap that prompt text cannot close, in which case it is a legitimate routing **discriminator** — glm-5.3 resolves this task where qwen3.8 does not — rather than a harness defect. Anything tried next (a metadata-inspection affordance, an allowlist default, a retrieval step) gets measured probe-rate-first at k=10 before outcomes are quoted.
