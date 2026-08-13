# Design 04 — The discriminating eval tier (tier-2: mined-from-history tasks)

*Drafted 2026-08-13. Status: **proposal — open decisions at the end.** Inputs: research track [08-discriminating-evals](../research/08-discriminating-evals.md) and the six-repo feasibility probe (kb-03197). Sessions: kb-03196.*

## Problem

The 10-fixture suite (tier-1) is pass-rate-saturated five models deep: Opus 4.8, Sonnet 5, GLM-5.2, Haiku 4.5, and qwen3.6:35b all score ~100%. It remains valuable as a fast **routing gate** (it kicked out gpt-oss:20b and caught nemotron's false-dones) but cannot discriminate within the top tier — which is where our engine-routing questions now live (Opus vs Sonnet, GLM vs Opus, talos vs Claude Code).

## Feasibility (probed 2026-08-13)

Six parallel agents probed photoqueue, flickrasync, agent_gtd, grit-mile, cleanr, and this repo as task supply. Verdict: **~250–350 viable candidates, dozens genuinely frontier-challenging, all six repos hermetic enough for offline sandboxed eval.** Squash-merge discipline means one main commit = one complete feature/fix + its tests = one mineable task: the commit's new tests are FAIL_TO_PASS, the pre-existing suite is PASS_TO_PASS — the SWE-bench dual-test gate falls out of our own git history.

| repo | viable | ceiling | obstacle |
|---|---|---|---|
| grit-mile | 80–150 | high | own eval harness + `it.fails` PBTs poison workspace-level PASS_TO_PASS — scope to `packages/core`/`shared`, per-file baselines |
| agent_gtd | 80–120 | high | 5–10 min suite (bcrypt/JWT) — mine at test-file granularity; exclude 2 Postgres-only test files |
| photoqueue | 30–40 | high | commit subjects leak the fix — statement sanitization required |
| harness-design | 30–40 | high | **self-referential** — exclude engine-loop + eval-fixture commits, or score via the other harness |
| cleanr | ~20 | mid | torch/fastai = multi-GB provisioning (runtime is hermetic, 12.6s suite) |
| flickrasync | 10–15 quality | mid | mirror-heavy; best bug bundled in a 27-file lint sweep |

Pilot shortlist (hard rungs): grit-mile `323485d` (modality-aware ramp check), `3c724c6` (frequency-aware proportion), `183c9c4` (run-day budget cap); agent_gtd `d0a19947` (rollout FSM deadlock), `0b887eec` (dispatch attribution); photoqueue `6749b4e44` (IPTC/XMP privacy stripping), `9bda7228f` (Flickr `ignored`-flag). Mid/easy floor: cleanr `8b397857`/`2a26268d`, agent_gtd `450ff34e`, grit-mile `ed49e37`/`0ac61e5`, flickrasync `8e8519c12`, photoqueue `53f7637f1`. Full detail + runner-up bench: kb-03197.

## Proposed design

**Metric — a difficulty ladder, not a pass rate.** Tasks are grouped into rungs; the headline number per engine is *highest rung cleared at ≥50% pass* (a METR-time-horizon-shaped metric), with per-rung pass rates and iteration/token counts beneath. Rung membership is assigned **empirically** — initial placement from the probe's easy/mid/hard guess, re-ranked by observed pass rates once the matrix runs — never a priori. The ladder cannot saturate: when the top rung is cleared, we mine a harder rung; the suite grows upward instead of pinning at 100%. Tier-1 stays as-is: the fast, cheap routing gate every new model runs first.

**Task shape — SWE-bench-style, adapted to our TaskSpec.** A mined task is: (a) a workspace snapshot of the repo at the fix commit's parent; (b) a sanitized task statement (desired behavior, not the fix); (c) the mined FAIL_TO_PASS tests, **withheld from the agent and applied at scoring time** — they are the sealed holdout, same discipline that caught nemotron's false-dones; (d) a scoped PASS_TO_PASS baseline (specific test files/packages verified green at the parent commit — never the whole workspace). The agent sees the repo and the statement; it is free to write its own tests; scoring applies the mined tests plus the baseline.

**Task-statement synthesis — the craft step, run as a pipeline.** Draft from commit message + diff + (for agent_gtd) the original groomed GTD spec recovered via the item-id in the commit subject; sanitize so the statement describes behavior without naming the fix; critic pass screening for SWE-bench Verified's two dominant defects (underspecified statement, unfair/overly-specific tests — Verified discarded 68% of naively-mined tasks for these). Agent-driven with human spot-check on the pilot; difficulty lives in the work, never in the grading.

**Environments — pre-provisioned per-repo, snapshot per-task.** One cached base environment per repo (venv/node_modules/target with deps installed — this amortizes cleanr's torch pull and grit-mile's pnpm install); per-task setup is `git worktree` at the parent commit + copy-in of the cached deps. Runs on this box (the 5090 host) initially; dispatch-host portability is out of scope for the pilot.

**Contamination hygiene.** Private repos are contamination-proof by construction (SWE-bench Pro pays for this with GPL sourcing and legal agreements; we get it free). Two exceptions enforced at mining time: harness-design engine-loop and eval-fixture commits are excluded (or scored only via `claude_code_eval`, the non-talos harness); grit-mile's embedded eval harness and `it.fails` property tests are excluded from every given-test set and baseline.

## Open decisions (for the design conversation)

1. **Hidden vs visible FAIL_TO_PASS.** Proposal says hidden (SWE-bench shape; prevents test-gaming; reuses our sealed-holdout discipline). The alternative — visible failing tests like tier-1 — makes tasks easier and more like real dispatch (where AC are visible). Could split: statement carries the *behavioral* AC, tests stay hidden.
2. **Pilot scope.** Proposal: hand-mine ~8 tasks (2 easy, 3 mid, 3 hard from the shortlist), run the 5-engine matrix (opus / sonnet / glm / haiku / qwen, k=3), and check the one thing that matters: **does the ladder separate Opus from Sonnet from GLM?** Only after discrimination is proven do we automate the mining pipeline.
3. **Harness-vs-harness runs.** The same mined tasks serve talos-vs-claude-code comparisons (Harness-Bench frame: fix prompts/sandbox/budget/evaluator, vary harness). Include in the pilot matrix or defer?
4. **Where task artifacts live.** A new `evals/` repo (keeps benchmark content out of the harness repo and away from dispatch agents' eyes) vs `fixtures-v2/` here. The self-reference probe argues for a separate repo.
