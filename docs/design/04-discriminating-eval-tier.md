# Design 04 — The discriminating eval tier (tier-2: mined-from-history tasks)

*Drafted 2026-08-13; decisions ratified 2026-08-14. Status: **design of record — pilot next.** Inputs: research track [08-discriminating-evals](../research/08-discriminating-evals.md) and the six-repo feasibility probe (kb-03197). Sessions: kb-03196.*

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

**Spec-level ladder — the resolution of the known-solvable problem (decided 2026-08-14).** Jason's objection: nearly every mined commit was *built by headless dispatch with a detailed groomed spec* — so we already know Opus/Sonnet + CC + full spec solves them, and reproducing that setup cannot discriminate. Resolution: **a mined commit is a task family parameterized by how much of the grooming the statement reveals.** The known-solvable tuple is (task + full spec); the discrimination axis is withholding the grooming. Three spec levels per task:

- **S3 — full groomed spec** (AC + files_to_modify + scope + pinned constants). Known-solvable by construction; kept only as the *calibration floor* — an engine failing S3 is below the dispatch bar.
- **S2 — AC only.** Behavioral acceptance criteria, zero localization (no file paths, no component names, no constants). Strips exactly what the groom workflow's code-grounding critic contributes. The standard tier-2 statement.
- **S1 — intent only.** The pre-groom artifact: the inbox-capture seed, or for bugs the *symptom* with no cause stated. The agent grooms AND builds — disambiguate, scope, localize, root-cause, implement. Never been done headless; unknown even for Opus. (This mirrors SWE-bench, whose statements are the raw *issue text* — the pre-groom artifact — with the 68% discard as the warning that raw intent needs an underspecification screen to stay fairly gradable.)

Above S1 sits the **wave composite**: a multi-item rollout (groomed, DAG-ordered, manager-merged) collapsed into one task at wave-intent level, graded by the union of the wave's mined tests. Single items are known-solvable; whole features by one agent in one context are not — this is the long-horizon top rung, where frontier-vs-frontier discrimination most plausibly lives.

Two side benefits: **the eval measures the value of grooming itself** — score-vs-spec-level per engine yields the routing rubric's missing axis ("how much spec does this engine need?") as data; and **the dispatch record supplies per-task baselines free** (original engine, wall-clock, iterations, lead interventions from the perf log — tasks that needed redispatch/inline fixes are pre-labeled as empirically harder).

**Environments — pre-provisioned per-repo, snapshot per-task.** One cached base environment per repo (venv/node_modules/target with deps installed — this amortizes cleanr's torch pull and grit-mile's pnpm install); per-task setup is `git worktree` at the parent commit + copy-in of the cached deps. Runs on this box (the 5090 host) initially; dispatch-host portability is out of scope for the pilot.

**Contamination hygiene.** Private repos are contamination-proof by construction (SWE-bench Pro pays for this with GPL sourcing and legal agreements; we get it free). Two exceptions enforced at mining time: harness-design engine-loop and eval-fixture commits are excluded (or scored only via `claude_code_eval`, the non-talos harness); grit-mile's embedded eval harness and `it.fails` property tests are excluded from every given-test set and baseline.

## Decisions of record (ratified 2026-08-14)

1. **Hidden FAIL_TO_PASS** — mined tests are the sealed grader; the statement carries behavioral AC (at the spec level the rung dictates). Prevents test-gaming; reuses the holdout discipline that caught nemotron's false-dones.
2. **Pilot before automation.** Hand-mine ~8 tasks (2 easy / 3 mid / 3 hard from the kb-03197 shortlist) and sample the *spec grid*, not just the tasks: the 8 at S2, 2-3 of them re-run at S1, plus one wave composite. Run the 5-engine matrix (opus / sonnet / glm / haiku / qwen, k=3). The pilot answers: does spec-withholding discriminate, is S1 fair or just noisy, and are composites completable at all? Only after discrimination is proven do we automate mining.
3. **Harness-vs-harness deferred.** Once the ladder discriminates models on talos, the same tasks serve the talos-vs-claude-code comparison (Harness-Bench frame: fix prompts/sandbox/budget/evaluator, vary harness) as a follow-up matrix.
4. **Separate `evals` repo** for task artifacts (snapshots, statements, sealed tests). Keeps benchmark content out of dispatch agents' reach (the csv-ledger answer-key incident), keeps five other repos' code out of this repo's history, and absorbs continuous rung growth. The harness repo keeps the runner.
5. **grit-mile code may be used in eval tasks** (Jason, 2026-08-14) — same private trust boundary.

## Count-check arming and -q sealed gates (2026-08-20 grooming of 0595137d; ratified 2026-09-18)

**Mechanism.** A sealed `gate_command` ending in `-q` puts pytest at verbosity -1, where the trailing stats line (`N failed, M passed in 4.46s`) is written WITHOUT `===` padding, so `parse_pytest_summary_totals` returns `None` and `summary_count` stays `None` — while the injected `PYTEST_ADDOPTS=-rA` still produces the `short test summary info` section, so ids parse normally and the trial scores as a normal (never `parse-empty`) result. The count-mismatch cross-check (`resolve` step 2) therefore never fires: it is not armed, not agreeing. This is surfaced per trial as `count_check=off` on the trial line and aggregated as the `cnt_off` summary column (`MinedReport::count_check_off`).

**Cancellation.** A repo's own `[tool.pytest.ini_options] addopts` containing `-v` nets verbosity back to 0 and restores the `===`-padded stats banner — the cross-check arms itself and `cnt_off` returns to 0 for that task.

**Where it bites today.** The count-check is OFF for exactly the two tier-2 tasks whose repo lacks `-v` at the pinned parent commit: `cleanr-comment-filters` (parent bf19b47) and `cleanr-owner-comments` (parent 78c7358) — cleanr carries `addopts = "-m 'not integration'"` with no `-v`. The other six keep it ON: agent_gtd ×3 at 6ca92d3 / 604a66b / aec3f15 with `addopts = "--strict-markers -v"`; flickrasync at d5fbce9 with `"-v -m 'not integration'"`; photoqueue ×2 at e288acb / eedafb1 with `"--strict-markers -v"`.

**Recommendation.** Drop `-q` from the `gate_command` of those two `task.json` files in the separate talos-evals repo, so every task emits the padded banner and the tripwire arms uniformly — this is explicitly OUT OF SCOPE for the harness repo (cross-repo data change; it also alters the agent-visible gate command and therefore the task itself, so it needs its own grooming and a before/after score comparison on the captured corpus).

**No runtime rewrite.** The harness deliberately does NOT rewrite a sealed gate command at run time; the sealed command is provenance of record, and silently un-sealing it would change what the gate measured.

**Diagnostic pattern.** For the current eight-task ladder at k trials, expect the `cnt_off` column to read `k` on `cleanr-comment-filters` and `cleanr-owner-comments` and `0` on the other six. ANY other pattern (nonzero on a `-v` task, or zero on a cleanr task before its `task.json` drops `-q`) means either `parse_pytest_summary_totals` regressed or a repo's pinned `addopts` changed — open that trial's `gate-output.txt` (path already printed on every trial line as `gate_output:`) to tell which.
