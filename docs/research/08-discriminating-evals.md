# Track 08 — Discriminating Evals: how the field builds coding-agent benchmarks, and how to un-saturate ours

*Researched 2026-08-13 via a fan-out/adversarial-verify workflow (23 sources fetched, 99 claims extracted, 25 top claims put through 3-vote adversarial verification: 20 confirmed, 5 refuted). Every surviving finding below rests on primary sources (lab announcements, arXiv papers, benchmark docs) with unanimous 3-0 verification votes. Refuted claims are listed at the end — they are traps this doc exists to keep us out of.*

## Why this track exists

Our 10-fixture suite is **pass-rate-saturated five models deep**: Opus 4.8, Sonnet 5, GLM-5.2, Haiku 4.5, and qwen3.6:35b all score ~100%. The suite still earns its keep as a *routing gate* — it kicked out gpt-oss:20b (46%) and, in Session 14, caught nemotron-3.5-lightning's false-dones (kb-03195) — but it cannot discriminate within the top tier, which is exactly where our engine-routing questions now live (Opus vs Sonnet, GLM vs Opus, talos vs Claude Code). This track surveys how marquee evals are structured, what makes tasks discriminate at the frontier, and how the field evaluates *harnesses* as distinct from models.

## The common spine: dual-test gating on real repo history

The entire SWE-bench family scores one way: a task is **resolved** only if the FAIL_TO_PASS tests (tests that failed before the reference fix) now pass **and** the PASS_TO_PASS tests (the existing suite) still pass, inside a reproducible Docker environment. Tasks are mined from real GitHub issue→PR pairs — the repo's own history supplies the task, the reference solution, and the grading tests. [OpenAI SWE-bench Verified announcement; Scale SWE-bench Pro paper, arXiv 2509.16941]

This is worth internalizing because our fixtures already implement a miniature of it (visible gate + sealed holdout ≈ FAIL_TO_PASS + PASS_TO_PASS with a hidden component), but we *author* tasks while the field *mines* them. Mining from real history is how the marquee benchmarks get difficulty for free — real issues carry the ambiguity, cross-file coupling, and incidental complexity that hand-authored fixtures shed.

## SWE-bench Verified: the human-validation pattern — and what it actually filtered

Verified is a 500-sample subset distilled from the original SWE-bench by a **93-developer annotation campaign** (1,699 samples reviewed, 0–3 severity scale, 2–3 = remove) that filtered out **68.3%** of samples. The two dominant defects: **underspecified problem statements** (38.3% flagged) and **unfair/overly-specific unit tests** (61.1% flagged). [OpenAI, Aug 2024]

Two lessons for us. First, the filter rate is a warning about task *mining*: two-thirds of naively-mined tasks were unfit — if we mine tasks from our own repo history, expect to discard most candidates for the same two defects. Second — and this survived adversarial verification while its popular misreading was refuted — **Verified removed ambiguity, not difficulty**. The claim that Verified filtered out "too hard" tasks went 1-2 in verification and is excluded. The design principle: a discriminating task should be *hard but fairly graded* — difficulty lives in the work, never in the grading.

## SWE-bench Pro: contamination resistance by construction

Pro (1,865 tasks, 41 professional repos) is the current contamination-defense playbook: **731 public tasks from GPL/strong-copyleft repos** (license as a legal deterrent against training-data inclusion), **276 private tasks from 18 proprietary startup codebases**, and an **858-task held-out set** that never ships. Tasks flow through a four-stage pipeline — repo sourcing, Docker environment creation, commit-scraping harvest requiring FAIL_TO_PASS+PASS_TO_PASS transitions, and human-expert augmentation with three human-in-the-loop checkpoints (fixing exactly the underspecification defect Verified surfaced). Same dual-test scoring gate. [Scale, arXiv 2509.16941, late 2025]

Note the discipline required here: the tempting headline numbers about Pro ("frontier drops to ~23%", "public→private score drop proves contamination") were **refuted or unconfirmed** under verification (0-3 and 1-2 respectively). The *structure* is verified; the *leaderboard state* must be refetched live at point of use, never cited from a synthesis — including this one.

## Terminal-Bench: the task shape beyond fix-a-bug

Terminal-Bench 2.0 (arXiv 2601.11868) is 89 tasks in real-workflow-inspired terminal environments — each task a **unique Docker environment + instruction + human-written oracle solution + comprehensive verification tests**. This is the self-contained-sandboxed-environment task shape, one level up from single-file fixes: the agent must operate a *system*, not patch a crate. (The claim that frontier scores <65% on it — i.e., that it's unsaturated — was refuted 0-3; adopt the task shape, refetch the leaderboard.)

## METR's time horizon: the saturation-resistant metric

The single most important idea for our problem. METR's **time-horizon metric** replaces pass-rate-at-fixed-difficulty with **"the human-baseline task duration at which a model hits 50% success"** (logistic fit over task successes, bootstrap CIs over task families/tasks/runs). It resists saturation *structurally*: when models improve, you extend the suite with longer tasks rather than watching a fixed suite pin at 100%. METR did exactly this in Time Horizon 1.1 (Jan 2026): 170→228 tasks, with 8h+ tasks more than doubled (14→31), explicitly "adding more long tasks to prevent saturation as capabilities advance." The metric measures *serial human labor replaceable at 50% success* — not how long the AI runs (AIs usually finish faster than the human baseline). [metr.org TH1.1 blog + limitations note, Jan 2026]

The verified caveats matter as much as the metric: it is **imprecise** (historical error bars ~2× in each direction — METR could not distinguish whether Opus 4.5's true horizon was 3.5h or 6.5h), and it varies by **orders of magnitude across domains** (40–100× lower for visual computer-use than for math). Design consequences for us: duration-based metrics need many tasks and trials for tight CIs, and difficulty comparisons only hold **within one domain** — the same shape as our own "never compare across the think-config knob" rule.

## The harness-evaluation wave: our thesis is now a research field

The strongest external validation of this project's framing. The June 2026 survey **"From Question Answering to Task Completion"** (arXiv 2606.20683) names **bottleneck attribution** — does performance/failure live in the foundation model, the execution harness, or their coupling? — as the central open question in agent evaluation, and decomposes the harness into **six coupled runtime responsibilities: observation, context, control, action, state, and verification**. That taxonomy is a ready-made checklist for what a harness eval must exercise.

**Harness-Bench** (arXiv 2605.27922, Jan 2026) operationalizes it: 106 sandboxed offline tasks across eight workflow categories, run through a setup–execution–judge pipeline that records artifacts, traces, usage stats, and validator outputs — **holding task conditions fixed** (prompts, initial sandbox state, budget, timeout, evaluator) **while varying only the harness**, full-factorial across model backends (106 × 6 harnesses × 8 backends = 5,194 trajectories). This is precisely the experimental design behind our talos-vs-claude-code benchmarks (kb-03078, kb-03102), now an emerging standard.

And the leaderboard-scale evidence that the harness moves the score: across 80 approaches / 178 SWE-bench leaderboard entries (arXiv 2506.17208), **no single architecture consistently wins** — but the *presence* of agentic behavior matters statistically: non-agentic submissions median ~24.7% resolved on Verified vs 54–63% for scaffolded/agentic groups (Kruskal-Wallis p=0.007). Two implications: the harness demonstrably moves the score with the model held constant, and there is no one "correct" architecture — **a harness eval should measure outcomes, not conformance to a prescribed design**.

## What this means for our discriminating tier (synthesis)

The design axes the verified literature supports:

1. **Shift the discriminating axis from pass-rate to difficulty/duration scaling.** A fixed-difficulty suite saturates the moment the weakest frontier model clears it. Build a difficulty *ladder* whose upper rungs are unclimbed, and report "highest rung at ≥50% pass" (a time-horizon-shaped metric) alongside per-rung pass rates. Our iteration counts already discriminate where pass rates don't (frontier ~5 iters vs haiku ~8 vs qwen ~10 on the saturated suite) — the ladder formalizes that.
2. **Mine, don't only author.** Real repo history (ours: harness-design, agent_gtd, personal_kb, dispatch) is a supply of genuinely hard tasks with real reference solutions and real regression suites — the dual-test gate falls out of the git history (the fix commit's tests = FAIL_TO_PASS; the pre-existing suite = PASS_TO_PASS). Budget for a heavy human filter pass (Verified discarded 68%): screen for underspecification and unfair tests, the two dominant defects.
3. **Difficulty in the work, never the grading.** Hard-but-fairly-graded: ambiguity that requires *judgment* belongs in the task (underspecification the agent must resolve sensibly), never in the verifier.
4. **Keep and extend the sealed-holdout discipline.** Our holdout re-gate is a small-scale version of what Pro's held-out set and Harness-Bench's hidden validators do; nemotron's false-dones (kb-03195) proved its value. For mined tasks, contamination defense = private tasks from our own private repos — we get Pro's strongest defense for free, since our repos aren't in anyone's training data.
5. **Statistical humility at small n.** METR sees ~2× error bars at hundreds of tasks; at our scale, treat single-run deltas as noise, keep k≥3, compare only within one domain and one knob-set, and prefer coarse verdicts (rung cleared / not cleared) over decimal pass-rates.
6. **For harness-vs-harness runs, adopt the Harness-Bench frame explicitly:** fix prompts, sandbox, budget, timeout, and evaluator; vary only the harness; record traces and usage. We already do most of this informally — write it into the runner as an invariant.

## Refuted claims (do not cite)

Five plausible-sounding claims failed 3-vote adversarial verification and are deliberately excluded — kept here because they are exactly the kind of thing a future session might "remember" as true:

- "Frontier models drop to ~23% on SWE-bench Pro / named top-entry scores" — **0-3**. Leaderboard state must be refetched live.
- "Public→private score drop (e.g. Opus 4.1 22.7%→17.8%) evidences contamination" — **1-2**, unconfirmed.
- "SWE-bench Verified filtered out tasks for being too hard" — **1-2**. The filter targeted underspecification and unfair tests, not difficulty.
- "Agent Island's winner-take-all competition is structurally saturation-resistant" — **0-3**.
- "Frontier models score below 65% on Terminal-Bench (unsaturated)" — **0-3**.

## Open questions carried forward

1. **Live headroom**: what do SWE-bench Pro and Terminal-Bench leaderboards show *today*, and how much frontier headroom actually remains? (Refetch at point of use.)
2. **The long-horizon task-authoring recipe**: the axis (duration/difficulty) is confirmed, but no verified source gave a repeatable low-cost recipe for synthesizing long-horizon, cross-file-coupled tasks from repo history at small-team scale. This is the design problem for our next session.
3. **Statistical power at small n**: no verified source established the tasks×trials floor for distinguishing two near-frontier harnesses on a ~10–100 task suite.
4. **False-done / reward-hacking methodology depth**: Harness-Bench uses hidden reference artifacts and validator scripts, but canary-string and task-generation contamination defenses were not covered in adoptable depth.

## Primary sources

- OpenAI, *Introducing SWE-bench Verified* (Aug 2024) — openai.com/index/introducing-swe-bench-verified
- Scale, *SWE-bench Pro* (arXiv 2509.16941, late 2025) + labs.scale.com leaderboards (public/private)
- *From Question Answering to Task Completion: A Survey on Agent System and Harness Design* (arXiv 2606.20683, June 2026)
- *Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows* (arXiv 2605.27922, Jan 2026) + harness-bench.ai
- METR, *Time Horizon 1.1* (blog, Jan 2026) + *Time-horizon limitations* (note, Jan 2026)
- *Terminal-Bench 2.0* (arXiv 2601.11868) + tbench.ai
- *Dissecting the SWE-Bench Leaderboards* (arXiv 2506.17208)
- Anthropic Engineering, *Demystifying evals for AI agents*; AWS ML blog, *Evaluating AI agents* (practitioner guidance)
