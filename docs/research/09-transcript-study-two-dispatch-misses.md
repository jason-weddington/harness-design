# 09 — Transcript study: two talos dispatch misses on 2026-09-19

GTD item `050252ba` asked what actually happened in two talos dispatch failures whose control-plane error strings were undiagnosable. Both transcripts were read in full. **Neither failure is the thing its error string said it was**, and the two failures are unrelated to each other: one is an output-token cutoff, the other is a dispatch-worker bug that destroyed a complete, verified, gate-green unit of work.

Sources: `r7-research:/home/dispatch/.local/state/talos/<item-id>/transcript.jsonl`, the GTD `claude_runs` records, and the current `crates/harness` / `agent-gtd-dispatch` code.

## Failure 1 — PhotoQueue stats sweep: a `max_tokens` cutoff, not a stall

**Item** `c0016303` (PhotoQueue, workspace mode: photoqueue + cleanr), talos run `671658619ed3`, `talos-glm` (glm-5.3 on Ollama Cloud, think=high, num_ctx 1,048,576), 2026-09-19T12:18:26Z → 12:24:02Z, exit 20, `outcome=StoppedWithoutFinish`. Perf-log `kb-03337`.

The control-plane text — *"stalled / no convergence — consider re-decomposing into smaller single-subsystem items"* — is wrong on every count. The run never stalled, never looped, never lost the acceptance-criteria list, and never hit a tool error it failed to recover from. `peak_iters_since_tree_change` was 1.

### Where the 21 iterations went

| Iterations | What happened | Output tokens |
|---|---|---|
| 1–14 | Orientation: 20 read-only tool calls over the repo, the installed SDK wheel, the existing tests and the lint/type config | 1,693 total |
| 15 | One planning turn: the whole implementation designed in a single reasoning block | 12,278 |
| 16–20 | AC-1 executed: dependency pin bumped, `uv lock` regenerated, sync verified against the new SDK | 329 total |
| 21 | Planning turn two — cut off | 32,768 (capped) |

Iteration 21 produced a 106,478-character reasoning block drafting the entire new test file, ended mid-thought on the words *"Let me write the file."*, and hit the output cap before it could emit the `edit_file` call. `stop_reason` was `MaxTokens`; output tokens were exactly the configured `max_tokens` of 32,768. The turn contained no tool call, so the engine took the no-tool-call branch and terminated.

One `edit_file` call landed in the whole run. `run_checks` was never called, so the gate never ran and `gates_green_at_exit` was false. Total wall clock 6.5 minutes, of which 148 seconds was the fatal turn and 92 seconds was a single filesystem-wide `find /` in iteration 3 looking for the installed SDK.

### This exact terminal is already fixed — and the run predates the fix by 90 minutes

`5211a15` (2026-09-19 13:48 UTC, shipped in v0.11.0) adds `FailureMode::Truncated` precisely so a `max_tokens` cutoff stops masquerading as `StoppedWithoutFinish`. The run started at 12:18 UTC. The fleet now carries 0.11.0, so a repeat would be labelled honestly.

**But the label is all that changes — the run still dies.** The seam at `crates/harness/src/engine.rs:2714` returns `Truncated` as terminal with `recovery_facts` deliberately `None`, on the stated reasoning that nudging a turn that just exhausted its output budget wastes a model call. That reasoning is sound *for a finish nudge*, which is what the surrounding block does. It does not obviously hold for a truncation-specific re-prompt telling the model to stop planning and take one concrete action now — the discarded 32K tokens were design deliberation, not a partial tool call, and the message history is otherwise intact.

Also note: the dispatch worker never passes `--max-tokens` (`agent-gtd-dispatch/src/agent_gtd_dispatch/talos.py`, `build_talos_argv`), so every fleet talos run uses the 32,768 default.

### The control says the spec is the problem, not the harness

The identical spec was redispatched to `claude-code-glm` 3 minutes later — same model, stronger harness. It also failed: run `a8399964`, 13 minutes, `stopped_without_assertion`, no branch pushed. Neither harness shipped anything for `c0016303`; `git ls-remote` on photoqueue shows no such branch.

Twenty-four acceptance criteria across seven files — five new SQLite tables with literal DDL, five upsert helpers, a paginated sweep job with catch-up, count-gated event fetches, a Redis join, scheduler registration, a pin bump, and tests for each — is a spec wide enough to kill a frontier model in two different harnesses. The re-decomposition advice in the error string is right by accident.

### Second finding in the same header: workspace mode disables leg 3

The run's `tree_baseline` is `Unobservable`, reason `git status --porcelain exited 128: fatal: not a git repository`. In workspace mode the clone root holds `photoqueue/` and `cleanr/` side by side and is not itself a repository, so `observe_tree` cannot read it.

`classify_change` (`crates/harness/src/exec.rs:411`) fails **open** on `Unobservable` — the `done` claim is accepted on trust. So the third leg of the Done contract (`kb-03300`: asserted, verified, work observed) is silently absent on **every workspace-mode talos dispatch**. It is not agent-disableable, which is what the rule forbids, but it is environment-disabled without anything downstream noticing. The run record does carry `tree_baseline_unobservable: true` in stats, so the signal exists and nobody consumes it.

## Failure 2 — flickrasync stats models: the worker deleted a finished run

**Item** `192d4b13` (flickrasync), `talos-glm-flash`, two runs: `1b9ebc32` (12:06:30 → 12:07:30) and `f9b641de` (12:09:23 → 12:11:23). Both recorded `failed` with `error_msg` = `git commit failed:` followed by four `(no files to check) Skipped` hook lines. `kb-03330`, `kb-03331`.

**The talos run succeeded.** The transcript for `1b9ebc32` ends with `disposition: Done`, `change: TreeChanged`, `verification: Checks` passed in 3.9 s — 745 tests, mypy strict clean, 90.02% coverage, exactly the two files the spec allowed. Eighteen iterations, seven `edit_file` calls, the gate run twice inside the loop and once more by the harness to verify the claim.

The work was not lost to a failed edit. At iteration 16 the agent read AC-11, which pins a branch commit subject, reasonably concluded a commit was expected, and ran `git add` + `git commit` itself. It succeeded — commit `362473b` on `feat/192d4b13-…`.

Then the dispatch worker ran its own unconditional stage-and-commit (`agent-gtd-dispatch/src/agent_gtd_dispatch/main.py:1360-1397`): `git add -A` staged nothing because the agent had already committed everything, the pre-commit hooks all reported "(no files to check) Skipped", `git commit` exited non-zero with nothing to commit, and the worker marked the run `failed` and returned **before the push**.

A complete, verified, gate-green commit was sitting on the branch in the clone and was thrown away. Twice. `git ls-remote` on flickrasync confirms the branch never reached origin. `claude-code-glm` then redid the same work in 75 seconds (`kb-03333`).

The trigger is specific and will recur: **a spec that mentions a commit invites the agent to commit, and a talos agent that commits makes the worker's commit fail.** The talos system prompt says nothing about who owns the commit.

### On the error string

The 319-character `error_msg` is not truncated — that is the complete captured output (`git_output_excerpt` keeps 1,500 head characters of a 2,000-character budget). The problem is not width. Git's own reason never appears in it at all, so the operator sees four skipped hooks and cannot distinguish "nothing was staged" from "a hook failed". Widening the excerpt would not have helped; naming the condition would.

## Answers to the item's five questions

1. **Where did the 21 iterations go?** Fourteen orienting, one planning, five executing AC-1 of 24, one truncated. The gate never ran.
2. **Lost context, loop, or unrecovered tool error?** None of the three. An output-token cutoff mid-plan.
3. **Model give-up or recoverable?** Neither — not prose-instead-of-tool-call. A hard `max_tokens` cutoff, already renamed `Truncated` by `5211a15`, still terminal. Whether it *should* be recoverable is an open design call.
4. **Did the flash edits fail to apply or fail to be staged?** Neither. They applied, passed the gate, and were committed by the agent. The worker's redundant commit is what failed.
5. **Should the run record surface more?** Not more characters. The missing pieces are git's own failure reason, and a downstream consumer for `tree_baseline_unobservable`.

## Follow-ups

Unambiguous, captured as items:

- **talos prompt**: state that the harness/worker owns the commit, so the agent stages nothing and commits nothing. Closes failure 2 from talos's side.
- **workspace-mode tree observation**: when the workspace root is not a repository, observe each immediate child that is one and combine, instead of failing open. Restores leg 3 for workspace-mode dispatch.
- **agent-gtd-dispatch** (owned by the peer session, flagged not fixed): between `git add -A` and `git commit`, when nothing is staged but `HEAD` has moved past the base ref, skip the commit and proceed to push rather than failing the run. Name the "nothing staged" condition in the error.

Open design call, for the owner:

- **Is a truncated turn recoverable?** Options are a bounded truncation-specific re-prompt ("your turn was cut off; take one concrete action now, do not plan further"), a larger `--max-tokens` on the dispatch lane, a spec-width check at grooming time, or leaving it terminal and treating spec width as the defect. The control run says spec width is the real defect; the other three are about how loudly and how early that gets said.
