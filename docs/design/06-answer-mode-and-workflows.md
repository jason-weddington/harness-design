# 06 — Answer mode: talos as the sub-agent of a dynamic workflow

Status: **BOTH HALVES SHIPPED (2026-09-18 — library `bce155b`, CLI: GTD 118fa92c).** Answer mode is live in `talos run --mode answer --schema`; the Python client and the first ported workflow remain open.

## The question

Claude Code's `Workflow` tool runs a deterministic JS program that calls `agent(prompt, {schema})` and gets back *data*, with `parallel` and `pipeline` for fan-out. The groom-to-ready and review-against-intent workflows we lean on daily are built on it. Jason's question: could a Claude Code session write something like a workflow that, when run, used **talos** for the sub-agents? Not the JS verbatim — the concept. The sub-agents would run on the Anthropic API or Ollama Cloud, not the local qwen lane, because a single GPU cannot fan out.

The short answer is yes, and the shape is cleaner than expected, because the thing we shipped the day before turns out to be the missing piece.

## What a dynamic workflow is, stripped of the JS

Three properties matter. The orchestrator is a *program*, not a static graph: `review.findings.map(f => agent(verify f))` spawns one verifier per finding, and the number is unknown until the review returns. That is the "dynamic" in dynamic workflow, and it rules out a DAG file as the definition format. The sub-agent returns *structured data* validated against a caller-supplied schema, not prose the orchestrator has to interpret. And fan-out is cheap: sub-agents are independent, so N of them run concurrently.

Everything else — phases, labels, journals — is ergonomics.

## The one missing piece: a second deliverable shape

Talos has exactly one deliverable: a changed workspace. `1ec37b2` made that literal — `Disposition::Done` now requires an observed tree change (leg 3 of the completion contract). A workflow sub-agent that reads code and returns a JSON verdict would be *rejected* by that precondition, correctly, because it changed nothing.

So talos needs **answer mode**: `talos run --mode answer --schema verdict.json`, where `finish` accepts a new disposition, `answer`, carrying a `result` that must validate against the schema. Invalid or absent is bounced back to the model as a tool error, exactly as a red gate or an unchanged tree is bounced today, and the loop continues. The validated `result` rides on `RunSummary` to stdout, where the orchestrator reads it.

This is the same three-legs contract, not a new one. The agent asserts (`finish(answer)`); something mechanical verifies (the schema); the deliverable demonstrably exists (the validated payload). For build mode leg 3 is the diff. For answer mode leg 3 is the payload. Absence or invalidity is a distinct failure, never inferred. One more disposition, one more evidence type, and the design does not change.

There is a satisfying inversion for read-only enforcement. An answer agent must not modify the workspace — twenty of them will share one checkout. Dropping `edit_file` from the registry closes the common path, but `bash` can still write. Rather than trying to sandbox `bash`, reuse the leg-3 observation *inverted*: accepting `finish(answer)` requires the tree to be **unchanged** relative to the run-start baseline. An answer agent that modified the workspace is told so and must revert before its answer is accepted. The same primitive (`observe_tree` + `classify_change`) enforces "you must have changed something" in build mode and "you must not have changed anything" in answer mode. Unobservable still fails open and is recorded.

## The orchestrator: a library, not a runtime

The result already leaves talos as one JSON line on stdout, so `agent()` is a subprocess spawn: prompt on stdin, `--schema` on the command line, `RunSummary.result` parsed off stdout. Fan-out is N processes. A ~100-line Python module gives `agent()`, `parallel()`, and `pipeline()` on top of `asyncio.gather`, and a Claude Code session writes Python as readily as it writes JS. A groom workflow becomes:

```python
draft = await agent("Read the repo and draft a spec for ...", schema=SPEC, model="glm-5.3-flash", workspace=repo)
critiques = await parallel([
    agent(f"Code-grounding critic: {draft}", schema=CRITIQUE, workspace=repo),
    agent(f"Spec-rigor critic: {draft}", schema=CRITIQUE, workspace=repo),
])
final = await agent(f"Synthesize: {draft} {critiques}", schema=SPEC, workspace=repo)
```

I would **not** embed a scripting language in talos to make `talos workflow` self-contained. That is the YAGNI trap for this project: Python calling subprocesses is the right seam until a real limitation shows up. The learning question underneath — does an orchestrator want to be a library your session imports, or a runtime that hosts workflows? — is answered by building the library first and seeing what it cannot do.

## What it buys

**Model choice.** The `Workflow` tool only spawns Claude Code on Anthropic models. Talos sub-agents run on glm-5.3-flash at roughly a tenth of the price, or on Bedrock where the Anthropic API is unreachable. The tool's ≤15-agent guideline exists because of cost; at flash prices a 40-agent workflow is affordable, which changes what is worth writing — a review-against-intent that verifies every acceptance criterion individually, for instance.

**It runs headless.** Today groom-to-ready and review-against-intent exist only inside an open Claude Code session. With answer mode, "dispatch a groom" becomes a real GTD action: the Pi runs it on flash while the laptop is closed. That is the orchestration-failure lens applied to our own tooling — the grooming workflow currently requires the control plane to be awake, and there is no reason it should.

**Every sub-agent run is a first-class record.** Sqlite run record, JSONL transcript, token accounting, the same eval instruments. The `Workflow` tool's journal is a debugging aid; talos's records are the product, and they make a workflow's every step replayable and measurable.

## What it does not buy

Talos's strength is claim-vs-verify against a *mechanical* gate, which is strongest for build tasks where a gate exists. For research agents the only mechanical check is schema validity, which catches malformed output, not wrong output. Answer mode therefore gets cost, reproducibility, and fleet-hosting, but **not** a stronger correctness guarantee than a Claude Code sub-agent. Correctness for research agents comes from the workflow *shape* — critics, refute stages, adversarial verification — which is orchestration, not harness. Worth stating plainly so nobody expects the harness to do the workflow's job.

Talos also has bash, read, and edit, and nothing else: no web search, no MCP. It is a code-and-docs workflow tool, not a general one. For the two workflows we want first, groom-to-ready and review-against-intent, that is fine — both are code and document reading tasks that never touch the web. Jason's call, 2026-09-18: web access is not a blocker for now.

## First cut — the talos-side pieces

Two items, serial, groomed with the groom-to-ready workflow before any build.

1. **`finish(answer)` and `Disposition::Answer`** in the harness crate: a `FinishClaim::Answer { result }` variant; a `--schema`-supplied JSON schema validated at the claim with a `jsonschema`-class validator; rejection back to the model on invalid or missing `result`, same shape as a red gate; `Disposition::Answer { result, verification, change }` carrying the validated payload and the same evidence discipline as `Done`; `result` surfaced on `RunSummary`. Ralph handling decided explicitly, not inherited.
2. **`talos run --mode answer`**: the CLI flag and the `--schema` path; a read-only tool registry (no `edit_file`); the inverted precondition — `finish(answer)` requires `TreeUnchanged`, unobservable fails open and is recorded; an answer-mode system prompt as a distinct template with its own golden test, so the build prompt is untouched and the eval-parity rule gains no new surface; an exit code decided against the dispatch worker's contract.

The Python client and the first ported workflow (groom-to-ready) follow as a third item once these land, along with a decision on where the client lives.

## Open questions for the inquiry

- **Library or runtime?** Answered by building the library first. The tell will be the first thing a Python orchestrator cannot express that a hosted runtime could.
- **Shared checkout semantics.** Twenty answer agents on one checkout, enforced by the inverted precondition, versus one `git worktree` per agent (~100 ms each, `mined_eval` already does this). The precondition catches a modification after the fact; a worktree prevents it. Which one the first real workflow needs is an empirical question.
- **Evaluating research agents.** Tier-1 and tier-2 measure build tasks against a sealed gate. There is no equivalent oracle for a critique or a spec. The honest instrument may be the workflow's own downstream outcome — did the groomed spec one-shot? — which is slow and confounded. Worth thinking about before claiming any research-agent row.
- **Exit code for `Answer`.** Answer mode is not reachable from `build_talos_argv`, so `map_talos_result` never sees it today; but if "dispatch a workflow" arrives, the worker will. Decide it in the item rather than inheriting 0.
- **Do ralph and workflows converge?** Ralph is a single agent in an outer loop toward an oracle; a workflow is many agents in a program toward a schema. They may stay distinct. If a workflow step ever wants "grind until green," that is a ralph call inside a workflow, not a merge of the two.
- **Cost data.** Every sub-agent run records tokens. The first ported groom should produce a real flash-vs-Sonnet cost row for the same workflow, which is the number that decides whether headless grooming is worth routing.

## Sequencing

Groom now (Jason, 2026-09-18). Build after the leg-3 guard finishes, and ideally after the first production transcripts have been read — the same instrument that found the IPTC misconception in tier-2 is what would tell us what research sub-agents actually need from the harness, and it costs nothing to look first.
