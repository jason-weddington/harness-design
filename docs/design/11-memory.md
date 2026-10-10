# 11 — Memory as a harness feature

Status: decided 2026-10-10 (Jason, with the personal-kb lead session). Phase 1 is being built; phases 2–3 wait on KB-side contracts noted below.

## Decision

Talos consumes the personal-kb experience loop natively, inside the engine loop, not through MCP or hooks. Claude Code can only reach the KB through hooks (session start, PreToolUse deny, Stop transcript digest) and an MCP server; talos owns the tool dispatch, the context window, compaction and its own verdicts on the agent's work, so memory can be a first-class harness feature with more control than a hook can have.

## Why the harness and not hooks + MCP

- **Delivery at the moment of use, at no turn cost.** The Claude Code soft gate can only deny a Bash command matched on a two-word target class, and kb-bench measured that a deny on top of the session slice "adds nothing and costs a turn" (KB kb-03673). Talos sees every tool call before it runs and can attach the relevant fact to that tool's result instead, or deny once when the evidence says denying is better.
- **Memory that survives compaction.** Talos owns the compaction tiers (design 08), so delivered facts can be protected the way the system prompt and task spec already are.
- **Capture without a human.** Headless runs never produce the user corrections the KB's shape-2 capture relies on, but talos corrects its own agent all the time: a gate that goes red then green, a `finish(done)` rejected by claim verification, a no-change rejection, a nudge. Those are recorded with full reasoning, and they are the headless analogue of a user correction with ground truth attached.
- **It is measurable.** The Harbor `TalosAgent` is validated (design 09), so a talos arm in kb-bench runs on the same images and scenarios as the Claude Code arm.

## Decisions of record

- **Seam:** a generic `Memory` seam on `RunConfig`, a no-op by default, with one adapter (personal-kb over HTTP). Nothing speculative beyond that one adapter.
- **Delivery mode:** annotate the tool result first; deny-once exists as a separate mode so kb-bench can run it as its own arm. Release only after kb-bench shows the feature helps, ideally in deny-once mode.
- **Default off** (`talos run --memory off|annotate|deny-once`), per this repo's rule that harness features ship default-off and flip only after tier-2 shows benefit and the tier-1 perfect-spec guard stays clean.
- **Eval isolation:** tier-2 and kb-bench runs use the seeded offline KB or memory off, never the live KB, so it cannot leak facts about benchmark tasks into runs.
- **Relevance lives in the KB service,** shared by both harnesses. Talos keeps a local first stage (the cached index plus a cheap prefilter) so most tool calls touch no network, and asks the KB only about the rest, failing open at 1 s.
- **Agent access:** native read tools (`kb_search`, `kb_get`, `kb_ask`) when memory is on. Agents never store directly: a `kb_note` tool proposes a lesson, submitted at run end through the KB's capture pipeline (distiller, critic, 30-day expiry unless re-observed), so automated quality control applies to agent writes too.
- **Clef:** the KB lead is measuring clef-flash against the lexical matcher on real commands, with a pre-registered rule (higher recall at no worse precision, p95 under 1 s). If it wins, it runs behind the KB's per-call advice endpoint, not inside talos.

## Phases

1. **Read path (building now).** The `Memory` seam and engine wiring (session slice into the system prompt, `before_tool` advice applied as annotate or deny-once, transcript telemetry for every decision), the personal-kb adapter (`GET /api/kb/prevention` at run start, cached index, a port of the hook's lexical matcher), the read tools, the `--memory` flag and a `TalosAgent` kwarg. Measured as a talos arm in kb-bench.
2. **KB-side relevance.** Switch the second stage to the KB's per-call advice endpoint once the clef-vs-lexical eval decides its shape.
3. **Native capture.** Turn digests to `POST /api/kb/turn` with `harness: "talos"`, carrying `reasoning`, `harness_correction` (trigger `gate_red | claim_rejected | no_change | nudge`, detail, `resolved_by` tool-use ids, `resolved`) and `agent_note` (claim, evidence refs) items, plus the `kb_note` tool. Shadow first on the live KB, because fleet runs would write unattended.

## KB-side contracts (agreed with the KB lead, owned on the personal-kb board)

Talos sends its native tool names with `harness: "talos"` and the KB maps them to its canonical names (`bash`→Bash, `edit_file`→Edit, `write_file`→Write, `read_file`→Read, `list_files`→Glob; `run_checks` stays its own canonical tool). Until that map lands, the prevention index is Bash-only, so phase 1 only advises talos `bash` calls. The digest gains optional `reasoning`, `harness_correction` and `agent_note` item kinds; the KB treats `harness_correction` as a deterministic capture shape and `agent_note` grounding as observed only when every evidence ref resolves to a tool result in the same digest.
