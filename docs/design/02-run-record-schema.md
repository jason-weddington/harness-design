# Run Record Schema (v1)

*Drafted 2026-06-22. Status: proposal for review. Resolves research open-decisions
#4 (run-record schema), #5 (cross-window handoff), and #8 (bail-with-report).*

The run record is the **single serializable state the inner loop is a function
of** — the thing the stateless reducer reduces over (12-factor Factor 5 + 12). Get
it right and durability becomes bookkeeping: resume after a host restart is "load
the record, skip completed steps, continue." Get it wrong and you get the failure
class that's miserable to debug — silent state corruption on resume, double-applied
side effects, runs that can't be replayed or audited. Grounded in `docs/research/05`
(durability), LangGraph checkpointing, Temporal event-sourcing, and DBOS.

## Two stores, one source of truth

The design is a **hybrid**: an append-only event log (the source of truth for the
trajectory) plus a materialized state snapshot (a derived cache for O(1) resume).

- **Event log** — append-only, per run: every model call, tool-call-start,
  tool-call-result, phase transition, budget tick, and the final disposition. This
  is the audit trail, the observability feed (the stream tee'd to disk *before*
  parsing), the "readable journal" of a run, **and the substrate for the eval
  flywheel** — every failed dispatch becomes a replayable eval case from its log.
- **State snapshot** — the current reduced state, rewritten after each step.
  Derivable from the log in principle; materialized so resume is a single read,
  not a full replay.

The log is the source of truth; the snapshot is a cache. (The cheaper alternative
— snapshot only, no log — is called out under [Alternatives](#alternatives); I
think the log earns its place in v1 because observability and the eval flywheel are
core to *this* project, not nice-to-haves.)

## The state record

One serializable struct, keyed by a deterministic `run_id`, `schema_version`-tagged
for migration. The critical structural choice is the **split between durable state
and disposable context** — because there are two kinds of resume (below).

```
RunRecord {
  run_id          // deterministic: hash(task_id + attempt_n) — stable across restarts
  schema_version
  attempt_n

  // ---- frozen at dispatch (the seam from GTD) ----
  task            // groomed task snapshot: acceptance criteria, files, scope
  project_config  // toolchain commands run_checks runs; model-routing hint

  // ---- DURABLE STATE (survives a context reset) ----
  phase           // outer control-flow position: Init|Orient|InnerLoop|Checks|Finalize|Done
  durable_facts   // the cross-window carrier: a passes:false acceptance checklist,
                  //   established facts/decisions, "what I've tried" — NOT the transcript
  budgets         // consumed + limits: iterations, tokens, cost, wall_clock_start
  last_gate_result// latest run_checks structured result
  disposition     // None until finish: Done | Blocked(reason) | Failed(retryable) + report

  // ---- DISPOSABLE CONTEXT (scratch; may be dropped/compacted) ----
  messages        // the current model context window; rebuildable, not authoritative

  // pointers, not payloads:
  // the git working tree + filesystem are the ultimate durable state (the code itself).
}
```

The load-bearing idea: **`durable_facts` + git + filesystem are the real
cross-window state; `messages` is scratch.** That split is what lets us do a
fresh-context restart before context rot without losing the task.

## The acceptance checklist — claim vs. verify

`durable_facts` has two parts:

- **`checklist`** — one item per **GTD acceptance criterion**, `{ id, criterion,
  status, evidence }`. The AC items are **immutable**: the agent cannot add,
  delete, or reword them (anti-drift — it must not quietly redefine "done" as
  something easier than what was groomed). If the agent thinks an AC is wrong or
  incomplete, that is a `Blocked` disposition, not a silent edit.
- **`findings`** — append-only free-form memory (established facts, decisions,
  ruled-out approaches). The anti-context-rot carrier: on a fresh-context restart
  the agent reads `findings` + `checklist`, re-orients from git, and skips dead
  ends it already walked.

**Write authority is split — this is the enforcement.** There is **no
`set_verified` tool.** The per-criterion state machine:

```
NotStarted -> InProgress -> ClaimedDone --[harness runs the criterion's check]--> Verified
                                  ^                                                  |
                                  +---------------- check failed --------------------+
```

The agent may move a criterion up to **`ClaimedDone`** (a *claim* — "I think this
is met"). **Only the harness writes `Verified`,** and only after running that
criterion's check and seeing it pass. The agent literally cannot type `Verified`
into the record. `finish(Done)` requires every criterion `Verified`, so a claim
that never passes its check can't graduate to a completed run.

**Evidence is typed, in descending trust:**

- **`Verified(test)`** — a deterministic check passed. Strongest. The check is
  either (a) shipped with the AC ("`test_foo` passes", "clippy clean" — gold
  standard), or (b) a test the agent wrote as part of the work (the soft spot,
  below).
- **`Verified(judge)`** — a calibrated LLM-judge, separate from the builder and
  rubric-based, passed. Probabilistic; flagged as weaker.
- **`ClaimedDone(needs-human)`** — no automatable check exists (genuinely
  subjective AC: "the error message is actionable"). The harness **refuses to
  auto-verify** and leaves it for the outer review. Never silently promoted to
  `Verified`.

**The soft spot, named:** agent-authored tests (case b) can be gamed (`assert
true`). The inner gate alone doesn't catch this; the backstops are *outer* — the
**coverage gate** in `run_checks` (a no-op test fails coverage, so it can't carry
the run to Done) and the **outer review / isolated verifier** reading the diff
*including the test* (the ~21% verification gap a human PR reviewer normally
closes). Per-criterion `Verified` is strong evidence, not proof; the proof is
`Verified(test)` + outer review.

**Honest consequence:** a task with subjective AC means the inner harness **cannot
self-certify pure `Done`** — its best terminal is "all automatable criteria
`Verified(test)`, the rest `ClaimedDone(needs-human)`," handed to the outer review.
That's correct behavior, not a gap.

**Forward-link to grooming:** inner verifiability is bounded by how checkable the
groomed AC are. The fix lives *upstream* — grooming (`groom-to-ready`) should make
each AC mechanically checkable where possible, and explicitly mark the rest
`needs-human`. The outer harness's job grows slightly so the inner harness's "done"
can be honest.

## The `finish` disposition

`finish(status, report)` is the inner harness's output contract to GTD — the
inverse of grooming (groom is intent→spec; disposition is spec→outcome). Three
terminal states:

- **`Done`** — gates green and every criterion `Verified`.
- **`Blocked(needs-decision)`** — the spec or environment is the problem;
  *retrying unchanged cannot help* (ambiguous/contradictory AC, missing
  prerequisite, a call outside the agent's scope, missing access). Resolution needs
  an upstream change.
- **`Failed(retryable)`** — the run is the problem, the spec is fine; *retrying
  might work* (loop, budget exhausted mid-progress, persistent tool error,
  transient infra).

**The discriminator: "does running the same thing again have any chance of
working?"** No → `Blocked`. Maybe → `Failed`. That's what tells the outer harness
what to do next: `Blocked` → route to a human / re-groom; `Failed` → re-dispatch or
escalate (engine/budget), giving up after N attempts.

**Asymmetry of trust:** `finish(Done)` is a *claim that triggers validation*
(run_checks green + all criteria `Verified`); if it doesn't hold, the harness
rejects it and either keeps looping or converts to `Failed`. `Blocked`/`Failed` are
taken at face value — "I can't" is believable in a way "I'm done" is not.

**Report shape (differs by status):**

```
Disposition {
  status,            // Done | Blocked | Failed
  summary,           // short prose
  checklist_final,   // per-AC status + evidence
  budget_spent,
  event_log_ref,     // pointer to the trajectory (also the eval-case seed)
  // Blocked: the specific decision needed — a question a human can answer
  // Failed:  the failure mode — loop | budget_exhausted | persistent_tool_error | transient_infra
}
```

A WIP branch for `Blocked`/`Failed` is the **outer harness's call** in v1 — the
inner harness leaves the tree and reports; GTD decides whether to snapshot it.

Seam mapping: `Done` → complete the GTD item (after the outer commit/push);
`Blocked` → post the question as a comment, move the item to a needs-decision
state; `Failed` → re-dispatch or escalate.

## Two resume modes

A single record, two ways to continue from it:

1. **Crash-resume** (host restart, eviction, sleep) — reload the *entire* record
   including `messages`, reconcile any interrupted step (below), continue exactly
   where we left off. Goal: lose nothing.
2. **Fresh-context restart** (deliberate, before quality degrades ~100–150k tokens,
   or "one feature per window") — **drop `messages`**, keep `durable_facts` +
   `phase` + `budgets`, re-orient from git/filesystem, and continue. Goal: shed
   context rot without losing progress. This is the Anthropic long-running-harness
   / Ralph pattern, made a first-class operation rather than an accident.

Both fall out of the durable/disposable split for free.

## Step boundaries, checkpoint cadence, and the idempotency landmine

- A **step** is one inner-loop iteration (model call → tool exec → append result)
  or one outer-phase transition. We **checkpoint after each step completes** — the
  snapshot is written *before* the next side-effect-causing step begins.
- **The landmine** (from `docs/research/05`): naive replay re-runs the step you
  crashed *inside*. If we crash mid-`edit_file` or mid-`run_command`, resume must
  not double-apply.
- **v1 handling:** the event log records `tool_call_started{seq, name, args}`
  *before* execution and `tool_call_result{seq, ...}` *after*. On resume, a
  `started` with no matching `result` = an interrupted step → re-execute it,
  leaning on **tool idempotency** (full-file writes, git ops, and most build
  commands are naturally re-runnable). The `seq` is the idempotency key.
- **Out of v1:** compensation/saga handlers for genuinely non-idempotent external
  side effects. v1 build tasks are local and re-runnable; we record the started/
  result markers now so the data is there if we need richer reconciliation later.

## Determinism

- `run_id` is deterministic from `(task_id, attempt_n)` so a re-dispatch of the
  same attempt addresses the same record (idempotent dispatch).
- Each event carries a monotonic `seq`; tool side effects key off `seq`.
- Record serialization is **deterministic** (`BTreeMap`/ordered structs) — same
  discipline that keeps the prompt cache hitting.

## Persistence interface (deployment-agnostic)

The store sits behind a trait so an ephemeral-disk deployment (a container, a
Fargate task) can swap durable storage without touching the loop:

```
trait RunStore {
  async fn load(&self, run_id) -> Option<RunRecord>;
  async fn append_event(&self, run_id, event) -> Result<seq>;
  async fn checkpoint(&self, run_id, &RunRecord) -> Result<()>;
}
```

v1 impl: **SQLite** (zero-ops, single-file, single-process) — two tables:
`events(run_id, seq, ts, kind, payload)` append-only, and `runs(run_id,
schema_version, state_blob, updated_at)` holding the latest snapshot. Other impls
(Postgres, an object store, the task tracker itself) slot in behind the trait.

## Alternatives considered

- **Snapshot only, no event log** — simplest; satisfies crash-resume. Rejected for
  v1 because it loses the trajectory, which *this* project needs for failed-run
  observability and the eval flywheel ("every failed run becomes a regression
  case"). If you'd rather YAGNI it, this is the lever to pull.
- **Full event-sourcing (no materialized snapshot)** — purest; state is always a
  replay. Rejected for v1 because it makes *every* resume pay replay cost and puts
  the idempotency landmine on the hot path. The snapshot-as-cache sidesteps both.

## Open questions

- ~~**`durable_facts` shape**~~ — **resolved (research #5):** AC-anchored,
  immutable checklist + append-only `findings`, with the claim-vs-verify mechanism
  above. Cross-window handoff carrier = `findings` + `checklist` + git/filesystem.
- ~~**`finish` disposition schema**~~ — **resolved (research #8):** Done /
  Blocked(needs-decision) / Failed(retryable), discriminated by "can retrying
  unchanged work?"; report shape above.
- **`messages` persistence** — store the full transcript in the snapshot blob, or
  reconstruct it from the event log on crash-resume (keeping the snapshot lean)?
- **Checkpoint granularity** — per step is the default; do any sub-steps (a long
  `run_command`) need intermediate checkpoints, or is step-level enough for v1?
- **LLM-judge in v1?** — the `Verified(judge)` tier needs a calibrated judge; defer
  it (start with `Verified(test)` + `ClaimedDone(needs-human)`), or build a minimal
  judge now? (Lean: defer — judge calibration is its own eval problem.)
