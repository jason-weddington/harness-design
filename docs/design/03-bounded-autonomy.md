# Bounded Autonomy (v0.4.0)

*Drafted 2026-07-11. Status: proposal for review.*

Scope: the loop-internal safety rails that make a run **safe to leave alone** — the
last prerequisite before the harness runs unattended as a real dispatch engine.
Three mechanisms, one of them design-heavy:

1. **Finish-recovery protocol** (the centerpiece) — detect a done-but-not-claimed
   spin, nudge, and *never silently discard work*.
2. **Wall-clock budget** — self-terminate gracefully before the worker's hard
   kill (token/cost caps deferred — see Budget enforcement).
3. **Retry / backoff** — `is_retryable()` already exists; the loop must actually
   use it.

Grounded in the engine as it stands (`crates/harness/src/engine.rs`), the
run-record schema (`docs/design/02`), and the v0.3.6 five-model eval matrix.

## The motivating data

The eval matrix (kb-02971/73/76) put **finish discipline** as the *universal
residual failure* across all five models: the agent does the work, the gates go
green, the sealed holdout goes green — and then it never calls `finish`, runs to
`max_iterations`, and gets recorded as `Failed`. A real `Done`, mislabeled. A
false negative.

The tell that this is a **harness gap, not a model gap**: the same failure shows
up under `claude -p` dispatch, filed as its own task. Two unrelated model families
failing the same way is the signature of a broken workflow layer, not an
incapable model. So the fix belongs in the loop.

The cost asymmetry is the whole argument: **deleting a bad branch is cheap;
throwing away two hours of good work because the agent didn't say "done" is
expensive.** The harness should bias hard toward *preserving* work it isn't sure
about and surfacing it, rather than discarding it on a technicality.

## The finish-recovery protocol

### The invariant it must not break

Claim-vs-verify is the project's spine: `finish(done)` is a *claim* the harness
verifies mechanically (gates green against the ACs); a `Done` carries its evidence
by construction, and **the harness never manufactures a `Done` on its own**.

The naive fix — auto-`finish` the moment gates go green — would break exactly
this: it collapses the claim into the verification and asserts task completion the
model never asserted (gates ⊇ ACs is not guaranteed; the agent may know an AC is
unmet that no gate covers).

**The resolution: don't fabricate the claim in the harness — move it up to the
lead.** The harness detects the spin, tries to get the *model* to claim, and if
that fails, **preserves the work and surfaces it with everything the lead session
needs to make the finish call itself.** Claim-vs-verify stays inviolate; the work
survives. The claim just happens one level up.

### Detection — the high-precision trip

The trip condition is deliberately narrow, because a false trip is itself a
CI-invisible failure (a nudge fired mid-thought can derail a legitimately-long
run, and the derailed-but-plausible result looks fine):

> **Trip when: the last `run_checks` was green AND the working tree has not
> changed for K iterations.**

This is high precision by construction. A spinner that's still doing real work
mutates the tree — no trip. A task that legitimately needs many iterations hasn't
gone green yet — no trip. *Green-and-still-going with a static tree* is the
narrow case that is almost certainly done; that's the only thing we poke.

**Red gates + spinning is explicitly out of the nudge path.** It's a
lower-precision signal (genuinely stuck vs. merely slow are hard to tell apart)
and the downside of a false nudge is higher. In v0.4.0 that case falls through to
the budget caps below — no nudge, no special handling.

### Nudge — force a claim or a reason

On trip, inject a synthetic turn (not a silent kill):

> *"The quality gates are currently green. If the acceptance criteria are met,
> call `finish(done)` now. If they are not yet met, reply with a one-sentence
> status: what remains, and why you are still working."*

Forcing the one-sentence status is the load-bearing detail. It either jolts the
model into the `finish` it forgot, or it extracts a *real reason* the run isn't
done — and that reason is telemetry we do not currently have. Every nudge-status
is captured in the run record as a first-class field (see [Schema](#schema-and-seam-changes)).
Over many runs these statuses tell us *why* finish discipline fails — an AC the
model believes is uncovered? re-verification it can't stop doing? polishing? —
which feeds model-routing and prompt work.

Nudges are **bounded at N**. Each trip re-checks the condition (the tree may have
moved between nudges, resetting the counter).

### Terminal — `Failed`, but recoverable

If the agent still hasn't called `finish` after N nudges, the run terminates
`Failed`. The disposition is honest — the agent *did* fail to claim victory — but
the disposition **report** carries the recovery facts:

- `gates_green_at_exit: bool` — was the last gate run green when we gave up?
- `tree_dirty: bool` — is there uncommitted work to preserve?
- `nudge_statuses: Vec<String>` — the one-sentence reasons captured above.

The harness does not claim `Done`; it hands the lead a labeled, evidence-bearing
"this is probably done, here's why, here's the tree" package.

### The seam — harness reports, worker preserves

Git writes are not a harness tool (`docs/design/01`: the worker owns
commit/push; the inner loop only mutates the working tree). So recovery is two
halves:

- **Harness half (this repo, v0.4.0):** detection, nudge, and the recovery facts
  on the run record. That's all designed and built here.
- **Worker half (agent-gtd-dispatch, separate track):** on a `Failed` disposition
  whose report says `gates_green_at_exit && tree_dirty`, the worker pushes a WIP
  feature branch and comments the `nudge_statuses` back on the GTD item — even
  though the exit code is non-zero. The lead sees a branch + a status and decides:
  merge it, or delete it (cheap).

**No new exit code.** The exit code stays coarse (`0` Done / `10` Blocked / `20`
Failed / `1` infra). The recovery signal rides in the run record the worker
already reads. This keeps the disposition→exit-code map (ratified in 0.3.5)
untouched.

## Budget enforcement — wall-clock only (as shipped)

> **Scope narrowed during the wave (2026-07-11).** The original plan below was
> "enforce all four caps." It shipped as **wall-clock only**: `iterations` was
> already enforced pre-0.4.0, wall-clock landed here, and **`tokens` / `cost` are
> deferred** — token caps are inscrutable (no human-legible right value per task)
> and cost caps are blocked on a token→price table that doesn't exist
> (`consumed.cost_micros` is never incremented). The prod `BudgetLimits` literal
> still hardcodes `tokens: 0, cost_micros: 0` (unenforced). See the roadmap
> backlog and the deferred-caps follow-up. This section is kept as the record of
> the design; the paragraph below describes the wall-clock cap that landed.

The types are already in the schema (`run_record::BudgetLimits { iterations,
tokens, cost_micros }`, plus a new `wall_clock_secs`), and the loop already
*ticks* `consumed` every iteration via `BudgetTick` events. v0.4.0 adds the
**wall-clock** check: when `wall_clock_secs != 0 && elapsed >= wall_clock_secs`
the run terminates `Failed { mode: BudgetExhausted }` with a summary naming the
bound (`"wall-clock budget exhausted"`). The check runs at the same
end-of-iteration point where `consumed` is updated. Token and cost enforcement
would slot in at the same point (reusing `consumed.tokens` / `cost_micros`) if
and when they're built.

Wall-clock uses `Budgets::wall_clock_start` (already persisted). On resume the
consumed budgets carry over (0.3.0 already does this for accounting) — so a
budget is a whole-run bound, not a per-attempt one, and a run can't dodge its cap
by crashing and resuming.

## Retry / backoff

`BackendError::is_retryable()` returns `true` only for `Transient`, and it exists
today — but the loop never calls it. A transient blip currently goes straight to
`Failed { TransientInfra }` on first occurrence (`engine.rs` says so explicitly).

v0.4.0 wraps the backend turn in a bounded retry: on a retryable error, back off
and retry up to R times before giving up. Only after exhausting retries does the
run terminate `Failed { TransientInfra }`. Backoff is deterministic (no
wall-clock jitter that would break replay — a fixed schedule, or jitter seeded
from run id + attempt).

**`ContextLengthExceeded` is a separate path, likely deferred.** Its variant doc
promises "the loop detects it, mutates the request (prune/compact), and tries
again" — that's real context-engineering work (what to prune, how to compact),
not the same shape as transient-retry, and the v0.2.0 pre-flight context guard
already keeps us off that rock in practice. Flag it here; groom it separately;
probably not v0.4.0.

## Schema and seam changes

- **Recovery facts** on `DispositionReport` (or a nested `RecoveryFacts`):
  `gates_green_at_exit`, `tree_dirty`, `nudge_statuses`. This is the worker's
  read surface for the WIP-preserve decision. A schema bump (v2 → v3) unless we
  can add them backward-compatibly with `#[serde(default)]`.
- **`FailureMode` for the finish-discipline case:** `Loop` fits ("looped / made no
  progress") and is already defined but never produced — v0.4.0 is its first
  producer. Whether the green-gates-spin deserves its own mode vs. reusing `Loop`
  (with the recovery facts doing the real discrimination) is a grooming call; the
  facts, not the label, are what the worker keys on.

## Locked decisions

- Detection = **green gates + no tree delta for K iterations**. High precision by
  construction.
- **Red-gates + spinning is not nudged** — it falls to the budget caps.
- Nudge forces **`finish(done)` or a one-sentence status**; bounded at N;
  statuses captured as first-class telemetry.
- The harness **never fabricates `Done`** — the claim moves up to the lead;
  claim-vs-verify is untouched.
- **No new exit code** — recovery facts ride in the run record; the worker reads
  them and pushes the WIP branch.
- Budget enforcement ships **wall-clock only** (`BudgetExhausted` names the
  bound); `iterations` was already enforced; **token/cost deferred** (inscrutable
  / no pricing table — see the Budget-enforcement note and the roadmap backlog).
- Retry is **bounded + deterministic backoff** on `is_retryable()` only.

## Open questions (to resolve in grooming)

- **K and N** — start from a guess (K≈3 static-tree iterations, N≈2 nudges) and
  tune against real run data; record both on every eval/dispatch row.
- **`FailureMode` labeling** — reuse `Loop` or add a finish-specific mode?
- **Recovery-facts placement** — extend `DispositionReport` in place (with serde
  defaults, no version bump) vs. a `RecoveryFacts` sub-struct + schema v3.
- **`ContextLengthExceeded`** — confirm it's deferred out of v0.4.0.
- **Retry R + backoff schedule** — count and shape; how it interacts with the
  wall-clock budget (a retry's backoff spends wall-clock a paused run doesn't).
