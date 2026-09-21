# somnus — functional spec

*What the nightly map-maintenance loop does, for the session building the crate. Companion to `nightly-map-maintenance-design.md`, which holds the reasoning; this holds the contract. Written 2026-09-20.*

## What it is

One binary, one job: while nobody is working, consolidate the day's KB writes into the mental-map directory. It runs as a timer-invoked subprocess on the host running `kb-service`, reaches the KB only over authed HTTP, and calls the Anthropic API (Sonnet) directly. It is a crate in the `harness-design` workspace consuming `harness` as a library — a tool set and a gate, not a new harness.

Measured cost: **~$5.20 for a full backfill of 25 projects / 1,051 entries, ~$0.15 a night, ~$4.50/month.** No prompt-cache discount on the nightly path, because a 24-hour gap against a 5-minute TTL means every night starts cold.

## The shape, in one paragraph

Code narrows to a bounded set, the model picks within it, code materialises the pick. Rung 0 is a SQL predicate that decides which projects are in scope — no model. Rung 1 is one inference per project that groups entries into subject areas and names them. Rung 2 is one inference per cluster that emits **ops from a closed vocabulary**. Rung 3 is code composing the map body from those ops. The model never writes a map body and never emits a value the code could have computed.

## Rung 0 — eligibility (no model)

`GET /api/kb/map-eligibility` → `{"projects": [verdict, ...]}` — but somnus does NOT read that endpoint in Phase 2, because it is `Depends(require_admin)` while somnus runs as a plain non-admin principal; the loop's actual read is `GET /api/kb/map-loop-input?project_ref=<ref>`, which enforces the eligibility gate server-side per project (404 on an unknown ref, 409 on an ineligible one) and therefore never needs the admin-gated list. A non-admin way to enumerate the eligible refs remains a Phase 3 prerequisite and is tracked as an open question on this item.

Each verdict carries `evidence` (`project_ref`, `mappable`, `ingested`, `hand_authored`, `maps`, `top_prefix`, `top_prefix_share`, `is_ingest_corpus`, `is_too_thin`, `is_journal`, `computed_eligible`), plus `override`, `effective_eligible`, `decided_by`, `orphaned`.

**Work only on `effective_eligible == true`.** As of 2026-09-20 that is 25 of 47 project_refs. The loop **reads overrides and must never write one** — that row is the human's verdict, and a loop that can force its own eligibility has no gate.

## Rung 1 — cluster extraction (one inference per project)

Input is one shipped call: `GET /api/kb/map-loop-input?project_ref=<ref>` (authed like every `/api/kb` route, deliberately NOT admin-gated; GTD item `somnus-loop-input` is the owner and this paragraph is its contract of record). It returns 200 with `{"project_ref", "excerpt_chars", "entries", "maps", "pockets", "pockets_omitted_reason"}`, where `excerpt_chars` is pinned at 600, `entries` carry `{id, short_title, long_title, entry_type, tags, excerpt, details_length, unpointed}`, `maps` carry `{id, short_title, long_title, pointers, body, contributor, updated_by}` (NOT the `MapRef` shape: `body` is the full `knowledge_details` for the read-modify-write materializer and `strike_gap`'s "Not yet documented:" text, and `contributor`/`updated_by` are what make the authorship tier enforceable — no `machine_authored` boolean is computed server-side), and `pockets` carry `{member_entry_ids, mean_similarity, min_similarity, max_similarity, edges: [{a, b, similarity}]}` — no label/name/title/topic field on a pocket, because naming is the model's job and geometry provably cannot do it. Admission is enforced server-side: 404 `project_ref not found` and 409 `project_ref not map-eligible` (human overrides honoured in both directions), so somnus never needs the admin-gated eligibility list. `pockets_omitted_reason` distinguishes "no pockets found" (`None`) from "pockets not computed" (`"too-few-unpointed-entries"`, `"unpointed-set-too-large"`, `"non-postgres-backend"`) — the pair statement is skipped in all three. Where it runs is a deliberate supersession of the design's "what runs where" table line routing pocket detection off the Pi: all distance arithmetic executes in pgvector on the data-DB host ("the Pis are 4-core aarch64 Pi 4s, the data DB is not even on them"), there is no Python-side vector math, and `kb.db.execute` is awaited — what runs on the Pi is row assembly over at most 300 ids and 20,000 pairs. Two divergences are part of the response contract, not bugs: (1) `maps[].pointers` is every `kb-\d{5}` in the map body minus the map's own id, so it MAY contain ids absent from `entries` (cross-project, `NULL`-`project_ref`, or unresolvable — of which 105 exist and are invisible to every per-project query), and a consumer building `entries_by_id[pointer]` must tolerate a missing key; (2) the unpointed anti-join does not constrain the owning map's `project_ref`, so `unpointed == false` does not imply that an owning map appears in `maps`. Pockets come back UNFILTERED by the cluster/decline ledger — this endpoint does not read the ledger, so somnus owns the decline filter and applies it after fetching; a caller that skips that filter re-proposes declined clusters every night and pays Sonnet for them.

Inject that as synthetic tool-call/result events and **delete the read tools from the output union** — 12-factor factor 13. Two consequences: the model cannot wander the KB, and the unit of work becomes one inference rather than an agentic loop. It fits: the largest eligible project packs to ~43k tokens against a 200k window.

The model returns clusters as `{label, member_entry_ids[], owning_map_id | null}`. It may return "this entry belongs nowhere" — forced assignment is a defect, and there is a worked example (an entry sitting in `home-network` with 0.399 max similarity to anything in it).

**Overlap between clusters is expected and correct, not a defect to resolve.** The graph already carries 17 detail entries with two owning maps and 4 with three or four.

## Rung 2 — ops (one inference per cluster)

The model emits ops from a **closed vocabulary** and nothing else:

- `add_pointer(map_id, entry_id, gloss)`
- `create_map(cluster_id, title, orientation_prose)`
- `strike_gap(map_id, gap_text, closing_entry_id)` — requires citing the entry that closed it
- `propose_gap(cluster_id, reason)`
- `no_change(cluster_id)`

There is no `rewrite_body` and no `write_map`. That is the enforcement mechanism, borrowed from talos's own design: there is no `set_verified` tool, so the agent cannot type `Verified` into the record. Here the agent cannot type a map body.

## Rung 3 — code composes the body

The canonical map form has four parts and **code emits three of them**: a coarse `Lives in <package/dir>` line, the `Detail entries:` pointer list, and the prose `Not yet documented:` line. The model writes only the 2–3 sentences of orientation prose and the per-pointer glosses.

This is why the map is **born lint-clean by construction**: every token class the purity lint rejects (file paths, `ENV_VAR` tokens, dotted identifiers, quoted literals, slash-joined paths) can only appear in the three code-composed parts. It also fixes the length budget — the budget is compositional (`900 + 175 × pointer_count`, validated against all 27 live maps), so it scales with pointer count instead of punishing the best-covered maps.

## The write path — `POST /api/kb/map-op`

**One endpoint, machine-principal only, and every op is verified by a set comparison over `kb-XXXXX` refs — never by parsing the map grammar.** This is the contract the run body was blocked on; it was the one seam we specified in neither direction, because loop-input, the cluster ledger and the lint were each obviously read-or-gate and this one is the only genuine write.

The request carries the **whole composed body**, because Rung 3 says somnus's code composes it and the server must not become a second renderer. A server-side composer would have to parse an existing map body to re-render it on `add_pointer`, which contradicts the deliberate grammar-agnosticism of the lint and would mangle the 27 hand-written maps that predate any grammar. So the server never reads the body's structure. It reads the body's **`kb-` refs**, with kb-core's own pointer regex, and that single primitive is enough to verify all four ops.

**Why not just `POST /api/kb/store` with `update_entry_id`, which already works and already hard-lints the machine principal?** Because the design's additive-only invariant is *non-negotiable and substitutes for the revert that a live Postgres cannot offer*, and on the `/store` path nothing enforces it: a body that silently drops half a map's pointers is a valid store. The same goes for the per-night structural caps, which exist to turn a clustering mistake into a one-map event. Both are currently prompt instructions and both become machine-checked here. That is the whole reason this endpoint exists rather than reusing `/store` — not the ergonomics of an op-shaped API.

**403 for any caller who is not the configured machine principal.** The invariants below (exactly one pointer added per call, no pointer ever removed) are correct for an unattended loop and hostile to a human editing a map in the SPA, so humans keep `/store`, where the lint stays advisory.

Every op takes `body` (the full composed map body) and returns the same envelope: `{map_id, version, pointer_count, budget}`. Optional `base_version` on the three update ops: when present and stale, **409**. That is the cheap guard for the one race the design deferred a lease over — a manual run against the timer — and it costs nothing when omitted.

- **`create_map`** — `{op, project_ref, short_title, long_title, body}`. Server asserts ≥1 `kb-` ref (the cardinal rule, already enforced for `mental_map` on `/store`), lints, checks the per-night caps, stores. **201.**
- **`add_pointer`** — `{op, map_id, body, added_entry_id}`. Let `old` and `new` be the ref sets of the stored and submitted bodies. Server asserts `new ⊇ old` (additive-only), `new \ old == {added_entry_id}` (exactly one pointer per call — the closed-op discipline made structural rather than prompted), and that `added_entry_id` resolves to an active entry in the map's own `project_ref`. **200.**
- **`strike_gap`** — `{op, map_id, body, gap_text, closing_entry_id}`. Server asserts `new == old` — a gap op must not move pointers — and that `closing_entry_id` is in `old`, since the spec requires citing the entry that closed the gap and that entry is by definition already pointed at. **200.**
- **`propose_gap`** — `{op, map_id, body, gap_text}`. Server asserts `new == old`. **200.**
- **Both gap ops also assert the op did what it claims, by plain substring and not by parse:** `strike_gap` requires `gap_text` present in the stored body and absent from the submitted one, `propose_gap` the reverse. A substring test is grammar-agnostic in exactly the way the ref-set test is, so this catches a renderer that dropped the gap silently without the server ever learning what a gap line looks like.
- **`no_change`** — never reaches HTTP. No endpoint, no row, no request.

**The per-night caps are enforced here, in SQL, not in the prompt:** one new map per `project_ref` and three per KB, counted as `mental_map` entries created by the machine principal since **UTC midnight**. UTC rather than host-local because three Pis in one fleet must agree on where a night ends, and a rolling 24-hour window is wrong in the one case that matters — a retry just after midnight would inherit yesterday's count. Exceeding either cap is **409** with a machine-readable reason, which somnus treats as an ordinary admission outcome for the rest of the night rather than a fault to retry.

**Status semantics, flat and terminal:** 403 not the machine principal · 404 unknown `map_id` · 409 invariant violation, stale `base_version`, or cap exceeded · 422 lint findings · 200/201 success. Nothing here is retryable, which matches the ledger and loop-input endpoints so somnus keeps one rule for the whole KB surface.

**The one kb-core addition:** a public `map_pointer_ids(text) -> set[str]` beside `count_map_pointers`, so the ref extraction the invariants rest on is the same regex the lint and the budget already use. A second copy of that pattern in the service is exactly the silent drift this pair has now been bitten by twice.

## The gate — leg 2

`POST /api/kb/map-lint`. Shell out as `/bin/sh -c 'curl -sf -X POST ...'`; **the HTTP status is the verdict** — 422 on a failing body, 200 on a clean one, deliberately not a 200 carrying `valid:false`, which `curl -f` would read as green.

**Today: register that tool under the literal name `run_checks`**, because the engine writes `last_gate_green` only from that branch. Treat it as a pinned workaround with an expiry, not as design — an HTTP map-lint is not `run_checks` in any sense its author would pick, and the dependency is on a hardcoded match arm.

The replacement is already scoped as harness-design `49b4445e`: the consumer **declares which registered tool is the gate**, defaulting to `run_checks`, with everything downstream keyed off the declaration — the same defaulted-seam shape as the change observer. Migrate to the declaration when it lands and drop the magic name.

Note *why* the fix is a declaration rather than a looser match, because it constrains what to ask for: `last_gate_green` is deliberately written by that one branch only. A design that sniffed other tools for gate-like behaviour was explicitly rejected, since it would let an agent arm its own finish-recovery by running anything that exits zero.

**The gate is mandatory, not deferrable.** With `checks = None` and a non-git working directory — the configuration this loop otherwise lands in — leg 2 and leg 3 both go inert and a `finish(done)` is accepted on the model's word alone.

## Leg 3 — change evidence

Supply a custom `ChangeObserver` via `RunConfig::with_change_observer`. It returns the project's map-pointer count in the `porcelain` field. The git-flavoured field name is inherited and deliberate; note the reason at the construction site so nobody renames it.

**On observer error, fail closed by returning the cached run-start observation.** `classify_change` compares `porcelain` and `head` by whole-value equality, so the cached value yields `TreeUnchanged` and the finish path rejects — a loop that cannot prove work happened does not get to claim it did. Returning `Unobservable` fails open; a sentinel is worse, because a baseline of `"17"` against `"OBSERVE_FAILED"` differs and manufactures `TreeChanged`. **No field can signal "I failed" without breaking the equality the fail-closed property depends on.** If the baseline itself fails at run start, refuse to start.

Bound `observe` at **10s** — one indexed local count that cannot answer in ten seconds means the KB is unhealthy.

**Log loudly on every fallback and count CONSECUTIVE fallbacks so the log escalates.** A `no_change` rejection does not terminate the run, so a sustained Postgres outage reads in the transcript as "the agent kept claiming done with no changes" while the real cause is invisible. The trait has no error channel by design; this is the only mitigation.

## Dispositions

- `done` — ops applied, gate green, pointer count moved.
- `already_satisfied` — nothing needed changing; requires a reason and an unchanged count.
- `blocked` — **`propose_gap` maps here exactly**: the cluster is real, nothing chunky exists to point at, and no amount of retrying fixes it because it needs an upstream authoring run.
- `failed` — everything else.

Never fabricate `done`. A green gate with no claim terminates `failed` while recording recovery facts.

## State

Exactly one table: the cluster/decline ledger (GTD item `somnus-cluster-ledger`, in flight). Keyed on **member-set overlap, not label**, because labels drift between runs. A decline is permanent with one escape — reopen when the member set doubles.

The run record itself is **local SQLite** on the same host. It is the loop's own state, not KB data, which is what `StoreError` was designed for; only KB writes go over HTTP.

## What it must not do

Never `deactivate` a map (on the HTTP path that path deletes the map's outbound edges, orphaning every detail it owned from the listener's reverse lookup). Never remove a pointer. Never edit prose on a human-authored map — authorship-tiered: the loop owns only the maps it wrote. Never touch `NULL`-`project_ref` entries (105 of them, invisible to every per-project query). Never write a map-eligibility override. Never survey through `/api/kb/get` — it is the only writer of `last_accessed`, simultaneously the sole pull-through read signal and the decay anchor.

## Crate obligations

- `somnus --version` must emit the **same output shape** as `talos --version`; installers parse whitespace field 2 for the token. A different shape turns every update into an unconditional reinstall.
- **No daemon.** Timer-invoked subprocess, so the installer restarts nothing.
- Both `x86_64` and `aarch64` must build; the publisher ships both at one token or fails loudly.
- A `--project <ref>` flag for a single-project run, which is how the falsifying experiment runs.
- `max_nudges` from a **named constant set to 0**, not a literal. With the gate registered as `run_checks`, `last_gate_green` does get set; what stays permanently false is `tree_dirty`, so the staleness predicate is always true and the nudge would arm on the first green gate. Disable it — but behind a flag, because the underlying `last_gate_green` behaviour is filed as a defect and the nudge becomes live for tool sets like this one if that lands.
- **The cost ceiling is NOT this crate's job — it is a harness budget.** `BudgetLimits` in the run record already carries `tokens` and `cost_micros` beside the `wall_clock_secs` that was armed on 2026-09-20; the engine's construction site hardcodes both to 0 (unbounded) and nothing arms them. Arming them is tracked as harness-design `00b5b825`. The right home is the harness, not here: the engine already receives per-turn `Usage` from every backend and already accumulates it, so a consumer-side accumulator re-derives what the engine knows one layer further from the data and has to wrap the loop to terminate it. **Do not pin a consumption expression in this spec** — `Usage::input_tokens` in that library is the *uncached remainder*, not the full prompt, so Anthropic's wire-shape sum is not identity through its type. The harness item owns choosing and documenting the expression. What somnus must do is *set* a per-night limit and surface the termination, not compute the total.

  Worth keeping distinct from a thing Jason already killed: an arbitrary per-turn **output** ceiling truncates a generation mid-flight and makes output bind before context does. A cumulative **run** budget terminates cleanly between turns, exactly as the wall-clock budget does, and truncates nothing. The two collapse into "token caps are bad" very easily.
- No compaction. Per-unit context is ~11k tokens against a 200k window; the highest fill ever observed on real work in that corpus is 80.1%.
- **No git-related startup assertion.** An earlier draft required asserting the working directory is *not* a git repo, on the reasoning that `finish(done)` rejects `TreeUnchanged` and an HTTP-only agent changes no files, so a stray `.git` would make every run burn its budget. **That was true only before the observer seam existed.** With a custom `ChangeObserver` supplied, `observe_tree` is not called on the baseline or on any of the three finish-time observations — git is never consulted, a stray `.git` is inert, and an assertion refusing to start in a directory that happens to be a repo would block a legitimate deployment for nothing.

## Structural caps, per night

One new map per project, three per KB, three eligible projects. Plus the dollar ceiling. The caps convert a clustering mistake into a one-map event rather than a corpus-wide one; the ceiling catches a retry storm at 3am.

**The first two caps are enforced server-side** (`POST /api/kb/map-op`, counted since UTC midnight), so a somnus bug cannot exceed them and somnus need not track them across a crash. The third — three eligible projects per night — is somnus's worklist policy and has no server analogue, since the server sees one request at a time and cannot tell a night's third project from its thirtieth.

**The ceiling is a single harness budget field counting the billed sum** — input + output + cache-read + cache-write, not a split. Measured nightly is 33k in / 3k out = 36k tokens at $0.15. **Nightly: 550,000.** ~15× expected, so tripping it is unambiguously a bug signal rather than a throttle, and it bounds a runaway night at roughly $2. **Backfill: 1,100,000** (~$7, against a measured $3.40 for all 16 projects), which has to be a distinct value on a distinct subcommand: a nightly ceiling loose enough to admit the backfill is not a ceiling, and a backfill run under the nightly ceiling trips on its first night. The measured figures are the $0.15, the 36k and the $3.40; the 15× and 2× headroom multipliers are a judgement call recorded here so a later reader can retune them against real spend rather than re-derive them.

## How we know it worked

Read the three maps it writes. At ~$0.35 for a three-project slice, "build it and look" is cheaper and faster than any proxy. The first gate is a human reading three maps: does each orient without holding facts, does every pointer resolve to something chunky and related. The listener jury telemetry is the follow-on instrument and reads while the loop runs, not before it is built.
