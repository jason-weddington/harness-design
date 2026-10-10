# Wire contracts

## What this is

Golden request/response fixtures for the two endpoints a harness calls, `GET /api/kb/prevention` and `POST /api/kb/turn`. They are used today by the Claude Code hook and are intended for talos to vendor (adoption is a harness-design item), so a schema change breaks the other side's tests instead of drifting silently.

## File format

`prevention.json` and `turn.json` each carry the top-level keys `contract`, `contract_version`, `generated_by`, `method`, `path`, `request` and `response`; `turn.json` also has `request.max_body_bytes`.

Examples are listed in `examples` arrays: request and response examples are `{name, body}` envelopes, and the prevention request examples are `{name, query}` envelopes.

The schemas are pydantic `model_json_schema()` output (JSON Schema draft 2020-12) for `PreventionResponse`, `TurnDigestRequest` and `TurnDigestResponse`, including the docstring-derived `description` and `title` keys.

`request.query_schema` in `prevention.json` is derived from FastAPI's OpenAPI parameters and carries `additionalProperties: false`; body schemas do not, because pydantic silently drops unknown body keys.

## Invariants a schema cannot express

POST /api/kb/turn checks in this order: 422 (schema-invalid body), then 413 (raw request bytes above `max_body_bytes`, whitespace included, in any capture mode), then 200 with a `reason`; a DB failure is reason `write-failed` at 200.

`event_id` must equal `<session_id>:<turn_index>`, otherwise the answer is 422.

Unknown body keys are dropped silently, so a consumer must assert exact key sets per `$defs`, as `test_contracts_hook.py` does, and may omit an optional field only by listing it explicitly (`HOOK_OMITTED_REQUEST_FIELDS`).

GET /api/kb/prevention resolves the project as the non-blank `project` stripped, else the `cwd` basename lowercased with `_` turned into `-` and any dispatch run-id suffix stripped, else `""` (empty index and slice; gate settings and `surprise_capture` are still env-driven).

GET /api/kb/prevention never answers 5xx; on failure it returns an inert body (gate enabled false, shadow true, empty index and slice, zero diagnostics) that a client cannot tell from an empty result.

`harness=talos` returns `tool_map` (native name to canonical name) while the index stays in canonical Claude Code names, so a consumer translates its native names through `tool_map` before matching; without `harness`, `tool_map` is `{}`.

`gate.enabled` and `gate.shadow` reflect only the Claude Code gate switch, and a mapped harness receives the index regardless and applies its own mode.

An absent `tool_map` key means a server older than tool-name normalization, so the harness must not post native-named digests to it.

Turn `event_id`s must be unique per session within the server's turn-digest retention (90 days).

Undeclared query params are silently ignored, so a consumer must validate its query against `query_schema`.

Auth is `Authorization: Bearer <api key>`; with `KB_AUTH_MODE=none` the token is ignored (the hook sends `local-no-auth` on a loopback URL).

A request-side addition such as a new TurnItem kind or query param needs the kb-service release deployed before any sender uses it, because an older server answers 422 for the whole digest.

The talos sender rules for `harness_correction` are in the `TurnHarnessCorrectionItem` `$defs` description.

## Seeing drift in production

The server logs `turn_event reason=invalid` (422, loc and type only) and `turn_event reason=too-large` (413), and counts both in `route_outcomes` (plus `unmapped_tools`) on GET /api/kb/turn/heartbeat.

The hook's turn-digest log records `http_status` and `server_reason` per send.

A /api/kb/prevention failure shows only as the server WARNING `prevention_fetch failed`.

## Regenerating

Run `uv run python packages/kb-service/scripts/gen_contracts.py` from the repo root, with `--check` to verify, and never hand-edit the JSON.

A pydantic or fastapi upgrade or a model docstring edit can change the rendered schema and needs a regenerate.

The enforcing tests are `packages/kb-service/tests/test_contracts_service.py` and `packages/personal-kb-hook/tests/test_contracts_hook.py`.

## contract_version

`contract_version` is an integer per file, edited only in `CONTRACT_VERSIONS` in the generator, and bumped for a breaking change.

For requests a breaking change is a removed or renamed field, a narrowed type, a lower `max*`/`max_body_bytes` or higher `min*`, a newly required field or query param, a removed enum member or TurnItem kind, or a changed path or method.

For responses it is a removed or renamed field, a field becoming optional or nullable, a widened type, or a new member of any response enum (`TurnDigestResponse.reason`, `PreventionResponse.surprise_capture`).

These regenerate with no bump: a new optional request field or query param, a new TurnItem kind, a new response field with a default, and description or title-only changes.

The generator detects property-removed, required-added/removed, enum-removed/added, limit-tightened/added, kind-removed, def-removed, type-changed and endpoint-changed, and refuses to write until the version is bumped.

Those detectors are a floor: any other change in the breaking list (for example a response field gaining null in anyOf) needs a manual bump, and a model rename trips def-removed and is bumped anyway.

## Vendoring

Copy this directory verbatim, and re-copy whenever its content changes.
