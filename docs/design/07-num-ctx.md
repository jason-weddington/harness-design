# 07 — num_ctx policy: pin the model's advertised context, never a constant

Status: **DECIDED (2026-09-19)** — ratifies the shared resolver shipped in 3584df9 as the num_ctx policy and restores it to the CLI after d76792d silently reverted that half.

## The policy

The `num_ctx` an Ollama run is pinned with must come from the model's own advertised context window, read from the daemon, or from an explicit operator pin — never from a compiled constant. `harness::ollama::resolve_num_ctx` (`crates/harness/src/ollama.rs:547`) is the single implementation of that policy and the single call site every lane must share: an explicit non-empty `OLLAMA_NUM_CTX` `u32` wins verbatim (no HTTP request); an empty or whitespace-only value is treated as unset; unset against a `localhost`/`127.0.0.1` base URL probes `POST /api/show` and pins the advertised value; unset against any other base URL leaves `num_ctx` unset (Ollama's own default applies); a probe failure is a construction error that exits `1` — never a fallback constant and never a silent default. One function, so the shipped lane and the measured lane cannot drift: `talos run` and both eval runners (`crates/harness/examples/coding_eval.rs:138`, `crates/harness/examples/mined_eval.rs:185`) all call it.

The run record carries the provenance: `BackendSettings.num_ctx` / `.num_ctx_source` are `"explicit"` or `"probe"` (from `NumCtxSource::as_str`) when a value was pinned, both `None` when it was not — the `Default` source pins no value, so the string `"default"` never reaches a record. The structured `{"num_ctx": {...}}` stderr line carries the same resolution, including the `Default` source and the below-floor warning, and is diagnostic only.

## What the Ollama API actually exposes

`POST /api/show` on an Ollama daemon returns a payload whose `model_info` object carries the model family under `general.architecture` and the advertised window under `{arch}.context_length`. These are the only keys the resolver reads — a keyed lookup, deliberately with no fallback scan for keys ending in `.context_length`.

Measurements (lead-measured on 2026-09-19 against the workstation daemon; transcribed here verbatim — do not re-probe, a build agent has no such daemon): `qwen3.8:27b` → `general.architecture` = `qwen35`, `qwen35.context_length` = 262144; `gpt-oss:20b` → `general.architecture` = `gptoss`, `gptoss.context_length` = 131072, alongside `gptoss.rope.scaling.original_context_length` = 4096. That last key is the decoy the resolver's doc comment warns about, confirmed real in a live payload: a suffix scan would have resolved gpt-oss:20b to 4096 — a 32x wrong window — which is exactly why `resolve_context_length` (`crates/harness/src/ollama.rs:335`) reads `model_info["{arch}.context_length"]` by key and never scans (rationale at `crates/harness/src/ollama.rs:306-309`).

## Options considered

(a) **A model-aware default derived from the advertised window via `POST /api/show` — CHOSEN.** Already implemented as `resolve_num_ctx` (`crates/harness/src/ollama.rs:547`), shared by the CLI and both eval runners. The value is the daemon's own answer for that exact model, so it cannot drift from the model's real window.

(b) **Per-engine configuration only — REJECTED.** It is the status quo that produced the drift this policy exists to close, and it is what the dispatch worker does today: it exports `OLLAMA_NUM_CTX` per engine (262144 for the qwen lane, 1048576 for the glm and glm-flash lanes), so every dispatched run takes the Explicit branch and the advertised window is never consulted. That is exactly option (a)'s escape hatch being used as the primary mechanism — fine as an override, wrong as the policy.

(c) **Adaptive sizing off the pre-flight guard's estimate — REJECTED.** `estimate_prompt_tokens` (`crates/harness/src/ollama.rs:864`) is `chars / 4` — a tripwire that decides whether a request should be sent at all (`crates/harness/src/ollama.rs:216-220`), not a sizing oracle. Deriving `num_ctx` from it would reintroduce a computed constant that can drift from the model's real window, which is the failure mode option (a) exists to eliminate.

(d) **In-run context compaction / message-history pruning — DEFERRED** to its own roadmap item. It is named here so the reader knows it was considered and where it went; it is not specified in this record.

## The regression that motivated this record

`3584df9` ("resolve Ollama num_ctx from the model's advertised context, shared with the eval runners", 2026-09-18) shipped the shared resolver to the CLI and to both eval runners. The very next commit to touch `crates/talos/src/main.rs`, `d76792d`, silently reverted the CLI half — it deleted `num_ctx_stderr_line`, made `build_ollama_backend` synchronous again, and reinstated the `32_768` hardcode — while `crates/harness/examples/coding_eval.rs:138` and `crates/harness/examples/mined_eval.rs:185` kept calling `resolve_num_ctx`. The shipped-vs-measured drift that 3584df9 closed was therefore reopened at HEAD: a `talos run` against a localhost daemon with `OLLAMA_NUM_CTX` unset pinned 32768 for `qwen3.8:27b`, while both eval runners probed the same model to 262144 — an 8x measured-vs-shipped gap on the unpinned local path.

The dispatched lanes were NOT affected: the dispatch worker exports `OLLAMA_NUM_CTX` per engine (262144 for talos-qwen, 1048576 for glm and glm-flash), so every dispatched Ollama run takes the Explicit branch and the hardcode never reached it. The blast radius is the unpinned local path — a hand-run `talos run`, and any local-vs-eval comparison.

`d76792d`'s own commit message claims `num_ctx` and `num_ctx_source` "come from `NumCtxResolution` (3584df9)" while its diff does the opposite — reinstating the hardcode and dropping the resolver call. That contradiction is the mechanism by which the revert passed review, and it is why this record now exists: the decision is written down so the next reader does not have to re-derive it from a diff.

## Constants, pinned

`MIN_EXPECTED_NUM_CTX = 32_768` (`crates/harness/src/ollama.rs:453`) is an audit floor, never assigned as a value: a probed window below it proceeds with the verbatim advertised value plus a BELOW-FLOOR warning. `SHOW_TIMEOUT = 10s` (`crates/harness/src/ollama.rs:255`) bounds the probe so a hung daemon cannot wedge construction. `DEFAULT_MAX_TOKENS = 32_768` (`crates/harness/src/engine.rs:299`) is the generation budget, unrelated to context size. The trap to note: the reverted CLI default of `32_768` is numerically identical to the floor the library explicitly documents as "not a default" — same number, opposite meaning. And the localhost default is NOT a constant at all: it is the probed advertised value, which is why the shipped lane and the eval lane agree by construction rather than by coincidence.

## Known residue

These four gaps are deliberately not closed by this item; each is named so the next reader does not re-derive it.

(1) **Locality gate.** `is_local_ollama_url` (`crates/harness/src/ollama.rs:460`) matches only `localhost`/`127.0.0.1`, so a LAN daemon (the workstation GPU reached from a dispatch host by IP) and Ollama Cloud both take the non-local branch — no `num_ctx` and a disarmed pre-flight guard, since the guard only runs when `num_ctx` is `Some` (`crates/harness/src/ollama.rs:216`). The required remedy for those topologies is pinning `OLLAMA_NUM_CTX`, as `README.md` already states. Probing any base URL is a named follow-up, not built here: Ollama Cloud's `/api/show` payload shape is unverified (no cloud key available to check it), and a 10s `SHOW_TIMEOUT` would land on every cloud run's construction path.

(2) **Generation headroom.** The guard trips at `estimate_prompt_tokens(&body) >= num_ctx` and reserves nothing for `options.num_predict` (default `DEFAULT_MAX_TOKENS` = 32_768), so a request can pass the guard and still overflow the window with its own generation. The proposed trip condition is `estimate + body.options.num_predict as usize > num_ctx` — an exact reservation, not a fraction. Deferred to its own item because it would hard-fail any run pinned at 32_768: a correct signal, but a behaviour change shared with the eval lane the library serves.

(3) **Telemetry for (2).** `BackendError::ContextLengthExceeded` is a unit variant (`crates/harness/src/model.rs:301`), so a tripped guard records no estimate, no `num_ctx`, and no `num_predict`. Gap (2) cannot be evaluated from production telemetry until that variant carries its numbers — a follow-up item.

(4) **Recorded provenance.** `NumCtxResolution.desc` and `.warning` (including the BELOW-FLOOR marker) reach only stderr, not `BackendSettings`, so a record reading `num_ctx: 4096, num_ctx_source: "probe"` cannot distinguish a correct small window from a wrong-key resolution. Adding a `num_ctx_provenance` field is a follow-up, deliberately out of this item's minimal-change scope.

## How to verify in production

The regression's failure mode turns into a named check. (a) Over a dispatch transcript: `jq -c 'select(.event=="run_start") | .backend_settings' <transcript.jsonl>` — an unpinned local `qwen3.8:27b` run must show `"num_ctx": 262144, "num_ctx_source": "probe"`; anything showing `32768` on that topology is the hardcode back. (b) The parity check: the same `model` must show the same `num_ctx` on a `mined_eval` row and a talos run record — any difference is the drift tripwire the shared resolver exists to prevent.