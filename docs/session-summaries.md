# Session summaries

A running, chronological log of working sessions on harness-design — this is a
learning project, so we document the *process*, not just the output. Each entry
is a 3-4 sentence summary; the full per-session write-up (process, decisions,
handoff) lives in a dated `lesson_learned` KB entry under `project_ref:
harness-design`, linked here. Append newest at the bottom; when resuming, read
the latest entry and its linked KB entry first.

---

## Session 1 — 2026-06-20 → 06-23 — gates, research, design, foundation

Took the project from nothing to a built v1 foundation: established Rust quality
gates *before* any code (so headless agents can run safely), ran a multi-agent
research workflow into `docs/research/` plus a ~960-entry local KB braintrust
(`harness-design-research`), then locked the v1 design — the
workflow-around-open-loop shape, an 8-tool inventory (LSP deferred), and the
run-record schema with claim-vs-verify and the Done/Blocked/Failed disposition
(`docs/design/01` + `02`). Corrected two assumptions the research had baked in
(a permission/allowlist model we don't have; a Pi-specific deployment) before
grooming three GTD items and dispatching them as a headless wave (data model +
tool layer in parallel, then RunStore). All three merged clean to `main`
(`eb8a129`) — 42 tests, 97.9% coverage, zero quality misses; one build agent even
dodged a license landmine unsupervised. **Next:** the loop engine + model-backend
trait (held for an interactive session). Full write-up + handoff: **kb-02851**.

---

## Session 2 — 2026-06-23 → 06-24 — the model contract + first live loop

Designed the architecturally-live core interactively, then shipped it as an autonomous
dispatch wave. The load-bearing design call: draw the `ModelBackend` trait *high* — a
normalized `AssistantTurn` out, each adapter an anti-corruption layer owning all wire
translation, so the loop never sees provider-native shapes; role-precise content blocks
(`ContentBlock` vs `UserBlock`, `Message` an enum over role) make illegal message states
unrepresentable, and a four-variant `BackendError` classifies while the loop reacts.
Resolved open decision #2 — **direct tool-calling for v1**, code-mode deferred to re-enter
later as a registered sandboxed tool (not a rewrite), grounded in the research corpus
rather than priors (`kb-02852`) — and accepted `CDLA-Permissive-2.0` into `deny.toml` for
the rustls TLS stack (`kb-02854`). Built as a 4-item dependency wave (D model contract →
E1 Anthropic adapter ‖ E2 loop+finish → F eval harness); all merged clean to `main`, 100
tests, 98.83% coverage, with one inline `typos` fix the lead's merge-gate caught that the
E2 agent's gates missed. **The live eval is green: 5/5 pass^k against `claude-haiku-4-5`** —
the loop drives the real Anthropic API through a `finish` tool call end-to-end. Full
write-up + handoff: **kb-02855**.

---

## Session 3 — 2026-06-29 → 07-07 — the three seams, overnight waves, v0.1.0

Started from a prompt-engineering question (should we model prompts in BAML?) and ended
with a released build engine. Investigated BAML against its live repo — verdict: don't
adopt (single-shot structured-*output* layer, FFI-blob/sidecar footprint, and it replaces
exactly the layer this project exists to build) but steal its ideas: versioned templated
prompts, prompts-as-testable-fixtures, lenient parsing (parked for Ollama) (`kb-02870`).
Decided to release at a **product boundary** ("the harness autonomously completes a
coding task, verified mechanically"), not an engineering milestone. Designed three seams
interactively: the confined `Workspace` (path resolution owned in one tested place; a
clarifying question caught the offload-readback gap early), the **claim-vs-verify**
control flow (`finish(done)` is a claim; the harness runs the checks itself; rejection is
steering; `Done` carries evidence by construction), and the askama prompt layer
(compile-time-checked, versioned template files, load-bearing phrases pinned by tests).
Then three autonomous dispatch waves overnight — 5 parallel tools/prompt items, the
engine rework, the eval fixture with per-trial isolation — 7 dispatches, all clean, zero
quality misses. Morning boundary ritual: **3/3 pass^k** — haiku found and fixed the
planted bug and the harness verified `cargo test` green itself — then `./release.sh` cut
**v0.1.0** with the first CHANGELOG and pushed to GitHub. Full write-up + handoff:
**kb-02899**.
