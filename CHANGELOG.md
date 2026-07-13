# Changelog
All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

- - -
## 0.5.0 - 2026-07-13
#### Features
- (**eval**) two hard fixtures — tokenbucket (withheld-test) + eventbus (multi-file) - (7690dd4) - Jason Weddington, *claude-code-glm*, *Claude Opus 4.8*
- (**eval**) claude-code-glm eval runner — same fixtures, external holdout scoring - (c2052a1) - Jason Weddington, *claude-code-sonnet*, *Claude Opus 4.8*
- (**harness**) finish-recovery nudges at the StoppedWithoutFinish terminal too - (e6ef1c0) - Jason Weddington, *claude-code-sonnet*, *Claude Opus 4.8*
- (**harness**) RunStats.gates_green_at_exit — classify done-but-unclaimed stops - (bafb6ed) - Jason Weddington, *Claude Opus 4.8*
#### Documentation
- Session 9 log — finish-discipline completed + harness-vs-model benchmark - (c457798) - Jason Weddington, *Claude Opus 4.8*
- document the talos fleet-publish flow (release + manual mid-work push) - (ad3136b) - Jason Weddington, *Claude Opus 4.8*
#### Miscellaneous Chores
- gitignore .envrc (local direnv with secrets) - (85abcbb) - Jason Weddington, *Claude Opus 4.8*

- - -

## 0.4.1 - 2026-07-11
#### Features
- (**harness**) capture recovery_facts on a green-static MaxIterations - (f02f0e9) - Jason Weddington, *talos-glm*, *Claude Opus 4.8*
#### Documentation
- bring roadmap current to v0.4.0 + make roadmap upkeep a session-log step - (bc06eda) - Jason Weddington, *Claude Opus 4.8*
#### Miscellaneous Chores
- release.sh publishes fleet artifact before pushing the tag - (3a58014) - Jason Weddington, *Claude Opus 4.8*

- - -

## 0.4.0 - 2026-07-11
#### Features
- (**harness**) wall-clock budget — graceful self-termination with recovery facts - (e1cd5b5) - Jason Weddington, *claude-code-sonnet*, *Claude Opus 4.8*
- (**harness**) bounded deterministic retry with backoff on transient errors - (3ec0aea) - Jason Weddington, *claude-code-sonnet*, *Claude Opus 4.8*
- (**harness**) finish-recovery protocol — detect done-but-unclaimed spin - (758cf4a) - Jason Weddington, *claude-code-glm*, *Claude Opus 4.8*
- (**talos**) version-stamp + dual-arch publish to the dispatch fleet - (2d48267) - Jason Weddington, *Claude Opus 4.8*
#### Bug Fixes
- (**talos**) raise default --max-iterations 24 → 500 - (d59b1d9) - Jason Weddington, *Claude Opus 4.8*
#### Documentation
- (**design**) 0.4.0 bounded-autonomy design — finish-recovery protocol - (2178ce9) - Jason Weddington, *Claude Opus 4.8*
- (**harness**) reconcile 0.4.0 budget scope to wall-clock-only - (00dfa04) - Jason Weddington, *Claude Opus 4.8*
- Session 8 continued — 0.4.0 wave shipped, harness-vs-model proven, released - (cdc3456) - Jason Weddington, *Claude Opus 4.8*
- Session 8 — 0.4.0 bounded-autonomy design/groom + talos-glm harness-gap finding - (3e581d1) - Jason Weddington, *Claude Opus 4.8*

- - -

## 0.3.7 - 2026-07-11
#### Features
- (**tools**) rename run_command to bash with single command-string interface - (e80de7f) - Jason Weddington
#### Documentation
- (**design**) sync tool inventory to run_command → bash rename - (03ffa66) - Jason Weddington, *talos-glm*
- Session 7 third sitting — bash tool + talos-glm dispatch unblock - (dbb668f) - Jason Weddington

- - -

## 0.3.6 - 2026-07-10
#### Features
- (**eval**) walrus fixture — implement compact() in append-only KV store (tier 3) - (672f5d2) - Claude Agent
- (**eval**) calc fixture — right-associative power operator (tier 5) - (bb17d4e) - Claude Agent
- (**eval**) csv-ledger fixture — cross-file bug fix with distractor (tier 2) - (dfc0b28) - Claude Agent
- (**eval**) taskdeck fixture — finish a half-built task-tracker CLI (tier 4) - (8d6bbe5) - Claude Agent
- (**eval**) TaskSpec-shaped fixture prompts + sealed holdout re-gate - (ef920f1) - Claude Agent
#### Bug Fixes
- (**eval**) exclude fixture-root target/ and Cargo.lock from trial copy-in - (df085f4) - Jason Weddington
- (**eval**) strip answer-key spoilers from csv-ledger, fix taskdeck file set - (329ad50) - Jason Weddington
#### Documentation
- Session 7 second sitting — 5-model matrix, talos search-tool finding - (b7e0688) - Jason Weddington
- Session 7 summary — eval hardening, holdout re-gate, fixture ladder - (a25bf32) - Jason Weddington
#### Miscellaneous Chores
- (**eval**) add mean_wall column to the coding_eval summary table - (bd7fb82) - Jason Weddington
- (**eval**) make coding_eval iteration cap env-overridable - (d4e33ef) - Jason Weddington
- (**eval**) relax fixture-discovery test to containment - (c10e501) - Jason Weddington

- - -

## 0.3.5 - 2026-07-10
#### Features
- (**talos**) support --version (clap version flag) - (f49dbac) - Jason Weddington, *talos-haiku*
- (**talos**) add talos CLI — task spec in, disposition-mapped exit code out - (0d69997) - Jason Weddington, *Claude Fable 5*
- (**task_spec**) add TaskSpec wire type and groomed-item task prompt - (d9cb7cc) - Jason Weddington, *Claude Fable 5*
#### Bug Fixes
- (**prompt**) finish-discipline framing + max-iterations headroom for groomed items - (5a7107e) - Jason Weddington, *Claude Fable 5*
- (**talos**) --help/--version exit 0 with plain output, not JSON error - (a4c5b88) - Jason Weddington, *Claude Fable 5*
#### Documentation
- complete Session 6 summary — first patrols merged, capability claim true - (d5fb086) - Jason Weddington, *Claude Fable 5*
- add Session 6 summary (the 0.3.5 epic — Talos becomes a build engine) - (ef0a7fa) - Jason Weddington, *Claude Fable 5*
#### Tests
- (**talos**) integration coverage for --file spec input - (f9153cd) - Jason Weddington, *talos-haiku*

- - -

## 0.3.0 - 2026-07-08
#### Features
- (**engine**) crash-resume + fresh-context restart - (3dea195) - Claude Haiku 4.5
- (**engine**) run identity + checkpoint wiring into the loop - (688fc4f) - Claude Haiku 4.5
- (**run_record**) schema v2 + disposition unification - (bb17d2b) - Claude Haiku 4.5
#### Bug Fixes
- (**engine**) reconcile on snapshot shape, not log-tail shape - (7e31d2b) - Jason Weddington
- (**engine**) crash-resume reconciliation — filter tail to tool events, pair by call_id - (9ca028d) - Jason Weddington
#### Documentation
- add Session 5 summary (durability through the review gate) - (80474c9) - Jason Weddington
- purge surviving pre-D6 idempotency/replay language - (01b72a1) - Jason Weddington
- insert 0.3.5 first-dogfood milestone; bring roadmap current to v0.2.0 - (5ae3dd1) - Jason Weddington
#### Tests
- (**engine**) kill-and-resume proof deterministic integration test - (5c55fd6) - Claude Haiku 4.5
- strengthen event byte-stability, per-iteration ordering, D9 observed-prompt assertions - (99998ea) - Jason Weddington

- - -

## 0.2.0 - 2026-07-08
#### Features
- (**eval**) backend selection in coding_eval (EVAL_BACKEND=anthropic|ollama) - (4911790) - Jason Weddington
- (**eval**) add per-trial run metrics (RunStats, TrialResult, aggregates) - (86575a2) - Jason Weddington
- (**eval**) add three subtle-bug fixtures + multi-fixture coding_eval - (034b39c) - Jason Weddington
- (**ollama**) add Ollama ModelBackend adapter (native /api/chat, local + cloud) - (1171c10) - Jason Weddington
#### Documentation
- add Session 4 summary (second backend + the four-model matrix) - (fd0776f) - Jason Weddington
- add roadmap (capability-themed milestones to the GTD adapter) - (2d6a0e7) - Jason Weddington
- add Session 3 summary (three seams, overnight waves, v0.1.0) - (d1c1dfa) - Jason Weddington

- - -

## 0.1.0 - 2026-07-07
#### Features
- (**anthropic**) add non-streaming Anthropic ModelBackend adapter - (7667cd6) - Jason Weddington
- (**engine**) claim-vs-verify — harness-run verification of finish(done) - (3c8f424) - Jason Weddington
- (**engine**) add minimal agent loop + finish tool - (b00f9fb) - Jason Weddington
- (**eval**) add coding-task eval — fixture crate + per-trial isolation - (4b8a867) - Jason Weddington
- (**eval**) add pass^k eval harness wrapping the loop - (309da0a) - Jason Weddington
- (**exec**) add exec core + run_command/run_checks tools (the done-oracle) - (cf6f0e8) - Jason Weddington
- (**model**) add model-IO contract (AssistantTurn, Message, ModelBackend, BackendError) - (b982d74) - Jason Weddington
- (**prompt**) add askama-templated system + task prompts - (1a141bf) - Jason Weddington
- (**run-record**) core run-record data model types - (4831e62) - Jason Weddington
- (**store**) add RunStore trait + SQLite implementation - (eb8a129) - Jason Weddington
- (**tool**) add Tool trait, ToolResult, and ToolRegistry - (dd7a8f7) - Jason Weddington
- (**tools**) add list_files tool - (d1b7c31) - Jason Weddington
- (**tools**) add read_file tool - (1983011) - Jason Weddington
- (**tools**) add edit_file tool (exact-match replace + create) - (4743f69) - Jason Weddington
- (**workspace**) add confined path resolution + real disk offload sink - (ba26d3a) - Jason Weddington
#### Documentation
- add Session 2 summary (model contract + first live loop) - (0cb14bb) - Jason Weddington
- add session-summaries log + session-logging practice - (291df1b) - Jason Weddington
- encode claim-vs-verify checklist + finish disposition - (054150a) - Jason Weddington
- add v1 run-record schema; resolve run_command open question - (8b136ff) - Jason Weddington
- correct research assumptions + add v1 inner-loop & tool design - (3ece211) - Jason Weddington
- point CLAUDE.md at the harness-design-research KB braintrust - (37bd626) - Jason Weddington
- add agent harness design research corpus - (3d64027) - Jason Weddington
#### Miscellaneous Chores
- gitignore local .claude session state - (23fb5d7) - Jason Weddington
- allow Unicode-3.0 license in cargo-deny - (26d11d3) - Jason Weddington
- scaffold Rust workspace and quality-gate harness - (2c45b2d) - Jason Weddington

- - -

Changelog generated by [cocogitto](https://github.com/cocogitto/cocogitto).