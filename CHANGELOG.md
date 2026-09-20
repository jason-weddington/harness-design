# Changelog
All notable changes to this project will be documented in this file. See [conventional commits](https://www.conventionalcommits.org/) for commit guidelines.

- - -
## 0.13.0 - 2026-09-20
#### Features
- (**engine**) cap consecutive identical answer-schema rejections - (e312f8e) - Jason Weddington, *Claude Opus 5*
- (**engine**) arm the wall-clock budget by default so a long run terminates cleanly - (ac5d9e3) - Jason Weddington, *Claude Opus 5*
- (**engine**) let a library consumer supply its own leg-3 change evidence - (046b1f9) - Jason Weddington, *Claude Opus 5*
- (**publish**) add follower publisher for the macOS talos binary - (2289db3) - Jason Weddington, *Claude Sonnet 5*
- (**publish**) publish every workspace binary at one token, verify per-file before advancing latest - (cc26f3c) - Jason Weddington, *Claude Opus 5*
#### Bug Fixes
- (**engine**) arm finish recovery on observed work, not only on a green in-loop gate - (fd6ebf5) - Jason Weddington, *Claude Opus 5*
- (**exec**) observe each child repo so leg 3 works on workspace-mode dispatch - (4afb690) - Jason Weddington, *Claude Opus 5*
#### Documentation
- a hardcoded fleet host list fails in the quiet direction - (f0a15fa) - Jason Weddington, *Claude Opus 5*

- - -

## 0.12.0 - 2026-09-20
#### Features
- (**engine**) make the compaction trigger threshold a configurable knob - (aacef15) - Jason Weddington, *Claude Opus 5*
- (**engine**) Ollama-only in-run context compaction with disorientation telemetry - (11e8133) - Jason Weddington, *Claude Opus 5*
- (**engine**) resolve the per-turn output cap per backend and model - (f5a9f60) - Jason Weddington, *Claude Opus 5*
- (**prompt**) state that the harness owns the commit, not the agent - (31a07ba) - Jason Weddington, *Claude Opus 5*
#### Bug Fixes
- (**engine**) pair tier-2 compaction by adjacency, not global call ids - (ac2cfc3) - Jason Weddington, *Claude Opus 5*
#### Documentation
- (**design-06**) answer mode's read-only guarantee is filesystem-scoped, not general - (2820781) - Jason Weddington, *Claude Opus 5*
- (**design-07**) cloud /api/show verified; no Ollama output cap exists - (3fcffa8) - Jason Weddington, *Claude Opus 5*
- (**design-08**) the dose-response settles it — harmful only under a setting production cannot produce - (7a35620) - Jason Weddington, *Claude Opus 5*
- (**design-08**) first measured compaction runs — correct, but the telemetry is blind to the harm - (62f9704) - Jason Weddington, *Claude Opus 5*
- (**design-08**) compaction ships default-on at 90% with a toggle - (ceb95db) - Jason Weddington, *Claude Opus 5*
- (**design-08**) compaction is Ollama-only; the output cap still covers all three backends - (b7eddbf) - Jason Weddington, *Claude Opus 5*
- (**design-08**) context budget — resolved output cap and a tiered compaction scheme - (0bcc4d1) - Jason Weddington, *Claude Opus 5*
- (**research**) transcript study of two talos dispatch misses (item 050252ba) - (e198fd7) - Jason Weddington, *Claude Opus 5*
- (**roadmap**) context budget is decided, superseding "not planned" on compaction - (e40b677) - Jason Weddington, *Claude Opus 5*
- session 18 log and roadmap — the context budget - (0523643) - Jason Weddington, *Claude Opus 5*
- latest has independent consumers — scope the version check per host set - (b902449) - Jason Weddington, *Claude Opus 5*

- - -

## 0.11.0 - 2026-09-19
#### Features
- (**engine**) name a max_tokens truncation as FailureMode::Truncated instead of masking it as StoppedWithoutFinish - (5211a15) - Jason Weddington, *Claude Fable 5.1*
- (**engine**) answer mode — finish(answer) with a schema-validated result - (bce155b) - Jason Weddington, *Claude Fable 5.1*
- (**engine**) require observed tree change before accepting a done claim - (1ec37b2) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**engine**) opt-in one-shot acceptance audit on the first gate-green finish(done), default off - (712fb21) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**engine**) opt-in full run transcript (JSONL), default off - (d4651b6) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**mined-eval**) default the tier-2 iteration cap to 500 (production parity) - (00b03e3) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**mined-eval**) post-run agent-gate verdict (resolved vs shippable) + per-run capture paths - (eb333c1) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**mined-eval**) production-parity agent gate for tier-2 (run_checks + finish-recovery armed) - (a0c0128) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**ollama**) parse prompt_eval_cached_count so cached input is visible and priced correctly - (7b2c6ea) - Jason Weddington, *Claude Fable 5.1*
- (**prompt**) opt-in Criterion Coverage rules, default off everywhere - (2c74407) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**ralph**) retry a hook-rejected commit when the hook itself mutated the tree, bounded and observable - (d6e93e1) - Jason Weddington, *Claude Fable 5.1*
- (**record**) record the resolved backend settings on the run record - (d76792d) - Jason Weddington, *Claude Fable 5.1*
- (**talos**) resolve Ollama num_ctx from the model's advertised context, shared with the eval runners - (3584df9) - Jason Weddington, *Claude Fable 5.1*
- (**talos**) answer mode — `talos run --mode answer --schema` - (d349a61) - Jason Weddington, *Claude Fable 5.1*
- (**talos**) let bare --transcript default to the run's state dir - (33fee62) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**talos**) prune the XDG state dir on run start so run artifacts stop accumulating - (caf7549) - Jason Weddington, *Claude Opus 5 (1M context)*
#### Bug Fixes
- (**deps**) bump rustls 0.23.41 -> 0.23.45 for RUSTSEC-2026-0285 - (f33fa86) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**engine**) parse a finish(answer) result that a backend delivered as JSON text - (b034a63) - Jason Weddington, *Claude Fable 5.1*
- (**engine**) reject an unrecognized or missing finish disposition instead of coercing it to Failed - (4c8b637) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**eval**) give tier-1 trial workspaces a git baseline, like production - (2612751) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**mined-eval**) surface when the count-mismatch tripwire is armed, never silently off - (3bdc845) - Jason Weddington, *Claude Fable 5.1*
- (**mined-eval**) inject the per-trial state root instead of reading env in the library - (7b04ecb) - Claude Agent, *Claude Opus 5*
- (**talos**) restore the /api/show num_ctx resolver the CLI lost in d76792d, and record the num_ctx policy - (1f56186) - Jason Weddington, *Claude Fable 5.1*
#### Revert
- remove the measured-negative criterion-coverage and acceptance-audit knobs - (cbddb48) - Jason Weddington, *Claude Opus 5 (1M context)*
#### Documentation
- (**design**) note transcripts shipped as part of the false-done instrumentation - (1a3c328) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**design**) propose per-criterion tests + a one-shot acceptance audit to cut false dones - (00d5235) - Jason Weddington, *Claude Opus 5 (1M context)*
- (**design-06**) cache reads were unparsed, not absent — cost figures are upper bounds; arm C is this project's groom - (0662532) - Jason Weddington, *Claude Fable 5.1*
- (**roadmap**) evening paragraph and next-up for the session close - (6ed9dfc) - Jason Weddington, *Claude Fable 5.1*
- session close — arm C adopted as the project groom, Ollama caching answered, v0.11.0 roadmap header - (8c1f530) - Jason Weddington, *Claude Fable 5.1*
- the talos groom (flash draft / glm critics / flash synth) is this project's default groom - (810a0ab) - Jason Weddington, *Claude Fable 5.1*
- session log — groom-to-ready on talos, the Ollama answer-mode bug, four-arm groom comparison, three arm specs shipped (kb-03352) - (e55db53) - Jason Weddington, *Claude Fable 5.1*
- talos-flow client shipped — design 06 status, roadmap, session log - (f1126e8) - Jason Weddington, *Claude Fable 5.1*
- session log — Ollama rule flipped, first real flash and qwen dispatches (kb-03318) - (2d38881) - Jason Weddington, *Claude Fable 5.1*
- roadmap — fleet published on 0.10.0-27-g53b56fb, both gates verified - (77acce9) - Jason Weddington, *Claude Fable 5.1*
- session log — leg 3, git-native tier-1, answer mode shipped (kb-03311) - (53b56fb) - Jason Weddington, *Claude Fable 5.1*
- design 06 — answer mode, talos as the sub-agent of a dynamic workflow - (8679fc1) - Jason Weddington, *Claude Fable 5.1*
- drop the rollout relaunch-cap item from queued — fixed in agent-gtd-dispatch 1.24.1 - (b28a823) - Jason Weddington, *Claude Opus 5 (1M context)*
- session log — negative knobs removed, state-dir retention, transcripts on dispatch (kb-03293) - (f6d5844) - Jason Weddington, *Claude Opus 5 (1M context)*
- A′/B measured — negative result, neither default flips (kb-03283) - (8b13923) - Jason Weddington, *Claude Opus 5 (1M context)*
- evals stay harder than perfect specs; harness changes must not break the perfect-spec path (kb-03268) - (c4d4f98) - Jason Weddington, *Claude Opus 5 (1M context)*
- overnight results, transcript-based false-done diagnosis, revised design 05 (kb-03264) - (fe2934c) - Jason Weddington, *Claude Opus 5 (1M context)*
- first representative tier-2 matrix (cap 500, k=3) results (kb-03252) - (63fa78a) - Jason Weddington, *Claude Opus 5 (1M context)*
- tier-2 production-parity matrix results and next steps (kb-03240) - (1a67a6d) - Jason Weddington, *Claude Opus 5 (1M context)*
#### Build system
- (**lefthook**) skip the heavy gates on docs-only changesets, fail-safe - (055d187) - Jason Weddington, *Claude Fable 5.1*

- - -

## 0.10.0 - 2026-09-13
#### Features
- (**eval**) A/B toggle for the test-first guidance - (eee2d3f) - Jason Weddington, *Claude*
- (**mined-eval**) measure agent test authorship per trial - (efdb1d9) - Jason Weddington, *Claude*
- (**mined-eval**) surface finish-discipline telemetry per tier-2 trial - (220406a) - Claude Haiku 4.5, *Claude Opus 4.7*
- (**mined-eval**) tier-2 mined-task eval runner (mined_eval example) - (4d67846) - Jason Weddington, *Claude Fable 5*
- (**ollama**) resolve num_ctx from advertised context length via /api/show - (b556f6b) - Claude Haiku 4.5, *Claude Opus 4.7*
- (**prompt**) require a failing test first and stop treating a green gate as done - (be85c75) - Jason Weddington, *Claude*
- (**ralph**) stop after N consecutive backend errors (BackendErrorsExhausted) - (03b8970) - talos-glm, *Claude Opus 5 (1M context)*
- (**talos**) print a ralph Error terminal's failing command to stderr - (66551b7) - talos-glm-flash, *Claude Opus 5 (1M context)*
#### Bug Fixes
- (**deps**) bump h2 0.4.15 -> 0.4.17 for RUSTSEC-2026-0258 - (7e00a2e) - Jason Weddington
- (**mined-eval**) route test-first guidance into the tier-2 agent prompt - (5342e9a) - Jason Weddington, *Claude*
- (**mined-eval**) section-scope pytest parsing and score collection errors Unresolved - (e684080) - Jason Weddington, *Claude*
- (**mined-eval**) carry BackendError payload and strip full-width pytest banners - (74a506e) - Jason Weddington
#### Documentation
- (**design**) ratify eval-tier decisions — spec-level ladder resolves the known-solvable problem - (41ec5b8) - Jason Weddington, *Claude Fable 5*
- (**design**) discriminating eval tier proposal — mined-from-history ladder (kb-03197) - (d69f426) - Jason Weddington, *Claude Fable 5*
- (**research**) track 08 — discriminating evals (SWE-bench anatomy, METR time horizon, Harness-Bench) - (29ba1cc) - Jason Weddington, *Claude Fable 5*
- (**roadmap**) record the glm-5.3 / glm-5.3-flash / sonnet-5 talos lane cutover - (bb9e13f) - Jason Weddington, *Claude Opus 5 (1M context)*
- Session 17 log — test-first prompt, glm-5.3 lanes, stale fleet found (kb-03226) - (ab3c9fc) - Jason Weddington, *Claude Opus 5 (1M context)*
- Session 16 log — tier-2 matrix v1, num_ctx foot gun closed, finish-recovery found dead (kb-03205) - (5aa5b9c) - Jason Weddington
- Session 15 log — tier-2 pilot built end-to-end, matrix handoff (kb-03200) - (d04969b) - Jason Weddington, *Claude Fable 5*
- Session 14 log — nemotron-3.5-lightning eval, first false-dones, pivot to discriminating evals (kb-03196) - (2a4727a) - Jason Weddington, *Claude Fable 5*
#### Miscellaneous Chores
- (**eval**) default the GLM eval runners to glm-5.3:cloud - (152fdcb) - Jason Weddington, *Claude Opus 5 (1M context)*

- - -

## 0.9.0 - 2026-07-15
#### Features
- (**bedrock**) add AWS Bedrock model backend (Converse API, TALOS_BEDROCK-gated) - (886aa2f) - Jason Weddington, *talos-glm*, *Claude Opus 4.8*
#### Documentation
- Session 13 log — Sonnet 5 + AWS Bedrock backend, harness-is-the-variable proven (kb-03115) - (87b8b76) - Jason Weddington
- add AWS Bedrock to the CLAUDE.md model-support list - (4ca45f4) - Jason Weddington
- add SWE-bench Pro model-capability reference for engine routing - (10bc595) - Jason Weddington
- scrub obsolete CLAUDE.md — drop Status section, fix stale gate/version facts - (9560a83) - Jason Weddington
#### Miscellaneous Chores
- move canonical Sonnet reference to Sonnet 5 (claude-sonnet-5) - (ddba2d3) - Jason Weddington

- - -

## 0.8.1 - 2026-07-15
#### Bug Fixes
- (**ralph**) commit only green finishes; revert-to-green + N consecutive do-overs - (242a57f) - Jason Weddington, *talos-glm*, *Claude Opus 4.8*
#### Documentation
- Session 12 log — ralph CLI (0.8.0), do-over fix, qwen dogfood + 3 gaps (kb-03112) - (1dc5b25) - Jason Weddington
- roadmap — Ralph shipped + forward view (ralph-ability, tasks.md, dispatch mode) - (0322c43) - Jason Weddington
- document talos ralph mode in README - (6190f0c) - Jason Weddington

- - -

## 0.8.0 - 2026-07-14
#### Features
- (**talos**) add `talos ralph` subcommand — thin CLI over run_ralph - (ce422ff) - Jason Weddington, *talos-glm*, *Claude Opus 4.8*

- - -

## 0.7.0 - 2026-07-14
#### Features
- (**ralph**) add the Ralph outer loop (fresh-context restart, stop-command, breakers, per-iteration commit) - (1b4c2bb) - Jason Weddington, *talos-glm*, *Claude Opus 4.8*
#### Bug Fixes
- (**engine**) raise DEFAULT_MAX_TOKENS 4096 → 32768 for reasoning models - (1a7a6ac) - Jason Weddington, *Claude Opus 4.8*
- (**release**) push tags to github as well as origin - (3d07dcc) - Jason Weddington, *Claude Opus 4.8*
#### Documentation
- Ralph core shipped on talos-glm + max_tokens diagnosis (kb-03104) - (7230089) - Jason Weddington, *Claude Opus 4.8*
- add in-run context compaction to roadmap backlog (not planned yet) - (db1ebaf) - Jason Weddington, *Claude Opus 4.8*

- - -

## 0.6.0 - 2026-07-14
#### Features
- (**anthropic**) enable prompt caching with static + rolling cache_control breakpoints - (98fe789) - Jason Weddington, *claude-code-glm*, *Claude Opus 4.8*
- (**eval**) cache-token accounting + claude_code_eval sonnet/real-Anthropic mode - (619ef58) - Jason Weddington, *claude-code-glm*, *Claude Opus 4.8*
#### Documentation
- correct roadmap — GTD adapter shipped at 0.3.5, this release is 0.6.0 - (6b64ef8) - Jason Weddington, *Claude Opus 4.8*
- Session 11 log — checker-only gates decision, prompt caching, benchmark v2 (sonnet ~8x) - (a3e94e1) - Jason Weddington, *Claude Opus 4.8*
- Session 10 log — benchmark verdict, 0.5.1, talos-glm judgment + workspace, context-failure lesson - (3ec6ed5) - Jason Weddington

- - -

## 0.5.1 - 2026-07-13
#### Bug Fixes
- (**talos**) version-stamp from the git tag, not the frozen crate version - (e246511) - Jason Weddington, *Claude Opus 4.8*
#### Tests
- ratchet workspace line coverage to ≥98% (95→98 gate) - (e710cba) - Jason Weddington, *talos-glm*, *Claude Opus 4.8*

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