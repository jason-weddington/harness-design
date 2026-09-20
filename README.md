# harness-design

A learning project to understand agent harness design by building one in Rust —
intended to serve as a headless-dispatch build engine for Agent GTD, supporting
Anthropic API models (Haiku/Sonnet/Opus), AWS Bedrock (Converse API), and local Ollama models. See
[`CLAUDE.md`](./CLAUDE.md) for goals.

## Dev setup

Requires Rust (pinned in `rust-toolchain.toml`; install via
[rustup](https://rustup.rs)). Install the quality-gate tooling as prebuilt
binaries:

```bash
# cargo-binstall (one-time): https://github.com/cargo-bins/cargo-binstall
cargo binstall -y cargo-nextest cargo-llvm-cov cargo-deny cargo-machete \
  typos-cli cargo-sort cargo-release cocogitto
# lefthook + gitleaks are not on binstall — grab their GitHub release binaries
# and put them on PATH (e.g. ~/.cargo/bin).
```

Activate the git hooks (every fresh clone must do this):

```bash
lefthook install
```

## Build & test

```bash
cargo build --workspace
cargo nextest run --workspace      # fast test runner
cargo test --doc --workspace       # doctests (nextest skips these)
```

## Backends

`talos run` selects a model backend from the environment (see `backend_from_env` in `crates/talos/src/main.rs`). Precedence: a `TALOS_BEDROCK` value that is non-empty after trimming selects the AWS Bedrock backend (Converse API) ahead of everything else; otherwise `TALOS_BACKEND` picks `anthropic` (default) or `ollama`.

- `TALOS_BEDROCK` — set to any non-empty (after `.trim()`) value to run on AWS Bedrock (Converse API) instead of the Anthropic API or Ollama. This is for work machines that cannot call the Anthropic API directly. Credentials AND region resolve via the standard AWS chain (env/profile/SSO/IMDS — no keys in source). It wins over `TALOS_BACKEND` / `ANTHROPIC_*` / `OLLAMA_*`; an unset, empty, or whitespace-only value falls through to the `TALOS_BACKEND` match. Only `claude-haiku-4-5` / `claude-sonnet-5` / `claude-opus-4-8` (via `ANTHROPIC_MODEL`) are mapped to Bedrock inference-profile ids; anything else is rejected at construction.
- `TALOS_BACKEND` — `anthropic` (default when unset) | `ollama`.
- `ANTHROPIC_API_KEY` — required for the anthropic backend.
- `ANTHROPIC_MODEL` — optional; default `claude-haiku-4-5`. Used by both the anthropic and bedrock backends.
- `OLLAMA_MODEL` — required for ollama.
- `OLLAMA_BASE_URL` — optional; default `http://localhost:11434`.
- `OLLAMA_API_KEY` — optional bearer token.
- `OLLAMA_NUM_CTX` — optional. A non-empty `u32` is used verbatim and NO probe is made; empty/whitespace is treated as unset. When unset and `OLLAMA_BASE_URL` is a localhost/`127.0.0.1` URL, talos probes the daemon's `POST /api/show` and pins the model's own advertised context length — a probe failure exits `1` with a JSON error on stderr rather than falling back to a constant. When unset and the base URL is non-localhost (Ollama Cloud, a LAN host), no `num_ctx` is set (Ollama's own default applies), so set this variable explicitly for those topologies.
- `OLLAMA_THINK` — `off|on|low|medium|high|max`.
- `TALOS_STATE_RETENTION_DAYS` — optional `u64` days of age-based retention talos applies to its own XDG state dir (`run.sqlite`, `offload/`, transcripts) on every `talos run` start; precedence is `--state-retention-days` flag > this env var > the compiled default of `30`, `0` disables pruning entirely, the env fallback does NOT survive dispatch's sudo boundary (only `TALOS_BACKEND` is kept), and the result is recorded per host in `<state-root>/talos/prune-last.json`.
- `--transcript` — opt-in JSONL run transcript, off by default; a bare `--transcript` defaults to `transcript.jsonl` in the run's state dir next to `run.sqlite`, while `--transcript <path>` uses that path verbatim (no env fallback — see `RunArgs::transcript` in `crates/talos/src/main.rs`).

## Answer mode (`talos run --mode answer`)

**Answer mode** turns talos into a sub-agent that returns *data* instead of a diff: `talos run --mode answer --schema result.json` reads a free-text question (stdin or `--file`), investigates the workspace, and terminates with a `finish(answer)` whose `result` payload conforms to the JSON Schema you supplied. It is the talos-side half of the "talos as the sub-agent of a dynamic workflow" design — see `docs/design/06-answer-mode-and-workflows.md`.

```bash
echo 'Which crates depend on the exec module, and why?' | \
talos run \
  --workspace /path/to/repo \
  --mode answer \
  --schema /path/to/result-schema.json \
  --task-id answer-exec-deps
# exit 40; stdout carries the validated payload at .disposition.Answer.result
```

- **`--mode <build|answer>`** (default `build`) — `build` is the pre-existing `TaskSpec` path, unchanged. `answer` REQUIRES `--schema`; `build` REJECTS it. Both shape errors are checked *before* stdin is read, so a wrongly flagged invocation fails immediately instead of blocking on a pipe.
- **`--schema <path>`** — the JSON Schema the answer's `result` must satisfy. Its raw bytes are shown to the model (key order and formatting survive verbatim) and separately compiled into the validator the harness enforces; a schema-invalid `result` is fed back as a tool-result error, not a termination. The path is used exactly as supplied — resolved against the process CWD, not canonicalized, not confined to `--workspace`, the same as `--file`.
- **Read-only tool registry.** An answer run gets `read_file`, `list_files`, `bash` and `finish` — no `edit_file`, and no `run_checks` (there is no `TaskSpec`, hence no gate command, this cut). Dropping `edit_file` is the convenience, not the enforcement: it keeps many answer agents sharing one checkout off each other's toes and stops the prompt advertising a capability the run would then reject.
- **The tree-UNCHANGED precondition (the enforcement).** Build mode requires evidence that work *happened* before it accepts `finish(done)`. Answer mode inverts it: an accepted `finish(answer)` requires the working tree to be **unchanged** relative to the run-start baseline. A changed tree is rejected with the changed paths as evidence and the loop continues, so an agent that wrote a scratch file can revert and finish. The rule holds no matter which tool did the mutating — `bash` included. An *unobservable* workspace (not a git work tree, `git` missing, the status call timed out) fails **open** and is recorded: the accepted `Disposition::Answer` carries `change: Unobservable{reason}` and `RunStats::tree_baseline_unobservable` is `true`, so you can always tell a verified read-only answer from an unverifiable one. Symmetrically, `finish(done)` and `finish(already_satisfied)` are rejected in answer mode with steering toward `answer`.
- **Exit 40**, not 0, 20 or 30. `agent-gtd-dispatch`'s `talos.py::map_talos_result` sets `push=True` only on exit 0 with parseable stdout (talos.py:285-306) — and a validated answer has nothing to push, so 0 would be wrong. It is not a task failure, so 20 would be wrong. It is not an already-satisfied build run, so 30 would be wrong; keeping 40 distinct from 30 lets a future mapper arm tell answer-with-payload from already-satisfied. Today both 30 and 40 fall to that mapper's unknown-exit-code catch-all (talos.py:342-346), which fails safe (status `failed`, `push=False`) — the right landing spot until the worker grows an arm for answer mode. Answer mode is deliberately unreachable from dispatch in this cut.
- **Reading the result.** `RunSummary` gains no field: the payload rides out on stdout through the embedded, externally-tagged `Disposition`, so the orchestrator's read path is `.disposition.Answer.result`, with `.disposition.Answer.change` as the read-only evidence.

Two operator notes:

1. **Concurrent answer agents on one host MUST each pass a distinct `--task-id`** (or a distinct `--run-store` plus `--offload-dir`). The default `talos-run` resolves to one state dir and one run id (`talos-run:1`), so parallel runs would collide on the same `run.sqlite` and the same run record.
2. **`tree_dirty` and `mutating_iters` are not the read-only oracle.** In answer mode they count *successful `bash` calls*, not observed tree changes, so a run that only grepped will still report them set. The authoritative read-only evidence is `Disposition::Answer.change` (and, on a transcript, the `run_end` stats `edit_file_calls_ok` and any `finish_rejection: "modified_workspace"` rows).

## Ralph mode (`talos ralph`)

The **Ralph loop** drives an agent toward an objective by re-invoking the inner engine with a **fresh context every outer iteration** — durable state lives *outside* the context window (the code on disk, the git history, and a notes file the agent reads-then-appends), so each pass starts cold and still makes forward progress. Each iteration does exactly one unit of work; the **harness owns a git commit per iteration** (a deliberate ralph-only exception to the worker-owns-git rule). Distinct from finish-recovery (which nudges the *same* context when a gate is red) — Ralph *restarts* the context. Core: `crates/harness/src/ralph.rs`.

Two commands, deliberately **never collapsed** — get this wrong and the loop misbehaves:

- **`--gate` (inner, per-iteration):** the `run_checks` command the inner engine uses to verify a `finish(done)` claim. The harness forces the agent to loop until this is green *before* it can finish, so the tree is already green when the per-iteration commit fires.
- **`--stop-when` (outer objective oracle):** a command run via `/bin/sh -c` whose exit `0` means "objective met, stop the whole loop." This is the goal, not the per-iteration bar.

**The load-bearing gotcha (why gate ≠ stop-when):** ralph commits ONLY green `finish(done)` finishes — the harness guarantees the inner `--gate` set is green *before* it commits, so the per-iteration `git commit` (which runs the repo's pre-commit hook and is **not** `--no-verify`'d) sees a green tree. Any *non-green* inner outcome, OR a green commit whose pre-commit hook rejects it, is **reverted** to the last green commit (`git reset --hard HEAD` + `git clean -fd`, ignored files like `target/` preserved, the iteration's `PROGRESS.md` append discarded — a clean do-over) and the loop retries with a fresh context. After `--max-do-overs` (default 3) *consecutive* do-overs with no green commit between them, the loop terminates with a `DoOversExhausted` terminal (exit 20, a task-side failure like `Stuck` — not the exit-1 infra `Error`). So the pre-commit hook must be **check-only** (never a `--fix`/formatter hook that mutates — see kb-03099) and must match the inner `--gate` set. Two things must **not** be in the commit hook: (1) the `--stop-when` threshold (e.g. a coverage floor) — it's false until the objective is met, so it would fail every commit and burn do-overs; and (2) a **conventional-commit-msg** hook — ralph's commit messages are `ralph: iteration N — <objective>`, which such a hook rejects.

Example — grind an unhealthy repo up to 90% test coverage on a local Ollama model:

```bash
TALOS_BACKEND=ollama OLLAMA_MODEL=qwen3.6:35b OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_THINK=on \
talos ralph \
  --workspace /path/to/repo \
  --objective 'Raise coverage to 90%. Each iteration: run coverage, pick the single highest-value untested function, write ONE test for it, verify it passes, append a note to PROGRESS.md, then finish.' \
  --stop-when 'uv run --frozen pytest --cov=<pkg> --cov-fail-under=90 -q' \
  --gate 'uv run --frozen ruff check . && uv run --frozen ruff format --check . && uv run --frozen pytest -q' \
  --notes-file PROGRESS.md \
  --max-ralph-iterations 25
```

The workspace **must already be a git work tree** — `run_ralph` does *not* run `git init`. Backend selection reuses the same `TALOS_BEDROCK` / `TALOS_BACKEND` / `ANTHROPIC_*` / `OLLAMA_*` env as `talos run`. Other flags: `--inner-max-iterations` (inner cap per pass, default 500), `--stuck-k` (consecutive no-progress passes before giving up — progress = a git diff *outside* the notes file, default 3), `--max-do-overs` (consecutive non-green / rejected-green-commit do-overs before `DoOversExhausted` — each reverted to the last green commit — default 3), `--ralph-wall-clock-secs` (0 = unbounded; also `TALOS_RALPH_WALL_CLOCK_SECS`), `--stop-when-timeout-secs` / `--gate-timeout-secs` (default 300). Ralph is **not** run-record persisted this cut — it prints a `RalphSummary` JSON (objective / terminal / outer_iterations / total_inner_iterations) to stdout and exits: **0** `StopConditionMet` · **20** `Stuck` / `MaxIterationsExhausted` / `TimeBudgetExhausted` / `DoOversExhausted` · **1** `Error` (git/spawn/revert failure). Watch progress via the `ralph: iteration N` commits and the notes file, not stdout.

## Quality gates

Run by lefthook locally and re-run in CI (`.github/workflows/ci.yml`), which is
the real enforcement boundary since local hooks can be skipped with
`--no-verify`.

| Stage | Gates |
|---|---|
| commit-msg | conventional commits (`cog verify`) |
| pre-commit | `cargo fmt --check`, `cargo clippy -- -D warnings`, `typos`, `cargo sort --check`, `gitleaks`, `cargo nextest run` |
| pre-push | coverage `--fail-under-lines 98`, `cargo test --doc`, `cargo machete`, `cargo deny check` |
| CI (scheduled) | `cargo audit` (advisories disclosed after merge) |

What the Rust compiler enforces for free (so there's no separate gate): full
type checking, null-safety (`Option`), memory/thread safety (borrow checker),
match exhaustiveness, unused imports/variables, and `unsafe` is `forbid`-den
project-wide. The gates only add what rustc can't see.

**Docs-only skip:** `scripts/docs-only.sh` is a fail-safe predicate wired into `lefthook.yml` via `skip:` blocks: pre-commit `clippy` + `test` and all four pre-push gates (`coverage`, `doctest`, `machete`, `deny`) skip themselves when EVERY changed path is under `docs/` or is a top-level `*.md`. The skip is proof-based and fail-safe — an empty changeset, a mixed changeset, a rename out of a source directory, a missing upstream, or a broken/deleted predicate all run everything. The pre-push skip additionally requires a configured upstream, so the first `git push -u` of a new branch always runs every gate. CI always runs the full set and is the audit for a wrong local skip: if CI fails clippy/test/coverage on a commit whose local hook printed `(skip) by condition`, `scripts/docs-only.sh` has regressed — and `lefthook run pre-commit --verbose` shows the resolved path list and the `docs-only:` reason line.

**Coverage ratchet:** the `--fail-under-lines` literal lives in `lefthook.yml`
and `.github/workflows/ci.yml`. Bump both upward as coverage improves; never let
it regress.

## Release

Decoupled from deploy (matches the Python projects). At a meaningful boundary:

```bash
./release.sh   # cog bump --auto -> tag -> push origin main --tags
```

## Publishing talos to the dispatch fleet

talos is self-hosting — we dispatch talos features to talos — so the binary
changes every wave. Rather than each dispatch host compiling it (slow, especially
the aarch64 Pi), the fast x86_64 dev box builds **both** arches and publishes them
to the homelab artifact host (`pi-04`); the fleet's `talos-update.sh`
(in `agent-gtd-dispatch`) then just pulls the binary.

```bash
./scripts/publish-talos.sh    # gate -> build x86_64 + aarch64 -> scp to pi-04 -> advance 'latest'
```

- **Version token.** `talos --version` prints `talos <semver>-g<short-sha>`
  (stamped by `crates/talos/build.rs`). It is a single URL/path-safe string; the
  consumer reads it via `talos --version | awk '{print $2}'`.
- **pi-04 layout.** `pi-04:/srv/talos/<TOKEN>/<arch>/talos` where `<arch>` ∈
  `{x86_64, aarch64}` (matching `uname -m`), plus `pi-04:/srv/talos/latest`
  holding the current `<TOKEN>` on one line — advanced only after both arches
  upload. Artifacts are immutable per-`<TOKEN>` (a gate-failing or dirty-tree
  build never ships; an existing `<TOKEN>` dir is never overwritten).
- **One-time cross-toolchain setup** (on the x86_64 publisher):
  `rustup target add aarch64-unknown-linux-gnu` and install the linker
  `gcc-aarch64-linux-gnu` (Debian/Ubuntu). The linker is wired in
  `.cargo/config.toml`.
- **Target override.** `TALOS_PUBLISH_HOST` (default `jason@pi-04`) and
  `TALOS_PUBLISH_DIR` (default `/srv/talos`).

### macOS binary

`publish-talos.sh` cannot build for macOS (cross-compiling needs an Apple SDK
plus a C cross toolchain, and `ring` adds one), so the Mac builds its own. Run
this ON THE MAC after a Linux publish:

```bash
./scripts/publish-talos-mac.sh    # build the commit `latest` names -> pi-04:/srv/talos/<TOKEN>/aarch64-apple-darwin/talos
```

It is a **follower**: it reads `latest`, builds exactly that commit in a
detached worktree (persistent target dir, so rebuilds are incremental), asserts
`talos --version` equals the token, and uploads without overwriting. It never
touches `latest`, and the `latest` invariant stays "both Linux arches present",
so a Mac that lags never blocks the fleet. `talos-update.sh` ignores the extra
directory. Interactive machines (the Mac, jason-desktop) install with
`~/scripts/pull_talos.sh`, which reads the same artifact tree and places the
binary at `~/.local/bin/talos`.
