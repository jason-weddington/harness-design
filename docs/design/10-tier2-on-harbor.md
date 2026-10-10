# 10 — Tier-2 on Harbor

## Purpose

Harbor is the tier-2 instrument. The mined tier-2 tasks are emitted as Harbor task dirs, `TalosAgent` runs the published talos binary inside each trial container, and `mined_score.py` scores the sealed re-gate in the verifier. The in-process runner and the calibration report (`talos-harbor-calibrate`) were retired on 2026-10-10, after the result of record in [09-harbor-calibration.md](09-harbor-calibration.md) showed the Harbor path scoring the same tasks the same way (MATCH). `scripts/tier2-harbor.sh` is the one command for a run, and `talos-harbor-tier2-report` prints its table.

## Emit

On a host that has the private talos-evals checkout and the source repos at each task's `repo_path`, run `cargo run -p harness --example mined_to_harbor -- --tasks-dir <talos-evals>/tasks --out <dir> [--spec-level s2]` (s2 is the default and the tier-2 level). The emitter prints one `emitted` line per task, the `base image:` line, and a `tier2:` line with the `scripts/tier2-harbor.sh` invocation for that `--out`.

The output holds git bundles of private repos and must never be committed. The emitter refuses an `--out` inside a git work tree.

## Move

Copy the whole emitted dir, `_base/` included, to a Docker host that has a checkout of this repo with `harbor/` synced (`cd harbor && uv sync --frozen`). Put it at a path outside any git work tree; `scripts/tier2-harbor.sh` refuses a `--tasks` dir inside one.

## Run

`usage: scripts/tier2-harbor.sh --tasks <emitted-harbor-tasks-dir> --jobs <out-dir> [-k N (default 3)] [-n N (default 1)] [--talos-version <token> | --talos-path <binary>] [--task <id>]...`

Local-GPU Ollama (wrapped in `scripts/gpu-eval.sh`, which takes the shared GPU lock and refuses to start while another model is resident): `TALOS_BACKEND=ollama OLLAMA_BASE_URL=http://<gpu-host>:11434 OLLAMA_MODEL=<model> OLLAMA_NUM_CTX=<n> OLLAMA_THINK=<level> TALOS_ARTIFACT_BASE=<artifact-host> scripts/tier2-harbor.sh --tasks <tasks> --jobs <jobs>`

Ollama cloud (no GPU lock, and `-n` may be raised): `TALOS_BACKEND=ollama OLLAMA_BASE_URL=https://ollama.com OLLAMA_API_KEY=<key> OLLAMA_MODEL=<model> OLLAMA_THINK=<level> TALOS_ARTIFACT_BASE=<artifact-host> scripts/tier2-harbor.sh --tasks <tasks> --jobs <jobs>`

Anthropic: `TALOS_BACKEND=anthropic ANTHROPIC_API_KEY=<key> TALOS_ARTIFACT_BASE=<artifact-host> scripts/tier2-harbor.sh --tasks <tasks> --jobs <jobs> --talos-version <token>`

Without `--talos-version` or `--talos-path`, `TalosAgent` installs the `latest` binary from `TALOS_ARTIFACT_BASE`; `--talos-path` uploads a local binary instead. `--task <id>` (repeatable) restricts the run to those task dirs (Harbor's `-i`), and the report's expected set follows it.

`--jobs` must sit outside any git work tree, because jobs carry transcripts and gate output with private source; it must also be absent or empty, so one jobs dir holds one run. Before it does anything else, the script prints one decision line on stderr (`tier2-harbor: backend=... gpu_lock=... talos=... k=... tasks=... jobs=... include=...`) that records how the run was configured.

The script then builds the base image (`scripts/harbor-build-base.sh`), runs `harbor run`, and runs the report whenever Harbor wrote anything to the jobs dir. It persists nothing of its own.

## Parity knobs

A run measures the same row as the calibration of record: spec level s2, test-first on, agent gate on, `gate_timeout_secs=900`, `wall_clock_secs=3300`, `max_iterations=500`, `-k` default 3, `--max-retries 0` and `-n 1` (the default). The first three are baked into the emitted tasks; the rest are on the `harbor run` line in `scripts/tier2-harbor.sh`, and the test `tier2_script_carries_the_parity_kwargs` in `crates/harness/src/mined_harbor.rs` pins them to the Rust constants (`DEFAULT_AGENT_GATE_TIMEOUT`, `HARBOR_WALL_CLOCK_SECS`, `HARBOR_MAX_ITERATIONS`).

`--max-retries 0` matters: a Harbor retry re-runs a trial that failed on infrastructure and keeps the retry's score, which would turn infra-Invalid trials into scored ones.

`OLLAMA_NUM_CTX` and `OLLAMA_THINK` are forwarded from the host env into the trial, so they must be set deliberately; they show up in the report's `config:` line as `num_ctx=` and `think=`, and two runs that differ there are not the same row.

## Why -n 1

One local GPU serves one request at a time, so concurrent trials queue behind each other and the queueing time burns each trial's wall clock. The script therefore refuses `-n` other than 1 when it wraps the run in the GPU lock. Hosted backends (Anthropic, Bedrock, Ollama cloud) may raise `-n`. `scripts/gpu-eval.sh`'s lock only serialises runs launched on the host that holds `GPU_LOCK_FILE`; a run launched elsewhere against the same GPU is not stopped by it.

For any long local-GPU run, also set `GPU_TELEMETRY_SSH=<gpu-host>` (and `GPU_TELEMETRY_UNIT=<fan-controller unit>` if the host runs one, with `GPU_TELEMETRY_DIR=<jobs>/gpu-telemetry` to keep it beside the run). `gpu-eval.sh` then samples temperature, power, SM clock and the clock-event-reason bitmask every 5 s on the GPU host, saves the unit's journal for the run window, and prints one summary line counting thermal-slowdown samples (bits `0x20`/`0x40`). The cooling on the shared 5090 is tuned for noise against a no-throttling constraint, so a run that shows thermal slowdown, or coolant settling above 46 °C in the fan log, is worth reporting to whoever owns the host's fan curve; it may also have slowed the trials.

## Image reuse

Harbor 0.9.0's default `--delete` runs `docker compose down --rmi all` after every trial, so each task image is removed when its trial ends, while the base image `talos-evals-base:1` persists. A later build of the same task reuses Docker's build cache, so the `env.setup` and baseline-tripwire layers are not re-run; this is reliance on the build cache, not a measured guarantee. `docker build --no-cache` (or `docker builder prune`) resamples the tripwire.

## Reading the report

`talos-harbor-tier2-report <jobs> --expect-k N --tasks <dir>` can be re-run on a jobs dir at any time; the script runs it with its own `-k` and `--tasks` (and `--task` filters). It accepts the jobs dir or a single job dir inside it.

stdout is tab-separated (`\t` below), in order: the `tier2: <jobs> (<N> trials, <T> tasks)` line; the `config:` line (distinct `agents`, `talos` versions, `backends` with `num_ctx` and `think`, `spec_levels` and `agent_kwargs` across all trials); the header `task\tresolved/k\tunresolved\tinvalid\tclaimed-done-but-unresolved`; one row per task (an expected task with no trials shows `0/0`); the `TOTAL` row; one `invalid\t<task>\t<trial>\t<reason>` line per invalid trial; and, with `--expect-k`, one `k-mismatch\t<task>\t<k>/<N>` line per task whose trial count is not N. The last column counts trials where talos claimed Done but the sealed re-gate did not resolve (invalid trials never count).

Each trial is classified by the first rule that matches: (1) `metadata.talos.outcome` is `BackendError`: invalid `backend-error`; (2) Harbor recorded an exception: invalid `harbor-exception(<type>)`, plus the first line of its message; (3) no `verifier/score.json`: invalid `no-score-json`; (4) an unreadable score.json or an unknown score: invalid `score-json-unreadable`; (5) score `invalid`: invalid with the scorer's `detail` (or `score-invalid-no-detail`); (6) otherwise resolved or unresolved.

stderr WARN lines, in order: `task-id mismatch` (the score.json task id differs from Harbor's task name); `<n> harbor job dirs merged`; `no --expect-k/--tasks: missing trials cannot be detected`; `unexpected-task <task>` (a task outside the expected set); `mixed-config <field>` (more than one talos version, backend or agent kwargs set); `gate-not-armed <task> <trial>` (a scored trial whose agent gate was not armed); `no valid trials`; and `talos-error <task> <trial>: <error>`.

Exit codes: 0 a clean measurement; 2 an input error (one `error:` line on stderr, nothing on stdout); 3 not a clean measurement (a k mismatch, an unexpected task, mixed config, a gate that was not armed, or no valid trials); 75 GPU busy, passed through from `gpu-eval.sh`. `scripts/tier2-harbor.sh` exits with Harbor's code when Harbor failed, else with the report's.

Never use an exit-3 table for a default-flip decision. Fix the cause (re-run the missing trials into a fresh jobs dir, or pin the config) and report from a clean run.
