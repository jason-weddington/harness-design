# talos-harbor

`talos_harbor.agent:TalosAgent` is a [Harbor](https://github.com/harbor-framework/harbor) `BaseInstalledAgent` that runs the talos binary inside a Harbor trial container. Harbor owns the container and the verifier; talos owns the agent loop. The adapter installs a talos binary into `/installed-agent`, hands talos the task instruction verbatim through `talos run --instruction-file`, and copies talos's own outcome record into the trial's `AgentContext` after the run.

It is pinned to `harbor==0.9.0` and needs a talos binary that supports `talos run --instruction-file`.

## Using TalosAgent from another repo (kb-bench)

Add the package as a uv dependency straight from this repo's `harbor/` subdirectory (replace the placeholder remote with yours):

```toml
[project]
dependencies = ["talos-harbor @ git+ssh://<your-remote>/harness-design#subdirectory=harbor"]
```

Then point Harbor at the class:

```bash
uv run harbor run -p <task-or-dataset> --agent-import-path talos_harbor.agent:TalosAgent --ak wall_clock_secs=540
```

### Agent kwargs (`--ak key=value`)

| kwarg | default | meaning |
|---|---|---|
| `talos_path` | `None` | Local path to a Linux talos binary. Uploaded as-is; wins over `artifact_base` when both are set. |
| `talos_version` | `"latest"` | Artifact token to install from `artifact_base`. `latest` reads `<artifact_base>/latest` first; anything else is used verbatim (e.g. `0.14.0-gabc1234`). |
| `artifact_base` | `$TALOS_ARTIFACT_BASE` | Base URL of a talos artifact host laid out as `<base>/latest` and `<base>/<token>/<arch>/talos` (`x86_64` or `aarch64`). A trailing `/` is stripped. |
| `max_iterations` | unset | Passed as `talos run --max-iterations`. |
| `wall_clock_secs` | unset | Passed as `talos run --wall-clock-secs`. |
| `gate_timeout_secs` | unset | Passed as `talos run --gate-timeout-secs`. |
| `memory` | `"off"` | `off`, `annotate`, `deny-once` or `belief-check`; anything else (another casing, `deny_once`, `null`, a bool or a number) raises `ValueError` when the agent is constructed, before any container exec. `off` passes no flag; `annotate`, `deny-once` and `belief-check` pass `talos run --memory <mode>`. In a Harbor YAML job config the value must be quoted (`memory: "off"`), because YAML 1.1 reads a bare `off` as false, which is rejected. |
| `kb_project` | `None` | A KB project ref passed as `talos run --kb-project`, and only with memory on: with `memory=off` it raises `ValueError`. It must match `[A-Za-z0-9][A-Za-z0-9._-]*`. When it is unset, talos resolves the project by walking up from the workspace to `.kb_project`. |

One of `talos_path` or `artifact_base`/`TALOS_ARTIFACT_BASE` is required; the constructor raises `ValueError` otherwise. The binary is downloaded on the host (consumer images often have neither curl nor wget), into a host temp dir rather than the logs dir, and then uploaded into the container. Unset `max_iterations`/`wall_clock_secs`/`gate_timeout_secs` emit no flag, so talos's own defaults apply.

Set `wall_clock_secs` below the task's `[agent] timeout_sec`. A Harbor agent timeout cancels the exec client but does not kill talos, so without talos's own wall-clock budget the verifier could score a tree that talos is still editing.

### Model selection

Harbor's `-m` / `model_name` is ignored. talos picks its backend and model from the environment it is given, which TalosAgent forwards from the host:

- only the selected backend's keys, chosen from the host env with talos's precedence (`metadata.talos.backend_selected` records it): **bedrock** (non-empty `TALOS_BEDROCK`) forwards `TALOS_BEDROCK`, `ANTHROPIC_MODEL` and every `AWS_*` variable; **ollama** (`TALOS_BACKEND=ollama`) forwards `TALOS_BACKEND`, `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, `OLLAMA_NUM_CTX`, `OLLAMA_THINK`; **anthropic** (otherwise) forwards `TALOS_BACKEND`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`. Other backends' credentials are never forwarded. A `TALOS_BACKEND` other than unset, `anthropic` or `ollama` raises `ValueError`;
- every `--ae KEY=VALUE` extra, verbatim, in every mode (so `--ae TALOS_TOKEN_BUDGET=5` reaches talos). An `--ae` value wins over the host value for the same key.

Unset keys are omitted, never sent empty. Nothing else from the host environment is forwarded.

### The gate command

talos's gate (`--gate-command`, registered as `run_checks`) comes from `AGENT_GATE_COMMAND` inside the container, never from the host. On a Dockerfile-only task, set it in `task.toml`:

```toml
[environment]
env = { AGENT_GATE_COMMAND = "uv run pytest -q" }
```

A task with its own `docker-compose.yaml` does not get `[environment] env` on its execs, so it must also set `AGENT_GATE_COMMAND` in the main service's `environment:`. An unset or empty `AGENT_GATE_COMMAND` runs talos with no checks; `metadata.talos.gate_armed` records which happened.

### Memory (personal-kb)

Memory is off by default and `off` emits no flag, so any talos binary works and tier-2 runs are unchanged.

`annotate` and `deny-once` need a talos binary whose `talos run` has `--memory`; `belief-check` needs a talos binary whose `--memory` accepts `belief-check`. An older binary exits with a CLI usage error, which raises `NonZeroAgentExitCodeError` (an Invalid trial).

TalosAgent never forwards `PERSONAL_KB_URL` or `PERSONAL_KB_API_KEY` from the host; set them with `--ae`. A trial image's own `ENV` also reaches talos, because the exec inherits it, and `metadata.talos.forwarded_env_keys` cannot show that.

For kb-bench's seeded in-container KB daemon, pass both explicitly as `--ae PERSONAL_KB_URL=http://127.0.0.1:8765 --ae PERSONAL_KB_API_KEY=local-no-auth`, the same pair kb-bench's Claude Code arm uses; inside the trial container, 127.0.0.1 is the container's own loopback. A real key goes in the template form `--ae 'PERSONAL_KB_API_KEY=${PERSONAL_KB_API_KEY}'`, so the key never appears on the command line: Harbor resolves `${VAR}` from the host env when it creates the agent.

TalosAgent does not configure the KB daemon. kb-service returns a Bash cue index only when its soft gate is enabled (`KB_SOFT_GATE_ENABLED=TRUE` in the daemon's environment). Otherwise only the session slice reaches talos and per-call advice never fires. In kb-bench those daemon switches are set by kb-bench's own arm wrapper (`kb-bench-seed configure`), not by TalosAgent.

Eval isolation: tier-2 and kb-bench run with memory off, or against a seeded offline KB daemon, never the live KB. TalosAgent does not enforce this. The field to audit is `metadata.talos.memory.source`, talos's own `memory_slice.source`, which names the KB talos actually used. kb-bench's contamination audit reads `agent/trajectory.json`, which TalosAgent does not write, so it reports talos trials as unaudited.

Fail-open: if the KB is unreachable, auth fails or the response is malformed, talos runs the whole trial with memory off. `metadata.talos.memory.memory_degraded` then carries the reason, and `metadata.talos.error` gains `memory degraded to off (<reason>)`. kb-service answers a server-side failure or an unresolvable project with an empty 200 response, not an error, so that case need not show as degraded; its signature is `memory.slice_items == 0` with memory on.

`talos_harbor.tier2_report` prints `metadata.talos.error` as `WARN: talos-error` lines and does not reclassify the trial. A memory-arm analysis must report as Invalid every trial whose `metadata.talos.memory.memory_degraded` is set or whose `metadata.talos.error` contains `memory`.

kb-bench scenarios write `/app/.kb_project`, so `kb_project` is normally left unset there.

### Trial-image requirements

- glibc >= 2.39: `ubuntu:24.04`, Debian trixie (e.g. `node:22-trixie-slim`), or newer.
- `bash`, and `git` on `PATH`.
- The task workdir is a git repo with at least one commit, and is either owned by the agent user or marked with `git config --system --add safe.directory <workdir>`. Otherwise talos cannot observe its own changes and will not report Done.

### Outputs

Everything lands in the trial's agent logs dir (`/logs/agent` in the container):

- `run-summary.json`: talos's stdout run summary.
- `transcript.jsonl`: the full talos transcript.
- `talos-stderr.log`: talos's stderr.
- `gate-command.txt`: the gate command talos was given (empty when none).
- `run.sqlite`: the talos run store.
- `offload/`: talos's offloaded tool outputs.
- `setup/install.json`: where the binary came from (`source`, `talos_path`, `requested_version`, `resolved_token`, `arch`, `url`, `sha256`).

Token counts go to `AgentContext`: `n_input_tokens` = input + cache read + cache write, `n_cache_tokens` = cache read, `n_output_tokens` = output, all from the transcript's `run_end` stats. `cost_usd` is left unset.

`AgentContext.metadata["talos"]` holds:

| key | value |
|---|---|
| `exit_code` | talos's exit code, or `None` if none was recorded (timeout, or the exec never ran). |
| `outcome` | `run-summary.json` `outcome`: `Finished`, `StoppedWithoutFinish`, `MaxIterations`, `BudgetExhausted` or `BackendError`. |
| `disposition` | the disposition variant: `Done`, `AlreadySatisfied`, `Answer`, `Blocked` or `Failed`. |
| `claimed_done` | `disposition == "Done"`; `None` (not `False`) when there is no disposition. |
| `iterations` | from the run summary. |
| `nudges_fired`, `gates_green_at_exit`, `tree_dirty`, `mutating_iters`, `edit_file_calls_ok`, `bash_calls_ok` | from the transcript's `run_end` stats. |
| `version` | the installed talos version (`talos --version`). |
| `gate_command` | contents of `gate-command.txt`, `None` if missing or empty. |
| `gate_armed` | whether a gate command was given. |
| `backend` | the run summary's `backend_settings`, verbatim. |
| `backend_selected` | `bedrock`, `ollama` or `anthropic`: which backend's env was forwarded. |
| `forwarded_env_keys` | the sorted names (never values) of the env keys forwarded to talos. |
| `install` | the parsed `setup/install.json`. |
| `tokens` | raw `{input, cache_read, cache_write, output}` from `run_end`, or `None`. |
| `compactions`, `highest_compaction_tier` | from the run summary. |
| `memory` | a 13-key record: `mode`, `kb_project`, `effective_mode`, `memory_annotations`, `memory_denies`, `memory_kb_tool_calls`, `memory_deny_overrides`, `memory_degraded`, `source`, `slice_items`, `slice_chars`, `memory_tools`, `memory_tools_dropped`. `mode` and `kb_project` are the agent kwargs as passed: `kb_project` is `None` when unset, even when talos resolved a project from `.kb_project`. `effective_mode` is `run_start.config.memory_mode`, the mode talos actually ran in, or `None` when it is absent. The four counters come from `run_end` stats and are `None` when `run_end` or the key is missing. `memory_degraded` is the degradation reason or `None`. `source`, `slice_items`, `slice_chars`, `memory_tools` and `memory_tools_dropped` come from the transcript's `memory_slice` event and are `None` when it is absent. |
| `stderr_tail` | the last 2048 bytes of `talos-stderr.log`, only when the run did not reach an outcome or `run_end` is missing. |
| `error` | present only when something is off: missing or unparsable outputs, no exit code, an exit-code/disposition mismatch, an install token that differs from the binary's version, or a memory cause: memory on but `run_end` lacks the counters, memory degraded, `run_start` lacks `memory_mode`, a mode mismatch, memory on with no `memory_slice`, memory off with memory activity, or an invariant violation. |

### Exit codes and Invalid trials

talos exit codes 0 (Done), 10 (Blocked), 20 (Failed / stopped / out of budget), 30 (AlreadySatisfied) and 40 (Answer) are outcomes, and the trial proceeds to the verifier normally. Exit code 1 (talos BackendError) and any other code raise `NonZeroAgentExitCodeError` with the tail of `talos-stderr.log` in the message.

Harbor still runs the verifier after `NonZeroAgentExitCodeError`. Treat a trial whose `exception_info` is `NonZeroAgentExitCodeError` or `AgentTimeoutError`, or whose `metadata.talos.outcome` is `BackendError`, as an infra failure and report it as Invalid, not as a model failure.

## Development

The package's gate chain, run from `harbor/` (lefthook runs it on commits touching `harbor/**` and on every push, and CI runs it as the `harbor` job):

```bash
cd harbor && uv sync --frozen && uv run --frozen ruff check . && uv run --frozen ruff format --check . && uv run --frozen mypy && uv run --frozen pytest -q
```

The tests are hermetic: no Docker, network, model or real talos binary. Coverage must stay at or above 90%.

`scripts/harbor-smoke.sh` is an opt-in end-to-end run of `examples/hello-talos` in Docker. It needs a running Docker daemon, a talos binary supporting `--instruction-file` (`TALOS_PATH=<local binary>` or `TALOS_ARTIFACT_BASE=<artifact host>`), and a backend env such as `ANTHROPIC_API_KEY`. It prints `PASS <jobs-dir>` when the trial scores 1 and talos wrote its run summary. No hook, CI job or test runs it.

## Running the tier-2 mined tasks on Harbor

`cargo run -p harness --example mined_to_harbor` converts talos-evals tier-2 mined tasks into Harbor task dirs. It reads the private talos-evals checkout and the source repos at each task's `repo_path`, so it is opt-in and no hook, CI job or test runs it. Its output contains git bundles of private repos and must never be committed; the emitter refuses any `--out` inside a git work tree.

The recipe, on a host that has the talos-evals checkout and the source repos:

```bash
cargo run -p harness --example mined_to_harbor -- --tasks-dir ~/git/talos-evals/tasks --out /tmp/harbor-tasks   # [--task <ID>] [--spec-level s1|s2|s3], default s2
TALOS_BACKEND=anthropic ANTHROPIC_API_KEY=... TALOS_ARTIFACT_BASE=... scripts/tier2-harbor.sh --tasks /tmp/harbor-tasks --jobs /tmp/harbor-jobs
```

The emitter prints a `tier2:` line, the `scripts/tier2-harbor.sh` invocation for its `--out` (runbook: [docs/design/10-tier2-on-harbor.md](../docs/design/10-tier2-on-harbor.md)). Each task.toml records `spec_level`, `test_first` and `agent_gate` in `[metadata]`.

`scripts/harbor-build-base.sh` builds `<out>/_base` as `talos-evals-base:1`, which every emitted Dockerfile builds `FROM`. The image is never pushed, so the build must use the default `docker` buildx driver (the local image store); a `docker-container` builder makes the task builds fail with 'pull access denied'.

A task whose `uv.lock` (the primary's at `parent_sha`, or a sibling's at its pin) has a `source = { git = "<url>" }` over ssh (`ssh://…` or scp-style `user@host:path`) gets that dependency vendored, because the image build has no ssh client or keys and must not get them. The emitter clones the url on the host (the only network access, and only in this opt-in tool) and writes the pinned commit's ancestry to `environment/gitdeps/<name>.bundle`; the Dockerfile unpacks it into a local bare repo and sets `url."file://…".insteadOf "<url>"` system-wide, so `uv sync --frozen` resolves offline from the pinned commit. A missing commit fails the build with `TALOS_EVAL_INVALID reason=gitdep-missing-commit`. An ssh source without a `#<sha>` pin, or one url pinned at two commits, is an emit error. https git sources and registry deps are left alone.

Each task's image build is also its environment check. A bundle whose `HEAD` is not the expected sha, a failing `env.setup`, or an agent gate that is red at the parent commit fails the image build with a `TALOS_EVAL_INVALID reason=<prefix>` line (`baseline-head-mismatch`, `setup-failed`, `agent-gate-red-at-parent`) in the trial's exception message; `talos-harbor-tier2-report` classifies those trials as Invalid, not as model failures. The agent-gate tripwire runs once per image build, not per trial, so a cached image does not re-check it; `docker build --no-cache` resamples it.

The verifier (`tests/test.sh`) writes its outputs to `/logs/verifier`:

| File | Contents |
|---|---|
| `steps.jsonl` | one `{"step", "rc"}` line per verifier step (the `gate` step also has `gate_ms`). |
| `agent-gate-post.out` | the agent gate re-run after the agent finished. |
| `porcelain.txt`, `porcelain.err` | `git status --porcelain -uall` of `/app` before the sealed copy. |
| `authored.json` | the agent's added and modified test files, from `mined_score.parse_authored_tests`. |
| `verify-meta.json` | `HEAD`, the `talos-baseline` sha and the commits ahead of it. |
| `sealed-check.txt` | any sealed file that did not land byte-for-byte. |
| `gate.out`, `gate.err` | stdout and stderr of the sealed scoring gate. |
| `reward.txt`, `score.json` | the scorer's verdict. |

A failed sealed copy, sealed check or gate-command read exits before the scorer, so no reward file is written and Harbor raises `RewardFileNotFoundError`; `talos-harbor-tier2-report` counts those trials as Invalid.
