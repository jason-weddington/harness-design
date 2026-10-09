# 09 — Harbor calibration: does the Harbor path score the same as mined_eval?

## Purpose

The tier-2 mined tasks now run two ways: in-process through `mined_eval` (the harness drives talos's engine against a local worktree and re-gates with the sealed tests), and through Harbor (`mined_to_harbor` emits a Harbor task dir per mined task, `TalosAgent` runs the talos binary in a trial container, and `mined_score.py` scores it in the verifier). Before any Harbor number is trusted as a tier-2 number, the two paths have to be shown to score the same tasks the same way. `talos-harbor-calibrate` is that check: it reads one Harbor job and one `mined_eval` run over the same tasks and says whether their resolved rates are statistically compatible.

## Parity knobs

A calibration is only meaningful when both runs measure the same row. These are the knobs that must match, and how each side sets them:

| knob | mined_eval | Harbor | checked by the report |
|---|---|---|---|
| backend + model | `EVAL_BACKEND` plus `ANTHROPIC_*` / `OLLAMA_*` | `TalosAgent` forwards `TALOS_BACKEND` plus the same model vars from the host | `parity model` (Harbor `metadata.talos.backend.model` must be a substring of every mined `backend_desc`) |
| spec level | `MINED_EVAL_SPEC_LEVEL=s2` | emitter `--spec-level s2` (recorded in each task.toml `[metadata] spec_level`) | `parity spec_level` (read from the task.toml behind `config.task.path`; `unknown` fails closed) |
| k | `MINED_EVAL_K` | `harbor run -k` | per task: `k-mismatch` |
| max iterations | `MINED_EVAL_MAX_ITERATIONS=500` | talos default 500 (or `--ak max_iterations=500`) | `parity max_iterations` |
| agent gate timeout | `DEFAULT_AGENT_GATE_TIMEOUT` = 900s (`MINED_EVAL_GATE_TIMEOUT_SECS`) | `--ak gate_timeout_secs=900` — talos itself defaults to 300 | `parity gate_timeout_secs` |
| wall clock | `MINED_EVAL_WALL_CLOCK_SECS=<n>` | `--ak wall_clock_secs=<n>`, below the task.toml `[agent] timeout_sec` | `parity wall_clock_secs` |
| agent gate | on (`MINED_EVAL_AGENT_GATE` unset or `1`) | on: every emitted task sets `AGENT_GATE_COMMAND` | `gate-parity` (mined `finish_recovery_armed` vs Harbor `gate_armed`, per task over valid trials) |
| test-first guidance | on (`MINED_EVAL_TEST_FIRST` unset or `1`) | on: the emitter renders the statement with test-first guidance | `parity test_first` |

## Command sequence

Both runs drive the same local GPU when the backend is a local Ollama, so wrap each in `scripts/gpu-eval.sh`, which holds an exclusive lock on `GPU_LOCK_FILE` for the whole run and refuses to start (exit 75) if another model is resident. Paths and hosts below are placeholders.

1. Run mined_eval with a dedicated state root, so talos's 30-day state-dir prune (which covers the default mined-eval subtree) cannot delete the records before you compare them:

```bash
OLLAMA_BASE_URL=<ollama-url> OLLAMA_MODEL=<model> EVAL_BACKEND=ollama \
TALOS_MINED_STATE_ROOT=<calibration-state-dir> \
MINED_EVAL_TASKS_DIR=<tasks-dir> MINED_EVAL_SPEC_LEVEL=s2 MINED_EVAL_K=<k> \
MINED_EVAL_MAX_ITERATIONS=500 MINED_EVAL_WALL_CLOCK_SECS=<n> MINED_EVAL_AGENT_GATE=1 \
MINED_EVAL_GATE_TIMEOUT_SECS=900 MINED_EVAL_TEST_FIRST=1 \
scripts/gpu-eval.sh cargo run --release -p harness --example mined_eval
```

The startup banner prints `run_dir=<...>`, and the last line printed is `trial records: <run_dir>/<task-id>/trial-<k>/trial-result.json`. Keep that `run_dir`.

2. Emit the Harbor tasks and build the base image:

```bash
cargo run -p harness --example mined_to_harbor -- --tasks-dir <tasks-dir> --out <harbor-tasks> --spec-level s2
scripts/harbor-build-base.sh <harbor-tasks>
```

3. Run the same tasks on Harbor:

```bash
cd harbor && OLLAMA_BASE_URL=<ollama-url> OLLAMA_MODEL=<model> TALOS_BACKEND=ollama \
../scripts/gpu-eval.sh uv run --frozen harbor run -p <harbor-tasks> --agent-import-path talos_harbor.agent:TalosAgent -k <k> -o <jobs> --max-retries 0 --ak gate_timeout_secs=900 --ak wall_clock_secs=<n> --ak max_iterations=500
```

`--max-retries 0` is required: a Harbor retry re-runs a trial that failed on infrastructure and keeps the retry's score, which would convert infra-Invalid trials into scored trials on the Harbor side only, while mined_eval never retries.

4. Compare:

```bash
cd harbor && uv run --frozen talos-harbor-calibrate --harbor-job <jobs> --mined-run <run_dir printed in the banner>
```

`--harbor-job` accepts either the `-o` jobs dir or a single job dir inside it.

## How each side classifies Invalid

Invalid means the measurement failed, so the trial leaves the denominator. Each trial line in the report's `trials:` block names the rule that fired.

Harbor side, first match wins: (1) `metadata.talos.outcome == "BackendError"` → `backend-error`; (2) a non-null `exception_info` → `harbor-exception(<exception_type>): <first line of the message>`; (3) no `verifier/score.json` → `no-score-json` (the verifier exited before the scorer, which Harbor reports as `RewardFileNotFoundError`); (4) `score.json` unreadable or with an unknown `score` → `score-json-unreadable`; (5) `score == "invalid"` → its `detail`; (6) otherwise resolved / unresolved from `score.json`.

Mined side, first match wins: (1) `score == "invalid"` → its `invalid_reason`; (2) `claimed_disposition` starting with `BackendError` → `backend-error`; (3) otherwise resolved / unresolved.

The BackendError rule is symmetric on purpose: the same root cause gets the same `backend-error` prefix on both sides, and both rules record the score the verifier would have given in the `overridden` column and the `overrides` lines. mined_eval itself still sealed-gate-scores a BackendError trial, so the report's mined pooled rate can differ from mined_eval's `print_summary` rate by exactly the BackendError trials reclassified to Invalid.

## Baseline-red tripwire asymmetry

mined_eval checks that the agent gate is green at the parent commit before every trial, and a red baseline makes that one trial Invalid with `agent-gate-red-at-parent`. The Harbor path checks it in the task image build, not in `tests/test.sh`: the emitted Dockerfile runs the agent gate once and fails the build with a `TALOS_EVAL_INVALID reason=agent-gate-red-at-parent` line, which reaches the report as a `harbor-exception(<type>)` Invalid for every trial of that task. Because the check runs once per image build, a cached image does not resample it, so a flaky baseline is per-trial on the mined side and all-or-nothing per image on the Harbor side; `docker build --no-cache` resamples it.

## Verdict and exit codes

The last line is contract 6 verbatim. Over the tasks present on both sides, each side's pooled resolved count R over its valid trials N gets a 95% Wilson interval (z = 1.96). The verdict is `MATCH` iff both intervals exist, they overlap (closed intervals), and no task's resolved count differs by more than 2 between the sides; otherwise `MISMATCH: <reasons>`, with reasons from `no-valid-trials: harbor`, `no-valid-trials: mined`, `intervals-disjoint` and `task-diff>2: <ids>`. Tasks whose k differs between the sides, or that are missing mined records, count towards the pooled rates but get no per-task diff.

The `comparability:` line is separate and fails closed. It is `ok`, or `DEGRADED:` followed by any of `config-mismatch: <knobs>`, `gate-parity: <ids>`, `reward-score-disagree: <n>` (Harbor's `reward` disagrees with `score.json`), `k-mismatch: <ids>`, `only-harbor: <ids>` and `only-mined: <ids>`. It never changes the MATCH/MISMATCH line, only the exit code.

Exit codes: `0` MATCH with comparability ok; `3` MATCH with comparability DEGRADED; `1` MISMATCH; `2` input error (unreadable or empty inputs, a mined schema mismatch, or inconsistent mined records), printed as one `error:` line on stderr with nothing on stdout.

## Tokens are not comparable

The mined record's `input_tokens` is raw input on successful turns and EXCLUDES cache tokens, while Harbor's `n_input_tokens` includes cache read and cache write. The report compares no token or cost columns for that reason.

## What MATCH does not prove

MATCH is a statistical-compatibility check at small k, not a proof of equivalence. At k = 3 over a handful of tasks the Wilson intervals are wide, so two paths that differ by a real 10–20 points can still MATCH. Treat MATCH as "no evidence the Harbor path scores differently", read the per-task table and the invalid breakdown alongside it, and re-run with a larger k before relying on a close call.
