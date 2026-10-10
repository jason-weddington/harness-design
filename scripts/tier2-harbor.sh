#!/usr/bin/env bash
# tier2-harbor.sh — the one command for a tier-2 run on Harbor (runbook: docs/design/10-tier2-on-harbor.md).
#
# It validates the environment, prints one decision line on stderr, builds the base image
# (scripts/harbor-build-base.sh), runs `harbor run` with the parity knobs of the calibration of
# record (wrapped in scripts/gpu-eval.sh only for a local-GPU Ollama model), then prints the
# per-task table with talos-harbor-tier2-report. It persists nothing of its own: the report can
# be re-run on the jobs dir at any time.
#
# The `-n "$N" --max-retries 0 --ak ...` run-time knobs are pinned to the Rust constants by
# crates/harness/src/mined_harbor.rs's tier2_script_carries_the_parity_kwargs test; keep them
# on one line.
#
# Exit codes: 0 clean; 2 usage, environment or input error; 3 the table is not a clean
# measurement; 75 GPU busy (from gpu-eval.sh); otherwise harbor's or the report's own code.
set -euo pipefail

USAGE='usage: scripts/tier2-harbor.sh --tasks <emitted-harbor-tasks-dir> --jobs <out-dir> [-k N (default 3)] [-n N (default 1)] [--talos-version <token> | --talos-path <binary>] [--task <id>]...'
USAGE_NOTE='  -n is the number of concurrent trials: it must be 1 when the backend is a single local GPU (queued trials would count against each trial'"'"'s wall clock); hosted backends may raise it.'

die() {
  echo "tier2-harbor: $1" >&2
  exit 2
}

usage_error() {
  echo "tier2-harbor: $1" >&2
  echo "$USAGE" >&2
  exit 2
}

TASKS=""
JOBS=""
K=3
N=1
TALOS_VERSION=""
TALOS_PATH=""
INCLUDE=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h | --help)
      echo "$USAGE"
      echo "$USAGE_NOTE"
      exit 0
      ;;
    --tasks | --jobs | -k | -n | --talos-version | --talos-path | --task)
      if [ "$#" -lt 2 ]; then
        usage_error "$1 needs a value"
      fi
      case "$1" in
        --tasks) TASKS=$2 ;;
        --jobs) JOBS=$2 ;;
        -k) K=$2 ;;
        -n) N=$2 ;;
        --talos-version) TALOS_VERSION=$2 ;;
        --talos-path) TALOS_PATH=$2 ;;
        --task) INCLUDE+=("$2") ;;
      esac
      shift 2
      ;;
    *)
      usage_error "unknown argument: $1"
      ;;
  esac
done

[ -n "$TASKS" ] || usage_error "--tasks is required"
[ -n "$JOBS" ] || usage_error "--jobs is required"
if [ -n "$TALOS_VERSION" ] && [ -n "$TALOS_PATH" ]; then
  usage_error "--talos-version and --talos-path are mutually exclusive"
fi
[[ "$K" =~ ^[1-9][0-9]*$ ]] || usage_error "-k must be a positive integer, got: $K"
[[ "$N" =~ ^[1-9][0-9]*$ ]] || usage_error "-n must be a positive integer, got: $N"

command -v git > /dev/null 2>&1 || die "git is not on PATH"

# Backend: mirrors TalosAgent._select_backend (a non-empty TALOS_BEDROCK always wins).
BACKEND="${TALOS_BACKEND:-}"
case "$BACKEND" in
  anthropic | ollama) ;;
  *) die "TALOS_BACKEND must be anthropic or ollama" ;;
esac
if [ -n "${TALOS_BEDROCK:-}" ] && [ "$BACKEND" != anthropic ]; then
  die "TALOS_BEDROCK is set, so talos would run on Bedrock; set TALOS_BACKEND=anthropic for a Bedrock run"
fi
GPU_LOCK=unwrapped
if [ "$BACKEND" = ollama ]; then
  [ -n "${OLLAMA_MODEL:-}" ] || die "OLLAMA_MODEL must be set for the ollama backend"
  [ -n "${OLLAMA_BASE_URL:-}" ] || die "OLLAMA_BASE_URL must be set for the ollama backend"
  CLOUD_URL=0
  case "$OLLAMA_BASE_URL" in
    https://ollama.com | https://ollama.com/*) CLOUD_URL=1 ;;
  esac
  if [ "$CLOUD_URL" = 1 ] && [ -z "${OLLAMA_API_KEY:-}" ]; then
    die "OLLAMA_API_KEY must be set for an ollama.com base URL"
  fi
  CLOUD_MODEL=0
  case "$OLLAMA_MODEL" in
    *:cloud) CLOUD_MODEL=1 ;;
  esac
  if [ "$CLOUD_URL" = 0 ] && [ "$CLOUD_MODEL" = 0 ]; then
    GPU_LOCK=wrapped
  fi
  BACKEND_LABEL=ollama
elif [ -n "${TALOS_BEDROCK:-}" ]; then
  BACKEND_LABEL=bedrock
else
  [ -n "${ANTHROPIC_API_KEY:-}" ] || die "ANTHROPIC_API_KEY must be set for the anthropic backend"
  BACKEND_LABEL=anthropic
fi
if [ "$GPU_LOCK" = wrapped ] && [ "$N" != 1 ]; then
  die "-n must be 1 on a local GPU backend"
fi

# Paths, made absolute and physical before any cd.
if [ -n "$TALOS_PATH" ]; then
  [ -f "$TALOS_PATH" ] || die "--talos-path is not an existing regular file: $TALOS_PATH"
  TALOS_PATH_ABS="$(cd "$(dirname "$TALOS_PATH")" && pwd -P)/$(basename "$TALOS_PATH")"
elif [ -z "${TALOS_ARTIFACT_BASE:-}" ]; then
  die "TALOS_ARTIFACT_BASE must be set unless --talos-path is given"
fi
[ -d "$TASKS" ] || die "--tasks is not an existing directory: $TASKS"
TASKS_ABS="$(cd "$TASKS" && pwd -P)"
if [ "$(git -C "$TASKS_ABS" rev-parse --is-inside-work-tree 2> /dev/null || true)" = true ]; then
  die "--tasks $TASKS_ABS is inside a git work tree; Harbor tasks carry private repo content and must never be committed"
fi
JOBS_PARENT="$(dirname "$JOBS")"
[ -d "$JOBS_PARENT" ] || die "the parent of --jobs does not exist: $JOBS_PARENT"
JOBS_ABS="$(cd "$JOBS_PARENT" && pwd -P)/$(basename "$JOBS")"
if [ -e "$JOBS_ABS" ] && [ ! -d "$JOBS_ABS" ]; then
  die "--jobs exists and is not a directory: $JOBS_ABS"
fi
if [ -d "$JOBS_ABS" ] && [ -n "$(ls -A "$JOBS_ABS")" ]; then
  die "--jobs is a non-empty directory: $JOBS_ABS"
fi
if [ "$(git -C "$(dirname "$JOBS_ABS")" rev-parse --is-inside-work-tree 2> /dev/null || true)" = true ]; then
  die "--jobs $JOBS_ABS is inside a git work tree; Harbor jobs carry private repo content and must never be committed"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [ -n "$TALOS_PATH" ]; then
  TALOS_LABEL="path:$TALOS_PATH_ABS"
elif [ -n "$TALOS_VERSION" ]; then
  TALOS_LABEL="version:$TALOS_VERSION"
else
  TALOS_LABEL=latest
fi
if [ "${#INCLUDE[@]}" -gt 0 ]; then
  INCLUDE_LABEL="$(
    IFS=,
    echo "${INCLUDE[*]}"
  )"
else
  INCLUDE_LABEL=all
fi
echo "tier2-harbor: backend=$BACKEND_LABEL gpu_lock=$GPU_LOCK talos=$TALOS_LABEL k=$K tasks=$TASKS_ABS jobs=$JOBS_ABS include=$INCLUDE_LABEL" >&2

# Step 1: the base image every emitted Dockerfile builds FROM.
"$ROOT/scripts/harbor-build-base.sh" "$TASKS_ABS" || exit $?

# Step 2: harbor run.
cd "$ROOT/harbor"
cmd=(uv run --frozen harbor run -p "$TASKS_ABS" --agent-import-path talos_harbor.agent:TalosAgent -k "$K" -n "$N" --max-retries 0 --ak gate_timeout_secs=900 --ak wall_clock_secs=3300 --ak max_iterations=500 -o "$JOBS_ABS")
if [ -n "$TALOS_PATH" ]; then
  cmd+=(--ak "talos_path=$TALOS_PATH_ABS")
elif [ -n "$TALOS_VERSION" ]; then
  cmd+=(--ak "talos_version=$TALOS_VERSION")
fi
report=(uv run --frozen talos-harbor-tier2-report "$JOBS_ABS" --expect-k "$K" --tasks "$TASKS_ABS")
for id in "${INCLUDE[@]}"; do
  cmd+=(-i "$id")
  report+=(--task "$id")
done
if [ "$GPU_LOCK" = wrapped ]; then
  cmd=("$ROOT/scripts/gpu-eval.sh" "${cmd[@]}")
fi
rc=0
"${cmd[@]}" || rc=$?

# Step 3: the report, whenever harbor wrote anything.
report_rc=""
if [ -d "$JOBS_ABS" ] && [ -n "$(ls -A "$JOBS_ABS")" ]; then
  report_rc=0
  "${report[@]}" || report_rc=$?
fi

if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
if [ -n "$report_rc" ]; then
  exit "$report_rc"
fi
echo "tier2-harbor: harbor wrote nothing to $JOBS_ABS" >&2
exit 1
