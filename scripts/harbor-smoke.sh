#!/usr/bin/env bash
# Opt-in Docker smoke run of the Harbor TalosAgent adapter against
# harbor/examples/hello-talos. NOT wired into any hook, CI job or test.
#
# Requirements:
#   - a running Docker daemon;
#   - a talos binary that supports `talos run --instruction-file`, supplied either as
#     TALOS_PATH=<local linux binary> or via TALOS_ARTIFACT_BASE=<artifact host base URL>;
#   - a backend env for talos (e.g. ANTHROPIC_API_KEY, or TALOS_BACKEND=ollama + OLLAMA_*),
#     which TalosAgent forwards into the trial container.
#
# Prints `PASS <jobs-dir>` and exits 0 when the trial's verifier scored 1 and talos wrote
# its run summary; otherwise prints `FAIL <reason> <jobs-dir>` and exits 1.
set -euo pipefail

if ! docker info >/dev/null 2>&1; then
  echo "harbor-smoke: docker is not available (docker info failed)" >&2
  exit 2
fi
if [ -z "${TALOS_PATH:-}" ] && [ -z "${TALOS_ARTIFACT_BASE:-}" ]; then
  echo "harbor-smoke: set TALOS_PATH or TALOS_ARTIFACT_BASE to supply a talos binary" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$(mktemp -d)"

args=(run -p examples/hello-talos --agent-import-path talos_harbor.agent:TalosAgent -o "$JOBS" --ak wall_clock_secs=540)
if [ -n "${TALOS_PATH:-}" ]; then
  args+=(--ak "talos_path=$TALOS_PATH")
fi

(cd "$ROOT/harbor" && uv run --frozen harbor "${args[@]}") || true

mapfile -t rewards < <(find "$JOBS" -path '*/verifier/reward.txt')
mapfile -t summaries < <(find "$JOBS" -path '*/agent/run-summary.json')

if [ "${#rewards[@]}" -ne 1 ]; then
  echo "FAIL expected exactly one verifier/reward.txt, found ${#rewards[@]} $JOBS"
  exit 1
fi
reward="$(tr -d '[:space:]' < "${rewards[0]}")"
if [ "$reward" != "1" ]; then
  echo "FAIL reward is '$reward', not 1 $JOBS"
  exit 1
fi
if [ "${#summaries[@]}" -ne 1 ]; then
  echo "FAIL expected exactly one agent/run-summary.json, found ${#summaries[@]} $JOBS"
  exit 1
fi
echo "PASS $JOBS"
