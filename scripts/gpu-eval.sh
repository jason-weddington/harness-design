#!/usr/bin/env bash
# gpu-eval.sh <cmd> [args...] — run an eval command that drives a shared local GPU.
#
# Opt-in wrapper for `harbor run` (scripts/tier2-harbor.sh) against a local Ollama GPU that other eval
# sessions share. It (1) takes an exclusive, non-blocking flock on GPU_LOCK_FILE (default
# $HOME/.cache/gpu-locks/rtx5090.lock) and holds it for the whole child run, (2) refuses to
# start unless every model Ollama currently has loaded is $OLLAMA_MODEL, a name variant of it
# ("$OLLAMA_MODEL-..."), or an embedding model, (3) optionally records GPU telemetry (see
# GPU_TELEMETRY_SSH below), then (4) runs the command and exits with its exit code. Exit 75 (EX_TEMPFAIL) means "GPU busy, try later"; exit 2 means misconfigured.
#
# The Ollama URL comes only from OLLAMA_BASE_URL; this script names no host.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: gpu-eval.sh <cmd> [args...]" >&2
  exit 2
fi

# (1) Lock, held on fd 9 for the life of this shell and inherited by the child.
lock="${GPU_LOCK_FILE:-$HOME/.cache/gpu-locks/rtx5090.lock}"
case "$lock" in
  */*) mkdir -p "${lock%/*}" ;;
esac
exec 9>"$lock"
if ! flock -n 9; then
  echo "gpu-eval: GPU busy (lock held: $lock)" >&2
  exit 75
fi

# (2) Pre-check: only our model (or an embedding model) may be resident.
if [ -z "${OLLAMA_BASE_URL:-}" ] || [ -z "${OLLAMA_MODEL:-}" ]; then
  echo "gpu-eval: OLLAMA_BASE_URL and OLLAMA_MODEL must be set" >&2
  exit 2
fi
if ! ps_json="$(curl -fsS "$OLLAMA_BASE_URL/api/ps")"; then
  echo "gpu-eval: cannot reach $OLLAMA_BASE_URL/api/ps" >&2
  exit 2
fi
foreign="$(printf '%s' "$ps_json" | python3 -c '
import json, sys
want = sys.argv[1]
models = json.load(sys.stdin).get("models") or []
names = [str(m.get("name", "")) for m in models if isinstance(m, dict)]
bad = [n for n in names if not (n == want or n.startswith(want + "-") or "embed" in n)]
print(" ".join(bad))
' "$OLLAMA_MODEL")"
if [ -n "$foreign" ]; then
  echo "gpu-eval: foreign model loaded: $foreign" >&2
  exit 75
fi

# (3) Optional GPU telemetry, so a long eval leaves evidence of whether the GPU throttled. With
# GPU_TELEMETRY_SSH set (an ssh target for the GPU host), nvidia-smi samples temperature, power,
# SM clock and the active clock-event-reason bitmask every 5 s into GPU_TELEMETRY_DIR (default
# ./gpu-telemetry-<UTC stamp>); with GPU_TELEMETRY_UNIT also set, that systemd unit's journal for
# the run window is saved beside it (e.g. the host's fan controller). Telemetry never fails the run.
tel_pid=""
tel_dir=""
if [ -n "${GPU_TELEMETRY_SSH:-}" ]; then
  start_epoch="$(date +%s)"
  tel_dir="${GPU_TELEMETRY_DIR:-gpu-telemetry-$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$tel_dir"
  ssh -o BatchMode=yes "$GPU_TELEMETRY_SSH" \
    nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks_event_reasons.active \
    --format=csv -l 5 > "$tel_dir/nvidia-smi.csv" 2> "$tel_dir/nvidia-smi.err" < /dev/null &
  tel_pid=$!
  echo "gpu-eval: GPU telemetry from $GPU_TELEMETRY_SSH -> $tel_dir" >&2
fi

# (4) Run the command with the lock still held; propagate its exit code.
rc=0
"$@" || rc=$?

if [ -n "$tel_pid" ]; then
  kill "$tel_pid" 2> /dev/null || true
  wait "$tel_pid" 2> /dev/null || true
  if [ -n "${GPU_TELEMETRY_UNIT:-}" ]; then
    ssh -o BatchMode=yes "$GPU_TELEMETRY_SSH" \
      journalctl -u "$GPU_TELEMETRY_UNIT" --since "@$start_epoch" --no-pager -o short-iso \
      > "$tel_dir/$GPU_TELEMETRY_UNIT.log" 2>&1 < /dev/null || true
  fi
  # Thermal slowdown is bit 0x20 (software) or 0x40 (hardware); 0x8 is a hardware slowdown of
  # any cause; 0x4 is only the power cap. A sample with no parseable bitmask is skipped.
  python3 - "$tel_dir/nvidia-smi.csv" >&2 << 'EOF' || true
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))[1:]
temps, thermal, hw = [], 0, 0
for r in rows:
    if len(r) < 5:
        continue
    try:
        temps.append(int(r[1])); bits = int(r[4].strip(), 16)
    except ValueError:
        continue
    thermal += bool(bits & 0x60); hw += bool(bits & 0x8)
print(f"gpu-eval: telemetry {len(temps)} samples, max {max(temps, default=0)}C, "
      f"thermal-slowdown samples {thermal}, hw-slowdown samples {hw}")
EOF
fi
exit "$rc"
