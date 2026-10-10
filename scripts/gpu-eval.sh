#!/usr/bin/env bash
# gpu-eval.sh <cmd> [args...] — run an eval command that drives a shared local GPU.
#
# Opt-in wrapper for `harbor run` (scripts/tier2-harbor.sh) against a local Ollama GPU that other eval
# sessions share. It (1) takes an exclusive, non-blocking flock on GPU_LOCK_FILE (default
# $HOME/.cache/gpu-locks/rtx5090.lock) and holds it for the whole child run, (2) refuses to
# start unless every model Ollama currently has loaded is $OLLAMA_MODEL, a name variant of it
# ("$OLLAMA_MODEL-..."), or an embedding model, then (3) runs the command and exits with its
# exit code. Exit 75 (EX_TEMPFAIL) means "GPU busy, try later"; exit 2 means misconfigured.
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

# (3) Run the command with the lock still held; propagate its exit code.
rc=0
"$@" || rc=$?
exit "$rc"
