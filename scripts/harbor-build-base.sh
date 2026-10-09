#!/usr/bin/env bash
# Build and smoke-check the base image for tasks emitted by mined_to_harbor.
#
# Usage: scripts/harbor-build-base.sh <out-dir>
#   <out-dir> is the --out of `cargo run -p harness --example mined_to_harbor`; this
#   script builds <out-dir>/_base and tags it with the contents of <out-dir>/_base/IMAGE.
#
# It needs a running Docker daemon. No hook, CI job or test invokes it: it is run by
# hand, and it must run before `harbor run` on the emitted tasks, because every emitted
# Dockerfile says `FROM talos-evals-base:1` and that image exists nowhere else.
#
# The image is never pushed to a registry, so the build must land in the local image
# store: use the default `docker` buildx driver. A `docker-container` builder keeps the
# result in its own cache, and the task builds then fail with 'pull access denied'.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: scripts/harbor-build-base.sh <out-dir>" >&2
  exit 2
fi
out=$1
for f in "$out/_base/Dockerfile" "$out/_base/IMAGE"; do
  if [ ! -f "$f" ]; then
    echo "error: $f is missing (run the mined_to_harbor emitter with --out $out first)" >&2
    exit 2
  fi
done
if ! docker info > /dev/null 2>&1; then
  echo "error: docker info failed; is the Docker daemon running?" >&2
  exit 2
fi

TAG=$(cat "$out/_base/IMAGE")
echo "buildx builder (needs the default 'docker' driver):"
docker buildx inspect | grep -E '^(Name|Driver):' || true
docker build -t "$TAG" "$out/_base"
docker run --rm "$TAG" bash -c 'set -e; git --version; python3 --version; uv --version; node --version; npm --version; ldd --version | head -1; redis-stack-server --port 6390 --save "" --appendonly no --daemonize yes; sleep 1; test "$(redis-cli -p 6390 JSON.SET k \$ 1)" = OK'
echo "PASS $TAG"
