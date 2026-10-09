#!/usr/bin/env bash
#
# Usage: check-glibc-floor.sh <floor> <binary>...
#
# Fail closed if any binary REQUIRES a GLIBC symbol version newer than <floor>
# (e.g. 2.39). The required versions are the verneed entries of the dynamic
# section, read with `readelf --version-info --wide` (binutils; present on any
# host that can build). Only the "Version needs section" is parsed, so versions
# a binary merely DEFINES are never counted. A missing tool, a non-ELF file, or
# zero GLIBC versions found is an error: a glibc-linked Rust binary always
# requires some, so "none" means the check saw nothing, not that all is well.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <floor> <binary>..." >&2
  exit 2
fi
floor="$1"; shift

if ! command -v readelf >/dev/null 2>&1; then
  echo "ERROR: readelf not found (install binutils); cannot verify glibc floor for $*" >&2
  exit 1
fi

fail=0
for bin in "$@"; do
  if [ ! -f "$bin" ]; then
    echo "ERROR: $bin is not a file; cannot verify glibc floor" >&2
    exit 1
  fi
  if ! info="$(readelf --version-info --wide "$bin" 2>/dev/null)"; then
    echo "ERROR: $bin is not a readable ELF; cannot verify glibc floor" >&2
    exit 1
  fi
  versions="$(printf '%s\n' "$info" \
    | awk '/^Version needs section/ {on=1; next} /^Version [a-z]+ section/ {on=0} on' \
    | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)+' | sort -u || true)"
  if [ -z "$versions" ]; then
    echo "ERROR: $bin: no GLIBC_ versions found in its verneed entries (not glibc-linked, or unreadable)" >&2
    exit 1
  fi
  max="$(printf '%s\n' "$versions" | sed 's/^GLIBC_//' | sort -V | tail -n 1)"
  echo "$bin: max GLIBC_$max (floor $floor)"
  highest="$(printf '%s\n%s\n' "$max" "$floor" | sort -V | tail -n 1)"
  if [ "$highest" != "$floor" ]; then
    echo "ERROR: $bin requires GLIBC_$max > floor $floor" >&2
    fail=1
  fi
done
exit "$fail"
