#!/usr/bin/env sh
# docs-only.sh — the fail-safe predicate behind the lefthook `skip:` blocks.
#
# Exits 0 (skip the gate) ONLY on positive proof that the resolved changeset
# is non-empty and entirely docs; exits 1 (run the gate) in every other case.
# Pure string classification, no filesystem access: a path is "docs" iff it
# begins with `docs/` or ends with `.md` and contains no `/` (byte-exact,
# case-sensitive — `README.MD` and `./docs/x.md` are NOT docs).
set -u

bad_args() {
    echo "docs-only: run mode=none reason=bad-args" >&2
    exit 1
}

# classify <path> — exit 0 iff <path> is a docs path.
classify() {
    case "$1" in
        docs/*) return 0 ;;
    esac
    # `*/*.md` catches every path containing a `/` (case patterns match `/`,
    # so this must run before the bare `*.md` form).
    case "$1" in
        */*.md) return 1 ;;
        *.md) return 0 ;;
        *) return 1 ;;
    esac
}

# Accumulate the resolved list: count, space-joined docs paths, first offender.
out=""
count=0
first_bad=""

mode=""
if [ "$#" -ge 1 ] && [ "$1" = "--" ]; then
    # The test form: `-- <file>...` (zero files allowed -> empty list).
    mode=list
    shift
elif [ "$#" -eq 1 ] && { [ "$1" = "--staged" ] || [ "$1" = "--push" ]; }; then
    mode=${1#--}
else
    # Every other argv is a usage error: no arguments, unknown flags, extra
    # arguments, or a bare file list with no leading `--` separator.
    bad_args
fi

case "$mode" in
    list)
        for p in "$@"; do
            count=$((count + 1))
            if classify "$p"; then
                if [ -z "$out" ]; then
                    out=$p
                else
                    out="$out $p"
                fi
            elif [ -z "$first_bad" ]; then
                first_bad=$p
            fi
        done
        ;;
    staged)
        # `--no-renames` is mandatory: rename detection would collapse
        # `git mv src/a.rs docs/a.md` to the docs path alone, silently
        # classifying a Rust-source deletion as docs-only. No `--relative`,
        # no `cd` — git prints repo-root-relative paths from any cwd.
        list=$(git diff --cached --name-only --no-renames 2>/dev/null) || {
            echo "docs-only: run mode=staged reason=git-failed" >&2
            exit 1
        }
        ;;
    push)
        # No upstream (brand-new branch before its first `git push -u`)
        # resolves to nothing — exit 1 without running any further git.
        upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{push}' \
            2>/dev/null) || {
            echo "docs-only: run mode=push reason=no-upstream" >&2
            exit 1
        }
        list=$(git diff --name-only --no-renames "$upstream..HEAD" 2>/dev/null) || {
            echo "docs-only: run mode=push reason=git-failed" >&2
            exit 1
        }
        ;;
esac

# Git modes: consume the resolved list one line at a time (never unquoted
# word-splitting of a command substitution, so a path with a space stays a
# single entry, agreeing with the explicit `-- <file>...` form).
if [ "$mode" != "list" ]; then
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        count=$((count + 1))
        if classify "$p"; then
            if [ -z "$out" ]; then
                out=$p
            else
                out="$out $p"
            fi
        elif [ -z "$first_bad" ]; then
            first_bad=$p
        fi
    done <<EOF
$list
EOF
fi

if [ "$count" -eq 0 ]; then
    echo "docs-only: run mode=$mode reason=empty-list" >&2
    exit 1
fi
if [ -n "$first_bad" ]; then
    echo "docs-only: run mode=$mode reason=non-docs:$first_bad" >&2
    exit 1
fi

echo "docs-only: skip mode=$mode n=$count paths=$out" >&2
exit 0