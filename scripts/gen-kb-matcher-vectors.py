"""Generate crates/harness/testdata/personal_kb_contracts/matcher_vectors.json.

Runs the personal-kb hook's Python originals so the talos matcher port can be
table-tested against them. Usage, from the harness-design root, with a personal_kb
checkout (or `git archive` extract) at the vendored tag:

    python3 -I scripts/gen-kb-matcher-vectors.py <personal_kb>/packages/personal-kb-hook/src "<tag> (<sha>)" \\
      | jq '.target_class |= map(select(.tool != "Bash"))' > crates/harness/testdata/personal_kb_contracts/matcher_vectors.json
"""
import json
import platform
import sys

src, commit = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from personal_kb_hook import cues_lite, prevention  # noqa: E402

TC_CASES = [
    ("Edit", "src/a.py"), ("Edit", "src/A.PY"), ("Edit", "Makefile"), ("Edit", ""),
    ("Edit", ".env"), ("Edit", "dir/archive.tar.gz"), ("Edit", "/abs/x.rs"),
    ("Edit", "a.b/c"), ("Edit", "docs/My Notes.md"), ("Read", "Cargo.toml"),
    ("Read", ".github/workflows/ci.yml"), ("Write", "x.json"), ("MultiEdit", "y.ts"),
    ("NotebookEdit", "n.ipynb"), ("LS", "."), ("LS", "src"), ("Glob", "**/*.rs"),
    ("run_checks", ""), ("finish", ""), ("Bash", "git push origin main"),
    ("Bash", "sudo FOO=1 systemctl restart nginx"), ("Bash", "cd repo && cargo test -p x"),
]
ARGS_CASES = [
    "src/a.py", "docs/My Notes.md", "", ".", "git push origin main",
    "git push -f origin feat/x", "sudo FOO=1 systemctl restart nginx",
    "cd repo && cargo test -p harness", "ls -la dir", "python3 -m pytest tests/ -q",
    "echo a\ngit push origin", "sudo -u dispatch FOO=1 git push x", "cd a; cd b || make all",
]
INDEX = [
    {"resolution_id": "kb-b1", "tool": "Bash", "target_class": "git push", "args_prefix": "github"},
    {"resolution_id": "kb-b2", "tool": "Bash", "target_class": "git push", "args_prefix": ""},
    {"resolution_id": "kb-e1", "tool": "Edit", "target_class": "ext:toml", "args_prefix": ""},
    {"resolution_id": "kb-e2", "tool": "Edit", "target_class": "ext:py", "args_prefix": ""},
    {"resolution_id": "kb-r1", "tool": "Read", "target_class": "ext:none", "args_prefix": ""},
    {"resolution_id": "kb-r2", "tool": "Read", "target_class": "ext:md", "args_prefix": "Notes.md"},
    {"resolution_id": "kb-l1", "tool": "LS", "target_class": "ext:none", "args_prefix": ""},
]
FM_CASES = [
    ("Bash", "git push github main"), ("Bash", "git push origin main"),
    ("Bash", "cargo build && git push github"), ("Bash", "cargo test"),
    ("Edit", "Cargo.toml"), ("Edit", "crates/x/Cargo.TOML"), ("Edit", "src/lib.rs"),
    ("Edit", "tools/gen.py"), ("Edit", ""), ("Read", "Cargo.toml"), ("Read", ".env"),
    ("Read", "Makefile"), ("Read", "docs/My Notes.md"), ("Read", "docs/notes.md"),
    ("Write", "a.py"), ("LS", "."), ("LS", "src"), ("run_checks", ""),
]

out = {
    "generated_by": "scripts/gen-kb-matcher-vectors.py",
    "source": f"personal_kb {commit}: personal-kb-hook cues_lite.target_class, cues_lite.bash_args_after_class, prevention._find_match",
    "python": platform.python_version(),
    "note": "Regenerate from the Python originals on any change to either side; never edit expectations by hand. Tool names here are CANONICAL (after tool_map translation).",
    "target_class": [{"tool": t, "target": s, "expected": cues_lite.target_class(t, s)} for t, s in TC_CASES],
    "bash_args_after_class": [{"target": s, "expected": cues_lite.bash_args_after_class(s)} for s in ARGS_CASES],
    "find_match": {
        "index": INDEX,
        "cases": [],
    },
}
for t, s in FM_CASES:
    entry, cls = prevention._find_match(t, s, INDEX)
    out["find_match"]["cases"].append(
        {"tool": t, "target": s, "expected_resolution_id": entry["resolution_id"] if entry else None, "expected_class": cls}
    )
json.dump(out, sys.stdout, indent=2)
print()
