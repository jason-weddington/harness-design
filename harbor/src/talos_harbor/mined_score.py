"""Self-contained Harbor verifier for talos tier-2 mined tasks.

A stdlib-only Python port of the tier-2 scorer. The source of truth is
``crates/harness/src/mined_eval.rs`` (``PytestParser``, ``resolve``,
``parse_authored_tests`` and their helpers). The Rust scorer and this script are
parallel implementations with no mechanism keeping them in sync: **a scoring
change must land in BOTH files**, or tier-2 numbers from the two paths stop
being comparable.

This file is copied ALONE into a trial container and run as::

    python3 mined_score.py --task <task_score.json> --gate-output <file> \\
        --authored <json-file> --out-dir <dir> \\
        [--gate-stderr <file>] [--gate-exit-code <int>]

It always writes ``<out-dir>/reward.txt`` (exactly ``1`` iff resolved, else
``0``, no trailing newline) and ``<out-dir>/score.json``, and never writes
``reward.json`` (Harbor 0.9.0's verifier prefers ``reward.json`` over
``reward.txt`` when both exist).

Gate capture contract: without ``--gate-stderr`` the ``--gate-output`` file
MUST be the gate's stdout FOLLOWED BY its stderr, not interleaved -- this
mirrors the Rust ``format!("{}{}", stdout, stderr)``. Interleaving can land a
stderr line inside the ``short test summary info`` section and end it early.
With ``--gate-stderr``, ``--gate-output`` is stdout only and the parse input is
stdout + stderr.
"""

import argparse
import enum
import hashlib
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# Rust `char::is_whitespace` (Unicode White_Space). Python's own str.split()
# / str.strip() additionally treat \x1c-\x1f as whitespace, so they are never
# used on parsed text.
_RUST_WS = "".join(
    chr(c)
    for c in (
        *range(0x09, 0x0E),  # \t \n \v \f \r
        0x20,
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
)
_WS_RUN = re.compile("[" + re.escape(_RUST_WS) + "]+")
_USIZE = re.compile(r"\+?[0-9]+")
_COUNTED_BUCKETS = frozenset(
    {"passed", "failed", "error", "skipped", "xfailed", "xpassed", "deselected"}
)


def _split_ws(s: str) -> list[str]:
    """Rust `str::split_whitespace`: split on runs of `_RUST_WS`, drop empties."""
    return [tok for tok in _WS_RUN.split(s) if tok]


def _rust_lines(s: str) -> list[str]:
    """Rust `str::lines`: split on "\\n", strip one trailing "\\r" per line."""
    if s == "":
        return []
    parts = s.split("\n")
    if s.endswith("\n"):
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


def _is_py(path: str) -> bool:
    """Rust `Path::extension` equals `py` (ASCII case-insensitive)."""
    return PurePosixPath(path).suffix.lower() == ".py"


# ===== test id parsing ======================================================


class TestStatus(enum.Enum):
    """One test's status from a pytest `-rA` short-summary line.

    XPASS is distinct and never treated as PASSED.
    """

    __test__ = False  # not a pytest test class

    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    XFAIL = "XFAIL"
    XPASS = "XPASS"


_TAGS: tuple[tuple[str, TestStatus], ...] = (
    ("PASSED ", TestStatus.PASSED),
    ("FAILED ", TestStatus.FAILED),
    ("ERROR ", TestStatus.ERROR),
    ("SKIPPED ", TestStatus.SKIPPED),
    ("XFAIL ", TestStatus.XFAIL),
    ("XPASS ", TestStatus.XPASS),
)


@dataclass
class ParseReport:
    """Result of `parse_pytest` (Rust `ParseReport`)."""

    statuses: dict[str, TestStatus]
    summary_count: int | None
    dropped_outside_section: int
    sections_seen: int


def parse_short_summary_line(line: str) -> tuple[TestStatus, str] | None:
    """Match a case-sensitive status tag prefix; return (status, rest)."""
    for tag, status in _TAGS:
        if line.startswith(tag):
            return status, line[len(tag) :]
    return None


def derive_nodeid(status: TestStatus, rest: str) -> str:
    """Map key for a status line's `rest` (Rust `derive_nodeid`).

    For `SKIPPED [N] <location>: <reason>` the key is the location token with
    ALL trailing ':' stripped (what the Rust code does). Otherwise the first
    whitespace token.
    """
    tokens = _split_ws(rest)
    if status is TestStatus.SKIPPED:
        first = tokens[0] if tokens else ""
        if first.startswith("[") and first.endswith("]"):
            return tokens[1].rstrip(":") if len(tokens) > 1 else ""
    return tokens[0] if tokens else ""


def parse_pytest_summary_totals(line: str) -> int | None:
    """Sum the counted buckets of a `=== N passed, M failed ... ===` line."""
    if not line.startswith("==="):
        return None
    rest = line[3:]
    idx = rest.rfind("===")
    if idx < 0:
        return None
    body = rest[:idx].strip("=").strip(_RUST_WS)
    total = 0
    found = False
    for raw_part in body.split(","):
        tokens = _split_ws(raw_part.strip(_RUST_WS))
        if len(tokens) < 2:
            continue
        if not _USIZE.fullmatch(tokens[0]):
            continue
        label = tokens[1].rstrip("s")
        if label in _COUNTED_BUCKETS:
            total += int(tokens[0])
            found = True
    return total if found else None


def parse_pytest(output: str) -> ParseReport:
    """Port of Rust `PytestParser::parse`: section-scoped `-rA` parsing."""
    statuses: dict[str, TestStatus] = {}
    summary_count: int | None = None
    in_section = False
    sections_seen = 0
    dropped = 0
    for line in _rust_lines(output):
        t = line.lstrip(_RUST_WS)
        tag = parse_short_summary_line(t)
        if t.startswith("=") and "short test summary info" in t:
            in_section = True
            sections_seen += 1
        elif in_section and (t.strip(_RUST_WS) == "" or tag is None):
            in_section = False
        if tag is not None:
            status, rest = tag
            nodeid = derive_nodeid(status, rest)
            if nodeid:
                if in_section:
                    statuses[nodeid] = status
                else:
                    dropped += 1
        n = parse_pytest_summary_totals(t)
        if n is not None:
            summary_count = n
    return ParseReport(
        statuses=statuses,
        summary_count=summary_count,
        dropped_outside_section=dropped,
        sections_seen=sections_seen,
    )


# ===== id matching ==========================================================


def normalize(nodeid: str) -> str:
    """Strip a leading `<file>.py::` segment (Rust `normalize`)."""
    idx = nodeid.find("::")
    if idx >= 0 and _is_py(nodeid[:idx]):
        return nodeid[idx + 2 :]
    return nodeid


def strip_param_suffix(nodeid: str) -> str:
    """Cut at the FIRST '[' only when the id ends with ']'."""
    open_idx = nodeid.find("[")
    if open_idx >= 0 and nodeid.endswith("]"):
        return nodeid[:open_idx]
    return nodeid


def is_file_qualified(task_id: str) -> bool:
    """True when the task id carries a `<file>.py::` prefix."""
    return normalize(task_id) != task_id


def match_task_id(raw_nodeid: str, normalized_nodeid: str, task_id: str) -> bool:
    """Match in raw space for file-qualified ids, normalized space otherwise."""
    side = raw_nodeid if is_file_qualified(task_id) else normalized_nodeid
    return strip_param_suffix(side) == strip_param_suffix(task_id)


def matches_exclusion(raw_nodeid: str, normalized_nodeid: str, excl: str) -> bool:
    """Literal (via `match_task_id`) or `Prefix::*` glob exclusion match."""
    if excl.endswith("::*"):
        class_prefix = excl[: -len("::*")]
        side = raw_nodeid if is_file_qualified(excl) else normalized_nodeid
        return side.startswith(f"{class_prefix}::")
    return match_task_id(raw_nodeid, normalized_nodeid, excl)


# ===== scoring ==============================================================


@dataclass(frozen=True)
class TaskScore:
    """The scoring-relevant subset of a mined task's `task.json`."""

    id: str
    fail_to_pass: list[str]
    positive_controls: list[str]
    pass_to_pass_exclusions: list[str]
    gate_command: str = ""
    sealed: list[str] = field(default_factory=list)


class Score(enum.Enum):
    """Three-member trial verdict; INVALID leaves the resolved/k denominator."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    INVALID = "invalid"


@dataclass(frozen=True)
class Verdict:
    """`resolve()` output: the score plus the Rust `ResolveDetail` evidence."""

    score: Score
    detail: str
    missing_fail_to_pass: list[str] = field(default_factory=list)
    uncovered_red: list[str] = field(default_factory=list)
    collection_errors: list[str] = field(default_factory=list)
    fail_to_pass_status: list[tuple[str, TestStatus | None]] = field(default_factory=list)
    positive_control_status: list[tuple[str, TestStatus | None]] = field(default_factory=list)
    excluded_red: list[tuple[str, str]] = field(default_factory=list)


_Normalized = list[tuple[str, str, TestStatus]]
_RED = (TestStatus.FAILED, TestStatus.ERROR)


def _require_str_list(obj: dict[str, Any], key: str) -> list[str]:
    if key not in obj:
        raise ValueError(f"missing required key `{key}`")
    value = obj[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"`{key}` must be a list of strings")
    return list(value)


def load_task_score(path: str) -> TaskScore:
    """Load and type-check a task_score.json; raises ValueError on bad input."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("top-level value is not an object")
    if "id" not in obj:
        raise ValueError("missing required key `id`")
    if not isinstance(obj["id"], str):
        raise ValueError("`id` must be a string")
    fail_to_pass = _require_str_list(obj, "fail_to_pass")
    positive_controls = _require_str_list(obj, "positive_controls")
    exclusions = _require_str_list(obj, "pass_to_pass_exclusions")
    gate_command = obj.get("gate_command", "")
    if not isinstance(gate_command, str):
        raise ValueError("`gate_command` must be a string")
    sealed_raw = obj.get("sealed", [])
    if not isinstance(sealed_raw, list):
        raise ValueError("`sealed` must be a list")
    sealed: list[str] = []
    for entry in sealed_raw:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("every `sealed` entry must be an object with a string `path`")
        sealed.append(entry["path"])
    return TaskScore(
        id=obj["id"],
        fail_to_pass=fail_to_pass,
        positive_controls=positive_controls,
        pass_to_pass_exclusions=exclusions,
        gate_command=gate_command,
        sealed=sealed,
    )


def find_status_for_task_id(normalized: _Normalized, task_id: str) -> TestStatus | None:
    """First match seeds; a later non-PASSED overrides a seeded PASSED."""
    seen: TestStatus | None = None
    for raw, nid, status in normalized:
        if match_task_id(raw, nid, task_id) and (
            seen is None or (seen is TestStatus.PASSED and status is not TestStatus.PASSED)
        ):
            seen = status
    return seen


def build_fail_to_pass_status(
    normalized: _Normalized, task: TaskScore
) -> list[tuple[str, TestStatus | None]]:
    """Each fail_to_pass id with its observed status (None when missing)."""
    return [(ftp, find_status_for_task_id(normalized, ftp)) for ftp in task.fail_to_pass]


def build_positive_control_status(
    normalized: _Normalized, task: TaskScore
) -> list[tuple[str, TestStatus | None]]:
    """Each positive control with its observed status (None when missing)."""
    return [(pc, find_status_for_task_id(normalized, pc)) for pc in task.positive_controls]


def build_missing_fail_to_pass(normalized: _Normalized, task: TaskScore) -> list[str]:
    """fail_to_pass ids that are missing OR present but not PASSED."""
    return [
        ftp
        for ftp in task.fail_to_pass
        if find_status_for_task_id(normalized, ftp) is not TestStatus.PASSED
    ]


def _red_not_fail_to_pass(normalized: _Normalized, task: TaskScore) -> _Normalized:
    return [
        (raw, nid, status)
        for raw, nid, status in normalized
        if status in _RED and not any(match_task_id(raw, nid, ftp) for ftp in task.fail_to_pass)
    ]


def build_unexcluded_red(normalized: _Normalized, task: TaskScore) -> list[str]:
    """FAILED/ERROR ids covered by neither fail_to_pass nor an exclusion."""
    seen: set[str] = set()
    out: list[str] = []
    for raw, nid, _status in _red_not_fail_to_pass(normalized, task):
        if any(matches_exclusion(raw, nid, e) for e in task.pass_to_pass_exclusions):
            continue
        if nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def build_excluded_red(normalized: _Normalized, task: TaskScore) -> list[tuple[str, str]]:
    """Red ids an exclusion covered, paired with the FIRST matching exclusion."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for raw, nid, _status in _red_not_fail_to_pass(normalized, task):
        excl = next(
            (e for e in task.pass_to_pass_exclusions if matches_exclusion(raw, nid, e)), None
        )
        if excl is None:
            continue
        if nid not in seen:
            seen.add(nid)
            out.append((nid, excl))
    return out


def build_collection_errors(normalized: _Normalized) -> list[str]:
    """ERROR ids with no '::' and a `.py` extension: file-level collection errors."""
    return [
        raw
        for raw, _nid, status in normalized
        if status is TestStatus.ERROR and "::" not in raw and _is_py(raw)
    ]


def resolve(statuses: dict[str, TestStatus], summary_count: int | None, task: TaskScore) -> Verdict:
    """Port of Rust `resolve()`: score a parsed status map against the task."""
    # (1) Empty parse.
    if not statuses:
        return Verdict(Score.INVALID, "parse-empty")
    # (2) Count mismatch (None means the cross-check is not armed).
    if summary_count is not None and summary_count != len(statuses):
        return Verdict(
            Score.INVALID, f"parse-mismatch: {len(statuses)} ids vs summary {summary_count}"
        )
    normalized: _Normalized = [(raw, normalize(raw), statuses[raw]) for raw in sorted(statuses)]
    ftp_status = build_fail_to_pass_status(normalized, task)
    pc_status = build_positive_control_status(normalized, task)
    excluded = build_excluded_red(normalized, task)

    def verdict(
        score: Score,
        detail: str,
        missing: list[str] | None = None,
        uncovered: list[str] | None = None,
        collection: list[str] | None = None,
    ) -> Verdict:
        return Verdict(
            score=score,
            detail=detail,
            missing_fail_to_pass=missing or [],
            uncovered_red=uncovered or [],
            collection_errors=collection or [],
            fail_to_pass_status=ftp_status,
            positive_control_status=pc_status,
            excluded_red=excluded,
        )

    # (3) Collection errors.
    collection_errors = build_collection_errors(normalized)
    if collection_errors:
        return verdict(
            Score.UNRESOLVED,
            "collection-error",
            missing=build_missing_fail_to_pass(normalized, task),
            uncovered=build_unexcluded_red(normalized, task),
            collection=collection_errors,
        )
    # (4) Positive controls: present and PASSED; absent is Invalid.
    for pc, status in pc_status:
        if status is None:
            return verdict(Score.INVALID, f"positive-control-uncollected: {pc}")
        if status is not TestStatus.PASSED:
            return verdict(
                Score.UNRESOLVED,
                "positive-control-red",
                missing=build_missing_fail_to_pass(normalized, task),
                uncovered=build_unexcluded_red(normalized, task),
            )
    # (5) fail_to_pass: present and PASSED.
    missing = build_missing_fail_to_pass(normalized, task)
    if missing:
        return verdict(
            Score.UNRESOLVED,
            "fail-to-pass-red",
            missing=missing,
            uncovered=build_unexcluded_red(normalized, task),
        )
    # (6) Every other red id must be covered by an exclusion.
    unexcluded = build_unexcluded_red(normalized, task)
    if unexcluded:
        return verdict(Score.UNRESOLVED, "unexcluded-red", uncovered=unexcluded)
    return verdict(Score.RESOLVED, "")


def exit_code_fault_reason(exit_code: int | None, stdout: str) -> str | None:
    """Map a gate exit code to an infra-fault reason, or None.

    NOT a port of Rust `gate_fault_reason`: only the prefix before the first
    ':' (`gate-timeout`, `gate-no-exit-code`) is shared with Rust's strings.
    124 is GNU `timeout`; >=128 with empty stdout is a signal kill with no
    output. When --gate-stderr is absent, `stdout` is the whole capture.
    """
    if exit_code is None:
        return None
    if exit_code == 124:
        return "gate-timeout: exit 124"
    if exit_code >= 128 and stdout.strip(_RUST_WS) == "":
        return f"gate-no-exit-code: exit {exit_code}"
    return None


# ===== agent test authorship ================================================


def is_test_path(path: str) -> bool:
    """Whether `path` looks like a pytest test file (Rust `is_test_path`)."""
    p = PurePosixPath(path)
    if _is_py(path) and (p.name.startswith("test_") or p.name.endswith("_test.py")):
        return True
    return any(part.lower() == "tests" for part in p.parts)


def parse_authored_tests(porcelain: str) -> tuple[list[str], list[str]]:
    """Split `git status --porcelain -uall` output into (added, modified) test paths.

    The input MUST be captured with `-uall` BEFORE the sealed copy: without
    `-uall` git collapses an untracked directory to a single `?? dir/` entry and
    every test file inside it is invisible here. `??` or any `A` in the status
    code is ADDED; anything else naming a test path is MODIFIED. A rename is
    attributed to its destination.
    """
    added: list[str] = []
    modified: list[str] = []
    for line in _rust_lines(porcelain):
        if len(line.encode("utf-8")) < 4:
            continue
        code = line[:2]
        path = line[2:].lstrip(_RUST_WS)
        path = path.rsplit(" -> ", 1)[-1]
        path = path.strip('"')
        if not is_test_path(path):
            continue
        if code == "??" or "A" in code:
            added.append(path)
        else:
            modified.append(path)
    return added, modified


def load_authored(path: str) -> tuple[list[str], list[str], str | None]:
    """Read the --authored JSON: (added, modified, authored_error)."""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return [], [], f"authored-missing: {path}"
    try:
        obj = json.loads(raw)
    except ValueError as exc:
        return [], [], f"authored-unparseable: {exc}"
    if not isinstance(obj, dict):
        return [], [], "authored-wrong-shape"
    lists: list[list[str]] = []
    for key in ("added", "modified"):
        value = obj.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return [], [], "authored-wrong-shape"
        lists.append(list(value))
    return lists[0], lists[1], None


# ===== CLI ==================================================================


def _count_check(summary_count: int | None, n_statuses: int) -> str:
    """Rust `CountCheck::label`."""
    if summary_count is not None:
        return "on"
    return "off" if n_statuses else "n/a"


def _status_list(pairs: list[tuple[str, TestStatus | None]]) -> list[dict[str, str | None]]:
    return [{"id": i, "status": None if s is None else s.name} for i, s in pairs]


def _pre_parse_fields() -> dict[str, Any]:
    return {
        "passed": [],
        "failed": [],
        "collection_errors": [],
        "missing_fail_to_pass": [],
        "uncovered_red": [],
        "fail_to_pass_status": [],
        "positive_control_status": [],
        "excluded_red": [],
        "summary_count": None,
        "sections_seen": 0,
        "dropped_outside_section": 0,
        "count_check": "n/a",
    }


def _evaluate(args: argparse.Namespace, out: dict[str, Any]) -> None:
    """Steps 1-4 of the scoring precedence; mutates `out` in place."""
    try:
        task = load_task_score(args.task)
    except ValueError as exc:
        out["detail"] = f"task-score-unreadable: {exc}"
        return
    out["task_id"] = task.id
    try:
        stdout_bytes = Path(args.gate_output).read_bytes()
        stderr_bytes = b"" if args.gate_stderr is None else Path(args.gate_stderr).read_bytes()
    except OSError as exc:
        out["detail"] = f"gate-output-missing: {exc}"
        return
    out["gate_output_bytes"] = len(stdout_bytes) + len(stderr_bytes)
    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    fault = exit_code_fault_reason(args.gate_exit_code, stdout_text)
    if fault is not None:
        out["detail"] = fault
        return
    report = parse_pytest(stdout_text + stderr_text)
    out["passed"] = sorted(k for k, v in report.statuses.items() if v is TestStatus.PASSED)
    out["failed"] = sorted(k for k, v in report.statuses.items() if v in _RED)
    out["summary_count"] = report.summary_count
    out["sections_seen"] = report.sections_seen
    out["dropped_outside_section"] = report.dropped_outside_section
    out["count_check"] = _count_check(report.summary_count, len(report.statuses))
    verdict = resolve(report.statuses, report.summary_count, task)
    out["score"] = verdict.score.value
    out["detail"] = verdict.detail
    out["missing_fail_to_pass"] = verdict.missing_fail_to_pass
    out["uncovered_red"] = verdict.uncovered_red
    out["collection_errors"] = verdict.collection_errors
    out["fail_to_pass_status"] = _status_list(verdict.fail_to_pass_status)
    out["positive_control_status"] = _status_list(verdict.positive_control_status)
    out["excluded_red"] = [{"id": i, "exclusion": e} for i, e in verdict.excluded_red]


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score a tier-2 mined task's sealed gate output.")
    p.add_argument("--task", required=True, help="task_score.json")
    p.add_argument("--gate-output", required=True, help="gate stdout (then stderr)")
    p.add_argument("--authored", required=True, help="authored-tests JSON")
    p.add_argument("--out-dir", required=True, help="where reward.txt/score.json go")
    p.add_argument("--gate-stderr", default=None, help="gate stderr, when captured apart")
    p.add_argument("--gate-exit-code", type=int, default=None, help="gate exit status")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: always writes reward.txt + score.json and returns 0."""
    args = _arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    added, modified, authored_error = load_authored(args.authored)
    out: dict[str, Any] = {
        "score": Score.INVALID.value,
        "detail": "",
        **_pre_parse_fields(),
        "agent_tests_added": added,
        "agent_tests_modified": modified,
        "authored_error": authored_error,
        "task_id": None,
        "gate_exit_code": args.gate_exit_code,
        "gate_output_bytes": None,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        _evaluate(args, out)
    except Exception as exc:
        traceback.print_exc()
        out.update(_pre_parse_fields())
        out["score"] = Score.INVALID.value
        out["detail"] = f"scorer-exception: {type(exc).__name__}: {exc}"
    reward = b"1" if out["score"] == Score.RESOLVED.value else b"0"
    (out_dir / "reward.txt").write_bytes(reward)
    (out_dir / "score.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"mined_score: task={out['task_id'] or '-'} score={out['score']} "
        f"detail={out['detail']!s} passed={len(out['passed'])} failed={len(out['failed'])} "
        f"count_check={out['count_check']} sections={out['sections_seen']}".replace("\n", "\\n")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
