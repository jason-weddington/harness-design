"""Tier-2 report: the per-task table for a Harbor jobs dir of emitted mined tasks.

``talos-harbor-tier2-report <jobs> [--expect-k N --tasks DIR [--task PATTERN]...]`` reads
Harbor 0.9.0 trials (``<job>/<trial>/{result.json, agent/, verifier/score.json}``),
classifies each one (rules 1-6, first match wins), and prints a config provenance line,
one row per task, a TOTAL row, the invalid trials and, with ``--expect-k``, every task
whose trial count is not N.

Exit codes: 0 a clean measurement; 3 the table is not a clean measurement (a k mismatch,
an unexpected task, mixed config, a trial without an armed gate, or no valid trials);
2 input error. ``scripts/tier2-harbor.sh`` runs it after ``harbor run``, and it can be
re-run on a jobs dir at any time. Runbook: ``docs/design/10-tier2-on-harbor.md``.

This module is stdlib-only: it reads raw JSON and never imports ``harbor``, so a schema
drift shows up as a classification, not an import error.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCORES = ("resolved", "unresolved", "invalid")


class InputError(Exception):
    """An unusable input: printed as one ``error:`` line on stderr, exit 2."""


@dataclass(frozen=True)
class Trial:
    """One Harbor trial's classification."""

    task: str
    trial_name: str
    score: str
    reason: str | None
    claimed_done: bool
    gate_armed: bool | None
    talos_error: str | None


@dataclass(frozen=True)
class _Found:
    trial: Trial
    result: dict[str, object]


# ---- tolerant lookups -------------------------------------------------------------------


def _dig(obj: object, *keys: str) -> object:
    """``obj[k1][k2]...`` where any missing key or non-dict intermediate gives None."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _distinct(values: Iterable[object]) -> list[str]:
    return sorted({str(v) for v in values})


def _join(values: Iterable[object], sep: str = ",", empty: str = "-") -> str:
    d = _distinct(values)
    return sep.join(d) if d else empty


def _one_line(text: str) -> str:
    """Tabs and newlines become single spaces, so a value cannot break a table row."""
    return re.sub(r"[\t\r\n]+", " ", text)


# ---- classification ---------------------------------------------------------------------


def _read_score_json(trial_dir: Path) -> tuple[bool, dict[str, object] | None]:
    """(exists, parsed object or None when unparseable / not an object)."""
    path = trial_dir / "verifier" / "score.json"
    if not path.is_file():
        return False, None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return True, None
    return True, value if isinstance(value, dict) else None


def _harbor_task_key(result: dict[str, object], score: dict[str, object] | None) -> str:
    sid = score.get("task_id") if score is not None else None
    if isinstance(sid, str):
        return sid
    return str(result.get("task_name"))


def _harbor_spec_level(result: dict[str, object]) -> str:
    path = _dig(result, "config", "task", "path")
    if not isinstance(path, str):
        return "unknown"
    toml_path = Path(path) / "task.toml"
    if not toml_path.is_file():
        return "unknown"
    try:
        data = tomllib.loads(toml_path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return "unknown"
    level = _dig(data, "metadata", "spec_level")
    return level if isinstance(level, str) else "unknown"


def classify_trial(trial_dir: Path, result: dict[str, object]) -> Trial:
    """Classify one Harbor trial (rules 1-6, first match wins)."""
    exists, score = _read_score_json(trial_dir)
    score_value = score.get("score") if score is not None else None
    talos = _dig(result, "agent_result", "metadata", "talos")

    cls: str
    reason: str | None = None
    exc = result.get("exception_info")
    if _dig(talos, "outcome") == "BackendError":
        cls, reason = "invalid", "backend-error"
    elif isinstance(exc, dict):
        lines = str(exc.get("exception_message") or "").splitlines()
        first = lines[0] if lines else ""
        reason = f"harbor-exception({exc.get('exception_type')})"
        if first:
            reason = f"{reason}: {first}"
        cls = "invalid"
    elif not exists:
        cls, reason = "invalid", "no-score-json"
    elif score is None or score_value not in SCORES:
        cls, reason = "invalid", "score-json-unreadable"
    elif score_value == "invalid":
        detail = score.get("detail")
        cls = "invalid"
        reason = detail if isinstance(detail, str) and detail else "score-invalid-no-detail"
    else:
        cls = "resolved" if score_value == "resolved" else "unresolved"

    gate = _dig(talos, "gate_armed")
    error = _dig(talos, "error")
    return Trial(
        task=_harbor_task_key(result, score),
        trial_name=str(result.get("trial_name")),
        score=cls,
        reason=reason,
        claimed_done=_dig(talos, "claimed_done") is True,
        gate_armed=gate if isinstance(gate, bool) else None,
        talos_error=error if isinstance(error, str) and error else None,
    )


# ---- discovery --------------------------------------------------------------------------


def _discover(root: Path, warnings: list[str]) -> list[_Found]:
    seen: dict[Path, Path] = {}
    for pattern in ("*/result.json", "*/*/result.json"):
        for p in root.glob(pattern):
            seen.setdefault(p.resolve(), p)
    found: list[_Found] = []
    job_dirs: set[Path] = set()
    for resolved in sorted(seen):
        path = seen[resolved]
        trial_dir = path.parent
        if not ((trial_dir / "agent").is_dir() or (trial_dir / "verifier").is_dir()):
            continue
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise InputError(f"unparseable harbor result.json: {path}: {exc}") from exc
        if not (
            isinstance(result, dict)
            and isinstance(result.get("trial_name"), str)
            and isinstance(result.get("task_name"), str)
        ):
            continue
        trial = classify_trial(trial_dir, result)
        task_name = result["task_name"]
        if trial.task != task_name:
            warnings.append(
                f"WARN: task-id mismatch {result['trial_name']}: "
                f"task_name={task_name} score.task_id={trial.task}"
            )
        job_dirs.add(resolved.parent.parent)
        found.append(_Found(trial=trial, result=result))
    if not found:
        raise InputError(f"no harbor trials found under {root}")
    if len(job_dirs) > 1:
        warnings.append(f"WARN: {len(job_dirs)} harbor job dirs merged")
    return found


def _expected_tasks(tasks: Path, patterns: list[str]) -> set[str]:
    """Task dirs under ``tasks`` that Harbor's ``-i`` filter would select (fnmatch on name)."""
    names = {d.name for d in tasks.iterdir() if d.is_dir() and (d / "task.toml").is_file()}
    if patterns:
        names = {n for n in names if any(fnmatch.fnmatchcase(n, p) for p in patterns)}
    if not names:
        raise InputError(f"no expected tasks under {tasks}")
    return names


# ---- report -----------------------------------------------------------------------------


def _config_fields(found: list[_Found]) -> dict[str, tuple[list[object], str]]:
    agents: list[object] = []
    versions: list[object] = []
    backends: list[object] = []
    levels: list[object] = []
    kwargs: list[object] = []
    for f in found:
        r = f.result
        agents.append(f"{_dig(r, 'agent_info', 'name')}@{_dig(r, 'agent_info', 'version')}")
        talos = _dig(r, "agent_result", "metadata", "talos")
        version = _dig(talos, "version")
        if version is not None:
            versions.append(version)
        b = _dig(talos, "backend")
        if isinstance(b, dict):
            kind_model = f"{b.get('kind')}/{b.get('model')}"
            backends.append(f"{kind_model} num_ctx={b.get('num_ctx')} think={b.get('think')}")
        levels.append(_harbor_spec_level(r))
        kw = _dig(r, "config", "agent", "kwargs")
        if isinstance(kw, dict):
            kwargs.append(json.dumps(kw, sort_keys=True))
    return {
        "agents": (agents, ","),
        "talos": (versions, ","),
        "backends": (backends, " | "),
        "spec_levels": (levels, ","),
        "agent_kwargs": (kwargs, " | "),
    }


def build_report(
    jobs_arg: str,
    found: list[_Found],
    expected: set[str] | None,
    expect_k: int | None,
    warnings: list[str],
) -> tuple[list[str], int]:
    """(stdout lines, exit code); appends this stage's WARN lines to ``warnings``."""
    trials = [f.trial for f in found]
    by_task: dict[str, list[Trial]] = {}
    for t in trials:
        by_task.setdefault(t.task, []).append(t)
    rows = set(by_task) | (expected or set())

    fields = _config_fields(found)
    joined = {name: _join(values, sep) for name, (values, sep) in fields.items()}
    out = [
        f"tier2: {jobs_arg} ({len(trials)} trials, {len(rows)} tasks)",
        f"config: agents={joined['agents']} talos={joined['talos']} "
        f"backends={joined['backends']} spec_levels={joined['spec_levels']} "
        f"agent_kwargs={joined['agent_kwargs']}",
        "task\tresolved/k\tunresolved\tinvalid\tclaimed-done-but-unresolved",
    ]
    totals = [0, 0, 0, 0, 0]
    mismatches: list[str] = []
    for task in sorted(rows):
        ts = by_task.get(task, [])
        k = len(ts)
        r = sum(t.score == "resolved" for t in ts)
        u = sum(t.score == "unresolved" for t in ts)
        i = sum(t.score == "invalid" for t in ts)
        f = sum(t.claimed_done and t.score == "unresolved" for t in ts)
        out.append(f"{task}\t{r}/{k}\t{u}\t{i}\t{f}")
        for idx, v in enumerate((r, k, u, i, f)):
            totals[idx] += v
        if expect_k is not None and k != expect_k:
            mismatches.append(f"k-mismatch\t{task}\t{k}/{expect_k}")
    out.append(f"TOTAL\t{totals[0]}/{totals[1]}\t{totals[2]}\t{totals[3]}\t{totals[4]}")
    for t in sorted(
        (t for t in trials if t.score == "invalid"), key=lambda t: (t.task, t.trial_name)
    ):
        out.append(f"invalid\t{t.task}\t{t.trial_name}\t{_one_line(str(t.reason))}")
    out.extend(mismatches)

    unclean = bool(mismatches)
    if expect_k is None and expected is None:
        warnings.append("WARN: no --expect-k/--tasks: missing trials cannot be detected")
    if expected is not None:
        for task in sorted(set(by_task) - expected):
            warnings.append(f"WARN: unexpected-task {task}")
            unclean = True
    for name in ("talos", "backends", "agent_kwargs"):
        values, sep = fields[name]
        if len(_distinct(values)) > 1:
            warnings.append(f"WARN: mixed-config {name}: {_join(values, sep)}")
            unclean = True
    by_name = sorted(trials, key=lambda t: (t.task, t.trial_name))
    for t in by_name:
        if t.score != "invalid" and t.gate_armed is not True:
            warnings.append(f"WARN: gate-not-armed {t.task} {t.trial_name}")
            unclean = True
    if totals[0] + totals[2] == 0:
        warnings.append("WARN: no valid trials")
        unclean = True
    for t in by_name:
        if t.talos_error is not None:
            warnings.append(
                f"WARN: talos-error {t.task} {t.trial_name}: {_one_line(t.talos_error)}"
            )
    return out, 3 if unclean else 0


# ---- CLI --------------------------------------------------------------------------------


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer of at least 1, got {raw!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module doc for exit codes."""
    parser = argparse.ArgumentParser(
        prog="talos-harbor-tier2-report",
        description="Per-task tier-2 table for a Harbor jobs dir (or a single job dir).",
    )
    parser.add_argument("jobs", help="Harbor jobs dir (harbor run -o) or a single job dir")
    parser.add_argument(
        "--expect-k", type=_positive_int, metavar="N", help="trials expected per task"
    )
    parser.add_argument("--tasks", metavar="DIR", help="the emitted Harbor tasks dir")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="PATTERN",
        help="restrict the expected tasks (fnmatch on the task dir name, like harbor -i)",
    )
    args = parser.parse_args(argv)
    if (args.expect_k is None) != (args.tasks is None):
        parser.error("--expect-k and --tasks must be given together")
    if args.task and args.tasks is None:
        parser.error("--task requires --tasks")

    warnings: list[str] = []
    try:
        for raw in (args.jobs, args.tasks):
            if raw is not None and not Path(raw).is_dir():
                raise InputError(f"not a directory: {raw}")
        expected = _expected_tasks(Path(args.tasks), args.task) if args.tasks else None
        found = _discover(Path(args.jobs), warnings)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines, code = build_report(args.jobs, found, expected, args.expect_k, warnings)
    for w in warnings:
        print(w, file=sys.stderr)
    print("\n".join(lines))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
