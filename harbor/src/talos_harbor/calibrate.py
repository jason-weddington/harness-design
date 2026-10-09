"""Calibration report: does a Harbor job score the same as an in-process mined_eval run?

``talos-harbor-calibrate --harbor-job <dir> --mined-run <dir>`` reads a Harbor 0.9.0 job
(``<job>/<trial>/{result.json, agent/, verifier/score.json}``) and a mined_eval run dir
(``<run_dir>/<task-id>/trial-<k>/trial-result.json``), classifies every trial on both
sides, and prints a per-trial audit block, a config-parity block, a per-task table, pooled
Wilson intervals and the verdict.

The verdict line is contract 6 verbatim: ``MATCH`` iff the two pooled 95% Wilson
intervals overlap AND no task's resolved count differs by more than 2. Everything else
that can make the comparison untrustworthy (config drift, gate parity, one-sided tasks,
k mismatches, reward/score disagreement) goes on the separate ``comparability:`` line and
turns a MATCH into exit code 3.

Exit codes: 0 MATCH + comparability ok; 3 MATCH + comparability DEGRADED; 1 MISMATCH;
2 input error.

This module is stdlib-only: it reads raw JSON and never imports ``harbor``, so a schema
drift shows up as a classification, not an import error. See
``docs/design/09-harbor-calibration.md`` for the runbook.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MINED_SCHEMA = "talos-mined-trial/1"
SCORES = ("resolved", "unresolved", "invalid")
CLAIMED_ROWS = ("done", "not-done", "unknown")
DEFAULT_MAX_ITERATIONS = 500
DEFAULT_GATE_TIMEOUT_SECS = 300

Side = Literal["harbor", "mined"]
Score = Literal["resolved", "unresolved", "invalid"]
Claimed = Literal["done", "not-done", "unknown"]


class InputError(Exception):
    """An unusable input: printed as one ``error:`` line on stderr, exit 2."""


@dataclass(frozen=True)
class TrialClass:
    """One trial's classification on either side."""

    side: Side
    task_id: str
    trial: str
    rule: int
    score: Score
    reason: str | None
    claimed: Claimed
    gate_armed: bool | None
    overridden_score: str | None
    reward_disagree: bool
    source: str


@dataclass(frozen=True)
class TaskTally:
    """Per-task counts on one side; ``k`` is the number of trials actually present."""

    k: int
    resolved: int
    invalid: int
    missing_records: int

    @property
    def valid(self) -> int:
        return self.k - self.invalid


@dataclass(frozen=True)
class ParityRow:
    """One knob of the config-parity block."""

    knob: str
    mined: str
    harbor: str
    ok: bool


@dataclass
class _HarborTrial:
    cls: TrialClass
    result: dict[str, object]
    score_json: dict[str, object] | None


@dataclass
class _MinedSide:
    trials: list[TrialClass] = field(default_factory=list)
    records: list[dict[str, object]] = field(default_factory=list)
    tallies: dict[str, TaskTally] = field(default_factory=dict)


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


def _ids(ids: Iterable[str]) -> str:
    return ",".join(sorted(ids))


# ---- statistics -------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for ``successes`` of ``n``; None when ``n == 0``."""
    if n == 0:
        return None
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Closed-interval overlap."""
    return a[0] <= b[1] and b[0] <= a[1]


# ---- Harbor side ------------------------------------------------------------------------


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


def classify_harbor_trial(trial_dir: Path, result: dict[str, object]) -> TrialClass:
    """Classify one Harbor trial (rules 1-6, first match wins)."""
    exists, score = _read_score_json(trial_dir)
    score_value = score.get("score") if score is not None else None
    scored = score_value if score_value in ("resolved", "unresolved") else None
    talos = _dig(result, "agent_result", "metadata", "talos")

    rule: int
    cls: Score
    reason: str | None = None
    overridden: str | None = None
    exc = result.get("exception_info")
    if _dig(talos, "outcome") == "BackendError":
        rule, cls, reason, overridden = 1, "invalid", "backend-error", scored
    elif isinstance(exc, dict):
        lines = str(exc.get("exception_message") or "").splitlines()
        first = lines[0] if lines else ""
        reason = f"harbor-exception({exc.get('exception_type')})"
        if first:
            reason = f"{reason}: {first}"
        rule, cls, overridden = 2, "invalid", scored
    elif not exists:
        rule, cls, reason = 3, "invalid", "no-score-json"
    elif score is None or score_value not in SCORES:
        rule, cls, reason = 4, "invalid", "score-json-unreadable"
    elif score_value == "invalid":
        detail = score.get("detail")
        rule, cls = 5, "invalid"
        reason = detail if isinstance(detail, str) and detail else "score-invalid-no-detail"
    else:
        rule, cls = 6, "resolved" if score_value == "resolved" else "unresolved"

    claimed_done = _dig(talos, "claimed_done")
    claimed: Claimed = (
        "done" if claimed_done is True else "not-done" if claimed_done is False else "unknown"
    )
    gate = _dig(talos, "gate_armed")
    reward = _dig(result, "verifier_result", "rewards", "reward")
    reward_disagree = (
        cls != "invalid"
        and isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and (reward == 1) != (cls == "resolved")
    )
    return TrialClass(
        side="harbor",
        task_id=_harbor_task_key(result, score),
        trial=str(result.get("trial_name")),
        rule=rule,
        score=cls,
        reason=reason,
        claimed=claimed,
        gate_armed=gate if isinstance(gate, bool) else None,
        overridden_score=overridden,
        reward_disagree=reward_disagree,
        source=str(trial_dir / "result.json"),
    )


def _discover_harbor(root: Path, warnings: list[str]) -> list[_HarborTrial]:
    seen: dict[Path, Path] = {}
    for pattern in ("*/result.json", "*/*/result.json"):
        for p in root.glob(pattern):
            seen.setdefault(p.resolve(), p)
    trials: list[_HarborTrial] = []
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
        cls = classify_harbor_trial(trial_dir, result)
        _, score = _read_score_json(trial_dir)
        task_name = result["task_name"]
        if cls.task_id != task_name:
            warnings.append(
                f"WARN: task-id mismatch {result['trial_name']}: "
                f"task_name={task_name} score.task_id={cls.task_id}"
            )
        job_dirs.add(resolved.parent.parent)
        trials.append(_HarborTrial(cls=cls, result=result, score_json=score))
    if not trials:
        raise InputError(f"no harbor trials found under {root}")
    if len(job_dirs) > 1:
        warnings.append(f"WARN: {len(job_dirs)} harbor job dirs merged")
    return trials


# ---- mined side -------------------------------------------------------------------------


def classify_mined_record(rec: dict[str, object], source: str) -> TrialClass:
    """Classify one mined trial record (rules 1-3, first match wins)."""
    score = rec.get("score")
    disposition = rec.get("claimed_disposition")
    rule: int
    cls: Score
    reason: str | None = None
    overridden: str | None = None
    if score == "invalid":
        why = rec.get("invalid_reason")
        rule, cls = 1, "invalid"
        reason = why if isinstance(why, str) and why else "mined-invalid-no-reason"
    elif isinstance(disposition, str) and disposition.startswith("BackendError"):
        rule, cls, reason, overridden = 2, "invalid", "backend-error", str(score)
    else:
        rule, cls = 3, "resolved" if score == "resolved" else "unresolved"
    gate = rec.get("finish_recovery_armed")
    return TrialClass(
        side="mined",
        task_id=str(rec.get("task_id")),
        trial=f"trial-{rec.get('trial')}",
        rule=rule,
        score=cls,
        reason=reason,
        claimed="done" if rec.get("claimed_done") is True else "not-done",
        gate_armed=gate if isinstance(gate, bool) else None,
        overridden_score=overridden,
        reward_disagree=False,
        source=source,
    )


def _discover_mined(root: Path, warnings: list[str]) -> _MinedSide:
    side = _MinedSide()
    by_task: dict[str, list[tuple[dict[str, object], TrialClass]]] = {}
    for path in sorted(root.glob("*/trial-*/trial-result.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise InputError(f"unparseable mined trial-result.json: {path}: {exc}") from exc
        found = rec.get("schema") if isinstance(rec, dict) else type(rec).__name__
        if found != MINED_SCHEMA or not isinstance(rec, dict):
            raise InputError(f"mined schema mismatch: {path}: {found}")
        if rec.get("score") not in SCORES:
            raise InputError(f"mined schema mismatch: {path}: score={rec.get('score')!r}")
        cls = classify_mined_record(rec, str(path))
        by_task.setdefault(cls.task_id, []).append((rec, cls))
    if not by_task:
        raise InputError(f"no mined trial records found under {root}")
    for task in sorted(by_task):
        entries = by_task[task]
        ks = {str(rec.get("k")) for rec, _ in entries}
        if len(ks) > 1:
            raise InputError(
                f"inconsistent mined records for {task}: k disagrees ({','.join(sorted(ks))})"
            )
        k = entries[0][0].get("k")
        if not isinstance(k, int) or isinstance(k, bool):
            raise InputError(f"inconsistent mined records for {task}: k is not an int ({k!r})")
        trials = Counter(str(rec.get("trial")) for rec, _ in entries)
        dupes = sorted(t for t, n in trials.items() if n > 1)
        if dupes:
            raise InputError(
                f"inconsistent mined records for {task}: duplicate trial {','.join(dupes)}"
            )
        if len(entries) > k:
            raise InputError(
                f"inconsistent mined records for {task}: {len(entries)} records > k={k}"
            )
        missing = k - len(entries)
        if missing:
            warnings.append(f"WARN: missing-records: {task}={missing}")
        classes = [c for _, c in entries]
        side.tallies[task] = _tally(classes, missing)
        side.trials.extend(classes)
        side.records.extend(rec for rec, _ in entries)
    return side


# ---- tallies, verdict, comparability ----------------------------------------------------


def _tally(classes: list[TrialClass], missing_records: int = 0) -> TaskTally:
    return TaskTally(
        k=len(classes),
        resolved=sum(c.score == "resolved" for c in classes),
        invalid=sum(c.score == "invalid" for c in classes),
        missing_records=missing_records,
    )


def _tallies(classes: list[TrialClass]) -> dict[str, TaskTally]:
    by_task: dict[str, list[TrialClass]] = {}
    for c in classes:
        by_task.setdefault(c.task_id, []).append(c)
    return {t: _tally(cs) for t, cs in by_task.items()}


def _k_mismatch(h: TaskTally, m: TaskTally) -> bool:
    return h.k != m.k or h.missing_records > 0 or m.missing_records > 0


def _pooled(tallies: dict[str, TaskTally], tasks: Iterable[str]) -> tuple[int, int]:
    ts = [tallies[t] for t in tasks]
    return sum(t.resolved for t in ts), sum(t.valid for t in ts)


def _task_diffs(harbor: dict[str, TaskTally], mined: dict[str, TaskTally]) -> dict[str, int]:
    """Raw resolved-count diff per intersection task that is not k-mismatch."""
    return {
        t: abs(harbor[t].resolved - mined[t].resolved)
        for t in sorted(harbor.keys() & mined.keys())
        if not _k_mismatch(harbor[t], mined[t])
    }


def verdict(harbor: dict[str, TaskTally], mined: dict[str, TaskTally]) -> tuple[bool, list[str]]:
    """Contract 6: pooled intervals overlap AND no task differs by more than 2 of k."""
    common = sorted(harbor.keys() & mined.keys())
    hi = wilson_interval(*_pooled(harbor, common))
    mi = wilson_interval(*_pooled(mined, common))
    reasons: list[str] = []
    if hi is None:
        reasons.append("no-valid-trials: harbor")
    if mi is None:
        reasons.append("no-valid-trials: mined")
    if hi is not None and mi is not None and not intervals_overlap(hi, mi):
        reasons.append("intervals-disjoint")
    over = [t for t, d in _task_diffs(harbor, mined).items() if d > 2]
    if over:
        reasons.append(f"task-diff>2: {_ids(over)}")
    return (not reasons, reasons)


def _gate_parity_ids(
    harbor_trials: list[TrialClass], mined_trials: list[TrialClass], common: Iterable[str]
) -> list[str]:
    out: list[str] = []
    for task in common:
        h = [c.gate_armed for c in harbor_trials if c.task_id == task and c.score != "invalid"]
        m = {c.gate_armed for c in mined_trials if c.task_id == task and c.score != "invalid"}
        if None in h or set(h) != m:
            out.append(task)
    return sorted(out)


def comparability(
    harbor: dict[str, TaskTally],
    mined: dict[str, TaskTally],
    parity: list[ParityRow],
    harbor_trials: list[TrialClass],
    mined_trials: list[TrialClass],
) -> list[str]:
    """Fail-closed comparability reasons, in pinned order; empty means ok."""
    common = sorted(harbor.keys() & mined.keys())
    reasons: list[str] = []
    diffs = [p.knob for p in parity if not p.ok]
    if diffs:
        reasons.append(f"config-mismatch: {','.join(diffs)}")
    gate = _gate_parity_ids(harbor_trials, mined_trials, common)
    if gate:
        reasons.append(f"gate-parity: {_ids(gate)}")
    disagree = sum(c.reward_disagree for c in harbor_trials)
    if disagree:
        reasons.append(f"reward-score-disagree: {disagree}")
    kmis = [t for t in common if _k_mismatch(harbor[t], mined[t])]
    if kmis:
        reasons.append(f"k-mismatch: {_ids(kmis)}")
    only_h = harbor.keys() - mined.keys()
    if only_h:
        reasons.append(f"only-harbor: {_ids(only_h)}")
    only_m = mined.keys() - harbor.keys()
    if only_m:
        reasons.append(f"only-mined: {_ids(only_m)}")
    return reasons


# ---- config parity ----------------------------------------------------------------------


_GATE_ON = re.compile(r"^on\(timeout=(\d+)s\)$")


def _mined_gate_timeout(agent_gate: object) -> str:
    m = _GATE_ON.match(str(agent_gate))
    return m.group(1) if m else str(agent_gate)


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


def _kwarg(result: dict[str, object], key: str, default: object) -> object:
    kwargs = _dig(result, "config", "agent", "kwargs")
    if isinstance(kwargs, dict) and kwargs.get(key) is not None:
        return kwargs[key]
    return default


def _harbor_model(result: dict[str, object]) -> str:
    model = _dig(result, "agent_result", "metadata", "talos", "backend", "model")
    return str(model) if model is not None else "unknown"


def parity_rows(records: list[dict[str, object]], harbor: list[_HarborTrial]) -> list[ParityRow]:
    """The parity block knobs, in pinned order."""
    results = [h.result for h in harbor]

    def plain(knob: str, mined_vals: Iterable[object], harbor_vals: Iterable[object]) -> ParityRow:
        m, h = _join(mined_vals), _join(harbor_vals)
        return ParityRow(knob, m, h, m == h)

    rows = [
        plain(
            "max_iterations",
            (r.get("max_iterations") for r in records),
            (_kwarg(x, "max_iterations", DEFAULT_MAX_ITERATIONS) for x in results),
        )
    ]
    gate_m = _join(_mined_gate_timeout(r.get("agent_gate")) for r in records)
    if gate_m == "off":
        rows.append(ParityRow("gate_timeout_secs", gate_m, "n/a", True))
    else:
        gate_h = _join(_kwarg(x, "gate_timeout_secs", DEFAULT_GATE_TIMEOUT_SECS) for x in results)
        rows.append(ParityRow("gate_timeout_secs", gate_m, gate_h, gate_m == gate_h))
    rows.append(
        plain(
            "wall_clock_secs",
            (r.get("wall_clock_secs") for r in records),
            (_kwarg(x, "wall_clock_secs", "unset") for x in results),
        )
    )
    rows.append(plain("test_first", (r.get("test_first") for r in records), [True]))
    mined_models = _distinct(r.get("backend_desc") for r in records)
    harbor_models = _distinct(_harbor_model(x) for x in results)
    model_ok = (
        bool(mined_models)
        and bool(harbor_models)
        and "unknown" not in harbor_models
        and all(h in m for h in harbor_models for m in mined_models)
    )
    rows.append(ParityRow("model", _join(mined_models), _join(harbor_models), model_ok))
    spec_m = _join(r.get("spec_level") for r in records)
    spec_h = _join(_harbor_spec_level(x) for x in results)
    spec_ok = spec_m == spec_h and "unknown" not in (spec_m.split(",") + spec_h.split(","))
    rows.append(ParityRow("spec_level", spec_m, spec_h, spec_ok))
    return rows


# ---- report -----------------------------------------------------------------------------


def _side_cell(side: str, t: TaskTally | None) -> str:
    if t is None:
        return f"{side} -"
    return f"{side} {t.resolved}/{t.valid} (invalid {t.invalid}, k {t.k})"


def _pooled_line(side: str, tallies: dict[str, TaskTally], common: list[str]) -> str:
    r, n = _pooled(tallies, common)
    iv = wilson_interval(r, n)
    if iv is None:
        return f"pooled {side}: 0/0 = n/a"
    return f"pooled {side}: {r}/{n} = {r / n:.3f} [{iv[0]:.3f}, {iv[1]:.3f}]"


def _trial_line(c: TrialClass) -> str:
    gate = "unknown" if c.gate_armed is None else str(c.gate_armed)
    return (
        f"  {c.side} {c.task_id} {c.trial} rule={c.rule} class={c.score} "
        f"claimed={c.claimed} gate_armed={gate} reason={c.reason or '-'} "
        f"overridden={c.overridden_score or '-'}"
    )


def _side_blocks(side: str, trials: list[TrialClass], warnings: list[str]) -> list[str]:
    lines = [f"false-done cross-tab ({side}): claimed \u00d7 score"]
    for claimed in CLAIMED_ROWS:
        row = Counter(c.score for c in trials if c.claimed == claimed)
        lines.append(
            f"  {claimed} resolved={row['resolved']} unresolved={row['unresolved']} "
            f"invalid={row['invalid']}"
        )
    false_dones = sum(c.claimed == "done" and c.score == "unresolved" for c in trials)
    lines.append(f"false_dones {side}={false_dones}")
    prefixes = Counter((c.reason or "").split(":", 1)[0] for c in trials if c.score == "invalid")
    if prefixes:
        groups = ", ".join(f"{p}={prefixes[p]}" for p in sorted(prefixes))
        lines.append(f"invalid {side}: {groups}")
    else:
        lines.append(f"invalid {side}: none")
    over = [c for c in trials if c.overridden_score is not None]
    backend = sum(c.reason == "backend-error" for c in over)
    harbor_exc = sum((c.reason or "").startswith("harbor-exception(") for c in over)
    would_be = sum(c.overridden_score == "resolved" for c in over)
    lines.append(
        f"overrides {side}: backend-error-over-scored={backend} "
        f"harbor-exception-over-scored={harbor_exc} (would-be resolved={would_be})"
    )
    if over:
        warnings.append(f"WARN: {len(over)} scored trials discarded by override on {side}")
    return lines


def _parser_health(side: str, checks: Iterable[object]) -> str:
    n = Counter(c for c in checks if c in ("on", "off", "n/a"))
    return f"parser-health {side}: count_check on={n['on']} off={n['off']} n/a={n['n/a']}"


def _harbor_headers(harbor: list[_HarborTrial]) -> tuple[str, str]:
    agents = _join(
        f"{_dig(h.result, 'agent_info', 'name')}@{_dig(h.result, 'agent_info', 'version')}"
        for h in harbor
    )
    backends: list[str] = []
    kwargs: list[str] = []
    versions: list[object] = []
    for h in harbor:
        talos = _dig(h.result, "agent_result", "metadata", "talos")
        backend = _dig(talos, "backend")
        if isinstance(backend, dict):
            backends.append(f"{backend.get('kind')}/{backend.get('model')}")
        kw = _dig(h.result, "config", "agent", "kwargs")
        if isinstance(kw, dict):
            kwargs.append(json.dumps(kw, sort_keys=True))
        version = _dig(talos, "version")
        if version is not None:
            versions.append(version)
    header = (
        f"harbor header: agents={agents} backends={_join(backends)} "
        f"agent_kwargs={_join(kwargs, ' | ')}"
    )
    return header, _join(versions)


def build_report(
    harbor_arg: str, mined_arg: str, harbor: list[_HarborTrial], mined: _MinedSide
) -> tuple[list[str], list[str], int]:
    """(stdout lines, stderr warnings, exit code) for already-discovered inputs."""
    warnings: list[str] = []
    h_trials = [h.cls for h in harbor]
    m_trials = mined.trials
    h_tally = _tallies(h_trials)
    m_tally = mined.tallies
    recs = mined.records
    common = sorted(h_tally.keys() & m_tally.keys())

    out = [
        f"calibration: harbor={harbor_arg} ({len(h_trials)} trials) "
        f"mined={mined_arg} ({len(m_trials)} trials)",
        f"mined header: backend={_join((r.get('backend_desc') for r in recs), ' | ')} "
        f"spec_level={_join(r.get('spec_level') for r in recs)} "
        f"agent_gate={_join(r.get('agent_gate') for r in recs)} "
        f"test_first={_join(r.get('test_first') for r in recs)} "
        f"max_iterations={_join(r.get('max_iterations') for r in recs)} "
        f"wall_clock_secs={_join(r.get('wall_clock_secs') for r in recs)}",
    ]
    harbor_header, harbor_versions = _harbor_headers(harbor)
    out.append(harbor_header)
    out.append(
        f"versions: mined={_join(r.get('harness_version') for r in recs)} harbor={harbor_versions}"
    )
    out.append("trials:")
    for c in sorted(h_trials + m_trials, key=lambda c: (c.side, c.task_id, c.trial)):
        out.append(_trial_line(c))

    parity = parity_rows(recs, harbor)
    out.append("parity:")
    for p in parity:
        status = "ok" if p.ok else "DIFF"
        out.append(f"  parity {p.knob}: mined={p.mined} harbor={p.harbor} {status}")

    diffs = _task_diffs(h_tally, m_tally)
    gate_ids = set(_gate_parity_ids(h_trials, m_trials, common))
    out.append("tasks:")
    for task in sorted(h_tally.keys() | m_tally.keys()):
        h, m = h_tally.get(task), m_tally.get(task)
        d = diffs.get(task)
        row = f"  {task}  {_side_cell('harbor', h)}  {_side_cell('mined', m)}  diff "
        row += "-" if d is None else str(d)
        if m is None:
            row += " [only-harbor]"
        if h is None:
            row += " [only-mined]"
        if h is not None and m is not None and _k_mismatch(h, m):
            row += " [k-mismatch]"
        if d is not None and d > 2:
            row += " [task-diff>2]"
        if task in gate_ids:
            row += " [gate-parity]"
        out.append(row)

    out.append(_pooled_line("harbor", h_tally, common))
    out.append(_pooled_line("mined", m_tally, common))
    out.extend(_side_blocks("harbor", h_trials, warnings))
    out.extend(_side_blocks("mined", m_trials, warnings))
    out.append(_parser_health("harbor", [(h.score_json or {}).get("count_check") for h in harbor]))
    out.append(_parser_health("mined", (r.get("count_check") for r in recs)))

    comp = comparability(h_tally, m_tally, parity, h_trials, m_trials)
    out.append(f"comparability: DEGRADED: {'; '.join(comp)}" if comp else "comparability: ok")
    match, reasons = verdict(h_tally, m_tally)
    out.append("MATCH" if match else f"MISMATCH: {'; '.join(reasons)}")
    code = (3 if comp else 0) if match else 1
    return out, warnings, code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module doc for exit codes."""
    parser = argparse.ArgumentParser(
        prog="talos-harbor-calibrate",
        description="Compare a Harbor job against a mined_eval run on the same tasks.",
    )
    parser.add_argument("--harbor-job", required=True, help="Harbor jobs dir or single job dir")
    parser.add_argument("--mined-run", required=True, help="mined_eval run_dir (from its banner)")
    args = parser.parse_args(argv)
    warnings: list[str] = []
    try:
        for raw in (args.harbor_job, args.mined_run):
            if not Path(raw).is_dir():
                raise InputError(f"not a directory: {raw}")
        harbor = _discover_harbor(Path(args.harbor_job), warnings)
        mined = _discover_mined(Path(args.mined_run), warnings)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lines, report_warnings, code = build_report(args.harbor_job, args.mined_run, harbor, mined)
    for w in warnings + report_warnings:
        print(w, file=sys.stderr)
    print("\n".join(lines))
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
