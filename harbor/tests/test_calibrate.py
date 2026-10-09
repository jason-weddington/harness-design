"""Tests for talos_harbor.calibrate: classification rules, statistics, verdict and
comparability, discovery, input errors and the full stdout report.

Every fixture is written under tmp_path by the builders below; nothing here needs Docker,
a network, a model or a talos binary.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from talos_harbor import calibrate
from talos_harbor.calibrate import (
    ParityRow,
    TaskTally,
    TrialClass,
    classify_harbor_trial,
    classify_mined_record,
    comparability,
    intervals_overlap,
    main,
    verdict,
    wilson_interval,
)

UNSET: Any = object()

MODEL = "glm-5.3:cloud"
BACKEND_DESC = f"ollama {MODEL}"
KWARGS: dict[str, Any] = {"gate_timeout_secs": 900, "max_iterations": 500, "wall_clock_secs": 3300}


# ---- builders ---------------------------------------------------------------------------


def task_dir(root: Path, task: str, spec_level: str | None = "s2") -> Path:
    """A Harbor task dir with task.toml [metadata] spec_level (C2's shape)."""
    d = root / "_tasks" / task
    d.mkdir(parents=True, exist_ok=True)
    meta = f'[metadata]\nspec_level = "{spec_level}"\n' if spec_level else "[agent]\n"
    (d / "task.toml").write_text(meta)
    return d


def harbor_trial(
    job: Path,
    trial_name: str,
    task: str,
    *,
    score: str | None = "resolved",
    score_raw: str | None = None,
    detail: str = "",
    score_task_id: str | None = UNSET,
    task_name: str | None = None,
    count_check: str | None = "on",
    outcome: str = "Finished",
    claimed_done: Any = True,
    gate_armed: Any = True,
    exception: dict[str, Any] | None = None,
    reward: Any = UNSET,
    kwargs: dict[str, Any] | None = UNSET,
    model: str | None = MODEL,
    version: str = "0.14.0-gabc1234",
    task_path: Any = UNSET,
    agent_result: Any = UNSET,
) -> Path:
    """Write one Harbor trial dir under ``job`` and return it."""
    d = job / trial_name
    (d / "agent").mkdir(parents=True, exist_ok=True)
    (d / "verifier").mkdir(exist_ok=True)
    if score_raw is not None:
        (d / "verifier" / "score.json").write_text(score_raw)
    elif score is not None:
        obj: dict[str, Any] = {
            "score": score,
            "detail": detail,
            "task_id": task if score_task_id is UNSET else score_task_id,
        }
        if count_check is not None:
            obj["count_check"] = count_check
        (d / "verifier" / "score.json").write_text(json.dumps(obj))
        (d / "verifier" / "reward.txt").write_text("1" if score == "resolved" else "0")
    if reward is UNSET:
        reward = 1.0 if score == "resolved" else 0.0
    if agent_result is UNSET:
        backend: dict[str, Any] = {"kind": "ollama"}
        if model is not None:
            backend["model"] = model
        agent_result = {
            "n_input_tokens": 100,
            "metadata": {
                "talos": {
                    "outcome": outcome,
                    "claimed_done": claimed_done,
                    "gate_armed": gate_armed,
                    "backend": backend,
                    "version": version,
                }
            },
        }
    if task_path is UNSET:
        task_path = str(task_dir(job.parent, task))
    result = {
        "task_name": task if task_name is None else task_name,
        "trial_name": trial_name,
        "trial_uri": f"file://{d}",
        "task_id": {"path": str(task_path)},
        "task_checksum": "abc",
        "config": {
            "task": {"path": task_path},
            "agent": {"kwargs": dict(KWARGS) if kwargs is UNSET else kwargs},
        },
        "agent_info": {"name": "talos", "version": "0.1.0"},
        "agent_result": agent_result,
        "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
        "exception_info": exception,
    }
    (d / "result.json").write_text(json.dumps(result))
    return d


def job_level_result(job: Path) -> None:
    """Harbor's job-level result.json: no trial_name."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "result.json").write_text(json.dumps({"id": "job", "stats": {}}))


def mined_record(
    run: Path,
    task: str,
    trial: int,
    *,
    k: int = 3,
    score: str = "resolved",
    invalid_reason: str | None = None,
    claimed_disposition: str = "Done",
    finish_recovery_armed: bool = True,
    agent_gate: str = "on(timeout=900s)",
    wall_clock_secs: int = 3300,
    max_iterations: int = 500,
    test_first: bool = True,
    backend_desc: str = BACKEND_DESC,
    spec_level: str = "s2",
    count_check: str = "on",
    schema: str = "talos-mined-trial/1",
) -> Path:
    """Write one mined trial-result.json and return its path."""
    d = run / task / f"trial-{trial}"
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": schema,
        "task_id": task,
        "trial": trial,
        "k": k,
        "spec_level": spec_level,
        "backend_desc": backend_desc,
        "max_iterations": max_iterations,
        "agent_gate": agent_gate,
        "test_first": test_first,
        "wall_clock_secs": wall_clock_secs,
        "compact_threshold_pct": 90,
        "harness_version": "0.1.0",
        "score": score,
        "invalid_reason": invalid_reason,
        "missing_fail_to_pass": [],
        "unexcluded_red": [],
        "collection_errors": [],
        "claimed_disposition": claimed_disposition,
        "claimed_done": claimed_disposition == "Done",
        "finish_recovery_armed": finish_recovery_armed,
        "count_check": count_check,
    }
    p = d / "trial-result.json"
    p.write_text(json.dumps(rec, indent=2) + "\n")
    return p


def matched_pair(
    tmp_path: Path, scores: dict[str, list[str]], mined_scores: dict[str, list[str]] | None = None
) -> tuple[Path, Path]:
    """Jobs dir + mined run dir with the same tasks, k and knobs on both sides."""
    jobs = tmp_path / "jobs"
    job = jobs / "job-1"
    job_level_result(job)
    run = tmp_path / "run"
    for task, ss in scores.items():
        for i, s in enumerate(ss):
            harbor_trial(job, f"{task}__{i}", task, score=s)
    for task, ss in (mined_scores or scores).items():
        for i, s in enumerate(ss):
            mined_record(run, task, i, k=len(ss), score=s)
    return jobs, run


def run_main(capsys: pytest.CaptureFixture[str], jobs: Path, run: Path) -> tuple[int, str, str]:
    code = main(["--harbor-job", str(jobs), "--mined-run", str(run)])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def classify(tmp_path: Path, **kw: Any) -> TrialClass:
    d = harbor_trial(tmp_path / "job", "t__1", "foo", **kw)
    return classify_harbor_trial(d, json.loads((d / "result.json").read_text()))


def tc(
    side: str = "harbor",
    task: str = "a",
    score: str = "resolved",
    gate: bool | None = True,
    reward_disagree: bool = False,
) -> TrialClass:
    return TrialClass(
        side=side,  # type: ignore[arg-type]
        task_id=task,
        trial="t",
        rule=6,
        score=score,  # type: ignore[arg-type]
        reason=None,
        claimed="done",
        gate_armed=gate,
        overridden_score=None,
        reward_disagree=reward_disagree,
        source="x",
    )


def tally(k: int, resolved: int, invalid: int = 0, missing: int = 0) -> TaskTally:
    return TaskTally(k=k, resolved=resolved, invalid=invalid, missing_records=missing)


# ---- module hygiene ---------------------------------------------------------------------


def test_calibrate_imports_are_stdlib_only() -> None:
    tree = ast.parse(Path(calibrate.__file__).read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert "harbor" not in mods
    for m in mods:
        assert m in sys.stdlib_module_names or m == "talos_harbor", m


def test_calibrate_has_no_type_ignore() -> None:
    assert "type: ignore" not in Path(calibrate.__file__).read_text()


# ---- statistics -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("s", "n", "lo", "hi"),
    [
        (0, 10, 0.0, 0.277540),
        (10, 10, 0.722460, 1.0),
        (5, 10, 0.236590, 0.763410),
        (1, 3, 0.061490, 0.792345),
        (12, 24, 0.314271, 0.685729),
        (9, 24, 0.211591, 0.572904),
    ],
)
def test_wilson_vectors(s: int, n: int, lo: float, hi: float) -> None:
    iv = wilson_interval(s, n)
    assert iv is not None
    assert iv[0] == pytest.approx(lo, abs=1e-6)
    assert iv[1] == pytest.approx(hi, abs=1e-6)


def test_wilson_empty_is_none() -> None:
    assert wilson_interval(0, 0) is None


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ((0.2, 0.5), (0.5, 0.8), True),
        ((0.2, 0.49), (0.5, 0.8), False),
        ((0.3, 0.6), (0.1, 0.9), True),
    ],
)
def test_intervals_overlap(a: tuple[float, float], b: tuple[float, float], want: bool) -> None:
    assert intervals_overlap(a, b) is want


# ---- Harbor classification (AC-6) -------------------------------------------------------


def test_harbor_rule1_backend_error(tmp_path: Path) -> None:
    c = classify(tmp_path, outcome="BackendError", score="unresolved")
    assert (c.rule, c.score, c.reason, c.overridden_score) == (
        1,
        "invalid",
        "backend-error",
        "unresolved",
    )


def test_harbor_rule1_wins_over_exception(tmp_path: Path) -> None:
    exc = {"exception_type": "NonZeroAgentExitCodeError", "exception_message": "rc=1"}
    c = classify(tmp_path, outcome="BackendError", exception=exc)
    assert c.rule == 1
    assert c.reason == "backend-error"


def test_harbor_rule2_exception_first_line(tmp_path: Path) -> None:
    exc = {"exception_type": "AgentTimeoutError", "exception_message": "took too long\nmore"}
    c = classify(tmp_path, exception=exc, score="resolved")
    assert c.rule == 2
    assert c.reason == "harbor-exception(AgentTimeoutError): took too long"
    assert c.overridden_score == "resolved"


def test_harbor_rule2_empty_message(tmp_path: Path) -> None:
    c = classify(tmp_path, exception={"exception_type": "X"}, score=None)
    assert c.reason == "harbor-exception(X)"
    assert c.overridden_score is None


def test_harbor_rule3_no_score_json(tmp_path: Path) -> None:
    c = classify(tmp_path, score=None)
    assert (c.rule, c.score, c.reason) == (3, "invalid", "no-score-json")


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"score": "maybe"}'])
def test_harbor_rule4_unreadable(tmp_path: Path, raw: str) -> None:
    c = classify(tmp_path, score_raw=raw)
    assert (c.rule, c.score, c.reason) == (4, "invalid", "score-json-unreadable")


def test_harbor_rule5_invalid_detail(tmp_path: Path) -> None:
    c = classify(tmp_path, score="invalid", detail="parse-empty: nothing")
    assert (c.rule, c.score, c.reason) == (5, "invalid", "parse-empty: nothing")


def test_harbor_rule5_empty_detail(tmp_path: Path) -> None:
    c = classify(tmp_path, score="invalid", detail="")
    assert c.reason == "score-invalid-no-detail"


@pytest.mark.parametrize("score", ["resolved", "unresolved"])
def test_harbor_rule6_scored(tmp_path: Path, score: str) -> None:
    c = classify(tmp_path, score=score)
    assert (c.rule, c.score, c.reason, c.reward_disagree) == (6, score, None, False)


def test_harbor_claimed_and_gate(tmp_path: Path) -> None:
    assert classify(tmp_path / "a", claimed_done=False, gate_armed=False).claimed == "not-done"
    c = classify(tmp_path / "b", agent_result=None)
    assert (c.claimed, c.gate_armed) == ("unknown", None)
    c = classify(tmp_path / "c", agent_result={"metadata": None})
    assert (c.claimed, c.gate_armed) == ("unknown", None)
    assert classify(tmp_path / "d", claimed_done=None).claimed == "unknown"


def test_harbor_reward_disagree(tmp_path: Path) -> None:
    assert classify(tmp_path / "a", score="unresolved", reward=1.0).reward_disagree
    assert classify(tmp_path / "b", score="resolved", reward=0.0).reward_disagree
    assert not classify(tmp_path / "c", score="resolved", reward=None).reward_disagree
    assert not classify(tmp_path / "d", score="invalid", reward=1.0).reward_disagree


def test_harbor_task_key_prefers_score_task_id(tmp_path: Path) -> None:
    assert classify(tmp_path / "a", task_name="talos-evals/foo").task_id == "foo"
    assert classify(tmp_path / "b", task_name="bar", score_task_id=None).task_id == "bar"


def test_harbor_trial_from_real_trial_result_model(tmp_path: Path) -> None:
    """Realism check: a real harbor TrialResult dump classifies like the hand fixtures."""
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.id import LocalTaskId
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
    from harbor.models.verifier.result import VerifierResult

    tdir = task_dir(tmp_path, "foo")
    result = TrialResult(
        task_name="foo",
        trial_name="foo__abc",
        trial_uri="file:///x",
        task_id=LocalTaskId(path=tdir),
        task_checksum="c",
        config=TrialConfig(
            task=TaskConfig(path=tdir),
            agent=AgentConfig(import_path="talos_harbor.agent:TalosAgent", kwargs=KWARGS),
        ),
        agent_info=AgentInfo(name="talos", version="0.1.0"),
        agent_result=AgentContext(
            metadata={
                "talos": {
                    "outcome": "Finished",
                    "claimed_done": True,
                    "gate_armed": True,
                    "backend": {"kind": "ollama", "model": MODEL},
                    "version": "v",
                }
            }
        ),
        verifier_result=VerifierResult(rewards={"reward": 1}),
        exception_info=ExceptionInfo(
            exception_type="NonZeroAgentExitCodeError",
            exception_message="exit 1\ntail",
            exception_traceback="tb",
            occurred_at=datetime(2026, 1, 1),
        ),
    )
    d = tmp_path / "jobs" / "job" / "foo__abc"
    (d / "verifier").mkdir(parents=True)
    (d / "verifier" / "score.json").write_text('{"score": "resolved", "task_id": "foo"}')
    (d / "result.json").write_text(result.model_dump_json())
    c = classify_harbor_trial(d, json.loads((d / "result.json").read_text()))
    assert (c.rule, c.reason) == (2, "harbor-exception(NonZeroAgentExitCodeError): exit 1")
    assert (c.claimed, c.gate_armed, c.overridden_score) == ("done", True, "resolved")
    rows = calibrate.parity_rows(
        [json.loads(mined_record(tmp_path / "run", "foo", 0).read_text())],
        [
            calibrate._HarborTrial(
                cls=c, result=json.loads(result.model_dump_json()), score_json=None
            )
        ],
    )
    assert all(r.ok for r in rows), rows


# ---- mined classification (AC-7) --------------------------------------------------------


def mrec(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "foo",
        "trial": 0,
        "score": "resolved",
        "invalid_reason": None,
        "claimed_disposition": "Done",
        "claimed_done": True,
        "finish_recovery_armed": True,
    }
    base.update(kw)
    return base


def test_mined_rule1_invalid() -> None:
    c = classify_mined_record(mrec(score="invalid", invalid_reason="setup-failed: x"), "s")
    assert (c.rule, c.score, c.reason, c.trial) == (1, "invalid", "setup-failed: x", "trial-0")
    c = classify_mined_record(mrec(score="invalid", invalid_reason=None), "s")
    assert c.reason == "mined-invalid-no-reason"


def test_mined_rule2_backend_error() -> None:
    c = classify_mined_record(
        mrec(score="unresolved", claimed_disposition="BackendError(x)", claimed_done=False), "s"
    )
    assert (c.rule, c.score, c.reason, c.overridden_score) == (
        2,
        "invalid",
        "backend-error",
        "unresolved",
    )
    assert c.claimed == "not-done"


def test_mined_rule3_scored() -> None:
    c = classify_mined_record(mrec(score="unresolved", finish_recovery_armed=False), "s")
    assert (c.rule, c.score, c.claimed, c.gate_armed) == (3, "unresolved", "done", False)
    assert classify_mined_record(mrec(), "s").score == "resolved"


# ---- verdict + comparability (AC-9) -----------------------------------------------------


def test_verdict_clean_match() -> None:
    t = {"a": tally(3, 2), "b": tally(3, 1)}
    assert verdict(t, dict(t)) == (True, [])


def test_verdict_no_valid_trials_each_side() -> None:
    assert verdict({"a": tally(2, 0, invalid=2)}, {"a": tally(2, 1)}) == (
        False,
        ["no-valid-trials: harbor"],
    )
    assert verdict({"a": tally(2, 1)}, {"a": tally(2, 0, invalid=2)}) == (
        False,
        ["no-valid-trials: mined"],
    )
    both = verdict({"a": tally(1, 0, invalid=1)}, {"a": tally(1, 0, invalid=1)})
    assert both == (False, ["no-valid-trials: harbor", "no-valid-trials: mined"])


def test_verdict_intervals_disjoint() -> None:
    assert verdict({"a": tally(24, 24)}, {"a": tally(24, 0)})[1][0] == "intervals-disjoint"


def test_verdict_task_diff_ids_and_threshold() -> None:
    h = {"b": tally(5, 5), "a": tally(5, 0), "c": tally(5, 3), "d": tally(5, 2)}
    m = {"b": tally(5, 2), "a": tally(5, 3), "c": tally(5, 1), "d": tally(5, 4)}
    ok, reasons = verdict(h, m)
    assert not ok
    assert reasons == ["task-diff>2: a,b"]
    assert verdict({"a": tally(5, 4)}, {"a": tally(5, 2)}) == (True, [])


def test_verdict_k_mismatch_tasks_have_no_diff() -> None:
    assert verdict({"a": tally(5, 5)}, {"a": tally(4, 2)}) == (True, [])


def rows(*diff: str) -> list[ParityRow]:
    return [ParityRow(k, "1", "1" if k not in diff else "2", k not in diff) for k in ("x", "y")]


def test_comparability_ok() -> None:
    t = {"a": tally(1, 1)}
    assert comparability(t, t, rows(), [tc()], [tc("mined")]) == []


def test_comparability_each_reason_in_order() -> None:
    h = {"a": tally(1, 1), "b": tally(2, 1), "h": tally(1, 1)}
    m = {"a": tally(1, 1), "b": tally(2, 1, missing=1), "m": tally(1, 1)}
    got = comparability(
        h,
        m,
        rows("x", "y"),
        [tc(task="a", gate=None), tc(task="b", reward_disagree=True)],
        [tc("mined", task="a"), tc("mined", task="b")],
    )
    assert got == [
        "config-mismatch: x,y",
        "gate-parity: a",
        "reward-score-disagree: 1",
        "k-mismatch: b",
        "only-harbor: h",
        "only-mined: m",
    ]


def test_comparability_gate_parity_ignores_invalid_trials() -> None:
    t = {"a": tally(2, 1, invalid=1)}
    h = [tc(task="a"), tc(task="a", gate=None, score="invalid")]
    m = [tc("mined", task="a"), tc("mined", task="a", gate=False, score="invalid")]
    assert comparability(t, t, rows(), h, m) == []
    assert comparability(t, t, rows(), [tc(task="a", gate=False)], m) == ["gate-parity: a"]


# ---- discovery + input errors (AC-4, AC-5, AC-7) ----------------------------------------


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--harbor-job" in capsys.readouterr().out


def assert_input_error(
    capsys: pytest.CaptureFixture[str], jobs: Path, run: Path, prefix: str
) -> str:
    code, out, err = run_main(capsys, jobs, run)
    assert code == 2
    assert out == ""
    assert err.count("\n") == 1
    assert err.startswith(prefix), err
    return err


def test_error_not_a_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    err = assert_input_error(capsys, tmp_path / "nope", run, "error: not a directory: ")
    assert err.strip().endswith("nope")
    assert_input_error(capsys, jobs, tmp_path / "nope2", "error: not a directory: ")


def test_error_empty_sides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    empty = tmp_path / "empty"
    empty.mkdir()
    assert_input_error(capsys, empty, run, "error: no harbor trials found under ")
    assert_input_error(capsys, jobs, empty, "error: no mined trial records found under ")


def test_error_corrupt_harbor_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    (jobs / "job-1" / "a__0" / "result.json").write_text("{not json")
    assert_input_error(capsys, jobs, run, "error: unparseable harbor result.json: ")


def test_error_corrupt_mined_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    (run / "a" / "trial-0" / "trial-result.json").write_text("nope")
    assert_input_error(capsys, jobs, run, "error: unparseable mined trial-result.json: ")


def test_error_mined_schema_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    mined_record(run, "a", 0, k=1, schema="talos-mined-trial/0")
    err = assert_input_error(capsys, jobs, run, "error: mined schema mismatch: ")
    assert err.strip().endswith("talos-mined-trial/0")
    (run / "a" / "trial-0" / "trial-result.json").write_text("[]")
    assert_input_error(capsys, jobs, run, "error: mined schema mismatch: ")
    mined_record(run, "a", 0, k=1, score="weird")
    assert_input_error(capsys, jobs, run, "error: mined schema mismatch: ")


def test_error_inconsistent_mined(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved", "resolved"]})
    mined_record(run, "a", 1, k=3)
    err = assert_input_error(capsys, jobs, run, "error: inconsistent mined records for ")
    assert "a: k disagrees" in err
    # duplicate trial: two dirs whose records both claim trial 0
    mined_record(run, "a", 1, k=2)
    p = mined_record(run, "a", 0, k=2)
    rec = json.loads(p.read_text())
    (run / "a" / "trial-1" / "trial-result.json").write_text(json.dumps(rec))
    err = assert_input_error(capsys, jobs, run, "error: inconsistent mined records for ")
    assert "duplicate trial 0" in err
    # count > k
    mined_record(run, "a", 0, k=1)
    mined_record(run, "a", 1, k=1)
    err = assert_input_error(capsys, jobs, run, "error: inconsistent mined records for ")
    assert "2 records > k=1" in err
    # non-int k
    mined_record(run, "a", 0, k="x")  # type: ignore[arg-type]
    mined_record(run, "a", 1, k="x")  # type: ignore[arg-type]
    err = assert_input_error(capsys, jobs, run, "error: inconsistent mined records for ")
    assert "k is not an int" in err


def test_discovery_single_job_dir_and_jobs_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"] * 3, "b": ["unresolved"] * 3})
    # a trial whose agent/result.json is not JSON is ignored
    (jobs / "job-1" / "a__0" / "agent" / "result.json").write_text("not json")
    for root in (jobs, jobs / "job-1"):
        code, out, err = run_main(capsys, root, run)
        assert code == 0, (out, err)
        assert f"calibration: harbor={root} (6 trials) mined={run} (6 trials)" in out
        assert err == ""


def test_discovery_skips_non_trial_objects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    stray = jobs / "job-1" / "stray"
    (stray / "verifier").mkdir(parents=True)
    (stray / "result.json").write_text('{"trial_name": 1, "task_name": "a"}')
    code, out, _ = run_main(capsys, jobs, run)
    assert code == 0
    assert "(1 trials)" in out


def test_discovery_warns_on_merged_jobs_and_task_id_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, run = matched_pair(tmp_path, {"foo": ["resolved"]})
    harbor_trial(jobs / "job-2", "foo__x", "foo", task_name="talos-evals/foo")
    mined_record(run, "foo", 0, k=2)
    mined_record(run, "foo", 1, k=2)
    code, out, err = run_main(capsys, jobs, run)
    assert code == 0, out
    assert "WARN: 2 harbor job dirs merged\n" in err
    assert "WARN: task-id mismatch foo__x: task_name=talos-evals/foo score.task_id=foo\n" in err
    assert "  foo  harbor 2/2 (invalid 0, k 2)  mined 2/2 (invalid 0, k 2)  diff 0" in out


def test_missing_records_is_k_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(
        tmp_path, {"a": ["resolved"] * 2}, mined_scores={"a": ["resolved"] * 2}
    )
    mined_record(run, "a", 0, k=3)
    mined_record(run, "a", 1, k=3)
    code, out, err = run_main(capsys, jobs, run)
    assert "WARN: missing-records: a=1\n" in err
    assert "diff - [k-mismatch]" in out
    assert "comparability: DEGRADED: k-mismatch: a" in out
    assert code == 3


# ---- end to end (AC-10, AC-11) ----------------------------------------------------------


def test_end_to_end_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(
        tmp_path,
        {"a": ["resolved", "resolved", "unresolved"], "b": ["resolved", "unresolved", "invalid"]},
    )
    code, out, err = run_main(capsys, jobs, run)
    assert code == 0, out
    assert out.rstrip("\n").splitlines()[-1] == "MATCH"
    assert "comparability: ok" in out
    assert err == ""


def test_end_to_end_mismatch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(
        tmp_path,
        {"a": ["resolved"] * 3, "b": ["resolved"] * 3},
        mined_scores={"a": ["unresolved"] * 3, "b": ["unresolved"] * 3},
    )
    code, out, _ = run_main(capsys, jobs, run)
    assert code == 1
    assert out.splitlines()[-1] == "MISMATCH: intervals-disjoint; task-diff>2: a,b"


def test_end_to_end_config_diff_is_exit_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    for task in ("a", "b"):
        for i in range(3):
            harbor_trial(
                jobs / "j",
                f"{task}__{i}",
                task,
                kwargs={"max_iterations": 500, "wall_clock_secs": 3300},
            )
            mined_record(run, task, i)
    code, out, _ = run_main(capsys, jobs, run)
    assert "  parity gate_timeout_secs: mined=900 harbor=300 DIFF" in out
    assert "comparability: DEGRADED: config-mismatch: gate_timeout_secs" in out
    assert out.splitlines()[-1] == "MATCH"
    assert code == 3


def test_only_harbor_task_is_match_but_degraded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    harbor_trial(jobs / "job-1", "z__0", "z")
    code, out, _ = run_main(capsys, jobs, run)
    assert "  z  harbor 1/1 (invalid 0, k 1)  mined -  diff - [only-harbor]" in out
    assert "comparability: DEGRADED: only-harbor: z" in out
    assert out.splitlines()[-1] == "MATCH"
    assert code == 3


def test_only_mined_and_gate_parity_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    harbor_trial(jobs / "j", "a__0", "a", gate_armed=None)
    mined_record(run, "a", 0, k=1)
    mined_record(run, "m", 0, k=1)
    code, out, _ = run_main(capsys, jobs, run)
    assert (
        "  a  harbor 1/1 (invalid 0, k 1)  mined 1/1 (invalid 0, k 1)  diff 0 [gate-parity]" in out
    )
    assert "  m  harbor -  mined 1/1 (invalid 0, k 1)  diff - [only-mined]" in out
    assert "comparability: DEGRADED: gate-parity: a; only-mined: m" in out
    assert code == 3


def test_task_diff_flag_in_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(
        tmp_path,
        {"a": ["resolved"] * 3, "b": ["unresolved"] * 3},
        mined_scores={"a": ["unresolved"] * 3, "b": ["unresolved"] * 3},
    )
    code, out, _ = run_main(capsys, jobs, run)
    assert (
        "  a  harbor 3/3 (invalid 0, k 3)  mined 0/3 (invalid 0, k 3)  diff 3 [task-diff>2]" in out
    )
    assert code == 1


def test_model_parity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    code, out, _ = run_main(capsys, jobs, run)
    assert f"  parity model: mined={BACKEND_DESC} harbor={MODEL} ok" in out
    assert code == 0
    harbor_trial(jobs / "job-1", "a__0", "a", model="qwen3")
    code, out, _ = run_main(capsys, jobs, run)
    assert f"  parity model: mined={BACKEND_DESC} harbor=qwen3 DIFF" in out
    assert "comparability: DEGRADED: config-mismatch: model" in out
    assert out.splitlines()[-1] == "MATCH"
    assert code == 3
    harbor_trial(jobs / "job-1", "a__0", "a", model=None)
    code, out, _ = run_main(capsys, jobs, run)
    assert "harbor=unknown DIFF" in out
    assert code == 3


def test_spec_level_parity_unknown_is_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, run = matched_pair(tmp_path, {"a": ["resolved"]})
    harbor_trial(jobs / "job-1", "a__0", "a", task_path=str(tmp_path / "gone"))
    code, out, _ = run_main(capsys, jobs, run)
    assert "  parity spec_level: mined=s2 harbor=unknown DIFF" in out
    assert code == 3
    for path in (None, str(task_dir(tmp_path, "nometa", spec_level=None))):
        harbor_trial(jobs / "job-1", "a__0", "a", task_path=path)
        code, out, _ = run_main(capsys, jobs, run)
        assert "harbor=unknown DIFF" in out
    bad = task_dir(tmp_path, "bad")
    (bad / "task.toml").write_text("[[[")
    harbor_trial(jobs / "job-1", "a__0", "a", task_path=str(bad))
    code, out, _ = run_main(capsys, jobs, run)
    assert "  parity spec_level: mined=s2 harbor=unknown DIFF" in out


def test_gate_off_renders_na(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    harbor_trial(jobs / "j", "a__0", "a", kwargs=None, gate_armed=False)
    mined_record(run, "a", 0, k=1, agent_gate="off", finish_recovery_armed=False, wall_clock_secs=0)
    code, out, _ = run_main(capsys, jobs, run)
    assert "  parity gate_timeout_secs: mined=off harbor=n/a ok" in out
    assert "  parity max_iterations: mined=500 harbor=500 ok" in out
    assert "  parity wall_clock_secs: mined=0 harbor=unset DIFF" in out
    assert "harbor header: agents=talos@0.1.0 backends=ollama/glm-5.3:cloud agent_kwargs=-" in out
    assert code == 3


def test_overrides_and_invalid_groups(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    j = jobs / "j"
    exc = "exception_type"
    harbor_trial(j, "a__0", "a", exception={exc: "NonZeroAgentExitCodeError"}, score="resolved")
    harbor_trial(j, "a__1", "a", exception={exc: "AgentTimeoutError"}, score=None)
    harbor_trial(j, "a__2", "a", outcome="BackendError", score="unresolved")
    harbor_trial(j, "a__3", "a")
    for i in range(4):
        mined_record(run, "a", i, k=4)
    mined_record(run, "a", 3, k=4, claimed_disposition="BackendError(boom)", score="resolved")
    code, out, err = run_main(capsys, jobs, run)
    assert (
        "invalid harbor: backend-error=1, harbor-exception(AgentTimeoutError)=1, "
        "harbor-exception(NonZeroAgentExitCodeError)=1"
    ) in out
    assert "invalid mined: backend-error=1" in out
    assert (
        "overrides harbor: backend-error-over-scored=1 harbor-exception-over-scored=1 "
        "(would-be resolved=1)"
    ) in out
    assert (
        "overrides mined: backend-error-over-scored=1 harbor-exception-over-scored=0 "
        "(would-be resolved=1)"
    ) in out
    assert "WARN: 2 scored trials discarded by override on harbor\n" in err
    assert "WARN: 1 scored trials discarded by override on mined\n" in err
    assert "  mined a trial-3 rule=2 class=invalid claimed=not-done gate_armed=True " in out
    assert "reason=backend-error overridden=resolved" in out
    assert code in (0, 3)


def test_harbor_exception_grouping(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    harbor_trial(jobs / "j", "a__0", "a", exception={"exception_type": "NonZeroAgentExitCodeError"})
    harbor_trial(jobs / "j", "a__1", "a", exception={"exception_type": "AgentTimeoutError"})
    harbor_trial(jobs / "j", "a__2", "a")
    for i in range(3):
        mined_record(run, "a", i)
    _, out, _ = run_main(capsys, jobs, run)
    assert (
        "invalid harbor: harbor-exception(AgentTimeoutError)=1, "
        "harbor-exception(NonZeroAgentExitCodeError)=1\n"
    ) in out


def test_golden_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    j = jobs / "job-1"
    job_level_result(j)
    harbor_trial(j, "foo__0", "foo", score="resolved", count_check="on")
    harbor_trial(j, "foo__1", "foo", score="unresolved", count_check="off")
    harbor_trial(j, "foo__2", "foo", score="invalid", detail="parse-empty: x", count_check="n/a")
    mined_record(run, "foo", 0, score="resolved")
    mined_record(run, "foo", 1, score="unresolved")
    mined_record(
        run,
        "foo",
        2,
        score="invalid",
        invalid_reason="gate-timeout: 900s",
        claimed_disposition="MaxIterations",
        count_check="n/a",
    )
    code, out, err = run_main(capsys, jobs, run)
    lines = out.splitlines()
    kw = json.dumps(KWARGS, sort_keys=True)
    assert lines[0] == f"calibration: harbor={jobs} (3 trials) mined={run} (3 trials)"
    assert lines[1] == (
        f"mined header: backend={BACKEND_DESC} spec_level=s2 agent_gate=on(timeout=900s) "
        "test_first=True max_iterations=500 wall_clock_secs=3300"
    )
    assert lines[2] == (
        f"harbor header: agents=talos@0.1.0 backends=ollama/{MODEL} agent_kwargs={kw}"
    )
    assert lines[3] == "versions: mined=0.1.0 harbor=0.14.0-gabc1234"
    assert lines[4] == "trials:"
    assert lines[5] == (
        "  harbor foo foo__0 rule=6 class=resolved claimed=done gate_armed=True "
        "reason=- overridden=-"
    )
    assert lines[10] == (
        "  mined foo trial-2 rule=1 class=invalid claimed=not-done gate_armed=True "
        "reason=gate-timeout: 900s overridden=-"
    )
    assert lines[11] == "parity:"
    assert lines[12:18] == [
        "  parity max_iterations: mined=500 harbor=500 ok",
        "  parity gate_timeout_secs: mined=900 harbor=900 ok",
        "  parity wall_clock_secs: mined=3300 harbor=3300 ok",
        "  parity test_first: mined=True harbor=True ok",
        f"  parity model: mined={BACKEND_DESC} harbor={MODEL} ok",
        "  parity spec_level: mined=s2 harbor=s2 ok",
    ]
    assert lines[18:20] == [
        "tasks:",
        "  foo  harbor 1/2 (invalid 1, k 3)  mined 1/2 (invalid 1, k 3)  diff 0",
    ]
    assert lines[20:22] == [
        "pooled harbor: 1/2 = 0.500 [0.095, 0.905]",
        "pooled mined: 1/2 = 0.500 [0.095, 0.905]",
    ]
    assert lines[22:29] == [
        "false-done cross-tab (harbor): claimed \u00d7 score",
        "  done resolved=1 unresolved=1 invalid=1",
        "  not-done resolved=0 unresolved=0 invalid=0",
        "  unknown resolved=0 unresolved=0 invalid=0",
        "false_dones harbor=1",
        "invalid harbor: parse-empty=1",
        "overrides harbor: backend-error-over-scored=0 harbor-exception-over-scored=0 "
        "(would-be resolved=0)",
    ]
    assert lines[29:36] == [
        "false-done cross-tab (mined): claimed \u00d7 score",
        "  done resolved=1 unresolved=1 invalid=0",
        "  not-done resolved=0 unresolved=0 invalid=1",
        "  unknown resolved=0 unresolved=0 invalid=0",
        "false_dones mined=1",
        "invalid mined: gate-timeout=1",
        "overrides mined: backend-error-over-scored=0 harbor-exception-over-scored=0 "
        "(would-be resolved=0)",
    ]
    assert lines[36:] == [
        "parser-health harbor: count_check on=1 off=1 n/a=1",
        "parser-health mined: count_check on=2 off=0 n/a=1",
        "comparability: ok",
        "MATCH",
    ]
    assert err == ""
    assert code == 0


def test_empty_pooled_and_invalid_none(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    run = tmp_path / "run"
    harbor_trial(jobs / "j", "a__0", "a", score=None)
    mined_record(run, "b", 0, k=1)
    code, out, _ = run_main(capsys, jobs, run)
    assert "pooled harbor: 0/0 = n/a" in out
    assert "pooled mined: 0/0 = n/a" in out
    assert "invalid mined: none" in out
    assert out.splitlines()[-1] == "MISMATCH: no-valid-trials: harbor; no-valid-trials: mined"
    assert code == 1
