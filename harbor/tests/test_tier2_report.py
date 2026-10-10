"""Tests for talos_harbor.tier2_report: classification rules, discovery, the expected-task
set, the full stdout table, every WARN and its exit-code effect, and input errors.

Every fixture is written under tmp_path by the builders below; nothing here needs Docker,
a network, a model or a talos binary.
"""

from __future__ import annotations

import ast
import json
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from talos_harbor import tier2_report
from talos_harbor.tier2_report import SCORES, Trial, classify_trial, main

UNSET: Any = object()

MODEL = "glm-5.3:cloud"
KWARGS: dict[str, Any] = {"gate_timeout_secs": 900, "max_iterations": 500, "wall_clock_secs": 3300}
KWARGS_JSON = json.dumps(KWARGS, sort_keys=True)
VERSION = "0.14.0-gabc1234"
HEADER = "task\tresolved/k\tunresolved\tinvalid\tclaimed-done-but-unresolved"
CONFIG = (
    f"config: agents=talos@0.1.0 talos={VERSION} "
    f"backends=ollama/{MODEL} num_ctx=65536 think=high spec_levels=s2 "
    f"agent_kwargs={KWARGS_JSON}"
)
NO_FLAGS = "WARN: no --expect-k/--tasks: missing trials cannot be detected\n"


# ---- builders ---------------------------------------------------------------------------


def task_dir(root: Path, task: str, spec_level: str | None = "s2") -> Path:
    """A Harbor task dir with task.toml [metadata] spec_level (the emitter's shape)."""
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
    outcome: str = "Finished",
    claimed_done: Any = True,
    gate_armed: Any = True,
    error: Any = UNSET,
    exception: dict[str, Any] | None = None,
    kwargs: dict[str, Any] | None = UNSET,
    model: str | None = MODEL,
    version: str | None = VERSION,
    task_path: Any = UNSET,
    agent_result: Any = UNSET,
    verifier: bool = True,
) -> Path:
    """Write one Harbor trial dir under ``job`` and return it."""
    d = job / trial_name
    (d / "agent").mkdir(parents=True, exist_ok=True)
    if verifier:
        (d / "verifier").mkdir(exist_ok=True)
    if score_raw is not None:
        (d / "verifier" / "score.json").write_text(score_raw)
    elif score is not None and verifier:
        obj: dict[str, Any] = {
            "score": score,
            "detail": detail,
            "task_id": task if score_task_id is UNSET else score_task_id,
        }
        (d / "verifier" / "score.json").write_text(json.dumps(obj))
    if agent_result is UNSET:
        backend: dict[str, Any] = {"kind": "ollama", "num_ctx": 65536, "think": "high"}
        if model is not None:
            backend["model"] = model
        talos: dict[str, Any] = {
            "outcome": outcome,
            "claimed_done": claimed_done,
            "gate_armed": gate_armed,
            "backend": backend,
            "version": version,
        }
        if error is not UNSET:
            talos["error"] = error
        agent_result = {"n_input_tokens": 100, "metadata": {"talos": talos}}
    if task_path is UNSET:
        task_path = str(task_dir(job.parent.parent, task))
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
        "verifier_result": {"rewards": {"reward": 1.0 if score == "resolved" else 0.0}},
        "exception_info": exception,
    }
    (d / "result.json").write_text(json.dumps(result))
    return d


def job_level_result(job: Path) -> None:
    """Harbor's job-level result.json: no trial_name."""
    job.mkdir(parents=True, exist_ok=True)
    (job / "result.json").write_text(json.dumps({"id": "job", "stats": {}}))


def classify(d: Path) -> Trial:
    return classify_trial(d, json.loads((d / "result.json").read_text()))


def run_main(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def two_task_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Tasks a, b and c (c never ran); 5 trials: a has 3, b has 2."""
    jobs = tmp_path / "jobs"
    job = jobs / "job-1"
    job_level_result(job)
    harbor_trial(job, "a__0", "a")
    harbor_trial(job, "a__1", "a", score="unresolved", claimed_done=True)
    harbor_trial(job, "a__2", "a", score="invalid", detail="setup-failed", claimed_done=True)
    harbor_trial(job, "b__0", "b")
    harbor_trial(job, "b__1", "b", score="unresolved", claimed_done=False)
    task_dir(tmp_path, "c")
    return jobs, tmp_path / "_tasks"


# ---- module hygiene ---------------------------------------------------------------------


def test_tier2_report_imports_are_stdlib_only() -> None:
    tree = ast.parse(Path(tier2_report.__file__).read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    assert "harbor" not in mods
    assert "talos_harbor" not in mods
    for m in mods:
        assert m in sys.stdlib_module_names or m == "__future__", m


def test_tier2_report_has_no_type_ignore() -> None:
    assert "type: ignore" not in Path(tier2_report.__file__).read_text()


def test_pyproject_scripts_are_exactly_the_tier2_report() -> None:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    assert data["project"]["scripts"] == {
        "talos-harbor-tier2-report": "talos_harbor.tier2_report:main"
    }


def test_scores() -> None:
    assert SCORES == ("resolved", "unresolved", "invalid")


# ---- classification ---------------------------------------------------------------------


def test_rule1_backend_error(tmp_path: Path) -> None:
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", outcome="BackendError")
    c = classify(d)
    assert (c.score, c.reason) == ("invalid", "backend-error")


def test_rule1_wins_over_exception(tmp_path: Path) -> None:
    exc = {"exception_type": "X", "exception_message": "boom"}
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", outcome="BackendError", exception=exc)
    assert classify(d).reason == "backend-error"


def test_rule2_exception_first_line(tmp_path: Path) -> None:
    exc = {"exception_type": "AgentTimeoutError", "exception_message": "timed out\nmore"}
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", exception=exc)
    c = classify(d)
    assert (c.score, c.reason) == ("invalid", "harbor-exception(AgentTimeoutError): timed out")


def test_rule2_empty_message(tmp_path: Path) -> None:
    exc = {"exception_type": "X", "exception_message": ""}
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", exception=exc)
    assert classify(d).reason == "harbor-exception(X)"


def test_rule3_no_score_json(tmp_path: Path) -> None:
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", score=None)
    c = classify(d)
    assert (c.score, c.reason) == ("invalid", "no-score-json")


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"score": "weird"}'])
def test_rule4_unreadable(tmp_path: Path, raw: str) -> None:
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", score_raw=raw)
    c = classify(d)
    assert (c.score, c.reason) == ("invalid", "score-json-unreadable")


def test_rule5_invalid_detail(tmp_path: Path) -> None:
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", score="invalid", detail="setup-failed")
    assert (classify(d).score, classify(d).reason) == ("invalid", "setup-failed")


def test_rule5_empty_detail(tmp_path: Path) -> None:
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", score="invalid")
    assert classify(d).reason == "score-invalid-no-detail"


def test_rule5_non_str_detail(tmp_path: Path) -> None:
    raw = json.dumps({"score": "invalid", "detail": 3, "task_id": "foo"})
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", score_raw=raw)
    assert classify(d).reason == "score-invalid-no-detail"


@pytest.mark.parametrize("score", ["resolved", "unresolved"])
def test_rule6_scored(tmp_path: Path, score: str) -> None:
    d = harbor_trial(tmp_path / "j" / "job", "t", "foo", score=score)
    c = classify(d)
    assert (c.score, c.reason) == (score, None)


def test_claimed_gate_and_error_fields(tmp_path: Path) -> None:
    job = tmp_path / "j" / "job"
    c = classify(harbor_trial(job, "t1", "foo", claimed_done=True, gate_armed=True, error="x"))
    assert (c.claimed_done, c.gate_armed, c.talos_error) == (True, True, "x")
    c = classify(harbor_trial(job, "t2", "foo", claimed_done="yes", gate_armed="on", error=""))
    assert (c.claimed_done, c.gate_armed, c.talos_error) == (False, None, None)
    c = classify(harbor_trial(job, "t3", "foo", agent_result=None, gate_armed=False))
    assert (c.claimed_done, c.gate_armed, c.talos_error) == (False, None, None)
    c = classify(harbor_trial(job, "t4", "foo", gate_armed=False, error=7))
    assert (c.gate_armed, c.talos_error) == (False, None)


def test_task_key_prefers_score_task_id(tmp_path: Path) -> None:
    job = tmp_path / "j" / "job"
    d = harbor_trial(job, "t", "foo", score_task_id="bar")
    assert classify(d).task == "bar"
    d = harbor_trial(job, "u", "foo", score_task_id=None)
    assert classify(d).task == "foo"
    d = harbor_trial(job, "v", "foo", score=None)
    assert classify(d).task == "foo"


def test_trial_from_real_trial_result_model(tmp_path: Path) -> None:
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
    c = classify(d)
    assert c == Trial(
        task="foo",
        trial_name="foo__abc",
        score="invalid",
        reason="harbor-exception(NonZeroAgentExitCodeError): exit 1",
        claimed_done=True,
        gate_armed=True,
        talos_error=None,
    )
    # And the same dump without the exception is a scored, gated trial with a config line.
    (d / "result.json").write_text(
        result.model_copy(update={"exception_info": None}).model_dump_json()
    )
    c = classify(d)
    assert (c.score, c.claimed_done, c.gate_armed) == ("resolved", True, True)
    fields = tier2_report._config_fields(
        [tier2_report._Found(trial=c, result=json.loads((d / "result.json").read_text()))]
    )
    assert fields["spec_levels"][0] == ["s2"]
    assert fields["agent_kwargs"][0] == [KWARGS_JSON]


# ---- full report ------------------------------------------------------------------------


def test_report_without_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, _ = two_task_fixture(tmp_path)
    code, out, err = run_main(capsys, jobs)
    assert code == 0
    assert out == (
        f"tier2: {jobs} (5 trials, 2 tasks)\n"
        f"{CONFIG}\n"
        f"{HEADER}\n"
        "a\t1/3\t1\t1\t1\n"
        "b\t1/2\t1\t0\t0\n"
        "TOTAL\t2/5\t2\t1\t1\n"
        "invalid\ta\ta__2\tsetup-failed\n"
    )
    assert err == NO_FLAGS


def test_report_with_expect_k_and_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs, tasks = two_task_fixture(tmp_path)
    code, out, err = run_main(capsys, jobs, "--expect-k", 3, "--tasks", tasks)
    assert code == 3
    assert out == (
        f"tier2: {jobs} (5 trials, 3 tasks)\n"
        f"{CONFIG}\n"
        f"{HEADER}\n"
        "a\t1/3\t1\t1\t1\n"
        "b\t1/2\t1\t0\t0\n"
        "c\t0/0\t0\t0\t0\n"
        "TOTAL\t2/5\t2\t1\t1\n"
        "invalid\ta\ta__2\tsetup-failed\n"
        "k-mismatch\tb\t2/3\n"
        "k-mismatch\tc\t0/3\n"
    )
    assert err == ""


def test_clean_run_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a")
    code, out, err = run_main(capsys, jobs, "--expect-k", 1, "--tasks", tmp_path / "_tasks")
    assert code == 0, (out, err)
    assert out.endswith("TOTAL\t1/1\t0\t0\t0\n")
    assert err == ""


def test_spec_level_is_read_through_config_task_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a", task_path=str(task_dir(tmp_path / "x", "a", "s3")))
    harbor_trial(jobs / "job-1", "a__1", "a", task_path=str(tmp_path / "nowhere"))
    harbor_trial(jobs / "job-1", "a__2", "a", task_path=None)
    bad = task_dir(tmp_path / "y", "a")
    (bad / "task.toml").write_text("not = = toml")
    harbor_trial(jobs / "job-1", "a__3", "a", task_path=str(bad))
    harbor_trial(jobs / "job-1", "a__4", "a", task_path=str(task_dir(tmp_path / "z", "a", None)))
    _, out, _ = run_main(capsys, jobs)
    assert " spec_levels=s3,unknown " in out


def test_f_counts_only_claimed_done_and_unresolved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "jobs" / "job-1"
    harbor_trial(job, "a__0", "a", score="unresolved", claimed_done=True)
    harbor_trial(job, "a__1", "a", score="invalid", claimed_done=True)
    harbor_trial(job, "a__2", "a", score="unresolved", claimed_done=False)
    harbor_trial(job, "a__3", "a", score="unresolved", claimed_done=None)
    harbor_trial(job, "a__4", "a", score="resolved", claimed_done=True)
    _, out, _ = run_main(capsys, tmp_path / "jobs")
    assert "a\t1/5\t3\t1\t1\n" in out


def test_agent_without_verifier_counts_as_no_score_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "jobs" / "job-1"
    harbor_trial(job, "a__0", "a")
    harbor_trial(job, "a__1", "a", verifier=False)
    _, out, _ = run_main(capsys, tmp_path / "jobs")
    assert "a\t1/2\t0\t1\t0\n" in out
    assert "invalid\ta\ta__1\tno-score-json\n" in out


def test_reason_tabs_and_newlines_become_spaces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "jobs" / "job-1"
    harbor_trial(job, "a__0", "a", score="invalid", detail="x\ty\nz")
    harbor_trial(job, "a__1", "a", error="bad\tthing\nhappened")
    _, out, err = run_main(capsys, tmp_path / "jobs")
    assert "invalid\ta\ta__0\tx y z\n" in out
    assert "WARN: talos-error a a__1: bad thing happened\n" in err


def test_jobs_dir_and_single_job_dir_give_same_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs, _ = two_task_fixture(tmp_path)
    _, by_jobs, _ = run_main(capsys, jobs)
    _, by_job, _ = run_main(capsys, jobs / "job-1")
    assert by_jobs.splitlines()[1:] == by_job.splitlines()[1:]
    assert by_job.startswith(f"tier2: {jobs / 'job-1'} (5 trials, 2 tasks)\n")


def test_discovery_skips_non_trial_objects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a")
    job_level_result(jobs / "job-1")
    # a trial whose agent/result.json is not JSON is ignored (no agent/ or verifier/ below it)
    (jobs / "job-1" / "a__0" / "agent" / "result.json").write_text("not json")
    stray = jobs / "job-1" / "stray"
    (stray / "verifier").mkdir(parents=True)
    (stray / "result.json").write_text('{"trial_name": 1, "task_name": "a"}')
    lst = jobs / "job-1" / "list"
    (lst / "agent").mkdir(parents=True)
    (lst / "result.json").write_text("[]")
    code, out, _ = run_main(capsys, jobs)
    assert code == 0
    assert out.startswith(f"tier2: {jobs} (1 trials, 1 tasks)\n")


# ---- WARNs ------------------------------------------------------------------------------


def test_task_id_mismatch_and_merged_jobs_warn_without_exit_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "foo__0", "foo")
    harbor_trial(jobs / "job-2", "foo__1", "foo", task_name="talos-evals/foo")
    code, out, err = run_main(capsys, jobs, "--expect-k", 2, "--tasks", tmp_path / "_tasks")
    assert code == 0, (out, err)
    assert err == (
        "WARN: task-id mismatch foo__1: task_name=talos-evals/foo score.task_id=foo\n"
        "WARN: 2 harbor job dirs merged\n"
    )
    assert "foo\t2/2\t0\t0\t0\n" in out


def test_unexpected_task_is_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a")
    harbor_trial(jobs / "job-1", "z__0", "z")
    harbor_trial(jobs / "job-1", "y__0", "y")
    code, out, err = run_main(
        capsys, jobs, "--expect-k", 1, "--tasks", tmp_path / "_tasks", "--task", "a"
    )
    assert code == 3
    assert err == "WARN: unexpected-task y\nWARN: unexpected-task z\n"
    assert "z\t1/1\t0\t0\t0\n" in out


def test_task_patterns_filter_the_expected_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "cleanr-1__0", "cleanr-1")
    harbor_trial(jobs / "job-1", "agtd-1__0", "agtd-1")
    for t in ("cleanr-2", "other"):
        task_dir(tmp_path, t)
    (tmp_path / "_tasks" / "_base").mkdir()
    (tmp_path / "_tasks" / "notes.txt").write_text("x")
    tasks = tmp_path / "_tasks"
    code, out, err = run_main(
        capsys, jobs, "--expect-k", 1, "--tasks", tasks, "--task", "cleanr-*", "--task", "agtd-1"
    )
    assert code == 3
    assert err == ""
    assert out.splitlines()[0] == f"tier2: {jobs} (2 trials, 3 tasks)"
    assert "cleanr-2\t0/0\t0\t0\t0\n" in out
    assert "other" not in out
    assert out.endswith("k-mismatch\tcleanr-2\t0/1\n")


@pytest.mark.parametrize(
    ("field", "override", "joined"),
    [
        ("talos", {"version": "v2"}, f"{VERSION},v2"),
        (
            "backends",
            {"model": "qwen"},
            f"ollama/{MODEL} num_ctx=65536 think=high | ollama/qwen num_ctx=65536 think=high",
        ),
        (
            "agent_kwargs",
            {"kwargs": {"max_iterations": 1}},
            f'{KWARGS_JSON} | {{"max_iterations": 1}}',
        ),
    ],
)
def test_mixed_config_is_exit_3(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    override: dict[str, Any],
    joined: str,
) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a")
    harbor_trial(jobs / "job-1", "a__1", "a", **override)
    code, _, err = run_main(capsys, jobs, "--expect-k", 2, "--tasks", tmp_path / "_tasks")
    assert code == 3
    assert err == f"WARN: mixed-config {field}: {joined}\n"


def test_missing_config_values_render_dash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a", agent_result=None, kwargs=None, gate_armed=True)
    _, out, _ = run_main(capsys, jobs)
    assert "talos=- backends=- spec_levels=s2 agent_kwargs=-\n" in out


def test_gate_not_armed_is_exit_3_for_scored_trials_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "jobs" / "job-1"
    harbor_trial(job, "a__1", "a", gate_armed=None, score="unresolved")
    harbor_trial(job, "a__0", "a", gate_armed=False)
    harbor_trial(job, "a__2", "a", gate_armed=False, score="invalid")
    code, _, err = run_main(
        capsys, tmp_path / "jobs", "--expect-k", 3, "--tasks", job.parent.parent / "_tasks"
    )
    assert code == 3
    assert err == "WARN: gate-not-armed a a__0\nWARN: gate-not-armed a a__1\n"


def test_no_valid_trials_is_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job = tmp_path / "jobs" / "job-1"
    harbor_trial(job, "a__0", "a", score="invalid", detail="d")
    code, out, err = run_main(
        capsys, tmp_path / "jobs", "--expect-k", 1, "--tasks", tmp_path / "_tasks"
    )
    assert code == 3
    assert err == "WARN: no valid trials\n"
    assert "TOTAL\t0/1\t0\t1\t0\n" in out


def test_talos_error_warns_without_exit_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = tmp_path / "jobs" / "job-1"
    harbor_trial(job, "a__1", "a", error="second")
    harbor_trial(job, "a__0", "a", error="first")
    code, _, err = run_main(
        capsys, tmp_path / "jobs", "--expect-k", 2, "--tasks", tmp_path / "_tasks"
    )
    assert code == 0
    assert err == "WARN: talos-error a a__0: first\nWARN: talos-error a a__1: second\n"


def test_warn_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jobs = tmp_path / "jobs"
    harbor_trial(jobs / "job-1", "a__0", "a", task_name="x/a", gate_armed=False, error="e")
    harbor_trial(jobs / "job-2", "a__1", "a", version="v2", score="invalid")
    code, _, err = run_main(capsys, jobs)
    assert code == 3
    assert err == (
        "WARN: task-id mismatch a__0: task_name=x/a score.task_id=a\n"
        "WARN: 2 harbor job dirs merged\n" + NO_FLAGS + f"WARN: mixed-config talos: {VERSION},v2\n"
        "WARN: gate-not-armed a a__0\n"
        "WARN: talos-error a a__0: e\n"
    )


# ---- input errors and argparse ----------------------------------------------------------


def assert_input_error(
    capsys: pytest.CaptureFixture[str], argv: list[object], expected: str
) -> None:
    code, out, err = run_main(capsys, *argv)
    assert code == 2
    assert out == ""
    assert err == expected + "\n"


def test_error_jobs_not_a_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing"
    assert_input_error(capsys, [missing], f"error: not a directory: {missing}")


def test_error_tasks_not_a_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing"
    assert_input_error(
        capsys,
        [tmp_path, "--expect-k", 1, "--tasks", missing],
        f"error: not a directory: {missing}",
    )


def test_error_no_trials(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job_level_result(tmp_path / "jobs")
    assert_input_error(
        capsys, [tmp_path / "jobs"], f"error: no harbor trials found under {tmp_path / 'jobs'}"
    )


def test_error_unparseable_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = harbor_trial(tmp_path / "jobs" / "job-1", "a__0", "a")
    (d / "result.json").write_text("{nope")
    code, out, err = run_main(capsys, tmp_path / "jobs")
    assert (code, out) == (2, "")
    assert err.startswith(f"error: unparseable harbor result.json: {d / 'result.json'}: ")
    assert err.count("\n") == 1


def test_error_no_expected_tasks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    harbor_trial(tmp_path / "jobs" / "job-1", "a__0", "a")
    tasks = tmp_path / "_tasks"
    assert_input_error(
        capsys,
        [tmp_path / "jobs", "--expect-k", 1, "--tasks", tasks, "--task", "zzz*"],
        f"error: no expected tasks under {tasks}",
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["j", "--expect-k", "0", "--tasks", "t"],
        ["j", "--expect-k", "x", "--tasks", "t"],
        ["j", "--expect-k", "1"],
        ["j", "--tasks", "t"],
        ["j", "--task", "a"],
        [],
    ],
)
def test_argparse_rejections(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--expect-k" in capsys.readouterr().out
