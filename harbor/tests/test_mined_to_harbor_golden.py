"""Harbor 0.9.0 parses the mined_to_harbor golden task dir the way the emitter intends.

The golden is the Rust emitter's rendered output for the synthetic fixture task
(crates/harness/testdata/mined_to_harbor/expected/fixture-task), read in place.
"""

import json
from pathlib import Path

from harbor.models.task.task import Task
from harbor.utils.env import resolve_env_vars

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "crates/harness/testdata/mined_to_harbor/expected/fixture-task"
FIXTURE = REPO / "crates/harness/testdata/mined_to_harbor/fixture-task"


def _fixture_agent_gate() -> str:
    with (FIXTURE / "task.json").open() as f:
        gate = json.load(f)["agent_gate_command"]
    assert isinstance(gate, str)
    return gate


def test_instruction_round_trips_byte_for_byte() -> None:
    task = Task(GOLDEN)
    assert task.instruction.encode("utf-8") == (GOLDEN / "instruction.md").read_bytes()


def test_environment_env_carries_the_agent_gate_verbatim() -> None:
    env = Task(GOLDEN).config.environment.env
    expected = {"AGENT_GATE_COMMAND": _fixture_agent_gate()}
    assert env == expected
    assert resolve_env_vars(env) == expected


def test_config_values() -> None:
    config = Task(GOLDEN).config
    assert config.environment.workdir == "/app"
    assert config.environment.allow_internet is True
    assert config.environment.build_timeout_sec == 1800
    assert config.agent.timeout_sec == 3600
    assert config.verifier.timeout_sec == 1800
    meta = config.metadata
    assert meta["mined_task_id"] == "fixture-task"
    assert meta["spec_level"] == "s2"
    assert meta["test_first"] is True
    assert meta["agent_gate"] == "on"
    assert meta["parent_sha"] == "1" * 40
    assert meta["sibling_shas"] == {"fixture-sib": "3" * 40}


def test_task_score_has_exactly_the_scorer_keys() -> None:
    with (GOLDEN / "tests/task_score.json").open() as f:
        score = json.load(f)
    assert set(score) == {
        "id",
        "gate_command",
        "fail_to_pass",
        "positive_controls",
        "pass_to_pass_exclusions",
        "sealed",
    }
