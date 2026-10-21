"""Hermetic tests for TalosAgent: no Docker, no network, no model, no real talos binary."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import shutil
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEnvironment, as_any, result
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext

from talos_harbor import agent as agent_mod
from talos_harbor.agent import TalosAgent, build_run_command

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://artifacts.example.invalid"
FORWARDED = [
    "TALOS_BACKEND",
    "TALOS_BEDROCK",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_KEY",
    "OLLAMA_NUM_CTX",
    "OLLAMA_THINK",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TALOS_ARTIFACT_BASE", raising=False)
    for key in FORWARDED:
        monkeypatch.delenv(key, raising=False)
    for key in ("PERSONAL_KB_URL", "PERSONAL_KB_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key in list(os.environ):
        if key.startswith("AWS_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def boom(url: str) -> bytes:
        calls.append(url)
        raise AssertionError(f"unexpected network call: {url}")

    monkeypatch.setattr(agent_mod, "_http_get", boom)
    return calls


def make(tmp_path: Path, **kwargs: Any) -> TalosAgent:
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    kwargs.setdefault("artifact_base", BASE)
    return TalosAgent(logs_dir=logs, **kwargs)


# ---------------------------------------------------------------- AC-5 identity


def test_identity() -> None:
    cls = importlib.import_module("talos_harbor.agent").TalosAgent
    assert cls.import_path() == "talos_harbor.agent:TalosAgent"
    assert TalosAgent.name() == "talos"
    assert TalosAgent.SUPPORTS_ATIF is False


def test_agent_module_has_no_tier2_assumptions() -> None:
    text = Path(agent_mod.__file__).read_text(encoding="utf-8")
    for word in ("task.json", "sealed", "fail_to_pass", "mined"):
        assert word not in text


# ------------------------------------------------------------------- AC-6 kwargs


def test_no_binary_source_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc:
        TalosAgent(logs_dir=tmp_path)
    assert "talos_path" in str(exc.value)
    assert "TALOS_ARTIFACT_BASE" in str(exc.value)


def test_empty_artifact_base_env_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALOS_ARTIFACT_BASE", "")
    with pytest.raises(ValueError):
        TalosAgent(logs_dir=tmp_path)


def test_artifact_base_trailing_slash_stripped(tmp_path: Path) -> None:
    agent = TalosAgent(logs_dir=tmp_path, artifact_base="https://artifacts.example.invalid/")
    assert agent.artifact_base == "https://artifacts.example.invalid"


def test_artifact_base_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALOS_ARTIFACT_BASE", BASE + "/")
    assert TalosAgent(logs_dir=tmp_path).artifact_base == BASE


def test_talos_path_alone_is_enough(tmp_path: Path) -> None:
    agent = TalosAgent(logs_dir=tmp_path, talos_path="/tmp/talos")
    assert agent.talos_path == "/tmp/talos"
    assert agent.artifact_base is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"max_iterations": "7"}, "--max-iterations 7"),
        (
            {"max_iterations": 7, "wall_clock_secs": 60, "gate_timeout_secs": 30},
            "--max-iterations 7 --wall-clock-secs 60 --gate-timeout-secs 30",
        ),
        ({}, ""),
    ],
)
def test_cli_flags(tmp_path: Path, kwargs: dict[str, Any], expected: str) -> None:
    assert make(tmp_path, **kwargs).build_cli_flags() == expected


# ---------------------------------------------------------- AC-7 install (path)


def test_install_from_talos_path(tmp_path: Path, no_network: list[str]) -> None:
    binary = tmp_path / "talos-bin"
    binary.write_bytes(b"\x7fELF fake talos")
    agent = make(tmp_path, talos_path=str(binary))  # wins over artifact_base
    env = FakeEnvironment()
    asyncio.run(agent.install(as_any(env)))

    assert no_network == []
    assert [(src, dst) for src, dst, _ in env.uploads] == [(str(binary), "/installed-agent/talos")]
    assert env.find("chmod 0755 /installed-agent/talos")[0].user == "root"
    record = json.loads((agent.logs_dir / "setup" / "install.json").read_text())
    assert record == {
        "source": "talos_path",
        "talos_path": str(binary),
        "requested_version": None,
        "resolved_token": None,
        "arch": None,
        "url": None,
        "sha256": hashlib.sha256(b"\x7fELF fake talos").hexdigest(),
    }


# ------------------------------------------------------ AC-8 install (artifact)


def fake_http(monkeypatch: pytest.MonkeyPatch, responses: dict[str, bytes]) -> list[str]:
    calls: list[str] = []

    def get(url: str) -> bytes:
        calls.append(url)
        return responses[url]

    monkeypatch.setattr(agent_mod, "_http_get", get)
    return calls


def no_talos_under(root: Path) -> bool:
    return not any(p.name == "talos" for p in root.rglob("*"))


def test_install_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"binary-bytes"
    calls = fake_http(
        monkeypatch,
        {
            f"{BASE}/latest": b"0.14.0-gabc1234\n",
            f"{BASE}/0.14.0-gabc1234/x86_64/talos": body,
        },
    )
    agent = make(tmp_path)
    env = FakeEnvironment(scripts={"uname -m": result(stdout="x86_64\n")})
    asyncio.run(agent.install(as_any(env)))

    assert calls == [f"{BASE}/latest", f"{BASE}/0.14.0-gabc1234/x86_64/talos"]
    assert len(env.uploads) == 1
    src, dst, data = env.uploads[0]
    assert dst == "/installed-agent/talos"
    assert data == body
    assert not Path(src).exists()  # the host temp dir is gone
    assert no_talos_under(agent.logs_dir)
    assert env.find("chmod 0755 /installed-agent/talos")
    record = json.loads((agent.logs_dir / "setup" / "install.json").read_text())
    assert record == {
        "source": "artifact_base",
        "talos_path": None,
        "requested_version": "latest",
        "resolved_token": "0.14.0-gabc1234",
        "arch": "x86_64",
        "url": f"{BASE}/0.14.0-gabc1234/x86_64/talos",
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def test_install_pinned_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"arm-binary"
    calls = fake_http(monkeypatch, {f"{BASE}/0.14.0-gabc1234/aarch64/talos": body})
    agent = make(tmp_path, talos_version="0.14.0-gabc1234")
    env = FakeEnvironment(scripts={"uname -m": result(stdout="aarch64")})
    asyncio.run(agent.install(as_any(env)))

    assert calls == [f"{BASE}/0.14.0-gabc1234/aarch64/talos"]
    assert env.uploads[0][2] == body
    assert no_talos_under(agent.logs_dir)
    record = json.loads((agent.logs_dir / "setup" / "install.json").read_text())
    assert record == {
        "source": "artifact_base",
        "talos_path": None,
        "requested_version": "0.14.0-gabc1234",
        "resolved_token": "0.14.0-gabc1234",
        "arch": "aarch64",
        "url": f"{BASE}/0.14.0-gabc1234/aarch64/talos",
        "sha256": hashlib.sha256(body).hexdigest(),
    }


@pytest.mark.parametrize("arch", ["armv7l", ""])
def test_install_unsupported_arch(tmp_path: Path, no_network: list[str], arch: str) -> None:
    env = FakeEnvironment(scripts={"uname -m": result(stdout=arch + "\n")})
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(make(tmp_path).install(as_any(env)))
    assert repr(arch) in str(exc.value)
    assert no_network == []


@pytest.mark.parametrize("latest", [b"", b"0.14.0/x", b"0.14 0"])
def test_install_bad_latest_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, latest: bytes
) -> None:
    calls = fake_http(monkeypatch, {f"{BASE}/latest": latest})
    env = FakeEnvironment(scripts={"uname -m": result(stdout="x86_64")})
    with pytest.raises(RuntimeError):
        asyncio.run(make(tmp_path).install(as_any(env)))
    assert calls == [f"{BASE}/latest"]
    assert env.uploads == []


def test_install_http_error_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def get(url: str) -> bytes:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(agent_mod, "_http_get", get)
    env = FakeEnvironment(scripts={"uname -m": result(stdout="x86_64")})
    with pytest.raises(urllib.error.URLError):
        asyncio.run(make(tmp_path).install(as_any(env)))


def test_http_get_reads_body(tmp_path: Path) -> None:
    page = tmp_path / "latest"
    page.write_bytes(b"0.14.0-gabc1234\n")
    assert agent_mod._http_get(page.as_uri()) == b"0.14.0-gabc1234\n"


# ------------------------------------------------------------------ AC-9 version


def test_version_command_and_parse(tmp_path: Path) -> None:
    agent = make(tmp_path)
    assert agent.get_version_command() == "/installed-agent/talos --version"
    assert agent.parse_version("talos 0.14.0-gabc1234\n") == "0.14.0-gabc1234"
    assert agent.parse_version("  0.14.0\n") == "0.14.0"


# ----------------------------------------------------------- AC-10 run command


def test_build_run_command_production_text() -> None:
    cmd = build_run_command("/installed-agent", "/logs/agent", "")
    assert cmd == (
        "printf '%s' \"${AGENT_GATE_COMMAND:-}\" > /logs/agent/gate-command.txt; set -- ; "
        'if [ -n "${AGENT_GATE_COMMAND:-}" ]; then '
        'set -- --gate-command "$AGENT_GATE_COMMAND"; fi; '
        '/installed-agent/talos run --workspace "$(pwd)" '
        "--instruction-file /installed-agent/instruction.md --task-id harbor "
        "--transcript /logs/agent/transcript.jsonl --run-store /logs/agent/run.sqlite "
        '--offload-dir /logs/agent/offload "$@" > /logs/agent/run-summary.json '
        "2> /logs/agent/talos-stderr.log"
    )
    assert "  " not in cmd
    with_flags = build_run_command("/installed-agent", "/logs/agent", "--max-iterations 7")
    assert '--offload-dir /logs/agent/offload --max-iterations 7 "$@" >' in with_flags


def run_built(tmp_path: Path, case_env: dict[str, str], cli_flags: str = "") -> tuple[Any, ...]:
    installed = tmp_path / "installed"
    logs = tmp_path / "logs"
    workspace = tmp_path / "workspace"
    for d in (installed, logs, workspace):
        d.mkdir()
    argv_file = tmp_path / "argv.txt"
    fake = installed / "talos"
    fake.write_text(
        "#!/bin/sh\n"
        f'for a in "$@"; do printf \'%s\\n\' "$a"; done > {argv_file}\n'
        "printf '{}'\n"
        "printf 'fake stderr\\n' >&2\n"
    )
    fake.chmod(0o755)
    cmd = build_run_command(str(installed), str(logs), cli_flags)
    env = {"PATH": os.environ["PATH"], **case_env}
    proc = subprocess.run(["sh", "-c", cmd], cwd=workspace, env=env, check=False)
    argv = argv_file.read_text().splitlines()
    return proc, argv, logs, workspace


@pytest.mark.parametrize("case_env", [{}, {"AGENT_GATE_COMMAND": ""}])
def test_run_command_without_gate(tmp_path: Path, case_env: dict[str, str]) -> None:
    proc, argv, logs, workspace = run_built(tmp_path, case_env)
    assert proc.returncode == 0
    assert "--gate-command" not in argv
    assert (logs / "gate-command.txt").read_bytes() == b""
    assert os.path.realpath(argv[argv.index("--workspace") + 1]) == os.path.realpath(workspace)
    assert argv[argv.index("--run-store") + 1] == f"{logs}/run.sqlite"
    assert argv[argv.index("--offload-dir") + 1] == f"{logs}/offload"
    assert argv[argv.index("--instruction-file") + 1].endswith("/installed/instruction.md")
    assert argv[argv.index("--task-id") + 1] == "harbor"
    assert argv[argv.index("--transcript") + 1] == f"{logs}/transcript.jsonl"
    assert (logs / "run-summary.json").read_text() == "{}"
    assert (logs / "talos-stderr.log").read_text() == "fake stderr\n"


def test_run_command_with_gate(tmp_path: Path) -> None:
    gate = 'cd sub && pytest -q "a b"'
    proc, argv, logs, workspace = run_built(
        tmp_path, {"AGENT_GATE_COMMAND": gate}, "--max-iterations 7"
    )
    assert proc.returncode == 0
    i = argv.index("--gate-command")
    assert argv[i + 1] == gate
    assert (logs / "gate-command.txt").read_bytes() == gate.encode()
    assert os.path.realpath(argv[argv.index("--workspace") + 1]) == os.path.realpath(workspace)
    assert argv[argv.index("--max-iterations") + 1] == "7"
    assert (logs / "run-summary.json").read_text() == "{}"


# ----------------------------------------------------------------- AC-11 run


def run_agent(agent: TalosAgent, env: FakeEnvironment, instruction: str = "do it") -> None:
    asyncio.run(agent.run(instruction, as_any(env), AgentContext()))


def test_run_uploads_instruction_verbatim(tmp_path: Path) -> None:
    instruction = "Café ✓ naïve\r\nsecond line\r\n\n"
    env = FakeEnvironment(scripts={"talos run": result(0)})
    run_agent(make(tmp_path), env, instruction)
    src, dst, data = env.uploads[0]
    assert dst == "/installed-agent/instruction.md"
    assert data == instruction.encode("utf-8")
    assert not Path(src).exists()
    chmod = env.find("chmod 0644 /installed-agent/instruction.md")
    assert chmod and chmod[0].user == "root"


def test_run_exec_call_shape(tmp_path: Path) -> None:
    env = FakeEnvironment(scripts={"talos run": result(0)})
    agent = make(tmp_path, max_iterations=7)
    run_agent(agent, env)
    (call,) = env.find("talos run")
    assert call.cwd is None
    assert call.user is None
    assert call.timeout_sec is None
    assert call.command == build_run_command(
        "/installed-agent", "/logs/agent", "--max-iterations 7"
    )


@pytest.mark.parametrize("rc", [0, 10, 20, 30, 40])
def test_run_outcome_codes_return(
    tmp_path: Path, rc: int, caplog: pytest.LogCaptureFixture
) -> None:
    env = FakeEnvironment(scripts={"talos run": result(rc)})
    agent = make(tmp_path)
    with caplog.at_level(logging.INFO):
        run_agent(agent, env)
    assert agent._talos_exit_code == rc
    assert env.find("tail -c") == []
    assert f"talos exit rc={rc} classified=outcome" in caplog.text


@pytest.mark.parametrize("rc", [1, 2, 137])
def test_run_non_outcome_codes_raise(
    tmp_path: Path, rc: int, caplog: pytest.LogCaptureFixture
) -> None:
    env = FakeEnvironment(
        scripts={
            "talos run": result(rc),
            "tail -c 2000": result(stdout="error: backend unreachable\n"),
        }
    )
    agent = make(tmp_path)
    with caplog.at_level(logging.INFO), pytest.raises(NonZeroAgentExitCodeError) as exc:
        run_agent(agent, env)
    message = str(exc.value)
    assert str(rc) in message
    assert "/logs/agent/talos-stderr.log" in message
    assert "error: backend unreachable" in message
    assert agent._talos_exit_code == rc
    assert f"talos exit rc={rc} classified=raise" in caplog.text


# --------------------------------------------------------- AC-12 env forwarding


def forwarded(tmp_path: Path) -> dict[str, str]:
    agent = make(tmp_path, extra_env={"TALOS_TOKEN_BUDGET": "5"})
    env = FakeEnvironment(scripts={"talos run": result(0)})
    run_agent(agent, env)
    (call,) = env.find("talos run")
    assert call.env is not None
    assert agent._forwarded_env_keys == sorted(call.env)
    return dict(call.env)


def test_env_forwarding_anthropic_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://x")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("HOME_SECRET", "x")
    assert forwarded(tmp_path) == {"ANTHROPIC_API_KEY": "k", "TALOS_TOKEN_BUDGET": "5"}


def test_env_forwarding_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALOS_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_THINK", "high")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "m")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    assert forwarded(tmp_path) == {
        "TALOS_BACKEND": "ollama",
        "OLLAMA_THINK": "high",
        "TALOS_TOKEN_BUDGET": "5",
    }


def test_env_forwarding_bedrock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALOS_BEDROCK", "1")
    monkeypatch.setenv("TALOS_BACKEND", "ollama")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "m")
    monkeypatch.setenv("OLLAMA_MODEL", "o")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    assert forwarded(tmp_path) == {
        "TALOS_BEDROCK": "1",
        "ANTHROPIC_MODEL": "m",
        "AWS_REGION": "us-west-2",
        "TALOS_TOKEN_BUDGET": "5",
    }


def test_env_forwarding_bogus_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALOS_BACKEND", "bogus")
    env = FakeEnvironment(scripts={"talos run": result(0)})
    with pytest.raises(ValueError, match="bogus"):
        run_agent(make(tmp_path), env)
    assert not env.find("talos run")
    with pytest.raises(ValueError, match="bogus"):
        asyncio.run(make(tmp_path).install(env))
    assert not env.execs and not env.uploads


def test_env_forwarding_extra_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALOS_BEDROCK", "1")
    monkeypatch.setenv("AWS_PROFILE", "host")
    agent = make(tmp_path, extra_env={"AWS_PROFILE": "extra", "ANTHROPIC_API_KEY": "e"})
    env = FakeEnvironment(scripts={"talos run": result(0)})
    run_agent(agent, env)
    assert env.find("talos run")[0].env == {
        "TALOS_BEDROCK": "1",
        "AWS_PROFILE": "extra",
        "ANTHROPIC_API_KEY": "e",
    }


def test_env_forwarding_backend_selected_from_extra_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TALOS_BACKEND", raising=False)
    monkeypatch.delenv("TALOS_BEDROCK", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-key")
    monkeypatch.setenv("OLLAMA_MODEL", "q")
    agent = make(tmp_path, extra_env={"TALOS_BACKEND": "ollama"})
    env = FakeEnvironment(scripts={"talos run": result(0)})
    run_agent(agent, env)
    assert env.find("talos run")[0].env == {"TALOS_BACKEND": "ollama", "OLLAMA_MODEL": "q"}
    assert agent._backend_selected == "ollama"


def test_env_forwarding_bogus_backend_in_extra_env(tmp_path: Path) -> None:
    env = FakeEnvironment(scripts={"talos run": result(0)})
    with pytest.raises(ValueError, match="nope"):
        run_agent(make(tmp_path, extra_env={"TALOS_BACKEND": "nope"}), env)
    assert not env.find("talos run")


# ---------------------------------------------- AC-13 populate_context_post_run


def stage(
    agent: TalosAgent,
    summary: str | None = "run-summary-done.json",
    transcript: str | None = "transcript-done.jsonl",
    gate: str | None = None,
    stderr: bool = False,
    install: dict[str, Any] | None = None,
) -> None:
    logs = agent.logs_dir
    if summary is not None:
        shutil.copy(FIXTURES / summary, logs / "run-summary.json")
    if transcript is not None:
        shutil.copy(FIXTURES / transcript, logs / "transcript.jsonl")
    if gate is not None:
        (logs / "gate-command.txt").write_text(gate)
    if stderr:
        shutil.copy(FIXTURES / "talos-stderr.log", logs / "talos-stderr.log")
    if install is not None:
        (logs / "setup").mkdir(exist_ok=True)
        (logs / "setup" / "install.json").write_text(json.dumps(install))


def populate(agent: TalosAgent, context: AgentContext | None = None) -> AgentContext:
    context = context or AgentContext()
    agent.populate_context_post_run(context)
    return context


TALOS_KEYS = {
    "exit_code",
    "outcome",
    "disposition",
    "claimed_done",
    "iterations",
    "nudges_fired",
    "gates_green_at_exit",
    "tree_dirty",
    "mutating_iters",
    "edit_file_calls_ok",
    "bash_calls_ok",
    "version",
    "gate_command",
    "gate_armed",
    "backend",
    "backend_selected",
    "forwarded_env_keys",
    "install",
    "tokens",
    "compactions",
    "highest_compaction_tier",
    "memory",
    "stderr_tail",
}


def test_populate_happy_path(tmp_path: Path) -> None:
    agent = make(tmp_path, version="0.14.0-gabc1234")
    agent._talos_exit_code = 0
    agent._forwarded_env_keys = ["ANTHROPIC_API_KEY"]
    agent._backend_selected = "anthropic"
    stage(agent, gate="test -f hello.txt", stderr=True)
    context = populate(agent, AgentContext(metadata={"keep": 1}))

    assert context.n_input_tokens == 1200 + 5000 + 800
    assert context.n_cache_tokens == 5000
    assert context.n_output_tokens == 340
    assert context.cost_usd is None
    assert context.metadata is not None
    assert context.metadata["keep"] == 1
    talos = context.metadata["talos"]
    assert set(talos) == TALOS_KEYS  # no "error"
    assert talos["exit_code"] == 0
    assert talos["outcome"] == "Finished"
    assert talos["disposition"] == "Done"
    assert talos["claimed_done"] is True
    assert talos["iterations"] == 2
    assert talos["nudges_fired"] == 1
    assert talos["gates_green_at_exit"] is True
    assert talos["tree_dirty"] is True
    assert talos["mutating_iters"] == 1
    assert talos["edit_file_calls_ok"] == 1
    assert talos["bash_calls_ok"] == 2
    assert talos["version"] == "0.14.0-gabc1234"
    assert talos["gate_command"] == "test -f hello.txt"
    assert talos["gate_armed"] is True
    assert (
        talos["backend"]
        == json.loads((FIXTURES / "run-summary-done.json").read_text())["backend_settings"]
    )
    assert talos["backend_selected"] == "anthropic"
    assert talos["forwarded_env_keys"] == ["ANTHROPIC_API_KEY"]
    assert talos["install"] is None
    assert talos["tokens"] == {"input": 1200, "cache_read": 5000, "cache_write": 800, "output": 340}
    assert talos["compactions"] == 0
    assert talos["highest_compaction_tier"] == 0
    assert talos["stderr_tail"] is None


def test_populate_blocked(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 10
    stage(agent, summary="run-summary-blocked.json")
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert talos["disposition"] == "Blocked"
    assert talos["claimed_done"] is False
    assert talos["gate_command"] is None
    assert talos["gate_armed"] is False
    assert "error" not in talos


def test_populate_mismatch(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 0
    stage(agent, summary="run-summary-blocked.json")
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert "exit_code/disposition mismatch: rc=0 disposition=Blocked" in talos["error"]


@pytest.mark.parametrize(
    ("rc", "summary", "mismatch"),
    [
        (20, "run-summary-done.json", True),
        (20, "run-summary-blocked.json", True),
        (1, "run-summary-done.json", False),
    ],
)
def test_populate_mismatch_matrix(tmp_path: Path, rc: int, summary: str, mismatch: bool) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = rc
    stage(agent, summary=summary)
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert ("mismatch" in talos.get("error", "")) is mismatch


def test_populate_rc20_with_failed_disposition(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 20
    stage(agent)
    summary = json.loads((agent.logs_dir / "run-summary.json").read_text())
    summary["disposition"] = {"Failed": {"mode": "Loop", "summary": "looped"}}
    (agent.logs_dir / "run-summary.json").write_text(json.dumps(summary))
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert talos["disposition"] == "Failed"
    assert talos["claimed_done"] is False
    assert "error" not in talos


def test_populate_never_ran(tmp_path: Path) -> None:
    agent = make(tmp_path)
    context = populate(agent)
    assert context.n_input_tokens is None
    talos = context.metadata["talos"]  # type: ignore[index]
    assert talos["claimed_done"] is None
    assert talos["exit_code"] is None
    assert talos["tokens"] is None
    assert talos["forwarded_env_keys"] is None
    assert talos["backend_selected"] is None
    assert talos["stderr_tail"] is None
    assert "run-summary.json" in talos["error"]
    assert "transcript" in talos["error"]
    assert "no talos exit code recorded (timeout or exec not reached)" in talos["error"]


def test_populate_truncated_transcript(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 0
    stage(agent, transcript="transcript-truncated.jsonl", stderr=True)
    context = populate(agent)
    talos = context.metadata["talos"]  # type: ignore[index]
    assert talos["tokens"] is None
    assert context.n_input_tokens is None
    assert talos["nudges_fired"] is None
    assert "run_end" in talos["error"]
    assert talos["stderr_tail"]  # run_end missing -> tail populated


def test_populate_backend_error_stderr_tail(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 1
    stage(agent, stderr=True)
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert talos["exit_code"] == 1
    assert talos["stderr_tail"] == (FIXTURES / "talos-stderr.log").read_text()


def test_populate_stderr_tail_is_bounded(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 137
    (agent.logs_dir / "talos-stderr.log").write_bytes(b"x" * 5000 + b"\xff" + b"y" * 10)
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert len(talos["stderr_tail"]) == 2048
    assert talos["stderr_tail"].endswith("�" + "y" * 10)


def test_populate_install_mismatch(tmp_path: Path) -> None:
    agent = make(tmp_path, version="0.13.0-gdead000")
    agent._talos_exit_code = 0
    install = {
        "source": "artifact_base",
        "talos_path": None,
        "requested_version": "latest",
        "resolved_token": "0.14.0-gabc1234",
        "arch": "x86_64",
        "url": f"{BASE}/0.14.0-gabc1234/x86_64/talos",
        "sha256": "0" * 64,
    }
    stage(agent, install=install)
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert talos["install"] == install
    assert (
        "install token/version mismatch: token=0.14.0-gabc1234 binary=0.13.0-gdead000"
        in talos["error"]
    )


def test_populate_install_match_has_no_error(tmp_path: Path) -> None:
    agent = make(tmp_path, version="0.14.0-gabc1234")
    agent._talos_exit_code = 0
    stage(agent, install={"source": "artifact_base", "resolved_token": "0.14.0-gabc1234"})
    assert "error" not in populate(agent).metadata["talos"]  # type: ignore[index]


def test_populate_unparsable_summary_and_bad_disposition(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 0
    stage(agent, summary=None)
    (agent.logs_dir / "run-summary.json").write_text("{not json")
    (agent.logs_dir / "setup").mkdir()
    (agent.logs_dir / "setup" / "install.json").write_text("garbage")
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert "run-summary.json unparsable" in talos["error"]
    assert talos["install"] is None

    (agent.logs_dir / "run-summary.json").write_text(json.dumps({"disposition": "Done"}))
    talos = populate(agent).metadata["talos"]  # type: ignore[index]
    assert talos["disposition"] is None
    assert talos["claimed_done"] is None
    assert "disposition is not a single-key dict" in talos["error"]


def test_populate_tolerates_odd_stats(tmp_path: Path) -> None:
    agent = make(tmp_path)
    agent._talos_exit_code = 0
    stage(agent, transcript=None)
    (agent.logs_dir / "transcript.jsonl").write_text(
        json.dumps({"event": "run_end", "stats": {"input_tokens": 3.0, "output_tokens": True}})
        + "\n[1]\n"
    )
    context = populate(agent)
    assert context.metadata["talos"]["tokens"] == {  # type: ignore[index]
        "input": 3,
        "cache_read": 0,
        "cache_write": 0,
        "output": 0,
    }


# ------------------------------------------------------------------- memory kwarg

SLICE_EVENT = {
    "items": 2,
    "chars": 137,
    "degraded": None,
    "source": "fake-kb p1",
    "memory_tools": ["kb_search"],
    "memory_tools_dropped": [],
}
ZERO_STATS = {
    "memory_annotations": 0,
    "memory_denies": 0,
    "memory_kb_tool_calls": 0,
    "memory_deny_overrides": 0,
    "memory_degraded": None,
}


def edit_transcript(
    agent: TalosAgent,
    *,
    config: dict[str, Any] | None = None,
    memory_slice: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> None:
    path = agent.logs_dir / "transcript.jsonl"
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if isinstance(event, dict) and event.get("event") == "run_start":
            if config is not None:
                event["config"] = {**event.get("config", {}), **config}
            out.append(json.dumps(event))
            if memory_slice is not None:
                out.append(json.dumps({"event": "memory_slice", **memory_slice}))
            continue
        if isinstance(event, dict) and event.get("event") == "run_end" and stats is not None:
            event["stats"].update(stats)
        out.append(json.dumps(event))
    path.write_text("".join(f"{x}\n" for x in out), encoding="utf-8")


def test_memory_defaults_off(tmp_path: Path) -> None:
    a = make(tmp_path)
    assert a.memory == "off"
    assert a.kb_project is None
    assert a.build_cli_flags() == ""


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ""),
        ({"memory": "off"}, ""),
        ({"memory": "annotate"}, "--memory annotate"),
        ({"memory": "deny-once"}, "--memory deny-once"),
        ({"memory": "belief-check"}, "--memory belief-check"),
        (
            {"memory": "annotate", "kb_project": "harness-design"},
            "--memory annotate --kb-project harness-design",
        ),
        (
            {"max_iterations": 7, "memory": "deny-once", "kb_project": "kb.bench_1"},
            "--max-iterations 7 --memory deny-once --kb-project kb.bench_1",
        ),
    ],
)
def test_memory_cli_flags(tmp_path: Path, kwargs: dict[str, Any], expected: str) -> None:
    assert make(tmp_path, **kwargs).build_cli_flags() == expected


@pytest.mark.parametrize(
    "value",
    [
        "ON",
        "Annotate",
        "deny_once",
        "belief_check",
        "",
        " annotate",
        "annotate\n",
        "none",
        None,
        True,
        False,
        1,
    ],
)
def test_memory_invalid_raises_at_construction(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError) as exc:
        make(tmp_path, memory=value)
    assert "invalid memory" in str(exc.value)
    assert "'deny-once'" in str(exc.value)
    assert "'belief-check'" in str(exc.value)


@pytest.mark.parametrize(
    "value", ["", "has space", "p1;id", "$(id)", "-p", ".hidden", "a/b", "p1\n", 7, True]
)
def test_kb_project_invalid_raises(tmp_path: Path, value: Any) -> None:
    with pytest.raises(ValueError) as exc:
        make(tmp_path, memory="annotate", kb_project=value)
    assert "invalid kb_project" in str(exc.value)


def test_kb_project_without_memory_raises(tmp_path: Path) -> None:
    for kwargs in ({}, {"memory": "off"}):
        with pytest.raises(ValueError) as exc:
            make(tmp_path, kb_project="harness-design", **kwargs)
        assert "needs memory" in str(exc.value)
        assert "'belief-check'" in str(exc.value)


def test_memory_validation_precedes_binary_source_check(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc:
        TalosAgent(logs_dir=tmp_path, memory="bogus")
    assert "invalid memory" in str(exc.value)
    assert "talos_path" not in str(exc.value)


def test_memory_validation_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as exc:
        make(tmp_path, memory="bogus", kb_project="has space")
    assert "invalid memory" in str(exc.value)
    assert "invalid kb_project" not in str(exc.value)
    with pytest.raises(ValueError) as exc:
        make(tmp_path, kb_project="has space")
    assert "invalid kb_project" in str(exc.value)
    assert "needs memory" not in str(exc.value)


def test_run_passes_memory_flags(tmp_path: Path) -> None:
    env = FakeEnvironment(scripts={"talos run": result(0)})
    agent = make(tmp_path, memory="deny-once", kb_project="p1")
    run_agent(agent, env)
    (call,) = env.find("talos run")
    assert call.command == build_run_command(
        "/installed-agent", "/logs/agent", "--memory deny-once --kb-project p1"
    )


def test_run_command_memory_argv(tmp_path: Path) -> None:
    proc, argv, _logs, _ws = run_built(
        tmp_path,
        {"AGENT_GATE_COMMAND": "true"},
        "--memory annotate --kb-project harness-design",
    )
    assert proc.returncode == 0
    assert argv[argv.index("--memory") + 1] == "annotate"
    assert argv[argv.index("--kb-project") + 1] == "harness-design"
    assert argv[argv.index("--gate-command") + 1] == "true"
    assert argv.count("--memory") == 1


def _kb_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("PERSONAL_KB_URL", "http://kb.example.invalid:8765")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "host-key")


def test_memory_env_never_forwarded_from_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kb_host_env(monkeypatch)
    env = FakeEnvironment(scripts={"talos run": result(0)})
    run_agent(make(tmp_path, memory="annotate"), env)
    assert env.find("talos run")[0].env == {"ANTHROPIC_API_KEY": "k"}


def test_memory_env_forwarded_from_extra_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _kb_host_env(monkeypatch)
    env = FakeEnvironment(scripts={"talos run": result(0)})
    agent = make(
        tmp_path,
        memory="deny-once",
        extra_env={"PERSONAL_KB_URL": "http://127.0.0.1:9000", "PERSONAL_KB_API_KEY": "extra-key"},
    )
    run_agent(agent, env)
    assert env.find("talos run")[0].env == {
        "ANTHROPIC_API_KEY": "k",
        "PERSONAL_KB_URL": "http://127.0.0.1:9000",
        "PERSONAL_KB_API_KEY": "extra-key",
    }
    assert agent._forwarded_env_keys == [
        "ANTHROPIC_API_KEY",
        "PERSONAL_KB_API_KEY",
        "PERSONAL_KB_URL",
    ]


def staged(tmp_path: Path, transcript: str = "transcript-done.jsonl", **kwargs: Any) -> TalosAgent:
    agent = make(tmp_path, **kwargs)
    agent._talos_exit_code = 0
    stage(agent, transcript=transcript)
    return agent


def memory_of(agent: TalosAgent) -> tuple[dict[str, Any], dict[str, Any]]:
    context = populate(agent)
    assert context.metadata is not None
    talos = context.metadata["talos"]
    return talos["memory"], talos


def cfg(mode: str) -> dict[str, Any]:
    return {"memory_mode": mode}


def test_populate_memory_off_defaults(tmp_path: Path) -> None:
    agent = staged(tmp_path)
    memory, talos = memory_of(agent)
    assert memory == {
        "mode": "off",
        "kb_project": None,
        "effective_mode": None,
        "memory_annotations": None,
        "memory_denies": None,
        "memory_kb_tool_calls": None,
        "memory_deny_overrides": None,
        "memory_degraded": None,
        "source": None,
        "slice_items": None,
        "slice_chars": None,
        "memory_tools": None,
        "memory_tools_dropped": None,
    }
    assert "error" not in talos
    edit_transcript(agent, config=cfg("off"), stats=ZERO_STATS)
    memory, talos = memory_of(agent)
    assert memory["effective_mode"] == "off"
    assert [memory[k] for k in agent_mod.MEMORY_STAT_KEYS] == [0, 0, 0, 0]
    assert "error" not in talos


def test_populate_memory_on_with_stats(tmp_path: Path) -> None:
    agent = staged(tmp_path, memory="deny-once", kb_project="p1")
    edit_transcript(
        agent,
        config=cfg("deny-once"),
        memory_slice=SLICE_EVENT,
        stats={
            "memory_annotations": 2,
            "memory_denies": 1,
            "memory_kb_tool_calls": 3,
            "memory_deny_overrides": 1,
            "memory_degraded": None,
        },
    )
    memory, talos = memory_of(agent)
    assert memory == {
        "mode": "deny-once",
        "kb_project": "p1",
        "effective_mode": "deny-once",
        "memory_annotations": 2,
        "memory_denies": 1,
        "memory_kb_tool_calls": 3,
        "memory_deny_overrides": 1,
        "memory_degraded": None,
        "source": "fake-kb p1",
        "slice_items": 2,
        "slice_chars": 137,
        "memory_tools": ["kb_search"],
        "memory_tools_dropped": [],
    }
    assert "error" not in talos


def test_populate_memory_degraded(tmp_path: Path) -> None:
    agent = staged(tmp_path, memory="annotate")
    edit_transcript(
        agent,
        config=cfg("annotate"),
        memory_slice={
            **SLICE_EVENT,
            "items": 0,
            "chars": 0,
            "degraded": "kb unreachable",
            "memory_tools": [],
        },
        stats={**ZERO_STATS, "memory_degraded": "kb unreachable"},
    )
    memory, talos = memory_of(agent)
    assert memory["memory_degraded"] == "kb unreachable"
    assert memory["memory_tools"] == []
    assert talos["error"] == "memory degraded to off (kb unreachable)"


def test_populate_memory_on_missing_telemetry(tmp_path: Path) -> None:
    agent = staged(tmp_path, memory="annotate")
    edit_transcript(agent, config=cfg("annotate"), memory_slice=SLICE_EVENT)
    memory, talos = memory_of(agent)
    assert [memory[k] for k in agent_mod.MEMORY_STAT_KEYS] == [None] * 4
    assert talos["error"] == (
        "run_end stats lack memory telemetry with memory=annotate: missing "
        "memory_annotations, memory_denies, memory_kb_tool_calls, memory_deny_overrides"
    )
    edit_transcript(agent, stats={"memory_annotations": 1})
    memory, talos = memory_of(agent)
    assert memory["memory_annotations"] == 1
    assert talos["error"] == (
        "run_end stats lack memory telemetry with memory=annotate: missing "
        "memory_denies, memory_kb_tool_calls, memory_deny_overrides"
    )


def test_populate_memory_on_no_run_end(tmp_path: Path) -> None:
    agent = staged(tmp_path, transcript="transcript-truncated.jsonl", memory="annotate")
    memory, talos = memory_of(agent)
    assert [memory[k] for k in agent_mod.MEMORY_STAT_KEYS] == [None] * 4
    assert memory["memory_degraded"] is None
    assert memory["effective_mode"] is None
    assert talos["error"] == (
        "transcript.jsonl has no run_end event (token counts unavailable); "
        "run_start config lacks memory_mode with memory=annotate"
    )


@pytest.mark.parametrize(
    ("kwarg", "effective", "slice_event"),
    [
        ("deny-once", "off", None),
        ("off", "annotate", None),
        ("annotate", "deny-once", SLICE_EVENT),
    ],
)
def test_populate_memory_mode_mismatch(
    tmp_path: Path, kwarg: str, effective: str, slice_event: dict[str, Any] | None
) -> None:
    agent = staged(tmp_path, memory=kwarg)
    edit_transcript(agent, config=cfg(effective), memory_slice=slice_event, stats=ZERO_STATS)
    _memory, talos = memory_of(agent)
    assert talos["error"] == (
        f"memory mode mismatch: requested {kwarg} but run_start.config.memory_mode={effective}"
    )


def test_populate_memory_run_start_lacks_mode(tmp_path: Path) -> None:
    agent = staged(tmp_path, memory="annotate")
    edit_transcript(agent, memory_slice=SLICE_EVENT, stats=ZERO_STATS)
    _memory, talos = memory_of(agent)
    assert talos["error"] == "run_start config lacks memory_mode with memory=annotate"
    edit_transcript(agent, config={"memory_mode": 5})
    memory, talos = memory_of(agent)
    assert memory["effective_mode"] is None
    assert talos["error"] == "run_start config lacks memory_mode with memory=annotate"


def test_populate_memory_on_no_slice(tmp_path: Path) -> None:
    agent = staged(tmp_path, memory="annotate")
    edit_transcript(agent, config=cfg("annotate"), stats=ZERO_STATS)
    _memory, talos = memory_of(agent)
    assert talos["error"] == "memory on but transcript has no memory_slice event"


@pytest.mark.parametrize(
    ("slice_event", "stats"),
    [(SLICE_EVENT, ZERO_STATS), (None, {**ZERO_STATS, "memory_denies": 1})],
)
def test_populate_memory_off_activity(
    tmp_path: Path, slice_event: dict[str, Any] | None, stats: dict[str, Any]
) -> None:
    agent = staged(tmp_path)
    edit_transcript(agent, config=cfg("off"), memory_slice=slice_event, stats=stats)
    _memory, talos = memory_of(agent)
    assert talos["error"] == "memory off requested but transcript shows memory activity"


@pytest.mark.parametrize(
    ("mode", "overrides", "expected"),
    [
        ("deny-once", {"memory_denies": 3}, "memory invariant violated: memory_denies=3 > 2"),
        (
            "annotate",
            {"memory_denies": 1},
            "memory invariant violated: memory_denies=1 under annotate",
        ),
        (
            "belief-check",
            {"memory_denies": 1},
            "memory invariant violated: memory_denies=1 under belief-check",
        ),
        (
            "annotate",
            {"memory_denies": 3},
            "memory invariant violated: memory_denies=3 > 2, memory_denies=3 under annotate",
        ),
        (
            "deny-once",
            {"memory_deny_overrides": 2, "memory_annotations": 1},
            "memory invariant violated: memory_deny_overrides=2 > memory_annotations=1",
        ),
        (
            "annotate",
            {"memory_annotations": 1, "memory_degraded": "kb unreachable"},
            "memory degraded to off (kb unreachable); "
            "memory invariant violated: counters non-zero while degraded",
        ),
        (
            "deny-once",
            {"memory_annotations": 2, "memory_denies": 2, "memory_deny_overrides": 1},
            None,
        ),
    ],
)
def test_populate_memory_invariants(
    tmp_path: Path, mode: str, overrides: dict[str, Any], expected: str | None
) -> None:
    agent = staged(tmp_path, memory=mode)
    edit_transcript(
        agent, config=cfg(mode), memory_slice=SLICE_EVENT, stats={**ZERO_STATS, **overrides}
    )
    _memory, talos = memory_of(agent)
    if expected is None:
        assert "error" not in talos
    else:
        assert talos["error"] == expected


def test_populate_memory_odd_values(tmp_path: Path) -> None:
    agent = staged(tmp_path, memory="deny-once")
    edit_transcript(
        agent,
        config=cfg("deny-once"),
        memory_slice={
            "items": True,
            "chars": "137",
            "degraded": None,
            "source": "",
            "memory_tools": ["kb_search", 3],
            "memory_tools_dropped": "echo",
        },
        stats={
            "memory_annotations": True,
            "memory_denies": 2.0,
            "memory_kb_tool_calls": "3",
            "memory_deny_overrides": None,
            "memory_degraded": "",
        },
    )
    memory, talos = memory_of(agent)
    assert [memory[k] for k in agent_mod.MEMORY_STAT_KEYS] == [0, 2, 0, 0]
    assert memory["memory_degraded"] is None
    assert memory["source"] is None
    assert memory["slice_items"] == 0
    assert memory["slice_chars"] == 0
    assert memory["memory_tools"] is None
    assert memory["memory_tools_dropped"] is None
    assert "error" not in talos
    edit_transcript(agent, stats={"memory_degraded": 5})
    memory, talos = memory_of(agent)
    assert memory["memory_degraded"] is None
    assert "error" not in talos
