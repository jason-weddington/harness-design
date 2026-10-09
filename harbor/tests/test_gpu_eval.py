"""Hermetic tests for scripts/gpu-eval.sh: the shared-GPU flock + /api/ps pre-check wrapper.

PATH is REPLACED (not prepended) by a tmp dir holding a stub ``curl`` that prints canned
/api/ps JSON, plus symlinks to the real bash/flock/python3/mkdir/cat, so no network or real
Ollama is touched.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "gpu-eval.sh"
REAL_TOOLS = ("bash", "flock", "python3", "mkdir", "cat")
MODEL = "glm-5.3:cloud"

pytestmark = pytest.mark.skipif(
    any(shutil.which(t) is None for t in REAL_TOOLS), reason="needs bash/flock/python3"
)


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in REAL_TOOLS:
        real = shutil.which(tool)
        assert real is not None
        (bindir / tool).symlink_to(real)
    curl = bindir / "curl"
    curl.write_text(
        f"#!{shutil.which('bash')}\n"
        'if [ -n "${FAKE_PS_FAIL:-}" ]; then exit 7; fi\n'
        'printf "%s" "$FAKE_PS_JSON"\n'
    )
    curl.chmod(0o755)
    return {
        "PATH": str(bindir),
        "HOME": str(tmp_path / "home"),
        "GPU_LOCK_FILE": str(tmp_path / "locks" / "gpu.lock"),
        "OLLAMA_BASE_URL": "http://ollama.invalid:11434",
        "OLLAMA_MODEL": MODEL,
        "FAKE_PS_JSON": ps(MODEL),
    }


def ps(*names: str) -> str:
    return json.dumps({"models": [{"name": n, "model": n} for n in names]})


def run(env: dict[str, str], *cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *cmd], env=env, capture_output=True, text=True, check=False, timeout=30
    )


def test_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)


def test_allowed_model_runs_child_and_propagates_exit_code(env: dict[str, str]) -> None:
    r = run(env, "bash", "-c", "echo child-ran; exit 7")
    assert r.returncode == 7
    assert r.stdout == "child-ran\n"
    r = run(env, "bash", "-c", "exit 0")
    assert r.returncode == 0


def test_nothing_loaded_is_allowed(env: dict[str, str]) -> None:
    env["FAKE_PS_JSON"] = ps()
    assert run(env, "bash", "-c", "exit 0").returncode == 0


def test_alias_variant_allowed(env: dict[str, str]) -> None:
    env["FAKE_PS_JSON"] = ps(f"{MODEL}-256k")
    assert run(env, "bash", "-c", "exit 0").returncode == 0


def test_embed_model_allowed(env: dict[str, str]) -> None:
    env["FAKE_PS_JSON"] = ps(MODEL, "nomic-embed-text:latest")
    assert run(env, "bash", "-c", "exit 0").returncode == 0


def test_foreign_model_refused(env: dict[str, str]) -> None:
    env["FAKE_PS_JSON"] = ps(MODEL, "qwen3:32b")
    r = run(env, "bash", "-c", "echo should-not-run")
    assert r.returncode == 75
    assert r.stdout == ""
    assert r.stderr == "gpu-eval: foreign model loaded: qwen3:32b\n"


def test_lock_held_elsewhere_is_busy(env: dict[str, str]) -> None:
    lock = Path(env["GPU_LOCK_FILE"])
    lock.parent.mkdir(parents=True)
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run(env, "bash", "-c", "echo should-not-run")
    assert r.returncode == 75
    assert r.stdout == ""
    assert r.stderr == f"gpu-eval: GPU busy (lock held: {lock})\n"


def test_lock_is_held_while_child_runs(env: dict[str, str]) -> None:
    lock = env["GPU_LOCK_FILE"]
    probe = f"flock -n {lock} -c true && echo free || echo held"
    r = run(env, "bash", "-c", probe)
    assert r.returncode == 0
    assert r.stdout == "held\n"


def test_missing_env_is_exit_2(env: dict[str, str]) -> None:
    del env["OLLAMA_MODEL"]
    r = run(env, "bash", "-c", "exit 0")
    assert r.returncode == 2
    assert r.stderr == "gpu-eval: OLLAMA_BASE_URL and OLLAMA_MODEL must be set\n"


def test_unreachable_ollama_is_exit_2(env: dict[str, str]) -> None:
    env["FAKE_PS_FAIL"] = "1"
    r = run(env, "bash", "-c", "exit 0")
    assert r.returncode == 2
    assert r.stderr == f"gpu-eval: cannot reach {env['OLLAMA_BASE_URL']}/api/ps\n"


def test_no_command_is_usage_error(env: dict[str, str]) -> None:
    r = run(env)
    assert r.returncode == 2
    assert r.stderr.startswith("usage: gpu-eval.sh")
