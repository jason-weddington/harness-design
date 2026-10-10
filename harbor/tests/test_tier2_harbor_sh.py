"""Hermetic tests for scripts/tier2-harbor.sh: the one tier-2-on-Harbor command.

PATH is REPLACED (not prepended) by a tmp dir holding stub ``uv``, ``docker`` and ``curl``
scripts that append to one log ($STUB_LOG), plus symlinks to the real tools the script,
scripts/harbor-build-base.sh and scripts/gpu-eval.sh need. No Docker, network or model is
touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "tier2-harbor.sh"
REAL_TOOLS = (
    "bash",
    "cat",
    "grep",
    "mkdir",
    "dirname",
    "basename",
    "ls",
    "flock",
    "python3",
    "git",
)
USAGE = (
    "usage: scripts/tier2-harbor.sh --tasks <emitted-harbor-tasks-dir> --jobs <out-dir> "
    "[-k N (default 3)] [-n N (default 1)] "
    "[--talos-version <token> | --talos-path <binary>] [--task <id>]..."
)

pytestmark = pytest.mark.skipif(
    any(shutil.which(t) is None for t in REAL_TOOLS), reason="needs the real tools"
)

UV_STUB = """\
printf 'uv\\t%s\\t%s\\n' "$PWD" "$*" >> "$STUB_LOG"
if [ "$1 $2 $3 $4" = "run --frozen harbor run" ]; then
  out=""
  prev=""
  for a in "$@"; do
    if [ "$prev" = -o ]; then out=$a; fi
    prev=$a
  done
  rc=${FAKE_HARBOR_RC:-0}
  if [ "$rc" = 0 ] || [ "${FAKE_HARBOR_WRITE:-}" = 1 ]; then mkdir -p "$out/job"; fi
  exit "$rc"
fi
if [ "$1 $2 $3" = "run --frozen talos-harbor-tier2-report" ]; then
  exit "${FAKE_REPORT_RC:-0}"
fi
exit 99
"""
DOCKER_STUB = 'printf \'docker\\t%s\\n\' "$*" >> "$STUB_LOG"\n'
CURL_STUB = 'printf \'curl\\t%s\\n\' "$*" >> "$STUB_LOG"\nprintf \'{"models":[]}\'\n'


class Env:
    def __init__(self, tmp_path: Path) -> None:
        tmp = tmp_path.resolve()
        self.bindir = tmp / "bin"
        self.bindir.mkdir()
        for tool in REAL_TOOLS:
            real = shutil.which(tool)
            assert real is not None
            (self.bindir / tool).symlink_to(real)
        bash = shutil.which("bash")
        for name, body in (("uv", UV_STUB), ("docker", DOCKER_STUB), ("curl", CURL_STUB)):
            stub = self.bindir / name
            stub.write_text(f"#!{bash}\n{body}")
            stub.chmod(0o755)
        self.log = tmp / "stub.log"
        self.tasks = tmp / "tasks"
        (self.tasks / "_base").mkdir(parents=True)
        (self.tasks / "_base" / "Dockerfile").write_text("FROM scratch\n")
        (self.tasks / "_base" / "IMAGE").write_text("talos-evals-base:1")
        (self.tasks / "a").mkdir()
        (self.tasks / "a" / "task.toml").write_text("[metadata]\n")
        self.jobs = tmp / "jobs"
        self.tmp = tmp
        self.vars: dict[str, str] = {
            "PATH": str(self.bindir),
            "HOME": str(tmp / "home"),
            "GPU_LOCK_FILE": str(tmp / "locks" / "gpu.lock"),
            "STUB_LOG": str(self.log),
            "TALOS_BACKEND": "anthropic",
            "ANTHROPIC_API_KEY": "sk-test-secret",
            "TALOS_ARTIFACT_BASE": "http://artifacts.invalid",
        }

    def ollama(self, url: str, model: str) -> None:
        del self.vars["ANTHROPIC_API_KEY"]
        self.vars.update(TALOS_BACKEND="ollama", OLLAMA_BASE_URL=url, OLLAMA_MODEL=model)

    def run(self, *args: object, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), *(str(a) for a in args)],
            env=self.vars,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )

    def std(self, *extra: object) -> subprocess.CompletedProcess[str]:
        return self.run("--tasks", self.tasks, "--jobs", self.jobs, *extra)

    def lines(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def uv_lines(self) -> list[str]:
        return [ln for ln in self.lines() if ln.startswith("uv\t")]

    def harbor_argv(self, k: int = 3, n: int = 1) -> str:
        return (
            f"run --frozen harbor run -p {self.tasks} "
            f"--agent-import-path talos_harbor.agent:TalosAgent -k {k} -n {n} --max-retries 0 "
            "--ak gate_timeout_secs=900 --ak wall_clock_secs=3300 --ak max_iterations=500 "
            f"-o {self.jobs}"
        )


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


def decision_line(stderr: str) -> str:
    lines = [ln for ln in stderr.splitlines() if ln.startswith("tier2-harbor: backend=")]
    assert len(lines) == 1, stderr
    return lines[0]


# ---- static checks ----------------------------------------------------------------------


def test_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK)


def test_bash_syntax() -> None:
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck() -> None:  # pragma: no cover - only where shellcheck is installed
    r = subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout


def test_help(env: Env) -> None:
    for flag in ("-h", "--help"):
        r = env.run(flag)
        assert r.returncode == 0
        assert r.stdout.splitlines()[0] == USAGE
        assert "-n" in r.stdout.splitlines()[1]
    assert env.lines() == []


# ---- usage, environment and path errors -------------------------------------------------


def _git_init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


USAGE_ERRORS: list[tuple[str, list[str]]] = [
    ("unknown-flag", ["--tasks", "{tasks}", "--jobs", "{jobs}", "--bogus"]),
    ("missing-value", ["--tasks", "{tasks}", "--jobs"]),
    ("no-tasks", ["--jobs", "{jobs}"]),
    ("no-jobs", ["--tasks", "{tasks}"]),
    (
        "both-talos",
        ["--tasks", "{tasks}", "--jobs", "{jobs}", "--talos-version", "v", "--talos-path", "x"],
    ),
    ("k-zero", ["--tasks", "{tasks}", "--jobs", "{jobs}", "-k", "0"]),
    ("k-word", ["--tasks", "{tasks}", "--jobs", "{jobs}", "-k", "three"]),
    ("k-leading-zero", ["--tasks", "{tasks}", "--jobs", "{jobs}", "-k", "03"]),
    ("n-zero", ["--tasks", "{tasks}", "--jobs", "{jobs}", "-n", "0"]),
]


@pytest.mark.parametrize(("name", "argv"), USAGE_ERRORS, ids=[n for n, _ in USAGE_ERRORS])
def test_usage_errors(env: Env, name: str, argv: list[str]) -> None:
    args = [a.format(tasks=env.tasks, jobs=env.jobs) for a in argv]
    r = env.run(*args)
    assert r.returncode == 2, (name, r.stderr)
    assert r.stderr.startswith("tier2-harbor: ")
    assert USAGE in r.stderr
    assert env.lines() == []


def _no_git(e: Env) -> list[object]:
    (e.bindir / "git").unlink()
    return []


Setup = Callable[["Env"], list[object]]


def _set(**kv: str) -> Setup:
    def apply(e: Env) -> list[object]:
        e.vars.update(kv)
        return []

    return apply


def _unset(*keys: str) -> Setup:
    def apply(e: Env) -> list[object]:
        for k in keys:
            e.vars.pop(k, None)
        return []

    return apply


def _ollama(url: str, model: str = "qwen-test", **kv: str) -> Setup:
    def apply(e: Env) -> list[object]:
        e.ollama(url, model)
        e.vars.update(kv)
        for k in [k for k, v in e.vars.items() if v == ""]:
            del e.vars[k]
        return []

    return apply


def _talos_path_missing(e: Env) -> list[object]:
    return ["--talos-path", e.tmp / "nope"]


def _tasks_missing(e: Env) -> list[object]:
    e.tasks = e.tmp / "no-tasks"
    return []


def _tasks_in_git(e: Env) -> list[object]:
    repo = _git_init(e.tmp / "repo")
    shutil.copytree(e.tasks, repo / "tasks")
    e.tasks = repo / "tasks"
    return []


def _jobs_parent_missing(e: Env) -> list[object]:
    e.jobs = e.tmp / "missing" / "jobs"
    return []


def _jobs_not_empty(e: Env) -> list[object]:
    (e.jobs / "old").mkdir(parents=True)
    return []


def _jobs_is_file(e: Env) -> list[object]:
    e.jobs.write_text("x")
    return []


def _jobs_in_git(e: Env) -> list[object]:
    e.jobs = _git_init(e.tmp / "repo") / "jobs"
    return []


def _local_gpu_n4(e: Env) -> list[object]:
    _ollama("http://ollama.invalid:11434")(e)
    return ["-n", "4"]


ENV_ERRORS = [
    ("no-git", _no_git, "git is not on PATH"),
    ("no-backend", _unset("TALOS_BACKEND"), "TALOS_BACKEND must be"),
    ("bad-backend", _set(TALOS_BACKEND="openai"), "TALOS_BACKEND must be"),
    (
        "bedrock-with-ollama",
        _ollama("http://ollama.invalid:11434", TALOS_BEDROCK="1"),
        "TALOS_BEDROCK is set",
    ),
    ("ollama-no-model", _ollama("http://ollama.invalid:11434", ""), "OLLAMA_MODEL"),
    ("ollama-no-url", _ollama(""), "OLLAMA_BASE_URL"),
    ("cloud-no-key", _ollama("https://ollama.com"), "OLLAMA_API_KEY"),
    ("cloud-path-no-key", _ollama("https://ollama.com/api"), "OLLAMA_API_KEY"),
    ("anthropic-no-key", _unset("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY"),
    ("no-artifact-base", _unset("TALOS_ARTIFACT_BASE"), "TALOS_ARTIFACT_BASE"),
    ("talos-path-missing", _talos_path_missing, "--talos-path"),
    ("tasks-missing", _tasks_missing, "--tasks is not an existing directory"),
    ("tasks-in-git", _tasks_in_git, "must never be committed"),
    ("jobs-parent-missing", _jobs_parent_missing, "the parent of --jobs"),
    ("jobs-not-empty", _jobs_not_empty, "non-empty"),
    ("jobs-is-file", _jobs_is_file, "not a directory"),
    ("jobs-in-git", _jobs_in_git, "must never be committed"),
    ("local-gpu-n4", _local_gpu_n4, "-n must be 1 on a local GPU backend"),
]


@pytest.mark.parametrize(("name", "setup", "needle"), ENV_ERRORS, ids=[n for n, _, _ in ENV_ERRORS])
def test_env_and_path_errors(env: Env, name: str, setup: Setup, needle: str) -> None:
    extra = setup(env)
    r = env.std(*extra)
    assert r.returncode == 2, (name, r.stderr)
    assert r.stderr.startswith("tier2-harbor: "), r.stderr
    assert needle in r.stderr
    assert USAGE not in r.stderr
    assert env.lines() == []


def test_local_gpu_n_refusal_is_exact(env: Env) -> None:
    env.ollama("http://ollama.invalid:11434", "qwen-test")
    r = env.std("-n", "4")
    assert r.returncode == 2
    assert r.stderr == "tier2-harbor: -n must be 1 on a local GPU backend\n"


# ---- run sequence -----------------------------------------------------------------------


def test_anthropic_run(env: Env) -> None:
    r = env.std("--talos-version", "v1", "--task", "a", "--task", "b")
    assert r.returncode == 0, r.stderr
    lines = env.lines()
    first_uv = next(i for i, ln in enumerate(lines) if ln.startswith("uv\t"))
    docker = [i for i, ln in enumerate(lines) if ln.startswith("docker\t")]
    assert docker and max(docker) < first_uv
    assert f"docker\tbuild -t talos-evals-base:1 {env.tasks}/_base" in lines
    assert env.uv_lines() == [
        f"uv\t{ROOT}/harbor\t{env.harbor_argv()} --ak talos_version=v1 -i a -i b",
        f"uv\t{ROOT}/harbor\trun --frozen talos-harbor-tier2-report {env.jobs} "
        f"--expect-k 3 --tasks {env.tasks} --task a --task b",
    ]
    assert not any(ln.startswith("curl\t") for ln in lines)
    assert decision_line(r.stderr) == (
        f"tier2-harbor: backend=anthropic gpu_lock=unwrapped talos=version:v1 k=3 "
        f"tasks={env.tasks} jobs={env.jobs} include=a,b"
    )
    assert "sk-test-secret" not in r.stdout + r.stderr


def test_relative_talos_path_is_made_absolute(env: Env) -> None:
    (env.tmp / "talos").write_text("#!/bin/sh\n")
    r = env.run("--tasks", "tasks", "--jobs", "jobs", "--talos-path", "talos", cwd=env.tmp)
    assert r.returncode == 0, r.stderr
    assert env.uv_lines()[0].endswith(f"-o {env.jobs} --ak talos_path={env.tmp}/talos")
    assert f"talos=path:{env.tmp}/talos " in decision_line(r.stderr)
    assert f"tasks={env.tasks} jobs={env.jobs} include=all" in decision_line(r.stderr)


def test_neither_talos_flag_ends_at_jobs(env: Env) -> None:
    r = env.std()
    assert r.returncode == 0, r.stderr
    uv = env.uv_lines()
    assert uv[0] == f"uv\t{ROOT}/harbor\t{env.harbor_argv()}"
    assert uv[1].endswith(f"--expect-k 3 --tasks {env.tasks}")
    assert " talos=latest " in decision_line(r.stderr)


def test_bedrock_runs_unwrapped_without_anthropic_key(env: Env) -> None:
    del env.vars["ANTHROPIC_API_KEY"]
    env.vars["TALOS_BEDROCK"] = "1"
    r = env.std()
    assert r.returncode == 0, r.stderr
    assert "backend=bedrock gpu_lock=unwrapped " in decision_line(r.stderr)
    assert not any(ln.startswith("curl\t") for ln in env.lines())


def test_local_ollama_is_wrapped(env: Env) -> None:
    env.ollama("http://ollama.invalid:11434", "qwen-test")
    r = env.std()
    assert r.returncode == 0, r.stderr
    lines = env.lines()
    curl = next(i for i, ln in enumerate(lines) if ln.startswith("curl\t"))
    harbor = next(i for i, ln in enumerate(lines) if "harbor run" in ln)
    assert curl < harbor
    assert lines[curl] == "curl\t-fsS http://ollama.invalid:11434/api/ps"
    assert "backend=ollama gpu_lock=wrapped " in decision_line(r.stderr)
    assert env.uv_lines()[0] == f"uv\t{ROOT}/harbor\t{env.harbor_argv()}"


def test_ollama_cloud_url_is_unwrapped(env: Env) -> None:
    env.ollama("https://ollama.com", "glm-5.3")
    env.vars["OLLAMA_API_KEY"] = "x"
    r = env.std()
    assert r.returncode == 0, r.stderr
    assert not any(ln.startswith("curl\t") for ln in env.lines())
    assert "gpu_lock=unwrapped" in decision_line(r.stderr)


def test_cloud_model_on_local_url_is_unwrapped(env: Env) -> None:
    env.ollama("http://ollama.invalid:11434", "glm-5.3:cloud")
    r = env.std()
    assert r.returncode == 0, r.stderr
    assert not any(ln.startswith("curl\t") for ln in env.lines())
    assert "gpu_lock=unwrapped" in decision_line(r.stderr)


def test_k_is_passed_to_harbor_and_report(env: Env) -> None:
    r = env.std("-k", "5")
    assert r.returncode == 0, r.stderr
    uv = env.uv_lines()
    assert uv[0] == f"uv\t{ROOT}/harbor\t{env.harbor_argv(k=5)}"
    assert uv[1].endswith(f"--expect-k 5 --tasks {env.tasks}")
    assert " k=5 " in decision_line(r.stderr)


def test_n_defaults_to_one_and_passes_through_on_a_cloud_url(env: Env) -> None:
    env.ollama("https://ollama.com", "glm-5.3")
    env.vars["OLLAMA_API_KEY"] = "x"
    r = env.std("-n", "4")
    assert r.returncode == 0, r.stderr
    assert env.uv_lines()[0] == f"uv\t{ROOT}/harbor\t{env.harbor_argv(n=4)}"


def test_harbor_failure_with_empty_jobs_skips_report(env: Env) -> None:
    env.jobs.mkdir()
    env.vars["FAKE_HARBOR_RC"] = "75"
    r = env.std()
    assert r.returncode == 75
    assert len(env.uv_lines()) == 1


def test_harbor_failure_that_wrote_jobs_still_reports(env: Env) -> None:
    env.vars.update(FAKE_HARBOR_RC="3", FAKE_HARBOR_WRITE="1", FAKE_REPORT_RC="0")
    r = env.std()
    assert r.returncode == 3
    assert "talos-harbor-tier2-report" in env.uv_lines()[1]


def test_report_code_is_the_exit_code(env: Env) -> None:
    env.vars["FAKE_REPORT_RC"] = "3"
    assert env.std().returncode == 3


def test_harbor_wrote_nothing_is_exit_1(env: Env) -> None:
    # A harbor that exits 0 but leaves the jobs dir absent.
    (env.bindir / "uv").write_text(
        f'#!{shutil.which("bash")}\nprintf \'uv\\t%s\\t%s\\n\' "$PWD" "$*" >> "$STUB_LOG"\nexit 0\n'
    )
    r = env.std()
    assert r.returncode == 1
    assert r.stderr.endswith(f"tier2-harbor: harbor wrote nothing to {env.jobs}\n")
    assert len(env.uv_lines()) == 1


def test_missing_base_image_tag_is_exit_2_without_uv(env: Env) -> None:
    (env.tasks / "_base" / "IMAGE").unlink()
    r = env.std()
    assert r.returncode == 2
    assert env.uv_lines() == []
