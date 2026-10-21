"""TalosAgent: a Harbor installed agent that runs the talos binary inside a trial container.

The adapter is deliberately thin. Harbor owns the container and the verifier; talos owns
the agent loop. This module installs a talos binary into ``/installed-agent``, hands it the
task instruction verbatim through ``--instruction-file``, and, once Harbor has synced
``/logs/agent`` back to the host, copies talos's own outcome record into the trial's
``AgentContext``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from harbor.agents.installed.base import BaseInstalledAgent, CliFlag, NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

INSTALLED_DIR = "/installed-agent"
"""Where the talos binary and the instruction file live inside the container."""

CONTAINER_LOGS_DIR = "/logs/agent"
"""The container path Harbor syncs back to the host as the trial's agent logs dir."""

TALOS_BIN = f"{INSTALLED_DIR}/talos"
INSTRUCTION_PATH = f"{INSTALLED_DIR}/instruction.md"
STDERR_LOG = f"{CONTAINER_LOGS_DIR}/talos-stderr.log"

SUPPORTED_ARCHES = ("x86_64", "aarch64")
"""Architectures the talos artifact host publishes a Linux binary for."""

OUTCOME_EXIT_CODES = frozenset({0, 10, 20, 30, 40})
"""talos exit codes that mean "the run reached an outcome" (any other code is infra failure)."""

EXIT_CODE_DISPOSITION = {0: "Done", 10: "Blocked", 30: "AlreadySatisfied", 40: "Answer"}
"""Exit code -> the one disposition it implies; 20 covers every other disposition."""

CLAIMED_DONE = "Done"
"""The disposition that counts as a claimed Done (talos's ``Done`` disposition)."""

FORWARDED_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "bedrock": ("TALOS_BEDROCK", "ANTHROPIC_MODEL"),
    "ollama": (
        "TALOS_BACKEND",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_API_KEY",
        "OLLAMA_NUM_CTX",
        "OLLAMA_THINK",
    ),
    "anthropic": ("TALOS_BACKEND", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
}
"""Per-backend host env keys forwarded to talos when set and non-empty.

The backend is chosen with talos's own precedence (non-empty ``TALOS_BEDROCK``, then
``TALOS_BACKEND=ollama``, else anthropic), so a run never carries another backend's
credentials. The bedrock set also forwards every ``AWS_*`` key; every extra_env key is
always forwarded unchanged."""

MEMORY_MODES: tuple[str, ...] = ("off", "annotate", "deny-once", "belief-check")
"""The ``talos run --memory`` values, spelled exactly like the harness ``MemoryMode::as_str``."""

KB_PROJECT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
"""Shape of a KB project ref, always applied with ``fullmatch``.

A ref matching it is one shell word that clap cannot read as a flag."""

MEMORY_STAT_KEYS: tuple[str, ...] = (
    "memory_annotations",
    "memory_denies",
    "memory_kb_tool_calls",
    "memory_deny_overrides",
)
"""The ``run_end.stats`` memory counter keys the harness emits."""

MAX_MEMORY_DENIES = 2
"""Mirrors the harness constant ``MAX_MEMORY_DENIES`` in crates/harness/src/memory.rs."""

ERROR_TAIL_BYTES = 2000
STDERR_TAIL_BYTES = 2048
HTTP_TIMEOUT_SECS = 120


def _http_get(url: str) -> bytes:
    """GET ``url`` on the host and return the body. Every HTTP call in this module goes here."""
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_SECS) as response:
        body: bytes = response.read()
        return body


def build_run_command(installed_dir: str, logs_dir: str, cli_flags: str) -> str:
    """Build the shell command that runs talos in the container.

    ``AGENT_GATE_COMMAND`` (from the task's ``[environment] env``) becomes ONE
    ``--gate-command`` argument via ``set --``/``"$@"``, so its quoting survives intact; when it
    is unset or empty no ``--gate-command`` flag is passed. The resolved gate command is also
    written to ``gate-command.txt`` so a trial records whether its gate was armed.
    """
    flags = f" {cli_flags}" if cli_flags else ""
    return (
        f"printf '%s' \"${{AGENT_GATE_COMMAND:-}}\" > {logs_dir}/gate-command.txt; "
        "set -- ; "
        'if [ -n "${AGENT_GATE_COMMAND:-}" ]; then '
        'set -- --gate-command "$AGENT_GATE_COMMAND"; fi; '
        f'{installed_dir}/talos run --workspace "$(pwd)" '
        f"--instruction-file {installed_dir}/instruction.md "
        "--task-id harbor "
        f"--transcript {logs_dir}/transcript.jsonl "
        f"--run-store {logs_dir}/run.sqlite "
        f"--offload-dir {logs_dir}/offload"
        f"{flags} "
        f'"$@" > {logs_dir}/run-summary.json 2> {logs_dir}/talos-stderr.log'
    )


class TalosAgent(BaseInstalledAgent):
    """Run talos as a Harbor installed agent.

    The binary comes either from a local file (``talos_path``) or from an artifact host
    (``artifact_base``, defaulting to ``$TALOS_ARTIFACT_BASE``) laid out as
    ``<base>/latest`` and ``<base>/<token>/<arch>/talos``.
    """

    CLI_FLAGS: ClassVar[list[CliFlag]] = [
        CliFlag("max_iterations", cli="--max-iterations", type="int"),
        CliFlag("wall_clock_secs", cli="--wall-clock-secs", type="int"),
        CliFlag("gate_timeout_secs", cli="--gate-timeout-secs", type="int"),
    ]

    def __init__(
        self,
        logs_dir: Path,
        *args: Any,
        talos_path: str | None = None,
        talos_version: str = "latest",
        artifact_base: str | None = None,
        memory: str = "off",
        kb_project: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not (isinstance(memory, str) and memory in MEMORY_MODES):
            raise ValueError(
                f"invalid memory {memory!r}: expected one of "
                "'off', 'annotate', 'deny-once', 'belief-check'"
            )
        if kb_project is not None and not (
            isinstance(kb_project, str) and KB_PROJECT_PATTERN.fullmatch(kb_project)
        ):
            raise ValueError(
                f"invalid kb_project {kb_project!r}: expected a KB project ref matching "
                "[A-Za-z0-9][A-Za-z0-9._-]*"
            )
        if kb_project is not None and memory == "off":
            raise ValueError(
                f"kb_project {kb_project!r} needs memory 'annotate', 'deny-once' or "
                "'belief-check', but memory is 'off'"
            )
        super().__init__(logs_dir, *args, **kwargs)
        if artifact_base is None:
            artifact_base = os.environ.get("TALOS_ARTIFACT_BASE")
        if artifact_base:
            artifact_base = artifact_base.rstrip("/")
        if talos_path is None and not artifact_base:
            raise ValueError(
                "TalosAgent needs a talos binary: pass talos_path=<local file> or set "
                "artifact_base / TALOS_ARTIFACT_BASE to a talos artifact host"
            )
        self.talos_path = talos_path
        self.talos_version = talos_version
        self.artifact_base = artifact_base or None
        self.memory = memory
        self.kb_project = kb_project

    def build_cli_flags(self) -> str:
        """Base CLI flags, then ``--memory`` and ``--kb-project`` when memory is on."""
        parts = [super().build_cli_flags()]
        if self.memory != "off":
            parts.append(f"--memory {self.memory}")
            if self.kb_project is not None:
                parts.append(f"--kb-project {self.kb_project}")
        return " ".join(part for part in parts if part)

    @staticmethod
    def name() -> str:
        return "talos"

    @classmethod
    def import_path(cls) -> str:
        return "talos_harbor.agent:TalosAgent"

    def get_version_command(self) -> str | None:
        return f"{TALOS_BIN} --version"

    def parse_version(self, stdout: str) -> str:
        version = stdout.strip()
        return version.removeprefix("talos ").strip()

    # ------------------------------------------------------------------ install

    async def install(self, environment: BaseEnvironment) -> None:
        self._select_backend()  # fail closed before any container exec
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        install_json = setup_dir / "install.json"

        if self.talos_path is not None:
            data = Path(self.talos_path).read_bytes()
            record: dict[str, Any] = {
                "source": "talos_path",
                "talos_path": self.talos_path,
                "requested_version": None,
                "resolved_token": None,
                "arch": None,
                "url": None,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            install_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            self.logger.info("installing talos from talos_path=%s", self.talos_path)
            await environment.upload_file(self.talos_path, TALOS_BIN)
            await self.exec_as_root(environment, f"chmod 0755 {TALOS_BIN}")
            return

        assert self.artifact_base is not None  # guaranteed by __init__
        uname = await self.exec_as_root(environment, "uname -m")
        arch = (uname.stdout or "").strip()
        if arch not in SUPPORTED_ARCHES:
            raise RuntimeError(
                f"unsupported container architecture {arch!r}: talos publishes only "
                f"{', '.join(SUPPORTED_ARCHES)}"
            )

        if self.talos_version == "latest":
            token = _http_get(f"{self.artifact_base}/latest").decode("utf-8").strip()
        else:
            token = self.talos_version
        if not token or "/" in token or any(ch.isspace() for ch in token):
            raise RuntimeError(f"invalid talos version token {token!r}")

        url = f"{self.artifact_base}/{token}/{arch}/talos"
        self.logger.info("installing talos token=%s arch=%s from %s", token, arch, url)
        data = _http_get(url)
        record = {
            "source": "artifact_base",
            "talos_path": None,
            "requested_version": self.talos_version,
            "resolved_token": token,
            "arch": arch,
            "url": url,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        install_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        # A host tempdir, never logs_dir: Docker bind-mounts logs_dir into the container.
        with tempfile.TemporaryDirectory(prefix="talos-harbor-") as tmp:
            binary = Path(tmp) / "talos"
            binary.write_bytes(data)
            await environment.upload_file(binary, TALOS_BIN)
        await self.exec_as_root(environment, f"chmod 0755 {TALOS_BIN}")

    # ---------------------------------------------------------------------- run

    def _select_backend(self) -> str:
        """Pick the backend with talos's precedence; fail closed on junk.

        extra_env wins over the host env for the two selector keys, because extra_env is
        what the container actually receives: a ``--ae TALOS_BACKEND=ollama`` run must not
        be classified (and credentialed) from the host's default.
        """

        def lookup(key: str) -> str | None:
            return self._extra_env[key] if key in self._extra_env else os.environ.get(key)

        if lookup("TALOS_BEDROCK"):
            return "bedrock"
        selected = lookup("TALOS_BACKEND")
        if selected == "ollama":
            return "ollama"
        if selected in (None, "", "anthropic"):
            return "anthropic"
        raise ValueError(
            f"unrecognised TALOS_BACKEND {selected!r}: expected unset, 'anthropic' or 'ollama'"
        )

    def _forwarded_env(self) -> dict[str, str]:
        backend = self._select_backend()
        self._backend_selected = backend
        env: dict[str, str] = {}
        for key in FORWARDED_ENV_KEYS[backend]:
            value = os.environ.get(key)
            if value:
                env[key] = value
        if backend == "bedrock":
            # Full AWS_* names (NOT _get_env_prefixed, which strips the prefix).
            for key, value in os.environ.items():
                if key.startswith("AWS_") and value:
                    env[key] = value
        env.update(self._extra_env)
        return env

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Deliberately NOT @with_prompt_template: talos sees the instruction bytes verbatim.
        env = self._forwarded_env()  # raises on a bogus TALOS_BACKEND before any exec
        self._forwarded_env_keys = sorted(env)
        with tempfile.TemporaryDirectory(prefix="talos-harbor-") as tmp:
            local = Path(tmp) / "instruction.md"
            with local.open("w", encoding="utf-8", newline="") as fh:
                fh.write(instruction)
            await environment.upload_file(local, INSTRUCTION_PATH)
        await self.exec_as_root(environment, f"chmod 0644 {INSTRUCTION_PATH}")

        # environment.exec directly, not _exec: _exec raises on every non-zero code, and
        # talos's 10/20/30/40 are outcomes, not failures.
        result = await environment.exec(
            command=build_run_command(INSTALLED_DIR, CONTAINER_LOGS_DIR, self.build_cli_flags()),
            cwd=None,
            env=env,
            timeout_sec=None,
            user=None,
        )
        rc = result.return_code
        self._talos_exit_code = rc
        classified = "outcome" if rc in OUTCOME_EXIT_CODES else "raise"
        self.logger.info("talos exit rc=%s classified=%s", rc, classified)
        if rc in OUTCOME_EXIT_CODES:
            return

        tail_result = await self.exec_as_agent(
            environment, f"tail -c {ERROR_TAIL_BYTES} {STDERR_LOG} 2>/dev/null || true"
        )
        tail = tail_result.stdout or ""
        raise NonZeroAgentExitCodeError(
            f"talos exited with code {rc} (not an outcome code); stderr tail from "
            f"{STDERR_LOG}:\n{tail}"
        )

    # ----------------------------------------------------------- post-run context

    def populate_context_post_run(self, context: AgentContext) -> None:
        errors: list[str] = []
        exit_code: int | None = getattr(self, "_talos_exit_code", None)

        summary = self._read_summary(errors)
        run_start, memory_slice, run_end = self._read_transcript(errors)

        disposition: str | None = None
        if summary is not None:
            raw = summary.get("disposition")
            if isinstance(raw, dict) and len(raw) == 1:
                disposition = str(next(iter(raw)))
            else:
                errors.append(f"run-summary.json disposition is not a single-key dict: {raw!r}")

        stats: dict[str, Any] = {}
        tokens: dict[str, int] | None = None
        if run_end is not None:
            raw_stats = run_end.get("stats")
            stats = raw_stats if isinstance(raw_stats, dict) else {}
            tokens = {
                "input": _as_int(stats.get("input_tokens")),
                "cache_read": _as_int(stats.get("cache_read_tokens")),
                "cache_write": _as_int(stats.get("cache_write_tokens")),
                "output": _as_int(stats.get("output_tokens")),
            }
            context.n_input_tokens = tokens["input"] + tokens["cache_read"] + tokens["cache_write"]
            context.n_cache_tokens = tokens["cache_read"]
            context.n_output_tokens = tokens["output"]

        if exit_code is None:
            errors.append("no talos exit code recorded (timeout or exec not reached)")
        elif disposition is not None and _exit_disposition_mismatch(exit_code, disposition):
            errors.append(
                f"exit_code/disposition mismatch: rc={exit_code} disposition={disposition}"
            )

        install = self._read_json(self.logs_dir / "setup" / "install.json")
        version = self.version()
        if (
            isinstance(install, dict)
            and install.get("source") == "artifact_base"
            and version is not None
            and version != install.get("resolved_token")
        ):
            errors.append(
                f"install token/version mismatch: token={install.get('resolved_token')} "
                f"binary={version}"
            )

        memory = self._memory_metadata(run_start, memory_slice, stats)
        errors.extend(
            self._memory_errors(
                memory, memory_slice, run_start is not None, run_end is not None, stats
            )
        )

        gate_command = self._read_text(self.logs_dir / "gate-command.txt") or None

        stderr_tail: str | None = None
        if exit_code is None or exit_code not in OUTCOME_EXIT_CODES or run_end is None:
            stderr_path = self.logs_dir / "talos-stderr.log"
            if stderr_path.is_file():
                data = stderr_path.read_bytes()[-STDERR_TAIL_BYTES:]
                stderr_tail = data.decode("utf-8", errors="replace")

        summary_or_empty = summary or {}
        talos: dict[str, Any] = {
            "exit_code": exit_code,
            "outcome": summary_or_empty.get("outcome"),
            "disposition": disposition,
            "claimed_done": (disposition == CLAIMED_DONE) if disposition is not None else None,
            "iterations": summary_or_empty.get("iterations"),
            "nudges_fired": stats.get("nudges_fired"),
            "gates_green_at_exit": stats.get("gates_green_at_exit"),
            "tree_dirty": stats.get("tree_dirty"),
            "mutating_iters": stats.get("mutating_iters"),
            "edit_file_calls_ok": stats.get("edit_file_calls_ok"),
            "bash_calls_ok": stats.get("bash_calls_ok"),
            "version": version,
            "gate_command": gate_command,
            "gate_armed": gate_command is not None,
            "backend": summary_or_empty.get("backend_settings"),
            "backend_selected": getattr(self, "_backend_selected", None),
            "forwarded_env_keys": getattr(self, "_forwarded_env_keys", None),
            "install": install if isinstance(install, dict) else None,
            "tokens": tokens,
            "compactions": summary_or_empty.get("compactions"),
            "highest_compaction_tier": summary_or_empty.get("highest_compaction_tier"),
            "memory": memory,
            "stderr_tail": stderr_tail,
        }
        if errors:
            talos["error"] = "; ".join(errors)

        metadata = dict(context.metadata or {})
        metadata["talos"] = talos
        context.metadata = metadata

    def _read_summary(self, errors: list[str]) -> dict[str, Any] | None:
        path = self.logs_dir / "run-summary.json"
        if not path.is_file():
            errors.append("run-summary.json missing")
            return None
        parsed = self._read_json(path)
        if not isinstance(parsed, dict):
            errors.append("run-summary.json unparsable")
            return None
        return parsed

    def _read_transcript(
        self, errors: list[str]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """One read of the transcript: first run_start, first memory_slice, last run_end."""
        path = self.logs_dir / "transcript.jsonl"
        if not path.is_file():
            errors.append("transcript.jsonl missing")
            return None, None, None
        run_start: dict[str, Any] | None = None
        memory_slice: dict[str, Any] | None = None
        run_end: dict[str, Any] | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # a truncated final line, or noise
            if not isinstance(event, dict):
                continue
            kind = event.get("event")
            if kind == "run_start" and run_start is None:
                run_start = event
            elif kind == "memory_slice" and memory_slice is None:
                memory_slice = event
            elif kind == "run_end":
                run_end = event
        if run_end is None:
            errors.append("transcript.jsonl has no run_end event (token counts unavailable)")
        return run_start, memory_slice, run_end

    def _memory_metadata(
        self,
        run_start: dict[str, Any] | None,
        memory_slice: dict[str, Any] | None,
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        """The 13-key ``metadata.talos.memory`` record."""
        effective_mode: str | None = None
        if run_start is not None:
            config = run_start.get("config")
            if isinstance(config, dict) and isinstance(config.get("memory_mode"), str):
                effective_mode = config["memory_mode"]
        degraded = stats.get("memory_degraded")
        source = memory_slice.get("source") if memory_slice is not None else None
        return {
            "mode": self.memory,
            "kb_project": self.kb_project,
            "effective_mode": effective_mode,
            **{key: _opt_int(stats, key) for key in MEMORY_STAT_KEYS},
            "memory_degraded": degraded if isinstance(degraded, str) and degraded else None,
            "source": source if isinstance(source, str) and source else None,
            "slice_items": _opt_int(memory_slice, "items"),
            "slice_chars": _opt_int(memory_slice, "chars"),
            "memory_tools": _str_list(memory_slice.get("memory_tools"))
            if memory_slice is not None
            else None,
            "memory_tools_dropped": _str_list(memory_slice.get("memory_tools_dropped"))
            if memory_slice is not None
            else None,
        }

    def _memory_errors(
        self,
        memory: dict[str, Any],
        memory_slice: dict[str, Any] | None,
        has_run_start: bool,
        has_run_end: bool,
        stats: dict[str, Any],
    ) -> list[str]:
        """Memory telemetry consistency checks, in their pinned order."""
        errors: list[str] = []
        on = self.memory != "off"
        effective = memory["effective_mode"]
        degraded = memory["memory_degraded"]
        counters = [memory[key] for key in MEMORY_STAT_KEYS]
        missing = [key for key in MEMORY_STAT_KEYS if key not in stats]
        if has_run_end and on and missing:
            errors.append(
                f"run_end stats lack memory telemetry with memory={self.memory}: "
                f"missing {', '.join(missing)}"
            )
        if degraded is not None:
            errors.append(f"memory degraded to off ({degraded})")
        if on and effective is None and has_run_start:
            errors.append(f"run_start config lacks memory_mode with memory={self.memory}")
        if effective is not None and effective != self.memory:
            errors.append(
                f"memory mode mismatch: requested {self.memory} but "
                f"run_start.config.memory_mode={effective}"
            )
        if on and effective == self.memory and memory_slice is None:
            errors.append("memory on but transcript has no memory_slice event")
        if not on and (memory_slice is not None or any(c is not None and c > 0 for c in counters)):
            errors.append("memory off requested but transcript shows memory activity")
        if all(c is not None for c in counters):
            annotations, denies, _calls, overrides = counters
            violations: list[str] = []
            if denies > MAX_MEMORY_DENIES:
                violations.append(f"memory_denies={denies} > {MAX_MEMORY_DENIES}")
            if self.memory in ("annotate", "belief-check") and denies > 0:
                violations.append(f"memory_denies={denies} under {self.memory}")
            if overrides > annotations:
                violations.append(
                    f"memory_deny_overrides={overrides} > memory_annotations={annotations}"
                )
            if degraded is not None and any(c > 0 for c in counters):
                violations.append("counters non-zero while degraded")
            if violations:
                errors.append(f"memory invariant violated: {', '.join(violations)}")
        return errors

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


def _as_int(value: Any) -> int:
    """Coerce a run_end stat to int; a missing or non-numeric stat counts as 0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def _opt_int(container: dict[str, Any] | None, key: str) -> int | None:
    """``_as_int`` of ``container[key]``; None when the container or the key is absent."""
    if container is None or key not in container:
        return None
    return _as_int(container[key])


def _str_list(value: Any) -> list[str] | None:
    """``value`` when it is a list of strings (possibly empty), else None."""
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def _exit_disposition_mismatch(exit_code: int, disposition: str) -> bool:
    """True when talos's exit code and its reported disposition disagree.

    Only outcome codes are checked: a non-outcome code (e.g. 1 = BackendError) has no
    disposition it implies, and is already flagged by the raise in ``run``.
    """
    if exit_code not in OUTCOME_EXIT_CODES:
        return False
    if exit_code == 20:
        return disposition in EXIT_CODE_DISPOSITION.values()
    return EXIT_CODE_DISPOSITION[exit_code] != disposition
