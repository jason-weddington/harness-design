"""A recording fake of harbor's BaseEnvironment: no Docker, no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.environments.base import ExecResult


@dataclass
class ExecCall:
    command: str
    cwd: str | None
    env: dict[str, str] | None
    timeout_sec: int | None
    user: str | int | None


@dataclass
class FakeEnvironment:
    """Records exec/upload_file calls and answers exec from a substring-keyed script.

    ``scripts`` maps a command substring to the ExecResult to return; the first key found
    in the command wins, and an unmatched command returns rc 0 with empty output.
    Uploaded bytes are copied at call time so tests can assert them after the
    adapter's temp files are gone.
    """

    scripts: dict[str, ExecResult] = field(default_factory=dict)
    execs: list[ExecCall] = field(default_factory=list)
    uploads: list[tuple[str, str, bytes]] = field(default_factory=list)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self.execs.append(ExecCall(command, cwd, env, timeout_sec, user))
        for key, result in self.scripts.items():
            if key in command:
                return result
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads.append((str(source_path), target_path, Path(source_path).read_bytes()))

    def commands(self) -> list[str]:
        return [c.command for c in self.execs]

    def find(self, needle: str) -> list[ExecCall]:
        return [c for c in self.execs if needle in c.command]


def result(rc: int = 0, stdout: str = "", stderr: str = "") -> ExecResult:
    return ExecResult(stdout=stdout, stderr=stderr, return_code=rc)


def as_any(env: FakeEnvironment) -> Any:
    """Pass the fake where a BaseEnvironment is expected."""
    return env
