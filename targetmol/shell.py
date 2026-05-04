"""Run external commands and save logs."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from targetmol.provenance import ProvenanceRecorder


@dataclass
class CommandResult:
    """External command execution result."""

    command: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


def run_command(
    *,
    step: str,
    command: list[str],
    cwd: Path,
    logs_dir: Path,
    provenance: ProvenanceRecorder,
    env: dict[str, str] | None = None,
) -> CommandResult:
    """Run an external command and write stdout/stderr logs."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    provenance.record_command(step=step, command=command, cwd=cwd)
    runtime_env = os.environ.copy()
    if env:
        runtime_env.update(env)
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, env=runtime_env)
    (logs_dir / f"{step}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (logs_dir / f"{step}.stderr.log").write_text(result.stderr, encoding="utf-8")
    return CommandResult(
        command=command,
        cwd=cwd,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
