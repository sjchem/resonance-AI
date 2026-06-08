"""Local execution helpers for generated CadQuery scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class ExecutionResult:
    script_path: Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def write_script(script: str, output_dir: Path, filename: str = "generated_cad.py") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / filename
    script_path.write_text(script, encoding="utf-8")
    return script_path


def run_script(script_path: Path, timeout_seconds: int = 120) -> ExecutionResult:
    script_path = script_path.resolve()
    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", str(script_path.parent / ".cache"))
    process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return ExecutionResult(
        script_path=script_path,
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )
