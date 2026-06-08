"""Open generated CAD viewer files on Linux, WSL, macOS, or Windows."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def open_viewer(path: Path) -> int:
    viewer_path = path.expanduser().resolve()
    if not viewer_path.exists():
        print(f"Viewer not found: {viewer_path}", file=sys.stderr)
        return 1

    command = _open_command(viewer_path)
    if command is None:
        print(f"Open this file from your browser: {viewer_path}", file=sys.stderr)
        return 2

    subprocess.Popen(command)
    print(f"Opening {viewer_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a generated Phase A CAD viewer.")
    parser.add_argument(
        "viewer",
        nargs="?",
        type=Path,
        default=Path("outputs/phase_a/bracket/viewer.html"),
        help="Path to viewer.html.",
    )
    args = parser.parse_args(argv)
    return open_viewer(args.viewer)


def _open_command(path: Path) -> list[str] | None:
    system = platform.system().lower()
    if _is_wsl() and shutil.which("explorer.exe"):
        return ["explorer.exe", _wsl_to_windows_path(path)]
    if system == "windows":
        return ["cmd", "/c", "start", "", str(path)]
    if system == "darwin" and shutil.which("open"):
        return ["open", str(path)]
    if shutil.which("xdg-open"):
        return ["xdg-open", str(path)]
    return None


def _is_wsl() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def _wsl_to_windows_path(path: Path) -> str:
    if shutil.which("wslpath"):
        result = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
