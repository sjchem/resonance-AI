"""Parse CalculiX modal results into natural frequencies.

CalculiX writes eigenfrequencies to the job ``.dat`` file under an
``E I G E N V A L U E   O U T P U T`` block. This module reads that block and
returns the natural frequencies (Hz) plus a small JSON/text report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import sys


@dataclass(frozen=True)
class Mode:
    """A single vibration mode."""

    mode_number: int
    frequency_hz: float
    eigenvalue: float


@dataclass(frozen=True)
class ModalResults:
    """Extracted natural frequencies for a part."""

    dat_file: Path
    modes: list[Mode]
    rigid_body_modes: int

    @property
    def fundamental_hz(self) -> float | None:
        flexible = [m.frequency_hz for m in self.modes if m.frequency_hz > 1.0]
        return flexible[0] if flexible else None

    def to_dict(self) -> dict:
        return {
            "dat_file": str(self.dat_file),
            "rigid_body_modes": self.rigid_body_modes,
            "fundamental_hz": self.fundamental_hz,
            "modes": [asdict(mode) for mode in self.modes],
        }

    def summary(self) -> str:
        if not self.modes:
            return "No modes found in result file."
        lines = [f"Natural frequencies from {self.dat_file.name}:"]
        for mode in self.modes:
            tag = "  (rigid body)" if mode.frequency_hz <= 1.0 else ""
            lines.append(f"  Mode {mode.mode_number:>2}: {mode.frequency_hz:10.3f} Hz{tag}")
        if self.fundamental_hz is not None:
            lines.append(f"Fundamental flexible frequency: {self.fundamental_hz:.3f} Hz")
        return "\n".join(lines)


# Matches CalculiX .dat eigenvalue rows:
#   mode_no   eigenvalue   omega(rad/time)   freq(cycles/time)
_ROW_PATTERN = re.compile(
    r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


def parse_dat(dat_file: Path) -> ModalResults:
    """Extract eigenfrequencies from a CalculiX ``.dat`` file."""

    dat_file = Path(dat_file).resolve()
    if not dat_file.exists():
        raise FileNotFoundError(f"Result file not found: {dat_file}")

    text = dat_file.read_text(encoding="utf-8", errors="ignore")
    modes: list[Mode] = []
    in_block = False

    for raw_line in text.splitlines():
        # CalculiX writes section headers as letter-spaced text
        # (e.g. "E I G E N V A L U E   O U T P U T"), so compare without spaces.
        compact = raw_line.upper().replace(" ", "")
        if "EIGENVALUEOUTPUT" in compact:
            in_block = True
            continue
        if not in_block:
            continue

        # Subsequent letter-spaced section headers terminate the eigenvalue block.
        if modes and ("PARTICIPATIONFACTORS" in compact or "EFFECTIVEMODALMASS" in compact):
            break

        match = _ROW_PATTERN.match(raw_line)
        if match:
            mode_no = int(match.group(1))
            eigenvalue = float(match.group(2))
            frequency_hz = float(match.group(4))  # cycles/time column = Hz
            modes.append(Mode(mode_number=mode_no, frequency_hz=frequency_hz, eigenvalue=eigenvalue))

    rigid_body_modes = sum(1 for mode in modes if mode.frequency_hz <= 1.0)
    return ModalResults(dat_file=dat_file, modes=modes, rigid_body_modes=rigid_body_modes)


def write_report(results: ModalResults, output_file: Path) -> Path:
    """Write the modal results to a JSON report file."""

    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse natural frequencies from a CalculiX .dat file.")
    parser.add_argument("dat_file", type=Path, help="CalculiX .dat result file.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path to write a JSON report.")
    args = parser.parse_args(argv)

    try:
        results = parse_dat(args.dat_file)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(results.summary())
    if args.json is not None:
        path = write_report(results, args.json)
        print(f"Wrote JSON report to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
