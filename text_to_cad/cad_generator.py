"""Phase A command line interface: prompt to local CadQuery CAD artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

try:
    from text_to_cad.cad_executor import run_script, write_script
    from text_to_cad.cad_templates import CadSpec, prompt_to_spec, render_cadquery_script
    from text_to_cad.viewer import write_viewer
except ModuleNotFoundError:
    from cad_executor import run_script, write_script
    from cad_templates import CadSpec, prompt_to_spec, render_cadquery_script
    from viewer import write_viewer


DEFAULT_OUTPUT_ROOT = Path("outputs") / "phase_a"


def generate_from_prompt(
    prompt: str,
    output_dir: Path,
    output_name: str = "bracket",
    execute: bool = True,
) -> int:
    spec = prompt_to_spec(prompt)
    script = render_cadquery_script(spec, output_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")
    (output_dir / "spec.json").write_text(spec.to_json() + "\n", encoding="utf-8")
    script_path = write_script(script, output_dir)
    _write_preview(spec, output_dir / "preview.png")

    print(f"Wrote prompt, spec, preview, and CadQuery script to {output_dir}")
    if not execute:
        print("Skipped CAD execution because --dry-run was supplied.")
        return 0

    result = run_script(script_path)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if not result.ok:
        print(
            "CadQuery execution failed. The generated script is still available for inspection.",
            file=sys.stderr,
        )
        return result.returncode

    viewer_path = write_viewer(output_dir / f"{output_name}.stl", spec, output_dir, output_name)
    print(f"Wrote interactive 3D viewer to {viewer_path}")
    print(f"CAD artifacts are in {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a simple Phase A CAD model from text.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Natural language part description.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a prompt text file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None, help="Output basename for STEP and STL files.")
    parser.add_argument("--dry-run", action="store_true", help="Write files but do not run CadQuery.")
    args = parser.parse_args(argv)

    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    output_name = args.name or _slug_from_prompt(prompt)
    return generate_from_prompt(
        prompt=prompt,
        output_dir=args.output_dir,
        output_name=output_name,
        execute=not args.dry_run,
    )


def _write_preview(spec: CadSpec, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    rect = Rectangle(
        (-spec.length_mm / 2.0, -spec.width_mm / 2.0),
        spec.length_mm,
        spec.width_mm,
        linewidth=1.8,
        edgecolor="#202124",
        facecolor="#d7e7f5",
    )
    ax.add_patch(rect)

    for x, y in _preview_holes(spec):
        ax.add_patch(
            Circle(
                (x, y),
                spec.hole_radius_mm,
                linewidth=1.4,
                edgecolor="#202124",
                facecolor="white",
            )
        )

    pad = max(spec.length_mm, spec.width_mm) * 0.12
    ax.set_xlim(-spec.length_mm / 2.0 - pad, spec.length_mm / 2.0 + pad)
    ax.set_ylim(-spec.width_mm / 2.0 - pad, spec.width_mm / 2.0 + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{spec.part_type.title()} top view ({spec.length_mm:g} x {spec.width_mm:g} x {spec.thickness_mm:g} mm)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _preview_holes(spec: CadSpec) -> list[tuple[float, float]]:
    if spec.hole_count <= 0:
        return []
    margin_x = max(spec.hole_diameter_mm * 2.0, spec.length_mm * 0.125)
    margin_y = max(spec.hole_diameter_mm * 2.0, spec.width_mm * 0.2)
    x = max(0.0, spec.length_mm / 2.0 - margin_x)
    y = max(0.0, spec.width_mm / 2.0 - margin_y)
    points = [
        (-x, -y),
        (x, -y),
        (-x, y),
        (x, y),
        (0.0, -y),
        (0.0, y),
        (-x, 0.0),
        (x, 0.0),
    ]
    return points[: spec.hole_count]


def _slug_from_prompt(prompt: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())[:4]
    return "_".join(words) if words else "part"


if __name__ == "__main__":
    raise SystemExit(main())
