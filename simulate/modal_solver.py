"""CalculiX modal (eigenfrequency) analysis from a tetrahedral mesh.

Given a volume mesh and a material, this module writes a CalculiX input deck
(`.inp`), runs the ``ccx`` solver, and returns the path to the result (`.dat`/
`.frd`) files. Natural-frequency extraction is handled in :mod:`simulate.results`.

Unit system: tonne-mm-s (see :mod:`simulate.materials`), so frequencies are Hz.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

try:
    from simulate.materials import Material, resolve_material
except ModuleNotFoundError:  # pragma: no cover - allow direct execution
    from materials import Material, resolve_material


# Boundary condition presets. "free" yields a free-free analysis (the first six
# modes are ~0 Hz rigid-body modes); the others clamp one outer face.
BOUNDARY_CONDITIONS = ("free", "fixed_bottom", "fixed_top", "encastre")


@dataclass(frozen=True)
class ModalSetup:
    """Inputs for a modal run."""

    mesh_file: Path
    material: Material
    num_modes: int = 10
    boundary: str = "fixed_bottom"


@dataclass(frozen=True)
class SolverRun:
    """Result of invoking CalculiX."""

    job_name: str
    work_dir: Path
    inp_file: Path
    dat_file: Path
    frd_file: Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.dat_file.exists()


def find_ccx() -> str | None:
    """Locate the CalculiX executable."""

    for candidate in ("ccx", "ccx_2.21", "ccx_2.20", "ccx_2.19", "calculix"):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def write_inp_deck(setup: ModalSetup, inp_file: Path) -> None:
    """Write a CalculiX modal input deck for the given mesh and material."""

    import meshio

    mesh = meshio.read(str(setup.mesh_file))
    points = np.asarray(mesh.points, dtype=float)

    elements, element_type = _extract_tetra(mesh)
    if elements.size == 0:
        raise ValueError("Mesh contains no tetrahedral elements to solve.")

    fixed_nodes = _boundary_nodes(points, elements, setup.boundary)

    inp_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("*HEADING")
    lines.append(f"Resonance AI modal analysis - {setup.material.name}")

    # Nodes (1-based numbering for CalculiX).
    lines.append("*NODE, NSET=NALL")
    for index, (x, y, z) in enumerate(points, start=1):
        lines.append(f"{index}, {x:.6f}, {y:.6f}, {z:.6f}")

    # Elements.
    lines.append(f"*ELEMENT, TYPE={element_type}, ELSET=EALL")
    for elem_index, connectivity in enumerate(elements, start=1):
        node_ids = ", ".join(str(int(n) + 1) for n in connectivity)
        lines.append(f"{elem_index}, {node_ids}")

    # Material.
    mat = setup.material
    lines.append(f"*MATERIAL, NAME={mat.name.upper()}")
    lines.append("*ELASTIC")
    lines.append(f"{mat.youngs_modulus_mpa:.6g}, {mat.poisson_ratio:.6g}")
    lines.append("*DENSITY")
    lines.append(f"{mat.density_t_per_mm3:.6e}")
    lines.append(f"*SOLID SECTION, ELSET=EALL, MATERIAL={mat.name.upper()}")

    # Boundary conditions.
    if fixed_nodes.size:
        lines.append("*NSET, NSET=NFIX")
        lines.extend(_chunk_ids(fixed_nodes + 1))
        lines.append("*BOUNDARY")
        lines.append("NFIX, 1, 3")

    # Frequency extraction step.
    lines.append("*STEP")
    lines.append("*FREQUENCY, STORAGE=YES")
    lines.append(f"{setup.num_modes}")
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*EL FILE")
    lines.append("S")
    lines.append("*END STEP")

    inp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_modal(setup: ModalSetup, output_dir: Path, job_name: str = "modal") -> SolverRun:
    """Write the deck and run CalculiX, returning the solver result paths."""

    ccx = find_ccx()
    if ccx is None:
        raise RuntimeError(
            "CalculiX (ccx) not found. Install it, e.g. 'sudo apt-get install calculix-ccx'."
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inp_file = output_dir / f"{job_name}.inp"
    write_inp_deck(setup, inp_file)

    # CalculiX is invoked with the job name (no extension) from the work dir.
    process = subprocess.run(
        [ccx, job_name],
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
    )

    return SolverRun(
        job_name=job_name,
        work_dir=output_dir,
        inp_file=inp_file,
        dat_file=output_dir / f"{job_name}.dat",
        frd_file=output_dir / f"{job_name}.frd",
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


def _extract_tetra(mesh) -> tuple[np.ndarray, str]:
    """Return tetra connectivity and the matching CalculiX element type."""

    for block in mesh.cells:
        if block.type == "tetra10":
            return np.asarray(block.data, dtype=int), "C3D10"
    for block in mesh.cells:
        if block.type == "tetra":
            return np.asarray(block.data, dtype=int), "C3D4"
    return np.empty((0, 4), dtype=int), "C3D4"


def _boundary_nodes(points: np.ndarray, elements: np.ndarray, boundary: str) -> np.ndarray:
    """Pick the node indices to clamp for the chosen boundary condition."""

    used = np.unique(elements)
    if boundary == "free" or used.size == 0:
        return np.empty(0, dtype=int)
    if boundary == "encastre":
        return used

    axis = 2  # Z
    coords = points[used, axis]
    span = float(coords.max() - coords.min())
    tol = max(span * 0.02, 1e-6)

    if boundary == "fixed_top":
        target = coords.max()
        mask = coords >= target - tol
    else:  # fixed_bottom (default)
        target = coords.min()
        mask = coords <= target + tol

    return used[mask]


def _chunk_ids(ids: np.ndarray, per_line: int = 8) -> list[str]:
    rows = []
    ids = np.asarray(ids, dtype=int)
    for start in range(0, len(ids), per_line):
        chunk = ids[start : start + per_line]
        rows.append(", ".join(str(int(value)) for value in chunk))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a CalculiX modal analysis on a mesh.")
    parser.add_argument("mesh_file", type=Path, help="Tetrahedral mesh (.msh/.inp/.vtk).")
    parser.add_argument("--material", default="steel", help="Material name or hint.")
    parser.add_argument("--modes", type=int, default=10, help="Number of modes to extract.")
    parser.add_argument(
        "--boundary",
        default="fixed_bottom",
        choices=BOUNDARY_CONDITIONS,
        help="Boundary condition preset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "simulation")
    parser.add_argument("--name", default="modal", help="Job name.")
    args = parser.parse_args(argv)

    setup = ModalSetup(
        mesh_file=args.mesh_file,
        material=resolve_material(args.material),
        num_modes=args.modes,
        boundary=args.boundary,
    )

    try:
        run = run_modal(setup, args.output_dir, job_name=args.name)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if run.stdout:
        print(run.stdout, end="")
    if not run.ok:
        if run.stderr:
            print(run.stderr, end="", file=sys.stderr)
        print("CalculiX run failed.", file=sys.stderr)
        return run.returncode or 1

    print(f"Solver finished. Results: {run.dat_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
