"""CalculiX modal (eigenfrequency) analysis from a solid volume mesh.

Given a volume mesh and a material, this module writes a CalculiX input deck
(`.inp`), runs the ``ccx`` solver, and returns the path to the result (`.dat`/
`.frd`) files. Natural-frequency extraction is handled in :mod:`simulate.results`.

Supported solid elements are tetrahedra (C3D4/C3D10) and hexahedra
(C3D8/C3D20). Gmsh quadratic 27-node hexes are reduced to the 20-node CalculiX
layout by dropping face/volume center nodes.

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
class ElementBlock:
    """A homogeneous CalculiX element block."""

    connectivity: np.ndarray
    calculix_type: str


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

    def failure_summary(self, *, max_lines: int = 14, max_chars: int = 1800) -> str:
        """Return the useful tail of a failed solver run for an API response."""

        combined = "\n".join(part.strip() for part in (self.stderr, self.stdout) if part.strip())
        if not combined:
            return f"CalculiX exited with code {self.returncode} without diagnostic output."

        lines = [line.strip() for line in combined.splitlines() if line.strip()]
        markers = ("error", "failed", "singular", "negative", "jacobian", "not enough", "fatal")
        marked = [line for line in lines if any(marker in line.lower() for marker in markers)]
        selected = marked[-max_lines:] if marked else lines[-max_lines:]
        summary = "\n".join(selected)
        if len(summary) > max_chars:
            summary = summary[-max_chars:]
        return f"CalculiX exited with code {self.returncode}. {summary}"


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

    element_blocks = _extract_solid_elements(mesh)
    if not element_blocks:
        raise ValueError("Mesh contains no tetrahedral or hexahedral solid elements to solve.")

    fixed_nodes = _boundary_nodes(points, element_blocks, setup.boundary)

    inp_file.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("*HEADING")
    lines.append(f"Resonance AI modal analysis - {setup.material.name}")

    # Nodes (1-based numbering for CalculiX).
    lines.append("*NODE, NSET=NALL")
    for index, (x, y, z) in enumerate(points, start=1):
        lines.append(f"{index}, {x:.6f}, {y:.6f}, {z:.6f}")

    # Elements. CalculiX allows multiple homogeneous sections to append to the
    # same element set, which lets us solve mixed tet/hex meshes from Gmsh.
    elem_index = 1
    for block in element_blocks:
        lines.append(f"*ELEMENT, TYPE={block.calculix_type}, ELSET=EALL")
        for connectivity in block.connectivity:
            node_ids = ", ".join(str(int(node) + 1) for node in connectivity)
            lines.append(f"{elem_index}, {node_ids}")
            elem_index += 1

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


def _extract_solid_elements(mesh) -> list[ElementBlock]:
    """Return supported solid element blocks in CalculiX-ready connectivity."""

    specs = {
        "hexahedron20": ("C3D20", 20),
        "hexahedron27": ("C3D20", 20),
        "hexahedron": ("C3D8", 8),
        "tetra10": ("C3D10", 10),
        "tetra": ("C3D4", 4),
    }
    blocks: list[ElementBlock] = []
    for mesh_block in mesh.cells:
        spec = specs.get(mesh_block.type)
        if spec is None:
            continue
        calculix_type, node_count = spec
        data = np.asarray(mesh_block.data, dtype=int)
        if data.size == 0:
            continue
        blocks.append(ElementBlock(connectivity=data[:, :node_count], calculix_type=calculix_type))
    return blocks


def _boundary_nodes(points: np.ndarray, element_blocks: list[ElementBlock], boundary: str) -> np.ndarray:
    """Pick the node indices to clamp for the chosen boundary condition."""

    used = np.unique(np.concatenate([block.connectivity.ravel() for block in element_blocks])) if element_blocks else np.empty(0, dtype=int)
    if boundary == "free" or used.size == 0:
        return np.empty(0, dtype=int)
    if boundary == "encastre":
        return used

    fixed_parts: list[np.ndarray] = []
    for component in _element_node_components(element_blocks):
        axis = 2  # Z
        coords = points[component, axis]
        span = float(coords.max() - coords.min())
        tol = max(span * 0.02, 1e-6)
        if boundary == "fixed_top":
            target = coords.max()
            fixed_parts.append(component[coords >= target - tol])
        else:  # fixed_bottom (default)
            target = coords.min()
            fixed_parts.append(component[coords <= target + tol])
    return np.unique(np.concatenate(fixed_parts)) if fixed_parts else np.empty(0, dtype=int)


def _element_node_components(element_blocks: list[ElementBlock]) -> list[np.ndarray]:
    """Group nodes connected by solid elements without an optional graph package."""

    all_cells = [np.asarray(block.connectivity, dtype=int) for block in element_blocks if block.connectivity.size]
    if not all_cells:
        return []
    used = np.unique(np.concatenate([cells.ravel() for cells in all_cells]))
    parent = {int(node): int(node) for node in used}

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for cells in all_cells:
        for cell in cells:
            anchor = int(cell[0])
            for node in cell[1:]:
                union(anchor, int(node))

    groups: dict[int, list[int]] = {}
    for node in used:
        groups.setdefault(find(int(node)), []).append(int(node))
    return [np.asarray(nodes, dtype=int) for nodes in groups.values()]


def _chunk_ids(ids: np.ndarray, per_line: int = 8) -> list[str]:
    rows = []
    ids = np.asarray(ids, dtype=int)
    for start in range(0, len(ids), per_line):
        chunk = ids[start : start + per_line]
        rows.append(", ".join(str(int(value)) for value in chunk))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a CalculiX modal analysis on a solid mesh.")
    parser.add_argument("mesh_file", type=Path, help="Solid mesh (.msh/.inp/.vtk) with tetra or hex cells.")
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
