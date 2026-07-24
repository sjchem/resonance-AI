"""Directional static stiffness analysis for structured bushing meshes.

The client convention used by Resonance AI defines X as the bushing
centerline. Parametric bushing meshes are generated around geometric Z, so the
engineering-to-mesh axis mapping is:

    Kx -> mesh Z, Ky -> mesh X, Kz -> mesh Y

The outer cylindrical interface is fixed. The inner cylindrical interface is
translated by a prescribed displacement while its other translations are held
at zero. CalculiX reaction forces on the inner interface are summed and divided
by the displacement to obtain stiffness in N/mm.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import pi
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

try:
    from simulate.materials import Material, resolve_material
    from simulate.modal_solver import ElementBlock, SolverRun, _chunk_ids, _extract_solid_elements, find_ccx
except ModuleNotFoundError:  # pragma: no cover - allow direct execution
    from materials import Material, resolve_material
    from modal_solver import ElementBlock, SolverRun, _chunk_ids, _extract_solid_elements, find_ccx


ENGINEERING_TO_MESH_AXIS = {"x": 2, "y": 0, "z": 1}
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_REACTION_ROW = re.compile(rf"^\s*(\d+)\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$")


@dataclass(frozen=True)
class StaticStiffnessSetup:
    """Inputs for three directional bushing stiffness load cases."""

    mesh_file: Path
    material: Material
    displacement_mm: float = 1.0
    inner_interface_length_mm: float | None = None
    outer_interface_length_mm: float | None = None


@dataclass(frozen=True)
class DirectionalStiffness:
    """Result for one prescribed translation."""

    engineering_axis: str
    mesh_axis: str
    displacement_mm: float
    reaction_force_n: float
    stiffness_n_per_mm: float
    reaction_vector_n: tuple[float, float, float]
    job_name: str
    dat_file: str


@dataclass(frozen=True)
class StaticStiffnessResult:
    """Combined Kx/Ky/Kz result and test assumptions."""

    material: str
    inner_node_count: int
    outer_node_count: int
    centerline_axis: str
    fixed_interface: str
    displaced_interface: str
    directions: tuple[DirectionalStiffness, ...]

    @property
    def kx_n_per_mm(self) -> float:
        return self._stiffness("x")

    @property
    def ky_n_per_mm(self) -> float:
        return self._stiffness("y")

    @property
    def kz_n_per_mm(self) -> float:
        return self._stiffness("z")

    def _stiffness(self, axis: str) -> float:
        return next(item.stiffness_n_per_mm for item in self.directions if item.engineering_axis == axis)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "kx_n_per_mm": self.kx_n_per_mm,
                "ky_n_per_mm": self.ky_n_per_mm,
                "kz_n_per_mm": self.kz_n_per_mm,
                "units": {"stiffness": "N/mm", "force": "N", "displacement": "mm"},
            }
        )
        return data


def run_static_stiffness(
    setup: StaticStiffnessSetup,
    output_dir: Path,
    *,
    job_name: str = "stiffness",
) -> StaticStiffnessResult:
    """Run X/Y/Z engineering load cases and return directional stiffness."""

    ccx = find_ccx()
    if ccx is None:
        raise RuntimeError("CalculiX (ccx) is required for static stiffness analysis.")
    if setup.displacement_mm <= 0:
        raise ValueError("Static stiffness displacement must be positive.")

    points, element_blocks = _read_solid_mesh(setup.mesh_file)
    inner_nodes, outer_nodes = radial_interface_nodes(
        points,
        element_blocks,
        inner_length_mm=setup.inner_interface_length_mm,
        outer_length_mm=setup.outer_interface_length_mm,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    directions: list[DirectionalStiffness] = []
    mesh_axis_names = ("X", "Y", "Z")

    for engineering_axis in ("x", "y", "z"):
        mesh_axis = ENGINEERING_TO_MESH_AXIS[engineering_axis]
        case_name = f"{job_name}_k{engineering_axis}"
        inp_file = output_dir / f"{case_name}.inp"
        write_static_deck(
            setup,
            inp_file,
            points=points,
            element_blocks=element_blocks,
            inner_nodes=inner_nodes,
            outer_nodes=outer_nodes,
            displacement_axis=mesh_axis,
        )
        run = _run_ccx(ccx, output_dir, case_name, inp_file)
        if not run.ok:
            raise RuntimeError(f"Static K{engineering_axis} solve failed: {run.failure_summary()}")

        reaction = parse_reaction_forces(run.dat_file, set_name="NINNER")
        force = abs(float(reaction[mesh_axis]))
        directions.append(
            DirectionalStiffness(
                engineering_axis=engineering_axis,
                mesh_axis=mesh_axis_names[mesh_axis],
                displacement_mm=setup.displacement_mm,
                reaction_force_n=force,
                stiffness_n_per_mm=force / setup.displacement_mm,
                reaction_vector_n=tuple(float(value) for value in reaction),
                job_name=case_name,
                dat_file=run.dat_file.name,
            )
        )

    result = StaticStiffnessResult(
        material=setup.material.name,
        inner_node_count=int(inner_nodes.size),
        outer_node_count=int(outer_nodes.size),
        centerline_axis="X",
        fixed_interface="outer core",
        displaced_interface="inner core",
        directions=tuple(directions),
    )
    report = output_dir / f"{job_name}_static_stiffness.json"
    report.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return result


def write_static_deck(
    setup: StaticStiffnessSetup,
    inp_file: Path,
    *,
    points: np.ndarray,
    element_blocks: list[ElementBlock],
    inner_nodes: np.ndarray,
    outer_nodes: np.ndarray,
    displacement_axis: int,
) -> None:
    """Write one CalculiX static displacement load case."""

    inp_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "*HEADING",
        f"Resonance AI static bushing stiffness - {setup.material.name}",
        "*NODE, NSET=NALL",
    ]
    for index, (x, y, z) in enumerate(points, start=1):
        lines.append(f"{index}, {x:.9g}, {y:.9g}, {z:.9g}")

    element_id = 1
    for block in element_blocks:
        lines.append(f"*ELEMENT, TYPE={block.calculix_type}, ELSET=EALL")
        for connectivity in block.connectivity:
            node_ids = ", ".join(str(int(node) + 1) for node in connectivity)
            lines.append(f"{element_id}, {node_ids}")
            element_id += 1

    material = setup.material
    lines.extend(
        [
            f"*MATERIAL, NAME={material.name.upper()}",
            "*ELASTIC",
            f"{material.youngs_modulus_mpa:.9g}, {material.poisson_ratio:.9g}",
            f"*SOLID SECTION, ELSET=EALL, MATERIAL={material.name.upper()}",
            "*NSET, NSET=NOUTER",
            *_chunk_ids(outer_nodes + 1),
            "*NSET, NSET=NINNER",
            *_chunk_ids(inner_nodes + 1),
            "*STEP",
            "*STATIC",
            "0.1, 1.0",
            "*BOUNDARY",
            "NOUTER, 1, 3, 0.0",
        ]
    )
    for axis in range(3):
        value = setup.displacement_mm if axis == displacement_axis else 0.0
        degree = axis + 1
        lines.append(f"NINNER, {degree}, {degree}, {value:.9g}")
    lines.extend(
        [
            "*NODE PRINT, NSET=NINNER",
            "RF",
            "*NODE FILE, NSET=NALL",
            "U, RF",
            "*EL FILE, ELSET=EALL",
            "S",
            "*END STEP",
        ]
    )
    inp_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_reaction_forces(dat_file: Path, *, set_name: str = "NINNER") -> np.ndarray:
    """Sum the final RF table for a named node set from a CalculiX DAT file."""

    text = Path(dat_file).read_text(encoding="utf-8", errors="replace")
    tables: list[list[np.ndarray]] = []
    current: list[np.ndarray] | None = None
    target = set_name.lower()

    for line in text.splitlines():
        lowered = line.lower()
        if target in lowered and ("force" in lowered or "rf" in lowered):
            current = []
            tables.append(current)
            continue
        if current is None:
            continue
        match = _REACTION_ROW.match(line)
        if match:
            current.append(
                np.asarray(
                    [float(value.replace("D", "E").replace("d", "e")) for value in match.groups()[1:]],
                    dtype=float,
                )
            )
        elif current and not line.strip():
            current = None

    populated = [table for table in tables if table]
    if not populated:
        raise ValueError(f"No reaction-force table for node set {set_name} was found in {dat_file.name}.")
    return np.vstack(populated[-1]).sum(axis=0)


def radial_interface_nodes(
    points: np.ndarray,
    element_blocks: list[ElementBlock],
    *,
    inner_length_mm: float | None = None,
    outer_length_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify inner and outer nodes on a structured mesh generated around Z."""

    used = np.unique(np.concatenate([block.connectivity.ravel() for block in element_blocks]))
    used_points = points[used]
    radii = np.linalg.norm(used_points[:, :2], axis=1)
    angles = np.mod(np.arctan2(used_points[:, 1], used_points[:, 0]), 2.0 * pi)
    z_values = used_points[:, 2]
    geometry_scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    key_tolerance = geometry_scale * 1e-7

    rays: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for node, radius, angle, z_value in zip(used, radii, angles, z_values, strict=True):
        key = (round(float(angle) / key_tolerance), round(float(z_value) / key_tolerance))
        rays.setdefault(key, []).append((int(node), float(radius)))

    inner: list[int] = []
    outer: list[int] = []
    for ray in rays.values():
        if len(ray) < 2:
            continue
        ordered = sorted(ray, key=lambda item: item[1])
        inner.append(ordered[0][0])
        outer.append(ordered[-1][0])

    inner_nodes = np.unique(np.asarray(inner, dtype=int))
    outer_nodes = np.unique(np.asarray(outer, dtype=int))
    if inner_nodes.size < 8 or outer_nodes.size < 8:
        raise ValueError(
            "Could not identify structured inner/outer bushing interfaces. "
            "Use a global or structured parametric bushing mesh."
        )

    inner_nodes = _filter_axial_interface(points, inner_nodes, inner_length_mm)
    outer_nodes = _filter_axial_interface(points, outer_nodes, outer_length_mm)
    overlap = np.intersect1d(inner_nodes, outer_nodes)
    if overlap.size:
        raise ValueError("Detected overlapping inner and outer interface nodes.")
    return inner_nodes, outer_nodes


def _filter_axial_interface(points: np.ndarray, nodes: np.ndarray, length_mm: float | None) -> np.ndarray:
    if length_mm is None or length_mm <= 0:
        return nodes
    z_values = points[nodes, 2]
    center = 0.5 * float(z_values.min() + z_values.max())
    full_span = float(z_values.max() - z_values.min())
    target = min(float(length_mm), full_span)
    tolerance = max(full_span * 1e-7, 1e-7)
    selected = nodes[np.abs(z_values - center) <= target * 0.5 + tolerance]
    if selected.size < 8:
        raise ValueError(f"Interface length {length_mm:g} mm selected too few mesh nodes.")
    return selected


def _read_solid_mesh(mesh_file: Path) -> tuple[np.ndarray, list[ElementBlock]]:
    import meshio

    mesh = meshio.read(str(mesh_file))
    points = np.asarray(mesh.points, dtype=float)
    element_blocks = _extract_solid_elements(mesh)
    if not element_blocks:
        raise ValueError("Mesh contains no CalculiX-supported tetrahedral or hexahedral solid elements.")
    return points, element_blocks


def _run_ccx(ccx: str, output_dir: Path, job_name: str, inp_file: Path) -> SolverRun:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run directional static bushing stiffness in CalculiX.")
    parser.add_argument("mesh_file", type=Path)
    parser.add_argument("--material", default="rubber")
    parser.add_argument("--displacement", type=float, default=1.0)
    parser.add_argument("--inner-length", type=float, default=None)
    parser.add_argument("--outer-length", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "static_stiffness")
    parser.add_argument("--name", default="bushing")
    args = parser.parse_args(argv)

    setup = StaticStiffnessSetup(
        mesh_file=args.mesh_file,
        material=resolve_material(args.material),
        displacement_mm=args.displacement,
        inner_interface_length_mm=args.inner_length,
        outer_interface_length_mm=args.outer_length,
    )
    try:
        result = run_static_stiffness(setup, args.output_dir, job_name=args.name)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
