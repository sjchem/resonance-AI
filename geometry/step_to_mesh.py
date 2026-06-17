"""Convert a STEP solid into a volumetric tetrahedral mesh with Gmsh.

This is the meshing stage of the local simulation workflow:

    STEP  ->  (this module)  ->  volume mesh (.msh / .inp / .vtk)

The output is a second-order tetrahedral mesh suitable for a CalculiX modal
or static analysis. The module exposes a small Python API plus a CLI so it can
be used directly or from the end-to-end pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class MeshResult:
    """Summary of a generated volume mesh."""

    step_file: Path
    mesh_file: Path
    node_count: int
    tetra_count: int
    min_edge_mm: float
    max_edge_mm: float

    def summary(self) -> str:
        return (
            f"Mesh: {self.mesh_file.name} | nodes={self.node_count} "
            f"tetra={self.tetra_count} | element size "
            f"{self.min_edge_mm:.3f}-{self.max_edge_mm:.3f} mm"
        )


def step_to_mesh(
    step_file: Path,
    output_file: Path,
    *,
    target_size_mm: float | None = None,
    min_size_mm: float | None = None,
    second_order: bool = True,
    optimize: bool = True,
    verbose: bool = False,
) -> MeshResult:
    """Mesh a STEP solid into a tetrahedral volume mesh.

    Parameters
    ----------
    step_file:
        Source STEP (.step / .stp) solid model.
    output_file:
        Destination mesh. The extension selects the writer
        (.msh for Gmsh, .inp for CalculiX/Abaqus, .vtk for ParaView).
    target_size_mm:
        Desired characteristic element size. Defaults to a fraction of the
        bounding-box diagonal when omitted.
    min_size_mm:
        Lower bound for element size. Defaults to ``target_size_mm`` / 4.
    second_order:
        Emit 10-node quadratic tetrahedra (recommended for stress/modal).
    optimize:
        Run Gmsh mesh optimization (Netgen) for better element quality.
    """

    import gmsh

    step_file = Path(step_file).resolve()
    output_file = Path(output_file).resolve()
    if not step_file.exists():
        raise FileNotFoundError(f"STEP file not found: {step_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(step_file.stem)
        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()

        diagonal = _bounding_box_diagonal(gmsh)
        size = target_size_mm if target_size_mm and target_size_mm > 0 else max(diagonal / 18.0, 1e-3)
        lower = min_size_mm if min_size_mm and min_size_mm > 0 else size / 4.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lower)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        if second_order:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

        gmsh.model.mesh.generate(3)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")

        gmsh.write(str(output_file))

        node_count, tetra_count, min_edge, max_edge = _mesh_stats(gmsh)
    finally:
        gmsh.finalize()

    return MeshResult(
        step_file=step_file,
        mesh_file=output_file,
        node_count=node_count,
        tetra_count=tetra_count,
        min_edge_mm=min_edge,
        max_edge_mm=max_edge,
    )


def _bounding_box_diagonal(gmsh_module) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh_module.model.getBoundingBox(-1, -1)
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _mesh_stats(gmsh_module) -> tuple[int, int, float, float]:
    node_tags, coords, _ = gmsh_module.model.mesh.getNodes()
    node_count = len(node_tags)

    coord_by_tag: dict[int, tuple[float, float, float]] = {}
    for index, tag in enumerate(node_tags):
        base = index * 3
        coord_by_tag[int(tag)] = (coords[base], coords[base + 1], coords[base + 2])

    # Element type 4 = 4-node tetra, 11 = 10-node tetra.
    tetra_count = 0
    min_edge = float("inf")
    max_edge = 0.0
    for tet_type in (4, 11):
        try:
            elem_tags, node_tags_flat = gmsh_module.model.mesh.getElementsByType(tet_type)
        except Exception:
            continue
        if len(elem_tags) == 0:
            continue
        nodes_per = 4 if tet_type == 4 else 10
        tetra_count += len(elem_tags)
        for elem_index in range(len(elem_tags)):
            corners = node_tags_flat[elem_index * nodes_per : elem_index * nodes_per + 4]
            points = [coord_by_tag[int(tag)] for tag in corners]
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = _distance(points[i], points[j])
                    min_edge = min(min_edge, edge)
                    max_edge = max(max_edge, edge)

    if min_edge == float("inf"):
        min_edge = 0.0
    return node_count, tetra_count, min_edge, max_edge


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mesh a STEP solid into a tetrahedral volume mesh.")
    parser.add_argument("step_file", type=Path, help="Input STEP/STP file.")
    parser.add_argument("output_file", type=Path, help="Output mesh (.msh, .inp, or .vtk).")
    parser.add_argument("--size", type=float, default=None, help="Target element size in mm.")
    parser.add_argument("--min-size", type=float, default=None, help="Minimum element size in mm.")
    parser.add_argument("--first-order", action="store_true", help="Emit linear tetrahedra (default is quadratic).")
    parser.add_argument("--no-optimize", action="store_true", help="Skip Netgen mesh optimization.")
    parser.add_argument("--verbose", action="store_true", help="Print Gmsh progress output.")
    args = parser.parse_args(argv)

    try:
        result = step_to_mesh(
            step_file=args.step_file,
            output_file=args.output_file,
            target_size_mm=args.size,
            min_size_mm=args.min_size,
            second_order=not args.first_order,
            optimize=not args.no_optimize,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ImportError:
        print("gmsh is required. Install it with: pip install gmsh", file=sys.stderr)
        return 3

    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
