"""Structured hex/swept meshing for suitable STEP solids.

This is intentionally conservative. Imported CAD can only be hex-meshed
automatically when Gmsh can discover a transfinite/recombined structure. If it
cannot produce a pure hexahedral volume mesh, callers should report that the CAD
needs sweepable/block decomposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class StructuredHexUnavailable(RuntimeError):
    """Raised when Gmsh cannot create a pure hexahedral mesh."""


@dataclass(frozen=True)
class HexMeshResult:
    """Summary of a generated structured mesh."""

    step_file: Path
    mesh_file: Path
    node_count: int
    element_counts: dict[str, int]
    min_edge_mm: float
    max_edge_mm: float

    @property
    def hex_count(self) -> int:
        return sum(count for name, count in self.element_counts.items() if name.startswith("hexahedron"))

    @property
    def non_hex_volume_count(self) -> int:
        return sum(
            count
            for name, count in self.element_counts.items()
            if name.startswith(("tetra", "wedge", "pyramid"))
        )


def step_to_swept_hex_mesh(
    step_file: Path,
    output_file: Path,
    *,
    target_size_mm: float | None = None,
    min_size_mm: float | None = None,
    verbose: bool = False,
) -> HexMeshResult:
    """Mesh a STEP solid with pure recombined/transfinite hexahedra."""

    import gmsh

    step_file = Path(step_file).resolve()
    output_file = Path(output_file).resolve()
    if not step_file.exists():
        raise FileNotFoundError(f"STEP file not found: {step_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(step_file.stem + "_hex")
        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()

        diagonal = _bounding_box_diagonal(gmsh)
        size = target_size_mm if target_size_mm and target_size_mm > 0 else max(diagonal / 14.0, 1e-3)
        lower = min_size_mm if min_size_mm and min_size_mm > 0 else size / 4.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lower)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads.
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal 3D, better for structured attempts.

        # Let Gmsh infer transfinite curves/surfaces/volumes where the imported
        # topology allows it, then ask each surface to recombine into quads.
        try:
            gmsh.model.mesh.setTransfiniteAutomatic()
        except Exception:
            pass
        for _, tag in gmsh.model.getEntities(2):
            try:
                gmsh.model.mesh.setRecombine(2, tag)
            except Exception:
                pass

        gmsh.model.mesh.generate(3)
        gmsh.write(str(output_file))

        node_count, element_counts, min_edge, max_edge = _mesh_stats(gmsh)
        result = HexMeshResult(
            step_file=step_file,
            mesh_file=output_file,
            node_count=node_count,
            element_counts=element_counts,
            min_edge_mm=min_edge,
            max_edge_mm=max_edge,
        )
        if result.hex_count <= 0:
            raise StructuredHexUnavailable("Gmsh did not produce hexahedral cells for this topology.")
        if result.non_hex_volume_count > 0:
            raise StructuredHexUnavailable(
                "Gmsh produced a mixed volume mesh instead of pure hex cells: "
                + ", ".join(f"{name}={count}" for name, count in result.element_counts.items())
            )
    finally:
        gmsh.finalize()

    return result


def _bounding_box_diagonal(gmsh_module) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh_module.model.getBoundingBox(-1, -1)
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _mesh_stats(gmsh_module) -> tuple[int, dict[str, int], float, float]:
    node_tags, coords, _ = gmsh_module.model.mesh.getNodes()
    node_count = len(node_tags)
    coord_by_tag: dict[int, tuple[float, float, float]] = {}
    for index, tag in enumerate(node_tags):
        base = index * 3
        coord_by_tag[int(tag)] = (coords[base], coords[base + 1], coords[base + 2])

    element_counts: dict[str, int] = {}
    min_edge = float("inf")
    max_edge = 0.0
    for elem_type, name, corner_count, edge_pairs in _ELEMENTS:
        try:
            elem_tags, node_tags_flat = gmsh_module.model.mesh.getElementsByType(elem_type)
        except Exception:
            continue
        if len(elem_tags) == 0:
            continue
        nodes_per = len(node_tags_flat) // len(elem_tags)
        element_counts[name] = element_counts.get(name, 0) + len(elem_tags)
        for elem_index in range(len(elem_tags)):
            corners = node_tags_flat[elem_index * nodes_per : elem_index * nodes_per + corner_count]
            points = [coord_by_tag[int(tag)] for tag in corners]
            for i, j in edge_pairs:
                edge = _distance(points[i], points[j])
                min_edge = min(min_edge, edge)
                max_edge = max(max_edge, edge)

    if min_edge == float("inf"):
        min_edge = 0.0
    return node_count, element_counts, min_edge, max_edge


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)


_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_HEX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
_WEDGE_EDGES = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5))
_PYRAMID_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4))

# Gmsh type id, meshio-like name, number of corner nodes, edge pairs.
_ELEMENTS = (
    (4, "tetra", 4, _TET_EDGES),
    (11, "tetra10", 4, _TET_EDGES),
    (5, "hexahedron", 8, _HEX_EDGES),
    (12, "hexahedron27", 8, _HEX_EDGES),
    (17, "hexahedron20", 8, _HEX_EDGES),
    (6, "wedge", 6, _WEDGE_EDGES),
    (13, "wedge18", 6, _WEDGE_EDGES),
    (7, "pyramid", 5, _PYRAMID_EDGES),
    (14, "pyramid14", 5, _PYRAMID_EDGES),
)
