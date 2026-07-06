"""Mapped hexahedral mesh for parametric rubber bushings."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, pi, sin
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BushingHexMeshResult:
    mesh_file: Path
    node_count: int
    element_counts: dict[str, int]
    min_edge_mm: float
    max_edge_mm: float

    @property
    def hex_count(self) -> int:
        return self.element_counts.get("hexahedron", 0)

    @property
    def non_hex_volume_count(self) -> int:
        return 0


def generate_bushing_hex_mesh(
    intent: dict[str, Any],
    output_file: Path | str,
    *,
    target_size_mm: float | None = None,
) -> BushingHexMeshResult:
    """Write a pure C3D8-compatible annular bushing mesh from structured JSON."""

    import meshio
    import numpy as np

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    geometry = intent.get("geometry", intent) if isinstance(intent, dict) else {}
    if not isinstance(geometry, dict):
        raise ValueError("Bushing hex mesh requires structured geometry JSON.")

    outer_diameter = _positive(geometry.get("outer_diameter_mm"), 76.0)
    inner_diameter = _positive(geometry.get("inner_diameter_mm"), 28.0)
    height = _positive(geometry.get("height_mm"), 40.0)
    if inner_diameter >= outer_diameter:
        raise ValueError("Inner diameter must be smaller than outer diameter for bushing hex mesh.")

    outer_radius = outer_diameter / 2.0
    inner_radius = inner_diameter / 2.0
    wall = outer_radius - inner_radius
    size = target_size_mm if target_size_mm and target_size_mm > 0 else max(min(wall / 3.0, height / 8.0), 1.0)
    radial_count = max(2, min(12, int(ceil(wall / size))))
    axial_count = max(3, min(40, int(ceil(height / size))))
    circum_count = max(16, min(96, int(ceil((2.0 * pi * outer_radius) / size))))
    if circum_count % 4:
        circum_count += 4 - (circum_count % 4)

    points: list[tuple[float, float, float]] = []
    node_index: dict[tuple[int, int, int], int] = {}
    for axial in range(axial_count + 1):
        z = -height / 2.0 + height * axial / axial_count
        for radial in range(radial_count + 1):
            radius = inner_radius + wall * radial / radial_count
            for circum in range(circum_count):
                angle = 2.0 * pi * circum / circum_count
                node_index[(axial, radial, circum)] = len(points)
                points.append((radius * cos(angle), radius * sin(angle), z))

    cells: list[list[int]] = []
    for axial in range(axial_count):
        for radial in range(radial_count):
            for circum in range(circum_count):
                next_circum = (circum + 1) % circum_count
                cells.append(
                    [
                        node_index[(axial, radial, circum)],
                        node_index[(axial, radial + 1, circum)],
                        node_index[(axial, radial + 1, next_circum)],
                        node_index[(axial, radial, next_circum)],
                        node_index[(axial + 1, radial, circum)],
                        node_index[(axial + 1, radial + 1, circum)],
                        node_index[(axial + 1, radial + 1, next_circum)],
                        node_index[(axial + 1, radial, next_circum)],
                    ]
                )

    point_array = np.asarray(points, dtype=float)
    cell_array = np.asarray(cells, dtype=int)
    meshio.write(str(output_file), meshio.Mesh(point_array, [("hexahedron", cell_array)]))

    min_edge, max_edge = _edge_range(point_array, cell_array)
    return BushingHexMeshResult(
        mesh_file=output_file,
        node_count=len(points),
        element_counts={"hexahedron": len(cells)},
        min_edge_mm=min_edge,
        max_edge_mm=max_edge,
    )


def _positive(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _edge_range(points, cells) -> tuple[float, float]:
    import numpy as np

    edge_pairs = (
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
    min_edge = float("inf")
    max_edge = 0.0
    for cell in cells:
        for left, right in edge_pairs:
            edge = float(np.linalg.norm(points[cell[left]] - points[cell[right]]))
            min_edge = min(min_edge, edge)
            max_edge = max(max_edge, edge)
    if min_edge == float("inf"):
        min_edge = 0.0
    return min_edge, max_edge
