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
    mesh_kind: str = "mapped_bushing_hex"

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
    """Write a pure C3D8-compatible bushing mesh from structured JSON.

    Slots are represented by removing aligned blocks from the polar hex grid.
    That keeps every remaining volume element a hexahedron while creating real
    slot faces on the mesh boundary.
    """

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
    inner_sleeve_thickness = _inner_sleeve_thickness(geometry, inner_diameter)
    outer_sleeve_thickness = _outer_sleeve_thickness(geometry)
    slot_spec = _slot_spec(geometry, wall, height)
    size = target_size_mm if target_size_mm and target_size_mm > 0 else max(min(wall / 3.0, height / 8.0), 1.0)
    radial_count = max(2, min(12, int(ceil(wall / size))))
    axial_count = max(3, min(40, int(ceil(height / size))))
    circum_count = max(16, min(96, int(ceil((2.0 * pi * outer_radius) / size))))
    if circum_count % 4:
        circum_count += 4 - (circum_count % 4)
    if slot_spec["count"]:
        circum_count = _align_circumferential_count(circum_count, slot_spec["count"])

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
    region_ids: list[int] = []
    for axial in range(axial_count):
        for radial in range(radial_count):
            for circum in range(circum_count):
                if _is_slot_cell(
                    axial=axial,
                    radial=radial,
                    circum=circum,
                    axial_count=axial_count,
                    radial_count=radial_count,
                    circum_count=circum_count,
                    inner_radius=inner_radius,
                    wall=wall,
                    height=height,
                    slot=slot_spec,
                ):
                    continue
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
                radial_center = inner_radius + wall * (radial + 0.5) / radial_count
                region_ids.append(_material_region_id(radial_center, inner_radius, outer_radius, inner_sleeve_thickness, outer_sleeve_thickness))
    if not cells:
        raise ValueError("Slot parameters removed every bushing hex cell; reduce slot count, width, or depth.")

    point_array = np.asarray(points, dtype=float)
    cell_array = np.asarray(cells, dtype=int)
    meshio.write(
        str(output_file),
        meshio.Mesh(
            point_array,
            [("hexahedron", cell_array)],
            cell_data={"material_region": [np.asarray(region_ids, dtype=int)]},
        ),
    )

    min_edge, max_edge = _edge_range(point_array, cell_array)
    return BushingHexMeshResult(
        mesh_file=output_file,
        node_count=len(points),
        element_counts={"hexahedron": len(cells)},
        min_edge_mm=min_edge,
        max_edge_mm=max_edge,
        mesh_kind="mapped_slotted_bushing_hex" if slot_spec["count"] else "mapped_bushing_hex",
    )


def _slot_spec(geometry: dict[str, Any], wall: float, height: float) -> dict[str, Any]:
    count = max(0, min(24, int(round(_number(geometry.get("slot_count"), 0.0)))))
    if count == 0:
        return {"count": 0}

    pitch = 360.0 / count
    width_deg = _positive(geometry.get("slot_width_deg"), min(18.0, pitch * 0.35))
    width_deg = max(1.0, min(width_deg, pitch * 0.82))
    depth = _positive(geometry.get("slot_depth_mm"), max(wall * 0.45, 0.5))
    radial_mode = str(geometry.get("slot_radial_mode") or "outer").strip().lower()
    if radial_mode not in {"outer", "through_wall"}:
        radial_mode = "outer"
    depth = wall if radial_mode == "through_wall" else max(0.1, min(depth, wall * 0.96))

    axial_mode = str(geometry.get("slot_axial_mode") or "through").strip().lower()
    if axial_mode not in {"through", "centered"}:
        axial_mode = "through"
    axial_height = _positive(geometry.get("slot_axial_height_mm"), height * 0.5)
    axial_height = height if axial_mode == "through" else max(0.1, min(axial_height, height * 0.98))

    return {
        "count": count,
        "width_deg": width_deg,
        "depth_mm": depth,
        "start_angle_deg": _number(geometry.get("slot_start_angle_deg"), 0.0),
        "radial_mode": radial_mode,
        "axial_mode": axial_mode,
        "axial_height_mm": axial_height,
    }


def _align_circumferential_count(circum_count: int, slot_count: int) -> int:
    if slot_count <= 0:
        return circum_count
    target = max(circum_count, slot_count * 8)
    aligned = int(ceil(target / slot_count) * slot_count)
    if aligned % 4:
        aligned += 4 - (aligned % 4)
    return min(144, max(slot_count * 8, aligned))


def _is_slot_cell(
    *,
    axial: int,
    radial: int,
    circum: int,
    axial_count: int,
    radial_count: int,
    circum_count: int,
    inner_radius: float,
    wall: float,
    height: float,
    slot: dict[str, Any],
) -> bool:
    count = int(slot.get("count") or 0)
    if count <= 0:
        return False

    radial_center = inner_radius + wall * (radial + 0.5) / radial_count
    slot_inner_radius = inner_radius if slot.get("radial_mode") == "through_wall" else inner_radius + max(0.0, wall - float(slot["depth_mm"]))
    if radial_center < slot_inner_radius:
        return False

    if slot.get("axial_mode") == "centered":
        z_center = -height / 2.0 + height * (axial + 0.5) / axial_count
        if abs(z_center) > float(slot["axial_height_mm"]) / 2.0:
            return False

    angle_deg = (circum + 0.5) * 360.0 / circum_count
    half_width = max(float(slot["width_deg"]) / 2.0, 180.0 / circum_count)
    pitch = 360.0 / count
    for index in range(count):
        center = float(slot["start_angle_deg"]) + pitch * index
        if _angle_distance_deg(angle_deg, center) <= half_width:
            return True
    return False


def _angle_distance_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _number(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed == parsed else fallback


def _positive(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _inner_sleeve_thickness(geometry: dict[str, Any], inner_diameter: float) -> float:
    explicit = _number(geometry.get("inner_sleeve_thickness_mm"), -1.0)
    if explicit >= 0:
        return explicit
    sleeve_diameter = _number(geometry.get("inner_sleeve_diameter_mm"), inner_diameter)
    return max(0.0, (sleeve_diameter - inner_diameter) / 2.0)


def _outer_sleeve_thickness(geometry: dict[str, Any]) -> float:
    return max(0.0, _number(geometry.get("outer_sleeve_thickness_mm", geometry.get("metal_sleeve_thickness_mm")), 0.0))


def _material_region_id(
    radial_center: float,
    inner_radius: float,
    outer_radius: float,
    inner_sleeve_thickness: float,
    outer_sleeve_thickness: float,
) -> int:
    if inner_sleeve_thickness > 0 and radial_center <= inner_radius + inner_sleeve_thickness:
        return 2
    if outer_sleeve_thickness > 0 and radial_center >= outer_radius - outer_sleeve_thickness:
        return 3
    return 1


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
