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
    global_compatible: bool = False
    template_id: str = ""
    circumferential_divisions: int = 0
    radial_divisions: int = 0
    axial_divisions: int = 0

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
    mesh_mode: str = "structured",
    template: dict[str, Any] | None = None,
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
    bore_spec = _bore_spec(geometry)
    mode = str(mesh_mode or "structured").strip().lower()
    global_mode = mode in {"global", "global_template", "dataset"}
    if global_mode:
        template_config = _global_template_config(template or geometry, slot_spec)
        radial_count = template_config["radial"]
        axial_count = template_config["axial"]
        circum_count = template_config["circumferential"]
    else:
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
            for circum in range(circum_count):
                angle = 2.0 * pi * circum / circum_count
                radius = _node_radius(
                    inner_radius=inner_radius,
                    outer_radius=outer_radius,
                    wall=wall,
                    radial_fraction=radial / radial_count,
                    angle_rad=angle,
                    z=z,
                    slot=slot_spec,
                    bore=bore_spec,
                    global_mode=global_mode,
                )
                node_index[(axial, radial, circum)] = len(points)
                points.append((radius * cos(angle), radius * sin(angle), z))

    cells: list[list[int]] = []
    region_ids: list[int] = []
    for axial in range(axial_count):
        for radial in range(radial_count):
            for circum in range(circum_count):
                if not global_mode and _is_slot_cell(
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
        mesh_kind=_mesh_kind(global_mode, slot_spec),
        global_compatible=global_mode,
        template_id=_template_id(circum_count, radial_count, axial_count, slot_spec) if global_mode else "",
        circumferential_divisions=circum_count,
        radial_divisions=radial_count,
        axial_divisions=axial_count,
    )


def _global_template_config(template: dict[str, Any], slot: dict[str, Any]) -> dict[str, int]:
    circum = _bounded_int(
        template.get("circumferential_divisions", template.get("global_circumferential_divisions")),
        fallback=96,
        minimum=24,
        maximum=192,
    )
    radial = _bounded_int(
        template.get("radial_divisions", template.get("global_radial_divisions")),
        fallback=8,
        minimum=2,
        maximum=32,
    )
    axial = _bounded_int(
        template.get("axial_divisions", template.get("global_axial_divisions")),
        fallback=16,
        minimum=3,
        maximum=64,
    )
    if circum % 4:
        circum += 4 - (circum % 4)
    slot_count = int(slot.get("count") or 0)
    if slot_count > 0 and circum < slot_count * 6:
        circum = slot_count * 6
        if circum % 4:
            circum += 4 - (circum % 4)
    return {"circumferential": circum, "radial": radial, "axial": axial}


def _bounded_int(value: Any, *, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _node_radius(
    *,
    inner_radius: float,
    outer_radius: float,
    wall: float,
    radial_fraction: float,
    angle_rad: float,
    z: float,
    slot: dict[str, Any],
    bore: dict[str, Any],
    global_mode: bool,
) -> float:
    local_inner = _bore_radius(inner_radius, angle_rad, bore)
    if not global_mode or int(slot.get("count") or 0) <= 0:
        return local_inner + max(0.1, outer_radius - local_inner) * radial_fraction
    influence = _slot_node_influence(angle_rad * 180.0 / pi, z, slot)
    max_depth = max(0.0, min(float(slot["depth_mm"]), wall * 0.94))
    local_outer = max(local_inner + max(wall * 0.06, 0.5), outer_radius - max_depth * influence)
    return local_inner + (local_outer - local_inner) * radial_fraction


def _bore_spec(geometry: dict[str, Any]) -> dict[str, Any]:
    shape = str(geometry.get("bore_shape") or "round").strip().lower()
    if shape not in {"round", "rounded_square"}:
        shape = "round"
    return {
        "shape": shape,
        "corner_radius_mm": max(0.0, _number(geometry.get("bore_corner_radius_mm"), 0.0)),
    }


def _bore_radius(inner_radius: float, angle_rad: float, bore: dict[str, Any]) -> float:
    if bore.get("shape") != "rounded_square":
        return inner_radius
    corner = max(0.0, float(bore.get("corner_radius_mm") or 0.0))
    exponent = 5.0 if corner <= 0 else max(3.2, min(8.0, inner_radius / max(corner, 1e-6)))
    denom = (abs(cos(angle_rad)) ** exponent + abs(sin(angle_rad)) ** exponent) ** (1.0 / exponent)
    if denom <= 1e-9:
        return inner_radius
    return inner_radius / denom


def _slot_node_influence(angle_deg: float, z: float, slot: dict[str, Any]) -> float:
    count = int(slot.get("count") or 0)
    if count <= 0:
        return 0.0
    half_width = max(float(slot["width_deg"]) / 2.0, 0.5)
    pitch = 360.0 / count
    angular = 0.0
    for index in range(count):
        center = float(slot["start_angle_deg"]) + pitch * index
        distance = _angle_distance_deg(angle_deg, center)
        if distance <= half_width:
            t = distance / half_width
            angular = max(angular, 0.5 * (1.0 + cos(pi * t)))
    if angular <= 0.0:
        return 0.0
    if slot.get("axial_mode") != "centered":
        return angular
    half_height = max(float(slot["axial_height_mm"]) / 2.0, 1e-6)
    distance_z = abs(z)
    if distance_z >= half_height:
        return 0.0
    axial = 0.5 * (1.0 + cos(pi * distance_z / half_height))
    return angular * axial


def _mesh_kind(global_mode: bool, slot: dict[str, Any]) -> str:
    if global_mode:
        return "global_slotted_bushing_hex" if slot.get("count") else "global_bushing_hex"
    return "mapped_slotted_bushing_hex" if slot.get("count") else "mapped_bushing_hex"


def _template_id(circum_count: int, radial_count: int, axial_count: int, slot: dict[str, Any]) -> str:
    slot_count = int(slot.get("count") or 0)
    family = f"{slot_count}-slot" if slot_count else "plain"
    return f"{family}-c{circum_count}-r{radial_count}-a{axial_count}"


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
