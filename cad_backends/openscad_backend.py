"""OpenSCAD backend for basic bushing exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


class OpenScadUnavailable(RuntimeError):
    """Raised when no configured OpenSCAD runner is available."""


class OpenScadExportError(RuntimeError):
    """Raised when OpenSCAD is available but cannot export the requested model."""


@dataclass(frozen=True)
class OpenScadBushing:
    scad_text: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class OpenScadExportResult:
    stl_path: Path | None
    png_path: Path | None
    warnings: list[str]


def generate_openscad_bushing(spec: dict[str, Any]) -> OpenScadBushing:
    """Generate OpenSCAD source and normalized parameters for a bushing spec."""

    part_type = str(spec.get("part_type", "bushing") or "bushing").lower()
    if part_type not in {"bushing", "rubber_mount"}:
        raise ValueError("OpenSCAD export currently supports bushing and rubber_mount parts only.")

    geometry = _geometry(spec)
    outer_diameter = _positive(geometry, "outer_diameter_mm", _positive(geometry, "length_mm", 60.0))
    inner_diameter = _positive(geometry, "inner_diameter_mm", _positive(geometry, "hole_diameter_mm", outer_diameter * 0.35))
    height = _positive(geometry, "height_mm", _positive(geometry, "thickness_mm", 40.0))
    if inner_diameter >= outer_diameter:
        raise ValueError("OpenSCAD bushing export requires inner_diameter_mm to be smaller than outer_diameter_mm.")

    wall = (outer_diameter - inner_diameter) / 2.0
    max_edge_break = max(0.0, min(height * 0.45, wall * 0.45))
    chamfer = min(max(0.0, _number(geometry.get("chamfer_mm"), 0.0)), max_edge_break)
    outer_sleeve_raw = geometry.get("outer_sleeve_thickness_mm", geometry.get("metal_sleeve_thickness_mm"))
    outer_sleeve_thickness = min(max(0.0, _number(outer_sleeve_raw, 0.0)), wall * 0.45)
    inner_sleeve_thickness = min(max(0.0, _number(geometry.get("inner_sleeve_thickness_mm"), 0.0)), wall * 0.45)
    inner_sleeve_length = min(height, max(0.0, _number(geometry.get("inner_sleeve_length_mm"), height)))
    outer_sleeve_length = min(height, max(0.0, _number(geometry.get("outer_core_length_mm"), height)))
    flange = str(geometry.get("flange", "none") or "none").lower()
    if flange not in {"none", "top", "bottom", "both"}:
        flange = "none"
    flange_diameter = _number(geometry.get("flange_diameter_mm"), 0.0) if flange != "none" else 0.0
    flange_thickness = _number(geometry.get("flange_thickness_mm"), 0.0) if flange != "none" else 0.0
    if flange_diameter <= outer_diameter or flange_thickness <= 0:
        flange = "none"
        flange_diameter = 0.0
        flange_thickness = 0.0
    bore_shape = str(geometry.get("bore_shape", "round") or "round").lower()
    if bore_shape not in {"round", "rounded_square"}:
        bore_shape = "round"
    bore_corner_radius = max(0.0, _number(geometry.get("bore_corner_radius_mm"), 0.0)) if bore_shape == "rounded_square" else 0.0
    slot_count = max(0, min(24, int(round(_number(geometry.get("slot_count"), 0.0)))))
    slot_width_deg = max(0.0, _number(geometry.get("slot_width_deg"), 0.0)) if slot_count else 0.0
    slot_depth = max(0.0, _number(geometry.get("slot_depth_mm"), 0.0)) if slot_count else 0.0
    slot_start_angle = _number(geometry.get("slot_start_angle_deg"), 0.0)
    slot_radial_mode = str(geometry.get("slot_radial_mode", "outer") or "outer").lower()
    if slot_radial_mode not in {"outer", "through_wall"}:
        slot_radial_mode = "outer"
    slot_axial_mode = str(geometry.get("slot_axial_mode", "through") or "through").lower()
    if slot_axial_mode not in {"through", "centered"}:
        slot_axial_mode = "through"
    slot_axial_height = height if slot_axial_mode == "through" else min(height, max(0.1, _number(geometry.get("slot_axial_height_mm"), height * 0.5)))

    parameters = {
        "cad_engine": "openscad",
        "part_type": part_type,
        "outer_diameter_mm": outer_diameter,
        "inner_diameter_mm": inner_diameter,
        "height_mm": height,
        "chamfer_mm": chamfer,
        "outer_sleeve_thickness_mm": outer_sleeve_thickness,
        "metal_sleeve_thickness_mm": outer_sleeve_thickness,
        "inner_sleeve_thickness_mm": inner_sleeve_thickness,
        "inner_sleeve_length_mm": inner_sleeve_length,
        "outer_sleeve_length_mm": outer_sleeve_length,
        "flange": flange,
        "flange_diameter_mm": flange_diameter,
        "flange_thickness_mm": flange_thickness,
        "bore_shape": bore_shape,
        "bore_corner_radius_mm": bore_corner_radius,
        "slot_count": slot_count,
        "slot_width_deg": slot_width_deg,
        "slot_depth_mm": slot_depth,
        "slot_start_angle_deg": slot_start_angle,
        "slot_radial_mode": slot_radial_mode,
        "slot_axial_mode": slot_axial_mode,
        "slot_axial_height_mm": slot_axial_height,
        "source_geometry": geometry,
    }
    return OpenScadBushing(scad_text=_render_bushing_scad(parameters), parameters=parameters)


def run_openscad_export(scad_path: Path | str, output_dir: Path | str) -> OpenScadExportResult:
    """Run OpenSCAD by local CLI or configured Docker image to export STL/PNG."""

    scad_path = Path(scad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / f"{scad_path.stem}.stl"
    png_path = output_dir / f"{scad_path.stem}.png"
    warnings: list[str] = []

    runner = _runner()
    if runner[0] == "cli":
        binary = runner[1]
        _run([binary, "-o", str(stl_path), str(scad_path)], "OpenSCAD STL export failed")
        try:
            _run(
                [binary, "-o", str(png_path), "--imgsize=1200,900", "--viewall", "--autocenter", str(scad_path)],
                "OpenSCAD PNG preview export failed",
            )
        except OpenScadExportError as exc:
            warnings.append(str(exc))
    else:
        image = runner[1]
        docker = runner[2]
        mount = f"{output_dir.resolve()}:/work"
        scad_name = scad_path.name
        stl_name = stl_path.name
        png_name = png_path.name
        _run(
            [docker, "run", "--rm", "-v", mount, "-w", "/work", image, "openscad", "-o", stl_name, scad_name],
            "OpenSCAD Docker STL export failed",
        )
        try:
            _run(
                [
                    docker,
                    "run",
                    "--rm",
                    "-v",
                    mount,
                    "-w",
                    "/work",
                    image,
                    "openscad",
                    "-o",
                    png_name,
                    "--imgsize=1200,900",
                    "--viewall",
                    "--autocenter",
                    scad_name,
                ],
                "OpenSCAD Docker PNG preview export failed",
            )
        except OpenScadExportError as exc:
            warnings.append(str(exc))

    return OpenScadExportResult(
        stl_path=stl_path if stl_path.exists() else None,
        png_path=png_path if png_path.exists() else None,
        warnings=warnings,
    )


def write_parameters_json(path: Path | str, parameters: dict[str, Any]) -> Path:
    """Write normalized OpenSCAD export parameters."""

    target = Path(path)
    target.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _geometry(spec: dict[str, Any]) -> dict[str, Any]:
    geometry = spec.get("geometry", spec)
    if hasattr(geometry, "model_dump"):
        geometry = geometry.model_dump()
    if not isinstance(geometry, dict):
        raise ValueError("OpenSCAD export requires a bushing parameter JSON object.")
    return dict(geometry)


def _positive(geometry: dict[str, Any], key: str, fallback: float) -> float:
    value = _number(geometry.get(key), fallback)
    return value if value > 0 else fallback


def _number(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed == parsed else fallback


def _runner() -> tuple[str, str] | tuple[str, str, str]:
    configured_bin = os.environ.get("OPENSCAD_BIN")
    binary = configured_bin or shutil.which("openscad")
    if binary:
        return ("cli", binary)

    image = os.environ.get("OPENSCAD_DOCKER_IMAGE")
    if image:
        docker = shutil.which("docker")
        if not docker:
            raise OpenScadUnavailable(
                "OpenSCAD Docker export is configured, but Docker is not available. "
                "Install Docker or set OPENSCAD_BIN to an OpenSCAD executable."
            )
        return ("docker", image, docker)

    raise OpenScadUnavailable(
        "OpenSCAD STL/PNG export needs OpenSCAD installed. Set OPENSCAD_BIN, install "
        "openscad on PATH, or set OPENSCAD_DOCKER_IMAGE for a Docker-based runner."
    )


def _run(command: list[str], failure_prefix: str) -> None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise OpenScadUnavailable(f"OpenSCAD runner was not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise OpenScadExportError(f"{failure_prefix}: command timed out after 120 seconds") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "OpenSCAD returned a non-zero exit code.").strip()
        raise OpenScadExportError(f"{failure_prefix}: {detail}")


def _render_bushing_scad(parameters: dict[str, Any]) -> str:
    od = parameters["outer_diameter_mm"]
    inner_diameter = parameters["inner_diameter_mm"]
    height = parameters["height_mm"]
    chamfer = parameters["chamfer_mm"]
    outer_sleeve = parameters["metal_sleeve_thickness_mm"]
    inner_sleeve = parameters["inner_sleeve_thickness_mm"]
    outer_sleeve_length = parameters["outer_sleeve_length_mm"]
    inner_sleeve_length = parameters["inner_sleeve_length_mm"]
    flange = parameters["flange"]
    flange_diameter = parameters["flange_diameter_mm"]
    flange_thickness = parameters["flange_thickness_mm"]
    bore_shape = parameters["bore_shape"]
    bore_corner_radius = parameters["bore_corner_radius_mm"]
    slot_count = parameters["slot_count"]
    slot_width_deg = parameters["slot_width_deg"]
    slot_depth = parameters["slot_depth_mm"]
    slot_start_angle = parameters["slot_start_angle_deg"]
    slot_radial_mode = parameters["slot_radial_mode"]
    slot_axial_height = parameters["slot_axial_height_mm"]
    ro = od / 2.0
    ri = inner_diameter / 2.0
    flange_radius = flange_diameter / 2.0
    rubber_outer = max(ri + 0.1, ro - outer_sleeve)
    rubber_inner = min(rubber_outer - 0.1, ri + inner_sleeve)

    return f"""// Generated by Resonance AI OpenSCAD backend.
// Units: millimeters.
$fn = 96;

outer_diameter = {_scad_num(od)};
inner_diameter = {_scad_num(inner_diameter)};
height = {_scad_num(height)};
chamfer = {_scad_num(chamfer)};
outer_sleeve_thickness = {_scad_num(outer_sleeve)};
inner_sleeve_thickness = {_scad_num(inner_sleeve)};
outer_sleeve_length = {_scad_num(outer_sleeve_length)};
inner_sleeve_length = {_scad_num(inner_sleeve_length)};
flange_mode = "{flange}";
flange_diameter = {_scad_num(flange_diameter)};
flange_thickness = {_scad_num(flange_thickness)};
bore_shape = "{bore_shape}";
bore_corner_radius = {_scad_num(bore_corner_radius)};
slot_count = {int(slot_count)};
slot_width_deg = {_scad_num(slot_width_deg)};
slot_depth = {_scad_num(slot_depth)};
slot_start_angle = {_scad_num(slot_start_angle)};
slot_radial_mode = "{slot_radial_mode}";
slot_axial_height = {_scad_num(slot_axial_height)};

module annular_prism(r_outer, r_inner, part_height, edge_break) {{
  safe_edge = min(max(edge_break, 0), min((r_outer - r_inner) * 0.45, part_height * 0.45));
  rotate_extrude(convexity = 8)
    polygon(points = safe_edge > 0 ? [
      [r_inner + safe_edge, -part_height / 2],
      [r_outer - safe_edge, -part_height / 2],
      [r_outer, -part_height / 2 + safe_edge],
      [r_outer, part_height / 2 - safe_edge],
      [r_outer - safe_edge, part_height / 2],
      [r_inner + safe_edge, part_height / 2],
      [r_inner, part_height / 2 - safe_edge],
      [r_inner, -part_height / 2 + safe_edge]
    ] : [
      [r_inner, -part_height / 2],
      [r_outer, -part_height / 2],
      [r_outer, part_height / 2],
      [r_inner, part_height / 2]
    ]);
}}

module rounded_square_bore_cut(size, corner_radius, cut_height) {{
    linear_extrude(height = cut_height, center = true, convexity = 8)
        offset(r = max(corner_radius, 0))
            square([max(size - 2 * corner_radius, 0.1), max(size - 2 * corner_radius, 0.1)], center = true);
}}

module slot_cutouts() {{
    if (slot_count > 0 && slot_width_deg > 0 && slot_depth > 0) {{
        slot_pitch = 360 / slot_count;
        tangential_width = 2 * {_scad_num(ro)} * sin(slot_width_deg / 2);
        radial_length = slot_radial_mode == "through_wall" ? {_scad_num(ro * 2.2)} : slot_depth;
        radial_center = slot_radial_mode == "through_wall" ? {_scad_num(ro * 0.22)} : {_scad_num(ro)} - radial_length / 2;
        for (i = [0 : slot_count - 1])
            rotate([0, 0, slot_start_angle + i * slot_pitch])
                translate([radial_center, 0, 0])
                    cube([radial_length, max(tangential_width, 0.2), slot_axial_height + 2], center = true);
    }}
}}

module bushing() {{
    difference() {{
        union() {{
            // Outer metal sleeve, rubber annulus, then inner metal sleeve.
            if (outer_sleeve_thickness > 0)
        color([0.62, 0.66, 0.70]) annular_prism({_scad_num(ro)}, {_scad_num(rubber_outer)}, outer_sleeve_length, chamfer);

            color([0.06, 0.06, 0.06]) annular_prism({_scad_num(rubber_outer)}, {_scad_num(rubber_inner)}, height, chamfer);

            if (inner_sleeve_thickness > 0)
        color([0.68, 0.72, 0.76]) annular_prism({_scad_num(rubber_inner)}, {_scad_num(ri)}, inner_sleeve_length, chamfer);

            if (flange_mode == "top" || flange_mode == "both")
        translate([0, 0, height / 2 + flange_thickness / 2])
            color([0.62, 0.66, 0.70]) annular_prism({_scad_num(flange_radius)}, {_scad_num(ri)}, flange_thickness, 0);

            if (flange_mode == "bottom" || flange_mode == "both")
        translate([0, 0, -height / 2 - flange_thickness / 2])
            color([0.62, 0.66, 0.70]) annular_prism({_scad_num(flange_radius)}, {_scad_num(ri)}, flange_thickness, 0);
        }}
        if (bore_shape == "rounded_square")
            rounded_square_bore_cut(inner_diameter, bore_corner_radius, height + 4);
        slot_cutouts();
    }}
}}

bushing();
"""


def _scad_num(value: float) -> str:
    return f"{float(value):.6g}"
