"""Deterministic prompt-to-CadQuery templates for Phase A.

The first milestone is intentionally template based. It gives us a stable
local Text-to-CAD loop before an LLM is introduced in later phases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Iterable


NUMBER_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:mm|millimeter|millimeters)?", re.I)
DIMENSION_PATTERN = re.compile(
    r"(?P<length>\d+(?:\.\d+)?)\s*(?:mm)?\s*[xX*]\s*"
    r"(?P<width>\d+(?:\.\d+)?)\s*(?:mm)?\s*[xX*]\s*"
    r"(?P<thickness>\d+(?:\.\d+)?)\s*(?:mm)?",
    re.I,
)

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
}


@dataclass(frozen=True)
class CadSpec:
    """Structured dimensions for a simple Phase A part."""

    part_type: str
    length_mm: float = 120.0
    width_mm: float = 60.0
    thickness_mm: float = 5.0
    hole_count: int = 0
    hole_diameter_mm: float = 8.0
    material_hint: str = "generic"

    @property
    def hole_radius_mm(self) -> float:
        return self.hole_diameter_mm / 2.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def prompt_to_spec(prompt: str) -> CadSpec:
    """Parse a narrow engineering prompt into a conservative CAD spec."""

    normalized = " ".join(prompt.strip().split())
    lowered = normalized.lower()

    length, width, thickness = _parse_dimensions(lowered)
    part_type = "bracket" if any(word in lowered for word in ("bracket", "mount", "bolt", "hole")) else "plate"
    hole_count = _parse_hole_count(lowered) if part_type == "bracket" else 0
    hole_diameter = _parse_hole_diameter(lowered)
    material_hint = _parse_material_hint(lowered)

    return CadSpec(
        part_type=part_type,
        length_mm=length,
        width_mm=width,
        thickness_mm=thickness,
        hole_count=hole_count,
        hole_diameter_mm=hole_diameter,
        material_hint=material_hint,
    )


def render_cadquery_script(spec: CadSpec, output_name: str) -> str:
    """Render a self-contained CadQuery script for the requested part."""

    step_name = f"{output_name}.step"
    stl_name = f"{output_name}.stl"
    holes = _hole_points(spec)
    hole_block = ""

    if holes:
        hole_block = f"""
part = (
    part
    .faces(">Z")
    .workplane()
    .pushPoints({holes!r})
    .hole({spec.hole_diameter_mm!r})
)
"""

    return f'''"""Generated CadQuery model for {output_name}."""

import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))

import cadquery as cq

LENGTH_MM = {spec.length_mm!r}
WIDTH_MM = {spec.width_mm!r}
THICKNESS_MM = {spec.thickness_mm!r}


part = cq.Workplane("XY").box(LENGTH_MM, WIDTH_MM, THICKNESS_MM)
{hole_block}
cq.exporters.export(part, str(OUTPUT_DIR / "{step_name}"))
cq.exporters.export(part, str(OUTPUT_DIR / "{stl_name}"))

print("Wrote {step_name} and {stl_name}")
'''


def _parse_dimensions(text: str) -> tuple[float, float, float]:
    match = DIMENSION_PATTERN.search(text)
    if match:
        return (
            float(match.group("length")),
            float(match.group("width")),
            float(match.group("thickness")),
        )

    thickness_match = re.search(
        r"(?P<thickness>\d+(?:\.\d+)?)\s*(?:mm|millimeter|millimeters)?\s+thick(?:ness)?",
        text,
        re.I,
    )
    thickness = float(thickness_match.group("thickness")) if thickness_match else 5.0
    numbers = [float(match.group("value")) for match in NUMBER_PATTERN.finditer(text)]
    if len(numbers) >= 2:
        return numbers[0], numbers[1], thickness
    return 120.0, 60.0, thickness


def _parse_hole_count(text: str) -> int:
    digit_match = re.search(r"(?P<count>\d+)\s+(?:bolt\s+)?holes?", text, re.I)
    if digit_match:
        return max(0, int(digit_match.group("count")))

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(?:bolt\s+)?holes?\b", text, re.I):
            return value

    return 4 if "hole" in text or "bolt" in text else 0


def _parse_hole_diameter(text: str) -> float:
    diameter_match = re.search(
        r"(?:hole|bolt)\s+(?:diameter|dia)\s+(?P<diameter>\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    if diameter_match:
        return float(diameter_match.group("diameter"))

    radius_match = re.search(r"(?:hole|bolt)\s+radius\s+(?P<radius>\d+(?:\.\d+)?)", text, re.I)
    if radius_match:
        return 2.0 * float(radius_match.group("radius"))

    return 8.0


def _parse_material_hint(text: str) -> str:
    for material in ("rubber", "steel", "aluminum", "aluminium", "plastic"):
        if material in text:
            return material
    return "generic"


def _hole_points(spec: CadSpec) -> list[tuple[float, float]]:
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
    return _dedupe_points(points[: spec.hole_count])


def _dedupe_points(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    seen = set()
    unique = []
    for point in points:
        rounded = (round(point[0], 6), round(point[1], 6))
        if rounded in seen:
            continue
        seen.add(rounded)
        unique.append(rounded)
    return unique
