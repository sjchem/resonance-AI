"""Structured CAD document schema for the Phase B agent."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from text_to_cad.cad_templates import CadSpec


Number = Annotated[float, Field(gt=0)]


class Vec3(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Size3(BaseModel):
    x: Number
    y: Number
    z: Number


class CubePrimitive(BaseModel):
    type: Literal["cube"]
    size: Size3


class CylinderPrimitive(BaseModel):
    type: Literal["cylinder"]
    radius: Number
    height: Number


class SpringPrimitive(BaseModel):
    type: Literal["spring"]
    coil_radius: Number = 12.0
    wire_radius: Number = 1.5
    height: Number = 40.0
    turns: Number = 6.0
    samples_per_turn: int = Field(default=18, ge=12, le=64)


class TubePrimitive(BaseModel):
    """A hollow cylinder: the core of a vibration-isolation bushing."""

    type: Literal["tube"]
    outer_radius: Number = 20.0
    inner_radius: Number = 8.0
    height: Number = 30.0
    chamfer: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def check_radii(self) -> "TubePrimitive":
        if self.inner_radius >= self.outer_radius:
            raise ValueError("Bushing inner_radius must be smaller than outer_radius.")
        max_chamfer = 0.49 * min(self.outer_radius - self.inner_radius, self.height)
        if self.chamfer > max_chamfer:
            self.chamfer = max(0.0, max_chamfer)
        return self


Primitive = CubePrimitive | CylinderPrimitive | SpringPrimitive | TubePrimitive


class HoleOperation(BaseModel):
    type: Literal["hole"]
    diameter: Number
    at: Vec3
    depth: float | None = Field(default=None, gt=0)


class BooleanOperation(BaseModel):
    type: Literal["union", "difference"]
    primitive: Primitive
    at: Vec3 = Field(default_factory=Vec3)


class EdgeOperation(BaseModel):
    type: Literal["fillet", "chamfer"]
    radius: Number


Operation = HoleOperation | BooleanOperation | EdgeOperation


class CadPart(BaseModel):
    name: str = Field(min_length=1)
    primitive: Primitive
    at: Vec3 = Field(default_factory=Vec3)
    operations: list[Operation] = Field(default_factory=list)


class AgentCadDocument(BaseModel):
    name: str = Field(min_length=1)
    units: Literal["mm"] = "mm"
    material_hint: str = "generic"
    description: str = ""
    parts: list[CadPart] = Field(min_length=1)

    @model_validator(mode="after")
    def require_supported_model(self) -> "AgentCadDocument":
        if len(self.parts) != 1:
            raise ValueError("Phase B currently supports one exported part per document.")
        return self


def document_from_spec(spec: CadSpec, name: str, description: str = "") -> AgentCadDocument:
    """Create a Phase B document from the Phase A deterministic parser."""

    if spec.part_type == "spring":
        return document_from_spring_spec(spec, name, description)
    if spec.part_type == "bushing":
        return document_from_bushing_spec(spec, name, description)

    operations: list[Operation] = []
    for x, y in _hole_points(spec):
        operations.append(
            HoleOperation(
                type="hole",
                diameter=spec.hole_diameter_mm,
                at=Vec3(x=x + spec.length_mm / 2.0, y=y + spec.width_mm / 2.0, z=0.0),
            )
        )

    return AgentCadDocument(
        name=name,
        material_hint=spec.material_hint,
        description=description,
        parts=[
            CadPart(
                name=name,
                primitive=CubePrimitive(
                    type="cube",
                    size=Size3(x=spec.length_mm, y=spec.width_mm, z=spec.thickness_mm),
                ),
                operations=operations,
            )
        ],
    )


def document_from_spring_spec(spec: CadSpec, name: str, description: str = "") -> AgentCadDocument:
    return AgentCadDocument(
        name=name,
        material_hint=spec.material_hint,
        description=description,
        parts=[
            CadPart(
                name=name,
                primitive=SpringPrimitive(
                    type="spring",
                    coil_radius=spec.length_mm / 2.0,
                    wire_radius=spec.hole_diameter_mm / 2.0,
                    height=spec.thickness_mm,
                    turns=max(1.0, float(spec.hole_count or 6)),
                ),
            )
        ],
    )


def document_from_bushing_spec(spec: CadSpec, name: str, description: str = "") -> AgentCadDocument:
    """Create a hollow-cylinder bushing document from the deterministic parser."""

    return AgentCadDocument(
        name=name,
        material_hint=spec.material_hint,
        description=description,
        parts=[
            CadPart(
                name=name,
                primitive=TubePrimitive(
                    type="tube",
                    outer_radius=spec.length_mm / 2.0,
                    inner_radius=spec.hole_diameter_mm / 2.0,
                    height=spec.thickness_mm,
                    chamfer=spec.chamfer_mm,
                ),
            )
        ],
    )


def spec_from_document(document: AgentCadDocument) -> CadSpec:
    """Convert a simple CAD document into viewer metadata."""

    part = document.parts[0]
    primitive = part.primitive
    chamfer = 0.0
    if primitive.type == "cube":
        length = primitive.size.x
        width = primitive.size.y
        thickness = primitive.size.z
    elif primitive.type == "cylinder":
        length = primitive.radius * 2.0
        width = primitive.radius * 2.0
        thickness = primitive.height
    elif primitive.type == "tube":
        length = primitive.outer_radius * 2.0
        width = length
        thickness = primitive.height
        chamfer = primitive.chamfer
    else:
        length = (primitive.coil_radius + primitive.wire_radius) * 2.0
        width = length
        thickness = primitive.height

    hole_ops = [op for op in part.operations if op.type == "hole"]
    if primitive.type == "tube":
        part_type = "bushing"
        hole_diameter = primitive.inner_radius * 2.0
        hole_count = 0
    else:
        hole_diameter = hole_ops[0].diameter if hole_ops else 8.0
        part_type = (
            "spring"
            if primitive.type == "spring"
            else "bracket"
            if hole_ops or "bracket" in document.name.lower()
            else "plate"
        )
        hole_count = len(hole_ops)
    return CadSpec(
        part_type=part_type,
        length_mm=length,
        width_mm=width,
        thickness_mm=thickness,
        hole_count=hole_count,
        hole_diameter_mm=hole_diameter,
        chamfer_mm=chamfer,
        material_hint=document.material_hint,
    )


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
    return points[: spec.hole_count]
