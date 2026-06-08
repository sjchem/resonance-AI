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


Primitive = CubePrimitive | CylinderPrimitive


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


def spec_from_document(document: AgentCadDocument) -> CadSpec:
    """Convert a simple CAD document into viewer metadata."""

    part = document.parts[0]
    primitive = part.primitive
    if primitive.type == "cube":
        length = primitive.size.x
        width = primitive.size.y
        thickness = primitive.size.z
    else:
        length = primitive.radius * 2.0
        width = primitive.radius * 2.0
        thickness = primitive.height

    hole_ops = [op for op in part.operations if op.type == "hole"]
    hole_diameter = hole_ops[0].diameter if hole_ops else 8.0
    part_type = "bracket" if hole_ops or "bracket" in document.name.lower() else "plate"
    return CadSpec(
        part_type=part_type,
        length_mm=length,
        width_mm=width,
        thickness_mm=thickness,
        hole_count=len(hole_ops),
        hole_diameter_mm=hole_diameter,
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
