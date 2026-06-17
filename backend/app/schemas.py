"""Pydantic models for CAD prompt parsing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CADGeometry(BaseModel):
    """Normalized dimensions extracted from a CAD prompt."""

    model_config = ConfigDict(extra="forbid")

    outer_diameter_mm: float | None = None
    inner_diameter_mm: float | None = None
    height_mm: float | None = None
    length_mm: float | None = None
    width_mm: float | None = None
    thickness_mm: float | None = None
    chamfer_mm: float | None = None
    fillet_mm: float | None = 0


class CADMaterial(BaseModel):
    """Material properties mentioned by the user."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    shore_a: float | None = None
    density_kg_m3: float | None = None


class SimulationHints(BaseModel):
    """Optional downstream CAD/mesh/simulation intent."""

    model_config = ConfigDict(extra="forbid")

    boundary_condition: str | None = None
    load_direction: str | None = None
    target_output: Literal["cad", "mesh", "simulation"] = "cad"


class CADPromptOutput(BaseModel):
    """Validated structured CAD intent returned by the parser."""

    model_config = ConfigDict(extra="forbid")

    part_type: Literal["bushing", "rubber_mount", "plate", "bracket", "unknown"]
    geometry: CADGeometry
    material: CADMaterial
    simulation_hints: SimulationHints
    missing_information: list[str] = Field(default_factory=list)


class CADPromptRequest(BaseModel):
    """Incoming natural-language CAD prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class ChatMessage(BaseModel):
    """A single turn in the engineering chat conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class CADChatRequest(BaseModel):
    """Incoming engineering chat message plus compiled CAD context."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class CADChatResponse(BaseModel):
    """Assistant reply for the engineering chat panel."""

    model_config = ConfigDict(extra="forbid")

    assistant_message: str
    cad_intent: CADPromptOutput
    preview_ready: bool
    preview_svg: str | None = None
