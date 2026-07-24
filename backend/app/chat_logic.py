"""Conversational helpers for the engineering chat workflow."""

from __future__ import annotations

from app.cad_preview import build_preview_svg
from app.knowledge_sources import KnowledgeSourceId, build_knowledge_context
from app.openai_client import (
    OpenAIConfigurationError,
    OpenAIRequestError,
    chat_reply,
    parse_cad_prompt,
)
from app.schemas import CADChatResponse, CADPromptOutput, ChatMessage


async def respond_to_cad_chat(
    message: str,
    prompt: str,
    history: list[ChatMessage] | None = None,
    knowledge_sources: list[KnowledgeSourceId] | None = None,
) -> CADChatResponse:
    """Return a conversational reply plus the current CAD intent state."""

    history = history or []
    parsed = await parse_cad_prompt(prompt)
    preview_ready = parsed.part_type != "unknown"
    source_context, consulted_sources = build_knowledge_context(knowledge_sources or [])

    assistant_message = await _build_conversational_message(
        parsed,
        message,
        history,
        source_context,
    )
    preview_svg = build_preview_svg(parsed) if preview_ready else None
    return CADChatResponse(
        assistant_message=assistant_message,
        cad_intent=parsed,
        preview_ready=preview_ready,
        preview_svg=preview_svg,
        consulted_sources=consulted_sources,
    )


async def _build_conversational_message(
    parsed: CADPromptOutput,
    message: str,
    history: list[ChatMessage],
    source_context: str = "",
) -> str:
    """Generate a natural chat reply, falling back to a templated one on failure."""

    turns = [{"role": turn.role, "content": turn.content} for turn in history]
    if not turns or turns[-1].get("content") != message:
        turns.append({"role": "user", "content": message})

    try:
        reply = await chat_reply(
            turns,
            parsed.model_dump_json(indent=2),
            source_context=source_context,
        )
    except (OpenAIConfigurationError, OpenAIRequestError):
        reply = None

    if reply:
        return reply
    return _build_assistant_message(parsed, message, parsed.part_type != "unknown")


def _build_assistant_message(parsed: CADPromptOutput, message: str, preview_ready: bool) -> str:
    """Turn the current CAD intent into a natural chat response."""

    if parsed.part_type == "unknown":
        return (
            "I can help with that, but I cannot confidently identify the CAD part yet. "
            "Tell me whether this should be a bushing, rubber mount, plate, or bracket, "
            "and include the main dimensions in mm."
        )

    part_name = parsed.part_type.replace("_", " ")
    material = parsed.material.name or "material not specified"
    dimensions = _format_dimensions(parsed)
    missing = parsed.missing_information
    approved = _is_approval(message)

    lines = [
        f"I am reading this as a {part_name} with {material}.",
        f"Current dimensions: {dimensions}.",
    ]

    if missing:
        lines.append(f"I still need or recommend clarification on: {'; '.join(missing)}.")
        lines.append("Reply with the missing values or corrections, and I will update the CAD intent.")
    elif approved:
        lines.append("I have enough information, so the structured CAD result and preview are updated on the right.")
    elif preview_ready:
        lines.append("I have enough information for a working preview. Reply with a correction or type proceed to lock this version in.")

    return "\n".join(lines)


def _format_dimensions(parsed: CADPromptOutput) -> str:
    geometry = parsed.geometry
    values = [
        ("OD", geometry.outer_diameter_mm),
        ("ID", geometry.inner_diameter_mm),
        ("height", geometry.height_mm),
        ("length", geometry.length_mm),
        ("width", geometry.width_mm),
        ("thickness", geometry.thickness_mm),
        ("chamfer", geometry.chamfer_mm),
        ("fillet", geometry.fillet_mm),
    ]
    parts = [f"{label} {value:g} mm" for label, value in values if value is not None]
    if geometry.coil_count is not None:
        parts.append(f"{geometry.coil_count:g} coils")
    return ", ".join(parts) if parts else "no dimensions captured yet"


def _is_approval(message: str) -> bool:
    return message.strip().lower() in {"proceed", "agree", "approved", "approve", "yes", "ok", "okay", "generate", "create", "go"}
