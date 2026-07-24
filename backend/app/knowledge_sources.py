"""Engineering knowledge sources available to the chat POC."""

from __future__ import annotations

from typing import Literal


KnowledgeSourceId = Literal["kiss_agent", "fair_explorer"]


KNOWLEDGE_SOURCES: dict[KnowledgeSourceId, dict[str, str]] = {
    "kiss_agent": {
        "name": "KISS Agent",
        "description": (
            "Engineering calculation guidance, sizing rules, and design assumptions "
            "made available by the KISS Agent."
        ),
    },
    "fair_explorer": {
        "name": "FAIR Explorer",
        "description": (
            "Traceable engineering datasets, metadata, and prior design evidence "
            "made available through FAIR Explorer."
        ),
    },
}


def build_knowledge_context(source_ids: list[KnowledgeSourceId]) -> tuple[str, list[str]]:
    """Return safe prompt context and display names for selected POC sources."""

    selected = [KNOWLEDGE_SOURCES[source_id] for source_id in dict.fromkeys(source_ids)]
    if not selected:
        return "", []

    lines = [
        "Selected engineering knowledge sources for this turn:",
        *[f"- {source['name']}: {source['description']}" for source in selected],
        (
            "Use only the supplied source context. Do not claim that a live record, "
            "calculation, or dataset was retrieved when no such evidence is included."
        ),
    ]
    return "\n".join(lines), [source["name"] for source in selected]
