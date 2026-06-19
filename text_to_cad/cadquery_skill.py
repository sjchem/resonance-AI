"""Loader for the local CadQuery LLM skill.

The skill (``skills/cadquery/``) teaches the model to write correct, idiomatic
CadQuery. This module reads ``SKILL.md`` once and exposes a compact preamble that
can be prepended to any system prompt that asks an LLM to produce CAD geometry.

See ``skills/cadquery/README.md`` for the full layout and attribution.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Repo root is two levels up from this file: text_to_cad/ -> <repo root>.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SKILL_DIR = _REPO_ROOT / "skills" / "cadquery"
_SKILL_FILE = _SKILL_DIR / "SKILL.md"


@lru_cache(maxsize=1)
def load_skill() -> str:
    """Return the full ``SKILL.md`` text, or an empty string if it is missing."""

    try:
        return _SKILL_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def skill_available() -> bool:
    """True when the skill file is present and non-empty."""

    return bool(load_skill())


def skill_preamble() -> str:
    """A compact, prompt-ready distillation of the skill.

    Kept deliberately short so it can ride along with task-specific system
    prompts without dominating the context window. The full skill lives in
    ``skills/cadquery/`` for humans and richer retrieval.
    """

    if not skill_available():
        return ""

    return (
        "CadQuery skill (apply when reasoning about geometry):\n"
        "- CadQuery is a B-Rep modeler on OpenCASCADE. Think 'select a face/edge, "
        "then apply a feature' — not 'union/subtract primitives'.\n"
        "- Prefer features (hole, fillet, chamfer, shell, sketch-extrude) over "
        "booleans. Booleans are the top cause of execution failures.\n"
        "- Place repeated holes with a construction rect + vertices (one feature), "
        "not a Python loop.\n"
        "- Selectors: > / < = position (max/min along axis); | parallel, # "
        "perpendicular, +/- normal direction; %Plane/%Circle = face type.\n"
        "- Keep fillet/chamfer smaller than half the thinnest adjacent edge, close "
        "every wire before extrude/revolve, keep revolve profiles off the axis.\n"
        "- Output millimeters, deterministic, no network/assets; export STEP + STL."
    )


def augment_system_prompt(system_prompt: str) -> str:
    """Prepend the skill preamble to an existing system prompt.

    Returns the prompt unchanged when the skill is unavailable so callers can use
    this transparently.
    """

    preamble = skill_preamble()
    if not preamble:
        return system_prompt
    return f"{preamble}\n\n{system_prompt}"
