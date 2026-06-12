"""Async OpenAI client for CAD prompt parsing."""

from __future__ import annotations

import os
from pathlib import Path

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, AuthenticationError, OpenAIError
from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas import CADPromptOutput


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """You are a CAD intent parser for vibroacoustic and mechanical components.

Convert the user's natural-language CAD request into strict JSON.

Rules:
1. Return JSON only.
2. Do not return Python code.
3. Use millimeters for all dimensions.
4. Do not invent missing critical dimensions.
5. If a dimension is missing, use null.
6. Supported part types: bushing, rubber_mount, plate, bracket, unknown.
7. Extract material if mentioned.
8. Extract chamfer, fillet, holes, load direction, and boundary condition if mentioned.
9. If important information is missing, add a short item to missing_information.
10. Keep target_output as "cad" unless the user explicitly asks for mesh or simulation.

Extraction details:
- "rubber bushing" means part_type is "bushing" and material.name is "rubber".
- "rubber mount" means part_type is "rubber_mount" and material.name is "rubber".
- "outer diameter 60 mm" means geometry.outer_diameter_mm is 60.
- "inner diameter 20 mm" means geometry.inner_diameter_mm is 20.
- "height 40 mm" means geometry.height_mm is 40.
- "chamfer 2 mm" means geometry.chamfer_mm is 2.
- "fillet 2 mm" means geometry.fillet_mm is 2.
- If chamfer or fillet size is given but location is not given, still extract the size. You may add the missing location to missing_information.
"""


class OpenAIConfigurationError(RuntimeError):
    """Raised when OPENAI_API_KEY or model configuration is missing."""


class OpenAIRequestError(RuntimeError):
    """Raised when the OpenAI API request fails."""


class CADValidationError(RuntimeError):
    """Raised when parsed model output does not match our CAD schema."""


def openai_status() -> dict[str, str | bool]:
    """Return local configuration state without exposing secrets."""

    return {
        "provider": "openai",
        "model": OPENAI_MODEL,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


async def parse_cad_prompt(prompt: str) -> CADPromptOutput:
    """Convert a natural-language CAD prompt into validated structured JSON."""

    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIConfigurationError("OPENAI_API_KEY is not set.")

    client = AsyncOpenAI()
    try:
        response = await client.responses.parse(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            text_format=CADPromptOutput,
            temperature=0,
        )
    except AuthenticationError as exc:
        raise OpenAIConfigurationError("OpenAI authentication failed. Check OPENAI_API_KEY.") from exc
    except APIConnectionError as exc:
        raise OpenAIRequestError("Could not connect to the OpenAI API.") from exc
    except APIStatusError as exc:
        raise OpenAIRequestError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
    except OpenAIError as exc:
        raise OpenAIRequestError(str(exc)) from exc

    parsed = response.output_parsed
    if parsed is None:
        raise CADValidationError("OpenAI did not return parsed CAD JSON.")

    try:
        return CADPromptOutput.model_validate(parsed)
    except ValidationError as exc:
        raise CADValidationError(str(exc)) from exc
