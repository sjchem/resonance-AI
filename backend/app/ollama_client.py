"""Small async client for the local Ollama API."""

from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import ValidationError

from app.schemas import CADPromptOutput


OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-coder:7b"

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
"""


class OllamaNotRunningError(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaModelNotAvailableError(RuntimeError):
    """Raised when qwen2.5-coder:7b is not available locally."""


class OllamaInvalidJSONError(RuntimeError):
    """Raised when the model response cannot be parsed as JSON."""


class CADValidationError(RuntimeError):
    """Raised when model JSON does not match our CAD schema."""


async def list_models() -> dict[str, Any]:
    """Return models installed in the local Ollama runtime."""

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        except httpx.ConnectError as exc:
            raise OllamaNotRunningError("Ollama is not running at http://localhost:11434.") from exc
        response.raise_for_status()
        return response.json()


async def parse_cad_prompt(prompt: str) -> CADPromptOutput:
    """Ask local Qwen2.5-Coder to convert a prompt into validated CAD JSON."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\nUser request:\n{prompt}",
        "format": CADPromptOutput.model_json_schema(),
        "stream": False,
        "options": {"temperature": 0},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaNotRunningError("Ollama is not running at http://localhost:11434.") from exc

    if response.status_code == 404:
        raise OllamaModelNotAvailableError(
            f"Model {OLLAMA_MODEL} is not available. Run: ollama pull {OLLAMA_MODEL}"
        )

    response_data = response.json()
    if "error" in response_data:
        message = str(response_data["error"])
        if "not found" in message.lower() or "pull" in message.lower():
            raise OllamaModelNotAvailableError(
                f"Model {OLLAMA_MODEL} is not available. Run: ollama pull {OLLAMA_MODEL}"
            )
        raise RuntimeError(message)

    raw_json = response_data.get("response")
    if not isinstance(raw_json, str) or not raw_json.strip():
        raise OllamaInvalidJSONError("Ollama returned an empty response.")

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise OllamaInvalidJSONError(f"Ollama returned invalid JSON: {exc}") from exc

    try:
        return CADPromptOutput.model_validate(parsed)
    except ValidationError as exc:
        raise CADValidationError(str(exc)) from exc
