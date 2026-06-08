"""FastAPI app for local Ollama CAD prompt parsing."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.ollama_client import (
    CADValidationError,
    OllamaInvalidJSONError,
    OllamaModelNotAvailableError,
    OllamaNotRunningError,
    list_models,
    parse_cad_prompt,
)
from app.schemas import CADPromptOutput, CADPromptRequest


app = FastAPI(title="Local CAD Prompt Parser", version="0.1.0")


@app.get("/")
async def health_check() -> dict[str, str]:
    """Simple health check for local development."""

    return {"status": "ok", "service": "local-cad-prompt-parser"}


@app.get("/models")
async def models() -> dict:
    """Return local Ollama models from /api/tags."""

    try:
        return await list_models()
    except OllamaNotRunningError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/parse-cad", response_model=CADPromptOutput)
async def parse_cad(request: CADPromptRequest) -> CADPromptOutput:
    """Parse a natural-language CAD prompt into validated structured JSON."""

    try:
        return await parse_cad_prompt(request.prompt)
    except OllamaNotRunningError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaModelNotAvailableError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except OllamaInvalidJSONError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CADValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
