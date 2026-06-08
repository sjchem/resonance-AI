"""LLM clients for the Phase B CAD agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any


class LlmConfigurationError(RuntimeError):
    """Raised when an LLM provider configuration is incomplete."""


def azure_openai_configured() -> bool:
    required = ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT")
    return all(os.getenv(name) for name in required)


def generate_json_with_azure(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Generate a CAD document JSON object using an Azure OpenAI deployment."""

    if not azure_openai_configured():
        raise LlmConfigurationError(
            "Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_DEPLOYMENT."
        )

    try:
        from openai import AzureOpenAI
    except ModuleNotFoundError as exc:
        raise LlmConfigurationError("Install the openai package to use Azure OpenAI.") from exc

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=float(os.getenv("RESONANCE_CAD_TEMPERATURE", "0.1")),
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Azure OpenAI returned an empty response.")
    return json.loads(content)


def generate_json_with_ollama(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Generate a CAD document JSON object using a local Ollama model."""

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(os.getenv("RESONANCE_CAD_TEMPERATURE", "0.1")),
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("OLLAMA_TIMEOUT", "120"))) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise LlmConfigurationError(
            "Ollama is not reachable. Install Ollama, run "
            "`ollama pull qwen2.5-coder:7b`, then start `ollama serve`."
        ) from exc

    content = (response_payload.get("message") or {}).get("content")
    if not content:
        raise RuntimeError("Ollama returned an empty response.")
    return _loads_json_object(content)


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _loads_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise
        return json.loads(match.group(0))
