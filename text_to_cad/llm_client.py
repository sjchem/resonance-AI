"""Azure OpenAI client for the Phase B CAD agent."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class LlmConfigurationError(RuntimeError):
    """Raised when Azure OpenAI configuration is incomplete."""


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


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
