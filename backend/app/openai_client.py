"""Async OpenAI/Azure OpenAI client for CAD prompt parsing."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
)
from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas import CADPromptOutput


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

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
    """Raised when OpenAI or Azure OpenAI configuration is missing."""


class OpenAIRequestError(RuntimeError):
    """Raised when the OpenAI API request fails."""


class CADValidationError(RuntimeError):
    """Raised when parsed model output does not match our CAD schema."""


def openai_status() -> dict[str, str | bool]:
    """Return local configuration state without exposing secrets."""

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_configured = all(
        [
            os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint,
            os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        ]
    )
    if azure_configured:
        return {
            "provider": _provider_name_for_endpoint(azure_endpoint),
            "model": os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
            "api_key_configured": True,
        }

    return {
        "provider": "openai",
        "model": OPENAI_MODEL,
        "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


async def parse_cad_prompt(prompt: str) -> CADPromptOutput:
    """Convert a natural-language CAD prompt into validated structured JSON."""

    client, model, provider = _build_client()
    try:
        if provider.startswith("azure_"):
            parsed = await _parse_with_azure_chat(client, model, prompt)
        else:
            parsed = await _parse_with_openai_responses(client, model, prompt)
    except AuthenticationError as exc:
        raise OpenAIConfigurationError("OpenAI authentication failed. Check the configured API key.") from exc
    except APIConnectionError as exc:
        raise OpenAIRequestError("Could not connect to the OpenAI API.") from exc
    except APIStatusError as exc:
        raise OpenAIRequestError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
    except OpenAIError as exc:
        raise OpenAIRequestError(str(exc)) from exc

    if parsed is None:
        raise CADValidationError("OpenAI did not return parsed CAD JSON.")

    try:
        return CADPromptOutput.model_validate(parsed)
    except ValidationError as exc:
        raise CADValidationError(str(exc)) from exc


async def _parse_with_openai_responses(
    client: AsyncOpenAI | AsyncAzureOpenAI, model: str, prompt: str
) -> CADPromptOutput | None:
    """Parse with the public OpenAI Responses API."""

    response = await client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=prompt,
        text_format=CADPromptOutput,
        temperature=0,
    )
    return response.output_parsed


async def _parse_with_azure_chat(
    client: AsyncOpenAI | AsyncAzureOpenAI, model: str, prompt: str
) -> CADPromptOutput | None:
    """Parse with Azure OpenAI Chat Completions structured outputs.

    The Azure deployment in this app uses API version 2024-10-21. Chat
    Completions is the compatible route for that setup, while the newer
    Responses route can return "Resource not found" on some Azure resources.
    """

    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format=CADPromptOutput,
        temperature=0,
    )
    return completion.choices[0].message.parsed


def _build_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str, str]:
    """Build the configured LLM client.

    Azure OpenAI is preferred when its environment variables are present because
    Azure deployment names are not the same thing as public OpenAI model names.
    """

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    if azure_api_key or azure_endpoint or azure_deployment:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_API_KEY": azure_api_key,
                "AZURE_OPENAI_ENDPOINT": azure_endpoint,
                "AZURE_OPENAI_DEPLOYMENT": azure_deployment,
            }.items()
            if not value
        ]
        if missing:
            raise OpenAIConfigurationError(
                f"Azure OpenAI configuration is incomplete. Missing: {', '.join(missing)}."
            )

        if _is_foundry_endpoint(azure_endpoint):
            return (
                AsyncOpenAI(
                    api_key=azure_api_key,
                    base_url=_normalize_foundry_base_url(azure_endpoint),
                    default_query={"api-version": AZURE_OPENAI_API_VERSION},
                ),
                azure_deployment,
                "azure_ai_foundry",
            )

        return (
            AsyncAzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=_normalize_azure_openai_endpoint(azure_endpoint),
                api_version=AZURE_OPENAI_API_VERSION,
            ),
            azure_deployment,
            "azure_openai",
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise OpenAIConfigurationError(
            "Set Azure OpenAI variables or set OPENAI_API_KEY for the public OpenAI API."
        )

    return AsyncOpenAI(), OPENAI_MODEL, "openai"


def _provider_name_for_endpoint(endpoint: str) -> str:
    if _is_foundry_endpoint(endpoint):
        return "azure_ai_foundry"
    return "azure_openai"


def _is_foundry_endpoint(endpoint: str | None) -> bool:
    return bool(endpoint and "services.ai.azure.com" in endpoint)


def _normalize_azure_openai_endpoint(endpoint: str) -> str:
    """Return the resource-level Azure OpenAI endpoint.

    The Azure OpenAI client expects the resource endpoint, for example:
    https://my-resource.openai.azure.com
    """

    return endpoint.rstrip("/")


def _normalize_foundry_base_url(endpoint: str) -> str:
    """Build an OpenAI-compatible base URL for Azure AI Foundry inference.

    Users sometimes paste a project endpoint such as:
    https://name.services.ai.azure.com/api/projects/proj-default
    For the OpenAI-compatible client path we only want the host, then `/models`.
    """

    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise OpenAIConfigurationError(
            "AZURE_OPENAI_ENDPOINT is not a valid URL. Expected an Azure AI Foundry endpoint."
        )
    return f"{parsed.scheme}://{parsed.netloc}/models"
