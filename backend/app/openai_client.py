"""Async OpenAI/Azure OpenAI client for CAD prompt parsing."""

from __future__ import annotations

import json
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
6. Supported part types: bushing, rubber_mount, plate, bracket, spring, unknown.
7. Extract material if mentioned.
8. Extract chamfer, fillet, holes, load direction, and boundary condition if mentioned.
9. If important information is missing, add a short item to missing_information.
10. Keep target_output as "cad" unless the user explicitly asks for mesh or simulation.

Extraction details:
- "rubber bushing" means part_type is "bushing" and material.name is "rubber".
- "rubber mount" means part_type is "rubber_mount" and material.name is "rubber".
- Vibracoustic bushings are almost always rubber or rubber-metal bonded. If the user says "bushing" without a material, set material.name to "rubber" by default. Only use a different material when the user explicitly says steel, aluminum, plastic, etc.
- "outer diameter 60 mm" means geometry.outer_diameter_mm is 60.
- "inner diameter 20 mm" means geometry.inner_diameter_mm is 20.
- "height 40 mm" means geometry.height_mm is 40.
- "chamfer 2 mm" means geometry.chamfer_mm is 2.
- "fillet 2 mm" means geometry.fillet_mm is 2.
- If chamfer or fillet size is given but location is not given, still extract the size. You may add the missing location to missing_information.

Bushing variants:
- "flanged bushing" or "flange diameter 90 mm" / "flange thickness 5 mm" means the bushing has a flange. Map to geometry.flange_diameter_mm and geometry.flange_thickness_mm.
- "rubber-metal bonded", "bonded bushing", or an outer metal sleeve thickness (e.g. "outer sleeve 2 mm") means there is a bonded outer steel sleeve. Map to geometry.metal_sleeve_thickness_mm.
- An inner steel sleeve / inner pipe (e.g. "inner sleeve 1.5 mm") maps to geometry.inner_sleeve_thickness_mm.
- "eccentric bore", "offset bore", or "bore offset 3 mm" means the inner bore is shifted from center. Map to geometry.bore_offset_mm.

Spring details:
- "spring", "compression spring", "coil spring", "helical spring" means part_type is "spring".
- The coil/mean diameter (e.g. "coil diameter 40 mm" or "radius 2 cm") maps to geometry.outer_diameter_mm (convert radius to diameter, cm to mm).
- The free length / overall length (e.g. "50 mm long") maps to geometry.height_mm.
- The wire diameter / wire thickness (e.g. "wire thickness 0.5 mm") maps to geometry.thickness_mm.
- The number of coils / turns (e.g. "10 coils") maps to geometry.coil_count.
"""

CHAT_SYSTEM_PROMPT = """You are Resonance AI, a friendly and concise CAD engineering assistant for \
vibroacoustic and mechanical components (bushings, rubber mounts, plates, brackets, springs).

You are having a natural, interactive conversation with an engineer, similar to ChatGPT or Claude. \
Your job is to guide them step by step toward a complete CAD intent before the model is generated.

You will be given the conversation so far and the current parsed CAD intent (as JSON). Reply with a \
short, natural chat message. Follow this conversational flow:

1. First, briefly acknowledge what the user said and confirm your understanding of the part and any \
dimensions captured so far.
2. If important details are still missing (dimensions, shape, material grade/Shore A, chamfer/fillet, \
holes, features), ask for them — but ask only for the 2-3 most important missing items at a time, as a \
short friendly question. Do not dump a long list.
3. Once you have the part type and the core dimensions needed to build it, summarize the full spec in \
one or two lines and ask the user to confirm, e.g. "Shall I go ahead and generate this CAD model? \
Reply 'proceed' to confirm."
4. If the user confirms (proceed / yes / go ahead / generate), reply that you are generating the CAD \
model now and that the preview and structured JSON are updated on the right.

Rules:
- Be warm, conversational, and brief (2-5 sentences). Use plain language, not bullet dumps.
- Ask follow-up questions naturally instead of listing every missing field.
- Never output JSON or code. Just the chat message.
- If the user changes or corrects something, acknowledge the change and continue.
- If the part is still unknown, ask the user what kind of part it is.
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
        if provider == "azure_ai_foundry":
            parsed = await _parse_with_foundry_fallbacks(model, prompt)
        elif provider == "azure_openai":
            parsed = await _parse_with_azure_openai_fallbacks(client, model, prompt)
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


async def chat_reply(
    history: list[dict[str, str]],
    cad_intent_json: str,
) -> str | None:
    """Generate a natural, conversational assistant reply for the chat panel.

    `history` is a list of {"role": "user"|"assistant", "content": str} turns, with the
    latest user message last. `cad_intent_json` is the current parsed CAD intent so the
    model can ground its reply in what has actually been captured.
    """

    client, model, provider = _build_client()

    context_message = {
        "role": "system",
        "content": (
            "Current parsed CAD intent (JSON). Use this to decide what is already known "
            "and what is still missing:\n" + cad_intent_json
        ),
    }
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        context_message,
        *history,
    ]

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
        )
    except AuthenticationError as exc:
        raise OpenAIConfigurationError(
            "OpenAI authentication failed. Check the configured API key."
        ) from exc
    except APIConnectionError as exc:
        raise OpenAIRequestError("Could not connect to the OpenAI API.") from exc
    except APIStatusError as exc:
        raise OpenAIRequestError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
    except OpenAIError as exc:
        raise OpenAIRequestError(str(exc)) from exc

    content = completion.choices[0].message.content
    return content.strip() if content else None


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


async def _parse_with_azure_chat_json_schema(
    client: AsyncOpenAI | AsyncAzureOpenAI, model: str, prompt: str
) -> CADPromptOutput | None:
    """Fallback for Azure chat using explicit JSON schema response_format."""

    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cad_prompt_output",
                "strict": True,
                "schema": CADPromptOutput.model_json_schema(),
            },
        },
        temperature=0,
    )
    content = completion.choices[0].message.content or ""
    if not content:
        return None
    return CADPromptOutput.model_validate(json.loads(content))


async def _parse_with_azure_openai_fallbacks(
    client: AsyncAzureOpenAI, model: str, prompt: str
) -> CADPromptOutput | None:
    """Try the Azure OpenAI parse helper first, then a plain JSON-schema chat call."""

    try:
        return await _parse_with_azure_chat(client, model, prompt)
    except APIStatusError as exc:
        if exc.status_code != 404:
            raise
    except OpenAIError:
        pass

    return await _parse_with_azure_chat_json_schema(client, model, prompt)


async def _parse_with_foundry_fallbacks(model: str, prompt: str) -> CADPromptOutput | None:
    """Try multiple Azure AI Foundry endpoint shapes before failing.

    Foundry deployments can be exposed either through a project endpoint such as
    `.../api/projects/<name>` or an OpenAI-compatible host-level route. Some
    combinations return 404 even with valid credentials, so we try the most
    likely structured-output paths in sequence.
    """

    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not azure_api_key or not azure_endpoint:
        raise OpenAIConfigurationError(
            "Azure AI Foundry configuration is incomplete. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
        )

    candidate_clients = [
        AsyncOpenAI(
            api_key=azure_api_key,
            base_url=_normalize_foundry_project_base_url(azure_endpoint),
            default_query={"api-version": AZURE_OPENAI_API_VERSION},
        ),
        AsyncOpenAI(
            api_key=azure_api_key,
            base_url=_normalize_foundry_base_url(azure_endpoint),
            default_query={"api-version": AZURE_OPENAI_API_VERSION},
        ),
    ]
    parsers = (_parse_with_azure_chat, _parse_with_openai_responses)
    last_error: OpenAIError | None = None

    for candidate in candidate_clients:
        for parser in parsers:
            try:
                return await parser(candidate, model, prompt)
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code == 404:
                    continue
                raise
            except OpenAIError as exc:
                last_error = exc
                continue

    if last_error:
        raise last_error
    return None


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


def _normalize_foundry_project_base_url(endpoint: str) -> str:
    """Return the exact Foundry project endpoint when the user configured one."""

    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise OpenAIConfigurationError(
            "AZURE_OPENAI_ENDPOINT is not a valid URL. Expected an Azure AI Foundry project endpoint."
        )
    path = parsed.path.rstrip("/")
    if path:
        return f"{parsed.scheme}://{parsed.netloc}{path}"
    return f"{parsed.scheme}://{parsed.netloc}"
