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
- A chamfer or fillet on the bushing edges is applied to BOTH the inner and outer top/bottom edges of the rubber body by default. Use geometry.chamfer_mm or geometry.fillet_mm for the size. If the user is ambiguous about which edges, do not put it in missing_information unless they actually ask.
- ALL arms / tabs / lugs / bracket arms / side ears attached to the bushing MUST be emitted as entries in geometry.arms (even if there is only one). Do NOT use the legacy arm_length_mm / arm_width_mm / arm_thickness_mm / arm_position fields — leave them null.
- Each arms entry has: length_mm, width_mm, thickness_mm, angle_deg (0 = +X, 90 = +Z, 180 = -X, 270 = -Z, measured around the bushing axis), and position ("centered" / "top" / "bottom").
- Position defaults to "centered" if the user does not say where vertically. "centered on height" / "in the middle" -> "centered". "at the top" -> "top". "at the bottom" -> "bottom".
- ONE arm with no angle given -> a single entry with angle_deg 0 and position "centered".
- "two arms on opposite sides" / "opposite side" / "180 degrees apart" / "both sides" -> TWO entries with identical length_mm/width_mm/thickness_mm, one at angle_deg 0 and one at angle_deg 180.
- "three arms equally spaced" -> three entries at 0, 120, 240.
- "four arms equally spaced" -> four entries at 0, 90, 180, 270.
- If the user says "add another arm same size on the opposite side", read the existing arm from the prior JSON and emit TWO entries: one at the original angle (0 if not stated) and one at angle + 180. Always re-emit the FULL arms list, not just the new arm.

Example for "bushing OD 50 ID 20 height 40 with two arms 60 mm long, 20 wide, 8 thick on opposite sides":
geometry.arms = [
  { "length_mm": 60, "width_mm": 20, "thickness_mm": 8, "angle_deg": 0,   "position": "centered" },
  { "length_mm": 60, "width_mm": 20, "thickness_mm": 8, "angle_deg": 180, "position": "centered" }
]
and arm_length_mm / arm_width_mm / arm_thickness_mm / arm_position are null.

Bolt holes (geometry.holes):
- When the user asks for bolt holes / mounting holes / fixing holes on a flange or on the top face, populate geometry.holes with one entry per hole pattern.
- Each entry has diameter_mm, pitch_circle_diameter_mm (the PCD), count, and start_angle_deg. "4 M8 holes on a 50 mm PCD" -> diameter_mm 8.5, pitch_circle_diameter_mm 50, count 4, start_angle_deg 0.

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
        validated = CADPromptOutput.model_validate(parsed)
    except ValidationError as exc:
        raise CADValidationError(str(exc)) from exc

    _normalize_arms(validated, prompt)
    return validated


_ARM_COUNT_WORDS = {
    "one": 1, "single": 1,
    "two": 2, "both": 2, "pair": 2, "double": 2,
    "three": 3, "triple": 3,
    "four": 4, "quad": 4, "quadruple": 4,
    "five": 5, "six": 6,
}


def _detect_requested_arm_count(prompt: str) -> int | None:
    """Return the number of arms the user explicitly asked for, or None if unclear."""

    import re

    text = prompt.lower()
    if not re.search(r"\b(arm|arms|tab|tabs|lug|lugs|ear|ears)\b", text):
        return None

    # Phrases that imply two arms symmetrically.
    if re.search(r"\b(opposite|both)\s+sides?\b", text):
        return 2
    if re.search(r"\b180\s*(deg|degrees|°)\s*apart\b", text):
        return 2

    # "two arms", "three arms equally spaced", "4 arms", "two tabs"...
    m = re.search(r"\b(\d+)\s+(?:arm|arms|tab|tabs|lug|lugs|ear|ears)\b", text)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 8:
                return n
        except ValueError:
            pass

    m = re.search(r"\b(one|single|two|both|pair|double|three|triple|four|quad|quadruple|five|six)\s+(?:of\s+)?(?:arm|arms|tab|tabs|lug|lugs|ear|ears)\b", text)
    if m:
        return _ARM_COUNT_WORDS.get(m.group(1))

    return None


def _normalize_arms(parsed: "CADPromptOutput", prompt: str) -> None:
    """Backfill geometry.arms from legacy fields and expand to the requested count.

    The LLM sometimes emits the legacy single-arm fields (arm_length_mm, etc.) instead of
    geometry.arms, or emits only one arm even when the user clearly asked for several.
    This helper:
      1. Promotes legacy arm_* fields into a single geometry.arms entry, then clears them.
      2. If the user's prompt explicitly asks for N>1 arms (e.g. "two arms on opposite sides")
         but the LLM returned 1 entry, duplicates that entry around the bushing axis.
    """

    geometry = parsed.geometry

    # Step 1: promote legacy single-arm fields to the arms list if needed.
    if not geometry.arms and any(
        v is not None
        for v in (geometry.arm_length_mm, geometry.arm_width_mm, geometry.arm_thickness_mm)
    ):
        length = geometry.arm_length_mm or 0.0
        width = geometry.arm_width_mm or 0.0
        thickness = geometry.arm_thickness_mm or 0.0
        if length > 0 and width > 0 and thickness > 0:
            from app.schemas import BushingArm  # local import to avoid cycle at module load

            geometry.arms = [
                BushingArm(
                    length_mm=length,
                    width_mm=width,
                    thickness_mm=thickness,
                    angle_deg=0.0,
                    position=geometry.arm_position or "centered",
                )
            ]

    # Always clear the legacy fields so the renderer uses geometry.arms exclusively.
    geometry.arm_length_mm = None
    geometry.arm_width_mm = None
    geometry.arm_thickness_mm = None
    geometry.arm_position = None

    # Step 2: expand to the requested arm count if the prompt asked for more.
    requested = _detect_requested_arm_count(prompt)
    if requested and geometry.arms and len(geometry.arms) < requested:
        base = geometry.arms[0]
        new_arms = []
        for i in range(requested):
            angle = (360.0 / requested) * i
            new_arms.append(
                base.model_copy(update={"angle_deg": angle})
            )
        geometry.arms = new_arms


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
