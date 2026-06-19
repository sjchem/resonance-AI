"""Phase B CLI: LLM-backed structured CAD generation with deterministic fallback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

try:
    from text_to_cad.agent_renderer import render_agent_cadquery_script
    from text_to_cad.agent_schema import AgentCadDocument, document_from_spec, spec_from_document
    from text_to_cad.cadquery_skill import augment_system_prompt
    from text_to_cad.cad_executor import run_script, write_script
    from text_to_cad.cad_templates import prompt_to_spec
    from text_to_cad.cad_generator import _write_preview
    from text_to_cad.llm_client import (
        LlmConfigurationError,
        azure_openai_configured,
        generate_json_with_azure,
        load_prompt_template,
    )
    from text_to_cad.viewer import write_viewer
except ModuleNotFoundError:
    from agent_renderer import render_agent_cadquery_script
    from agent_schema import AgentCadDocument, document_from_spec, spec_from_document
    from cadquery_skill import augment_system_prompt
    from cad_executor import run_script, write_script
    from cad_templates import prompt_to_spec
    from cad_generator import _write_preview
    from llm_client import (
        LlmConfigurationError,
        azure_openai_configured,
        generate_json_with_azure,
        load_prompt_template,
    )
    from viewer import write_viewer


DEFAULT_OUTPUT_ROOT = Path("outputs") / "phase_b"
DEFAULT_SYSTEM_PROMPT = Path(__file__).resolve().parent / "prompts" / "agent_system_prompt.txt"


def generate_with_agent(
    prompt: str,
    output_dir: Path,
    output_name: str,
    provider: str = "auto",
    execute: bool = True,
    max_repairs: int = 2,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    document, source = _create_document(prompt, output_name, provider)
    script = render_agent_cadquery_script(document, output_name)
    spec = spec_from_document(document)

    (output_dir / "prompt.txt").write_text(prompt.strip() + "\n", encoding="utf-8")
    (output_dir / "agent_document.json").write_text(
        json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "agent_source.txt").write_text(source + "\n", encoding="utf-8")
    script_path = write_script(script, output_dir, filename="agent_generated_cad.py")
    _write_preview(spec, output_dir / "preview.png")

    print(f"Wrote Phase B CAD document, preview, and script to {output_dir}")
    print(f"CAD document source: {source}")
    if not execute:
        print("Skipped CAD execution because --dry-run was supplied.")
        return 0

    result = _run_with_repairs(script_path, document, output_name, output_dir, max_repairs)
    if not result:
        return 1

    viewer_path = write_viewer(output_dir / f"{output_name}.stl", spec, output_dir, output_name)
    print(f"Wrote interactive 3D viewer to {viewer_path}")
    print(f"CAD artifacts are in {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase B LLM-backed CAD agent.")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Natural language part description.")
    prompt_group.add_argument("--prompt-file", type=Path, help="Path to a prompt text file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None, help="Output basename for STEP/STL files.")
    parser.add_argument(
        "--provider",
        choices=("auto", "azure", "fallback"),
        default="auto",
        help="CAD document source. auto uses Azure when configured, otherwise fallback.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write files but do not run CadQuery.")
    parser.add_argument("--max-repairs", type=int, default=2, help="Reserved for execution repair attempts.")
    args = parser.parse_args(argv)

    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    output_name = args.name or _slug_from_prompt(prompt)
    try:
        return generate_with_agent(
            prompt=prompt,
            output_dir=args.output_dir,
            output_name=output_name,
            provider=args.provider,
            execute=not args.dry_run,
            max_repairs=args.max_repairs,
        )
    except LlmConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def create_document_for_prompt(prompt: str, output_name: str, provider: str = "auto") -> AgentCadDocument:
    return _create_document(prompt, output_name, provider)[0]


def _create_document(prompt: str, output_name: str, provider: str) -> tuple[AgentCadDocument, str]:
    use_azure = provider == "azure" or (provider == "auto" and azure_openai_configured())
    if use_azure:
        try:
            system_prompt = augment_system_prompt(load_prompt_template(DEFAULT_SYSTEM_PROMPT))
            payload = generate_json_with_azure(system_prompt, prompt)
            return AgentCadDocument.model_validate(payload), "azure_openai"
        except (LlmConfigurationError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
            if provider == "azure":
                raise
            print(f"Azure CAD document generation failed, using deterministic fallback: {exc}", file=sys.stderr)

    spec = prompt_to_spec(prompt)
    return document_from_spec(spec, output_name, description=prompt.strip()), "deterministic_fallback"


def _run_with_repairs(
    script_path: Path,
    document: AgentCadDocument,
    output_name: str,
    output_dir: Path,
    max_repairs: int,
) -> bool:
    for attempt in range(max(1, max_repairs + 1)):
        result = run_script(script_path)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.ok:
            return True
        (output_dir / f"cad_error_attempt_{attempt + 1}.txt").write_text(
            result.stderr or result.stdout,
            encoding="utf-8",
        )
        if attempt < max_repairs:
            script_path = write_script(
                render_agent_cadquery_script(document, output_name),
                output_dir,
                filename="agent_generated_cad.py",
            )

    print("Phase B CAD execution failed. Error logs were written to the output directory.", file=sys.stderr)
    return False


def _slug_from_prompt(prompt: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())[:4]
    return "_".join(words) if words else "agent_part"


if __name__ == "__main__":
    raise SystemExit(main())
