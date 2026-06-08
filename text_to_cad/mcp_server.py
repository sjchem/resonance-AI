"""Minimal MCP stdio server for Resonance CAD Phase B tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from pydantic import ValidationError

from text_to_cad.agent_renderer import render_agent_cadquery_script
from text_to_cad.agent_schema import AgentCadDocument, spec_from_document
from text_to_cad.cad_agent import create_document_for_prompt
from text_to_cad.cad_executor import run_script, write_script
from text_to_cad.cad_generator import _write_preview
from text_to_cad.viewer import write_viewer


SERVER_NAME = "resonance-cad"
SERVER_VERSION = "0.1.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Resonance CAD MCP server over stdio.")
    parser.parse_args(argv)
    server = ResonanceCadMcpServer()
    server.serve()
    return 0


class ResonanceCadMcpServer:
    def serve(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle(request)
            except Exception as exc:
                response = self.error_response(None, -32603, str(exc))
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}

        if method == "initialize":
            return self.success_response(
                request_id,
                {
                    "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self.success_response(request_id, {"tools": self.tools()})
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            return self.success_response(request_id, self.call_tool(tool_name, arguments))

        return self.error_response(request_id, -32601, f"Unknown method: {method}")

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_resonance_cad_document",
                "description": "Create a validated structured CAD document from a natural-language prompt.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string"},
                        "name": {"type": "string"},
                        "provider": {"type": "string", "enum": ["auto", "azure", "fallback"]},
                    },
                    "required": ["prompt"],
                },
            },
            {
                "name": "inspect_resonance_cad",
                "description": "Inspect a structured CAD document and summarize dimensions/features.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"document": {"type": "object"}},
                    "required": ["document"],
                },
            },
            {
                "name": "export_resonance_cad",
                "description": "Export a structured CAD document to STEP, STL, preview, and viewer.html.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document": {"type": "object"},
                        "output_dir": {"type": "string"},
                        "name": {"type": "string"},
                    },
                    "required": ["document", "output_dir", "name"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "create_resonance_cad_document":
                prompt = str(arguments["prompt"])
                output_name = str(arguments.get("name") or "agent_part")
                provider = str(arguments.get("provider") or "auto")
                document = create_document_for_prompt(prompt, output_name, provider)
                return text_result(json.dumps(document.model_dump(mode="json"), indent=2))

            if name == "inspect_resonance_cad":
                document = AgentCadDocument.model_validate(arguments["document"])
                return text_result(json.dumps(_inspect_document(document), indent=2))

            if name == "export_resonance_cad":
                document = AgentCadDocument.model_validate(arguments["document"])
                output_dir = Path(str(arguments["output_dir"]))
                output_name = str(arguments["name"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "agent_document.json").write_text(
                    json.dumps(document.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                spec = spec_from_document(document)
                script = render_agent_cadquery_script(document, output_name)
                script_path = write_script(script, output_dir, filename="agent_generated_cad.py")
                _write_preview(spec, output_dir / "preview.png")
                result = run_script(script_path)
                if not result.ok:
                    (output_dir / "cad_error_attempt_1.txt").write_text(
                        result.stderr or result.stdout,
                        encoding="utf-8",
                    )
                    return text_result(result.stderr or result.stdout, is_error=True)
                write_viewer(output_dir / f"{output_name}.stl", spec, output_dir, output_name)
                return text_result(str(output_dir / "viewer.html"))

        except (KeyError, ValidationError, ValueError) as exc:
            return text_result(str(exc), is_error=True)

        return text_result(f"Unknown tool: {name}", is_error=True)

    @staticmethod
    def success_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _inspect_document(document: AgentCadDocument) -> dict[str, Any]:
    spec = spec_from_document(document)
    part = document.parts[0]
    return {
        "name": document.name,
        "units": document.units,
        "material_hint": document.material_hint,
        "part_count": len(document.parts),
        "operation_count": len(part.operations),
        "bounding_box_mm": {
            "length": spec.length_mm,
            "width": spec.width_mm,
            "height": spec.thickness_mm,
        },
        "holes": spec.hole_count,
        "hole_diameter_mm": spec.hole_diameter_mm if spec.hole_count else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
