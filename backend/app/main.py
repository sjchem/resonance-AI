"""FastAPI app for OpenAI-powered CAD prompt parsing."""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.cad_preview import build_preview_svg
from app.chat_logic import respond_to_cad_chat
from app.openai_client import (
    CADValidationError,
    OpenAIConfigurationError,
    OpenAIRequestError,
    openai_status,
    parse_cad_prompt,
)
from app.schemas import CADChatRequest, CADChatResponse, CADPromptOutput, CADPromptRequest
from app.upload_context import UploadContext, build_upload_context


app = FastAPI(title="Resonance AI", version="0.1.0")


@app.get("/")
async def health_check() -> dict[str, str]:
    """Simple health check for the deployed web app."""

    return {"status": "ok", "service": "openai-cad-prompt-parser"}


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> str:
    """Frontend for prompt parsing and CAD-style preview."""

    return UI_HTML


@app.get("/generate", response_class=HTMLResponse)
async def generate_ui() -> str:
    """Compatibility route for the existing Azure Web App URL."""

    return UI_HTML


@app.get("/models")
async def models() -> dict:
    """Return OpenAI provider configuration without exposing secrets."""

    return openai_status()


@app.post("/parse-cad", response_model=CADPromptOutput)
async def parse_cad(request: CADPromptRequest) -> CADPromptOutput:
    """Parse a natural-language CAD prompt into validated structured JSON."""

    try:
        return await parse_cad_prompt(request.prompt)
    except OpenAIConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CADValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/chat-cad", response_model=CADChatResponse)
async def chat_cad(request: CADChatRequest) -> CADChatResponse:
    """Return an interactive chat reply plus the current CAD state."""

    try:
        return await respond_to_cad_chat(request.message, request.prompt, request.history)
    except OpenAIConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CADValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/generate-cad")
async def generate_cad(request: CADPromptRequest) -> dict:
    """Parse prompt JSON and return a lightweight preview payload."""

    parsed = await parse_cad(request)
    return {
        "cad_intent": parsed.model_dump(),
        "preview_svg": build_preview_svg(parsed),
        "note": "Interactive preview only for the current web POC. STEP/STL export is not implemented in this deployed flow yet.",
    }


@app.post("/preview-cad")
async def preview_cad(parsed: CADPromptOutput) -> dict:
    """Return a CAD-style preview for already structured input."""

    return {
        "cad_intent": parsed.model_dump(),
        "preview_svg": build_preview_svg(parsed),
        "note": "Interactive preview only for the current web POC. STEP/STL export is not implemented in this deployed flow yet.",
    }


@app.post("/upload-context", response_model=UploadContext)
async def upload_context(file: UploadFile = File(...)) -> UploadContext:
    """Extract prompt context from an uploaded document, image, or CAD file."""

    try:
        return await build_upload_context(file)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read uploaded file: {exc}") from exc


def _safe_export_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "model").strip()).strip("_")
    return cleaned[:60] or "model"


@app.post("/export/step")
async def export_step(payload: dict) -> FileResponse:
    """Generate a real STEP file from the CAD prompt via the CadQuery pipeline.

    STEP is a B-Rep exchange format that cannot be produced from the browser
    preview mesh, so it is generated server-side. This requires the optional
    CadQuery dependency; when it is not installed the endpoint returns 501 and
    the UI falls back to the client-side formats (STL, GLB, DXF, PNG, PDF, JSON).
    """

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="A prompt is required to generate a STEP file.")

    name = _safe_export_name(str(payload.get("name", "model")))

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from text_to_cad.cad_agent import generate_with_agent
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                "STEP export needs the CadQuery pipeline, which is not installed in this "
                "deployment. Use STL, GLB, DXF, PNG, PDF, or JSON instead."
            ),
        ) from exc

    output_dir = Path(tempfile.mkdtemp(prefix="resonance_step_"))
    try:
        code = generate_with_agent(
            prompt=prompt,
            output_dir=output_dir,
            output_name=name,
            provider="auto",
            execute=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        raise HTTPException(status_code=500, detail=f"STEP generation failed: {exc}") from exc

    step_path = output_dir / f"{name}.step"
    if code != 0 or not step_path.exists():
        raise HTTPException(status_code=500, detail="STEP generation did not produce a file.")

    return FileResponse(step_path, media_type="application/step", filename=f"{name}.step")


UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resonance AI — Vibracoustic</title>
  <style>
    :root {
      --bg: #eef5ff;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #647084;
      --line: #d8e0ea;
      --brand: #071f3f;
      --brand-2: #0b2f5f;
      --accent: #d8222a;
      --accent-dark: #ad1720;
      --danger: #b42318;
      --cad: #0f766e;
      --cad-dark: #0b4f49;
      --soft: #e7f0fb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Segoe UI Variable", "Aptos", "Noto Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: #fff;
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 0 rgba(17, 24, 39, 0.04);
    }
    .topbar {
      width: min(1440px, calc(100% - 48px));
      margin: 0 auto;
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }
    .brand-lockup {
      display: flex;
      align-items: center;
      gap: 20px;
      min-width: 0;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      white-space: nowrap;
    }
    .vc-logo {
      display: block;
      width: 245px;
      height: auto;
      max-height: 58px;
      flex: 0 0 auto;
      object-fit: contain;
    }
    .product {
      padding-left: 18px;
      border-left: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      line-height: 1.1;
    }
    .product .name {
      font-size: 17px;
      font-weight: 700;
      color: var(--brand);
      white-space: nowrap;
    }
    .product .tag {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1.4px;
      margin-top: 2px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(30px, 4vw, 54px);
      line-height: 0.98;
      letter-spacing: 0;
      max-width: 720px;
    }
    main {
      width: min(1440px, calc(100% - 48px));
      margin: 0 auto 32px;
    }
    .hero {
      min-height: 220px;
      margin: 0 calc(50% - 50vw) 28px;
      padding: 32px 0;
      color: #0f2a49;
      background:
        linear-gradient(110deg, rgba(222, 236, 255, 0.96), rgba(201, 226, 255, 0.93) 55%, rgba(186, 216, 255, 0.9)),
        radial-gradient(circle at 72% 38%, rgba(255, 255, 255, 0.62), transparent 34%),
        #d9ecff;
      display: grid;
      align-items: end;
      overflow: hidden;
    }
    .hero-inner {
      width: min(1440px, calc(100% - 48px));
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.7fr);
      gap: 28px;
      align-items: end;
    }
    .eyebrow {
      margin: 0 0 14px;
      color: #3d658f;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 1.6px;
      text-transform: uppercase;
    }
    .hero-copy {
      margin: 0;
      max-width: 660px;
      color: #33587f;
      font-size: 16px;
      line-height: 1.6;
    }
    .hero-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1px;
      background: rgba(83, 128, 178, 0.24);
      border: 1px solid rgba(83, 128, 178, 0.28);
    }
    .metric {
      min-height: 92px;
      padding: 16px;
      background: rgba(245, 250, 255, 0.82);
    }
    .metric strong {
      display: block;
      font-size: 21px;
      line-height: 1;
      margin-bottom: 9px;
    }
    .metric span {
      display: block;
      color: #4c6f96;
      font-size: 12px;
      line-height: 1.35;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(360px, 460px) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }
    .workbench, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 22px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }
    .section-title {
      margin: 0 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .section-title strong {
      color: var(--brand);
      font-size: 16px;
    }
    .title-head {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .download {
      position: relative;
    }
    .download-btn {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line, #d8e0ea);
      background: #fff;
      color: var(--brand);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      padding: 7px 12px;
      border-radius: 9px;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
    }
    .download-btn:hover { background: #f3f6fb; }
    .download-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .download-btn .chev { font-size: 10px; }
    .download-menu {
      position: absolute;
      top: calc(100% + 6px);
      right: 0;
      z-index: 30;
      width: 290px;
      background: #fff;
      border: 1px solid var(--line, #d8e0ea);
      border-radius: 12px;
      box-shadow: 0 18px 40px rgba(15, 36, 64, 0.18);
      padding: 8px;
      display: none;
    }
    .download-menu.open { display: block; }
    .download-group + .download-group {
      margin-top: 6px;
      border-top: 1px solid #eef2f7;
      padding-top: 6px;
    }
    .download-group-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      padding: 4px 8px;
    }
    .download-item {
      display: flex;
      align-items: baseline;
      gap: 8px;
      width: 100%;
      text-align: left;
      border: none;
      background: none;
      font: inherit;
      padding: 8px;
      border-radius: 8px;
      cursor: pointer;
      color: var(--ink, #1c2733);
    }
    .download-item:hover { background: #f3f6fb; }
    .download-item .fmt { font-weight: 700; font-size: 13px; min-width: 44px; }
    .download-item .desc { font-size: 12px; color: var(--muted); }
    .download-item:disabled { opacity: 0.5; cursor: not-allowed; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: #0f5132;
      background: #e7f5ed;
      border: 1px solid #bde3cb;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 800;
    }
    .status-pill::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #16a34a;
    }
    label {
      display: block;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    textarea, input, select {
      width: 100%;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 12px;
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
      background: #fff;
    }
    textarea { min-height: 130px; }
    input, select { resize: none; }
    .field { margin-bottom: 12px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .divider {
      margin: 20px 0;
      border: 0;
      border-top: 1px solid var(--line);
    }
    button {
      width: 100%;
      margin-top: 14px;
      border: 0;
      border-radius: 3px;
      background: var(--brand);
      color: white;
      padding: 12px 16px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }
    button:hover { background: var(--brand-2); }
    button:disabled { opacity: 0.62; cursor: wait; }
    .primary-action {
      background: var(--accent);
    }
    .primary-action:hover { background: var(--accent-dark); }
    .chat {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .chat-shell {
      min-height: 520px;
      max-height: 620px;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: linear-gradient(135deg, #f9fbfd, #eef3f8);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .chat-shell.idle {
      min-height: 560px;
      max-height: none;
    }
    .chat-log {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .chat-shell.idle .chat-log {
      flex: 0 0 auto;
      justify-content: flex-start;
      padding: 16px;
    }
    .msg {
      padding: 10px 12px;
      border-radius: 3px;
      max-width: 85%;
      font-size: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--brand);
      color: #fff;
    }
    .msg.bot {
      align-self: flex-start;
      background: #fff;
      border: 1px solid var(--line);
      color: var(--ink);
    }
    .msg.intro {
      max-width: 100%;
      padding: 18px 20px;
      font-size: 15px;
      line-height: 1.6;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }
    .msg.bot.err { border-color: #fecdca; background: #fff7f6; color: var(--danger); }
    .chat-composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      align-items: end;
    }
    .chat-composer textarea {
      flex: 1;
      min-height: 96px;
      max-height: 220px;
      resize: none;
      margin: 0;
    }
    .chat-shell.idle .chat-composer {
      padding: 16px;
      grid-template-columns: 1fr;
      gap: 12px;
      border-top: 0;
      background: transparent;
    }
    .chat-shell.idle .chat-composer textarea {
      min-height: 240px;
      max-height: 280px;
      padding: 18px;
      font-size: 16px;
      line-height: 1.6;
    }
    .chat-actions {
      display: flex;
      align-items: end;
    }
    .chat-shell.idle .chat-actions {
      justify-content: flex-end;
    }
    .chat-composer button {
      width: auto;
      margin: 0;
      min-width: 104px;
      padding: 12px 18px;
    }
    .attachment-tools {
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px dashed #b8c8da;
      background: #f8fbff;
    }
    .label-with-info {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 0;
    }
    .info-dot {
      position: relative;
      display: inline-grid;
      place-items: center;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      color: #fff;
      background: var(--brand);
      font-size: 12px;
      font-weight: 800;
      cursor: help;
    }
    .info-dot:hover::after,
    .info-dot:focus::after {
      content: attr(data-tooltip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 9px);
      transform: translateX(-50%);
      z-index: 10;
      width: min(320px, 82vw);
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.45;
      text-align: left;
      white-space: normal;
    }
    .info-dot:hover::before,
    .info-dot:focus::before {
      content: "";
      position: absolute;
      left: 50%;
      bottom: calc(100% + 3px);
      transform: translateX(-50%);
      border: 6px solid transparent;
      border-top: 0;
      border-bottom-color: #fff;
      z-index: 11;
    }
    .attachment-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .attachment-row input {
      flex: 1;
      padding: 9px;
      background: #fff;
    }
    .attachment-row button {
      flex: 0 0 auto;
      width: auto;
      margin: 0;
      padding: 9px 14px;
      background: #385a7e;
    }
    .attachment-row button:hover {
      background: #2a4562;
    }
    .attachment-list {
      display: grid;
      gap: 8px;
    }
    .attachment-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 9px 10px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 13px;
    }
    .attachment-item strong {
      color: var(--brand);
      overflow-wrap: anywhere;
    }
    .attachment-item span {
      color: var(--muted);
      display: block;
      margin-top: 2px;
    }
    .attachment-item button {
      width: auto;
      margin: 0;
      padding: 6px 9px;
      color: var(--danger);
      background: #fff1f0;
    }
    .activity-panel {
      display: none;
      gap: 8px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #f8fbff;
    }
    .activity-panel.active {
      display: grid;
    }
    .activity-status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--brand);
      font-size: 13px;
      font-weight: 700;
    }
    .activity-phase {
      color: var(--muted);
      font-weight: 600;
      text-align: right;
    }
    .activity-track {
      width: 100%;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dbe6f2;
    }
    .activity-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #0f766e, #24a39a);
      transition: width 260ms ease;
    }
    .activity-panel.done .activity-fill {
      background: linear-gradient(90deg, #11895d, #34c77b);
    }
    .activity-panel.error .activity-fill {
      background: linear-gradient(90deg, #d8222a, #ef6a6f);
    }
    .preview-progress {
      display: none;
      gap: 8px;
      margin-bottom: 14px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #f4f9ff;
      border-radius: 3px;
    }
    .preview-progress.active {
      display: grid;
    }
    .preview-progress-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--brand);
      font-size: 13px;
      font-weight: 700;
    }
    .preview-progress-phase {
      color: var(--muted);
      font-weight: 600;
    }
    .preview-progress-track {
      width: 100%;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dbe6f2;
    }
    .preview-progress-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #0f766e, #24a39a);
      transition: width 260ms ease;
    }
    .preview-progress.done .preview-progress-fill {
      background: linear-gradient(90deg, #11895d, #34c77b);
    }
    .preview-progress.error .preview-progress-fill {
      background: linear-gradient(90deg, #d8222a, #ef6a6f);
    }
    .category-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .category-chip {
      width: auto;
      margin: 0;
      padding: 12px 14px;
      display: grid;
      gap: 4px;
      text-align: left;
      background: #f4f9ff;
      color: var(--brand);
      border: 1px solid var(--line);
      border-radius: 4px;
      cursor: pointer;
      transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
    }
    .category-chip strong {
      font-size: 14px;
      font-weight: 750;
      color: var(--brand);
    }
    .category-chip span {
      font-size: 12px;
      color: var(--muted);
      font-weight: 500;
    }
    .category-chip:hover {
      background: #e7f0fb;
      border-color: #b8c8da;
    }
    .category-chip.active {
      background: var(--brand);
      border-color: var(--brand);
    }
    .category-chip.active strong,
    .category-chip.active span {
      color: #fff;
    }
    .summary-box {
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      background: #fbfdff;
      padding: 14px;
      min-height: 112px;
    }
    .summary-box p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .summary-box dl {
      margin: 0;
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 7px 14px;
      font-size: 14px;
    }
    .summary-box dt {
      color: var(--muted);
      font-weight: 700;
    }
    .summary-box dd {
      margin: 0;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .stack {
      display: grid;
      gap: 18px;
    }
    .preview {
      min-height: 470px;
      display: grid;
      place-items: center;
      overflow: hidden;
      position: relative;
      background:
        linear-gradient(90deg, rgba(216, 224, 234, 0.34) 1px, transparent 1px),
        linear-gradient(rgba(216, 224, 234, 0.34) 1px, transparent 1px),
        #fff;
      background-size: 28px 28px;
    }
    .viewer3d {
      width: 100%;
      height: 100%;
      min-height: 470px;
    }
    .viewer3d canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    svg {
      width: 100%;
      max-height: 560px;
    }
    .placeholder {
      width: min(560px, 90%);
      display: grid;
      gap: 14px;
      color: var(--muted);
      text-align: center;
    }
    .placeholder svg {
      width: 100%;
      max-height: 300px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    pre {
      margin: 0;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .param-controls {
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: 420px;
      overflow: auto;
    }
    .param-row {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 6px 12px;
    }
    .param-row .param-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--brand);
    }
    .param-row .param-value {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      justify-self: end;
    }
    .param-row .param-number {
      width: 76px;
      padding: 5px 7px;
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 13px;
      color: var(--brand);
      text-align: right;
      background: #fff;
    }
    .param-row .param-unit {
      font-size: 12px;
      color: var(--muted);
    }
    .param-row .param-slider {
      grid-column: 1 / -1;
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .param-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding-top: 4px;
      border-top: 1px solid var(--line);
    }
    .param-reset {
      background: #fff;
      color: var(--brand);
      border: 1px solid var(--line);
      padding: 6px 14px;
      font-size: 13px;
    }
    .param-reset:hover {
      background: #eef3f9;
    }
    .sim-block {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sim-block strong {
      font-size: 13px;
      color: var(--brand);
    }
    .sim-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .sim-table th, .sim-table td {
      text-align: left;
      padding: 4px 6px;
      border-bottom: 1px solid var(--line);
    }
    .sim-table th {
      color: var(--muted);
      font-weight: 600;
    }
    .sim-table td.sim-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--brand);
    }
    .sim-stiff {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 18px;
      font-size: 13px;
      color: var(--muted);
    }
    .sim-stiff b {
      color: var(--brand);
      font-variant-numeric: tabular-nums;
    }
    .error-box {
      width: min(860px, 92%);
      margin: 0;
      padding: 18px;
      color: var(--danger);
      background: #fff7f6;
      border: 1px solid #fecdca;
      border-radius: 3px;
    }
    @media (max-width: 1080px) {
      .hero-inner, .workspace { grid-template-columns: 1fr; }
      .hero-metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      .topbar, main, .hero-inner { width: min(100% - 28px, 1440px); }
      .brand-lockup { align-items: flex-start; gap: 12px; flex-direction: column; }
      .product { padding-left: 0; border-left: 0; }
      .hero { padding-block: 26px; }
      .hero-metrics { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand-lockup">
        <div class="logo">
          <img
            src="https://vibracoustic.com/wp-content/uploads/2023/08/vibracoustic.webp"
            alt="Vibracoustic"
            class="vc-logo"
          >
        </div>
        <div class="product">
          <span class="name">Resonance AI</span>
          <span class="tag">CAD Prompt Studio</span>
        </div>
      </div>
    </div>
  </header>
  <main>
    <section class="hero">
      <div class="hero-inner">
        <div>
          <p class="eyebrow">NVH CAD Intelligence</p>
          <h1>Resonance AI</h1>
          <p class="hero-copy">Prompt-to-CAD intent capture for vibroacoustic components, tuned for fast engineering review.</p>
        </div>
        <div class="hero-metrics" aria-label="Workflow summary">
          <div class="metric"><strong>01</strong><span>Prompt intake</span></div>
          <div class="metric"><strong>02</strong><span>Parametric editor</span></div>
          <div class="metric"><strong>03</strong><span>Preview review</span></div>
        </div>
      </div>
    </section>

    <section class="workspace">
      <section class="workbench chat">
        <div class="section-title">
          <strong>Engineering Chat</strong>
          <span class="status-pill">Model ready</span>
        </div>
        <div class="attachment-tools">
          <label for="contextFile" class="label-with-info">
            Add document, image, or old CAD model
            <span
              class="info-dot"
              tabindex="0"
              aria-label="Supported file types: PDF, image jpg, JSON, STEP/STP/IGES/STL/OBJ/DXF/SCAD/FCStd"
              data-tooltip="Supported file types: PDF, image (jpg), JSON, STEP/STP/IGES/STL/OBJ/DXF/SCAD/FCStd"
            >i</span>
          </label>
          <div class="attachment-row">
            <input
              id="contextFile"
              type="file"
              accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.step,.stp,.iges,.igs,.stl,.obj,.dxf,.scad,.fcstd,.txt,.md,.json,.xml,.csv"
            >
            <button id="uploadContextButton" type="button">Attach File</button>
          </div>
          <div id="attachmentList" class="attachment-list"></div>
        </div>
        <div id="chatShell" class="chat-shell idle">
          <div id="chatLog" class="chat-log">
            <div class="msg bot intro">Start by adding a PDF, image, JSON status file, or old CAD model. I will extract context and draft a short prompt for approval. Images are reference-only in this POC, so please type the key dimensions in chat.</div>
          </div>
          <form id="chatForm" class="chat-composer">
            <textarea id="chatInput" placeholder="Describe the part, main dimensions, material, and anything important for the CAD model..." autocomplete="off"></textarea>
            <div class="chat-actions">
              <button type="submit" id="chatSend" class="primary-action">Send</button>
            </div>
          </form>
        </div>
        <div id="activityPanel" class="activity-panel" aria-live="polite">
          <div class="activity-status">
            <span id="activityLabel">Ready</span>
            <span id="activityPhase" class="activity-phase">Waiting</span>
          </div>
          <div class="activity-track" aria-hidden="true">
            <div id="activityFill" class="activity-fill"></div>
          </div>
        </div>
        <div class="summary-box" id="summaryBox">
          <p>Upload a file or write a request. I will summarize the proposed CAD intent before generating the model.</p>
        </div>
      </section>

      <section class="stack">
        <div class="panel" id="categoryPanel">
          <div class="section-title">
            <strong>Product family</strong>
            <span class="muted">Pick to load a starter prompt</span>
          </div>
          <div class="category-grid" id="categoryGrid">
            <button type="button" class="category-chip" data-category="bushing">
              <strong>Rubber bushing</strong>
              <span>Suspension &amp; chassis bushings</span>
            </button>
            <button type="button" class="category-chip" data-category="bonded-bushing">
              <strong>Bonded bushing</strong>
              <span>Rubber-metal bonded sleeve</span>
            </button>
            <button type="button" class="category-chip" data-category="flanged-bushing">
              <strong>Flanged bushing</strong>
              <span>Bushing with mounting flange</span>
            </button>
            <button type="button" class="category-chip" data-category="air-spring">
              <strong>Air spring</strong>
              <span>Air suspension element</span>
            </button>
            <button type="button" class="category-chip" data-category="coil-spring">
              <strong>Coil spring</strong>
              <span>Helical compression spring</span>
            </button>
            <button type="button" class="category-chip" data-category="damper">
              <strong>Damper / decoupler</strong>
              <span>Vibration damper element</span>
            </button>
          </div>
        </div>
        <div class="panel">
          <div class="section-title">
            <div class="title-head">
              <strong>3D CAD Model</strong>
              <span class="muted">Interactive preview</span>
            </div>
            <div class="download">
              <button type="button" id="downloadBtn" class="download-btn" aria-haspopup="true" aria-expanded="false" disabled>
                Download <span class="chev">&#9662;</span>
              </button>
              <div id="downloadMenu" class="download-menu" role="menu">
                <div class="download-group">
                  <div class="download-group-label">CAD model</div>
                  <button type="button" class="download-item" data-format="step" role="menuitem"><span class="fmt">STEP</span><span class="desc">Engineering CAD exchange (.step)</span></button>
                  <button type="button" class="download-item" data-format="stl" role="menuitem"><span class="fmt">STL</span><span class="desc">3D printing / mesh (.stl)</span></button>
                  <button type="button" class="download-item" data-format="glb" role="menuitem"><span class="fmt">GLB</span><span class="desc">Interactive 3D sharing (.glb)</span></button>
                  <button type="button" class="download-item" data-format="dxf" role="menuitem"><span class="fmt">DXF</span><span class="desc">2D profile / drawing (.dxf)</span></button>
                </div>
                <div class="download-group">
                  <div class="download-group-label">Documentation</div>
                  <button type="button" class="download-item" data-format="png" role="menuitem"><span class="fmt">PNG</span><span class="desc">CAD preview image (.png)</span></button>
                  <button type="button" class="download-item" data-format="pdf" role="menuitem"><span class="fmt">PDF</span><span class="desc">Technical summary (.pdf)</span></button>
                  <button type="button" class="download-item" data-format="json" role="menuitem"><span class="fmt">JSON</span><span class="desc">CAD parameters (.json)</span></button>
                </div>
                <div class="download-group">
                  <div class="download-group-label">Convenience</div>
                  <button type="button" class="download-item" data-format="zip" role="menuitem"><span class="fmt">ZIP</span><span class="desc">Download all files (.zip)</span></button>
                </div>
              </div>
            </div>
          </div>
          <div id="previewProgress" class="preview-progress" aria-live="polite">
            <div class="preview-progress-head">
              <span id="previewProgressLabel">Generating CAD model</span>
              <span id="previewProgressPhase" class="preview-progress-phase">Working</span>
            </div>
            <div class="preview-progress-track" aria-hidden="true">
              <div id="previewProgressFill" class="preview-progress-fill"></div>
            </div>
          </div>
          <div class="preview" id="preview">
            <div class="placeholder">
              <p class="muted">Interactive 3D preview will appear here after the CAD intent is parsed.</p>
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="section-title">
            <strong>Parametric Editor</strong>
            <span id="paramHint" class="muted">Drag a slider to resize the model live.</span>
          </div>
          <div id="paramControls" class="param-controls">
            <p class="muted">Adjustable dimensions will appear here once a model is generated. Use the Download menu to export the edited part.</p>
          </div>
          <div id="simResults"></div>
          <pre id="jsonOutput" hidden>{}</pre>
        </div>
      </section>
    </section>
  </main>

  <script type="module">
    const preview = document.getElementById("preview");
    const jsonOutput = document.getElementById("jsonOutput");
    const paramControls = document.getElementById("paramControls");
    const paramHint = document.getElementById("paramHint");
    const simResults = document.getElementById("simResults");
    const summaryBox = document.getElementById("summaryBox");
    const chatShell = document.getElementById("chatShell");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatLog = document.getElementById("chatLog");
    const chatSend = document.getElementById("chatSend");
    const activityPanel = document.getElementById("activityPanel");
    const activityLabel = document.getElementById("activityLabel");
    const activityPhase = document.getElementById("activityPhase");
    const activityFill = document.getElementById("activityFill");
    const previewProgress = document.getElementById("previewProgress");
    const previewProgressLabel = document.getElementById("previewProgressLabel");
    const previewProgressPhase = document.getElementById("previewProgressPhase");
    const previewProgressFill = document.getElementById("previewProgressFill");
    const contextFile = document.getElementById("contextFile");
    const uploadContextButton = document.getElementById("uploadContextButton");
    const attachmentList = document.getElementById("attachmentList");
    const categoryGrid = document.getElementById("categoryGrid");
    const downloadBtn = document.getElementById("downloadBtn");
    const downloadMenu = document.getElementById("downloadMenu");
    let activeViewer = null;
    let activityTimer = null;
    let requestContext = [];
    let attachmentContexts = [];
    let pendingDraftPrompt = "";
    let chatHistory = [];
    let lastExport = { mesh: null, canvas: null, intent: null, prompt: "", name: "model" };
    // Persistent camera so live parametric edits keep the same view angle.
    let viewerCamera = { rotationX: -0.55, rotationY: 0.78, zoom: 1 };
    // Parametric editor state.
    let currentEditIntent = null;
    let baseGeometry = null;
    let paramRenderQueued = false;
    // When true, ignore the uploaded mesh and render the parametric model instead
    // (set after "Convert to editable bushing").
    let preferParametric = false;
    // Mesh-warp editing: keep the real uploaded geometry but stretch OD/ID/height.
    let meshEditMode = false;
    let overrideMeshFaces = null;
    let editableMesh = null;

    uploadContextButton.addEventListener("click", uploadContextFile);
    contextFile.addEventListener("change", () => {
      if (contextFile.files && contextFile.files.length) {
        uploadContextFile();
      }
    });
    chatInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        chatForm.requestSubmit();
      }
    });
    chatInput.addEventListener("input", autoResizeChatInput);
    autoResizeChatInput();
    syncChatState();

    const categoryStarters = {
      "bushing": "I want a rubber suspension bushing. Outer diameter 60 mm, inner diameter 20 mm, height 40 mm. Material: rubber, Shore A 55. Please ask me anything else needed (chamfer, fillet, load direction).",
      "bonded-bushing": "I want a rubber-metal bonded bushing. Outer diameter 60 mm with a 2 mm outer steel sleeve, inner diameter 20 mm with a 1.5 mm inner steel sleeve, height 40 mm. Rubber Shore A 55 in between. Ask me for any missing details.",
      "flanged-bushing": "I want a flanged rubber bushing. Outer diameter 50 mm, inner diameter 16 mm, height 35 mm, flange diameter 70 mm, flange thickness 4 mm. Rubber Shore A 60. Ask me for any missing details.",
      "air-spring": "I want an air spring element for light-vehicle suspension. Top mounting diameter 120 mm, bellows diameter 150 mm, free height 180 mm, internal pressure 6 bar. Ask me for any missing geometry or mounting details.",
      "coil-spring": "I want a steel compression coil spring. Coil diameter 40 mm, free length 80 mm, wire thickness 4 mm, 8 coils, material steel. Ask me for any missing details.",
      "damper": "I want a hydraulic damper / decoupler mount. Body diameter 50 mm, height 60 mm, rubber Shore A 50, with an inner steel sleeve 12 mm bore. Ask me for any missing details."
    };

    categoryGrid.addEventListener("click", (event) => {
      const button = event.target.closest(".category-chip");
      if (!button) return;
      const key = button.dataset.category;
      const starter = categoryStarters[key];
      if (!starter) return;
      for (const chip of categoryGrid.querySelectorAll(".category-chip")) {
        chip.classList.toggle("active", chip === button);
      }
      chatInput.value = starter;
      autoResizeChatInput();
      chatInput.focus();
      chatInput.setSelectionRange(chatInput.value.length, chatInput.value.length);
    });

    downloadBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      if (downloadBtn.disabled) return;
      const open = downloadMenu.classList.toggle("open");
      downloadBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.addEventListener("click", (event) => {
      if (!downloadMenu.contains(event.target) && event.target !== downloadBtn) {
        downloadMenu.classList.remove("open");
        downloadBtn.setAttribute("aria-expanded", "false");
      }
    });
    downloadMenu.addEventListener("click", (event) => {
      const item = event.target.closest(".download-item");
      if (!item) return;
      downloadMenu.classList.remove("open");
      downloadBtn.setAttribute("aria-expanded", "false");
      Promise.resolve(handleDownload(item.dataset.format)).catch((error) => {
        appendMsg("bot", "Download failed: " + (error && error.message ? error.message : error));
      });
    });

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = chatInput.value.trim();
      if (!text) return;
      appendMsg("user", text);
      chatHistory.push({role: "user", content: text});
      const approvedDraft = pendingDraftPrompt && isApproval(text);
      const requestText = approvedDraft ? pendingDraftPrompt : text;
      requestContext.push(requestText);
      if (!approvedDraft) {
        pendingDraftPrompt = "";
      }
      chatInput.value = "";
      autoResizeChatInput();
      chatSend.disabled = true;
      chatSend.textContent = "Working...";
      startActivity("Processing CAD request", [
        "Reading request",
        "Parsing CAD intent",
        "Preparing preview"
      ]);
      const thinking = appendMsg("bot", "Reviewing your request...");
      try {
        const prompt = buildFullPrompt();
        const response = await fetch("/chat-cad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({message: text, prompt, history: chatHistory.slice(0, -1)})
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "CAD parsing failed");
        }
        const intent = payload.cad_intent || {};
        const previewReady = Boolean(payload.preview_ready);
        jsonOutput.textContent = JSON.stringify(intent, null, 2);
        lastExport.intent = intent;
        lastExport.prompt = prompt;
        lastExport.name = exportBaseName(intent);
        downloadBtn.disabled = !intent || Object.keys(intent).length === 0;
        // A fresh chat generation renders the parsed intent, not a prior mesh-warp edit.
        meshEditMode = false;
        overrideMeshFaces = null;
        if (previewReady) {
          try {
            await render3DPreview(intent);
            buildParamControls(intent);
          } catch (previewError) {
            cleanupViewer();
            preview.innerHTML = '<div class="placeholder"><p class="muted">The CAD intent was parsed, but the interactive preview library could not be loaded.</p></div>';
          }
        } else {
          cleanupViewer();
          currentEditIntent = null;
          buildParamControls(null);
          preview.innerHTML = '<div class="placeholder"><p class="muted">I need one more engineering detail before I can show a useful preview.</p></div>';
        }
        updateSummary(intent);
        thinking.textContent = payload.assistant_message || formatChatSummary(intent);
        chatHistory.push({role: "assistant", content: thinking.textContent});
        completeActivity(previewReady ? "Preview updated" : "More detail needed");
      } catch (error) {
        thinking.classList.add("err");
        thinking.textContent = error.message;
        summaryBox.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
        preview.innerHTML = `<pre class="error-box">${escapeHtml(error.message)}</pre>`;
        failActivity("Request failed");
      } finally {
        chatSend.disabled = false;
        chatSend.textContent = "Send";
        chatInput.focus();
      }
    });

    async function uploadContextFile() {
      if (uploadContextButton.disabled) {
        return;
      }
      const file = contextFile.files && contextFile.files[0];
      if (!file) {
        appendMsg("bot", "Choose a PDF, image, or CAD file first.");
        return;
      }

      // A new upload starts from its exact mesh again (undo any prior convert).
      preferParametric = false;
      meshEditMode = false;
      overrideMeshFaces = null;
      editableMesh = null;

      uploadContextButton.disabled = true;
      uploadContextButton.textContent = "Reading...";
      startActivity("Reading uploaded file", [
        "Uploading file",
        "Extracting context",
        "Drafting prompt"
      ]);
      try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/upload-context", {
          method: "POST",
          body: formData
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "File upload failed");
        }
        // Parse real geometry from STL uploads so the 3D viewer shows the actual part.
        let meshNote = "";
        const lowerName = (file.name || "").toLowerCase();
        if (lowerName.endsWith(".stl")) {
          try {
            const buffer = await file.arrayBuffer();
            const mesh = parseStlMesh(buffer, payload.filename);
            if (mesh) {
              payload.clientMesh = mesh;
              payload.measured = measuredSummaryFromMesh(mesh);
              const triNote = mesh.shown < mesh.original
                ? ` (preview simplified to ${mesh.shown.toLocaleString()} of ${mesh.original.toLocaleString()} triangles)`
                : ` (${mesh.original.toLocaleString()} triangles)`;
              meshNote = `Loaded the actual STL geometry into the 3D viewer${triNote}.`;
            } else {
              meshNote = "The STL could not be parsed for an exact preview, so I will show a sized model instead.";
            }
          } catch (meshError) {
            meshNote = "The STL could not be parsed for an exact preview, so I will show a sized model instead.";
          }
        } else if (lowerName.endsWith(".step") || lowerName.endsWith(".stp")) {
          meshNote = "STEP file received. I can read its dimensions, but an exact surface preview needs the geometry kernel, so I will show a sized model for now.";
        }

        attachmentContexts.push(payload);
        renderAttachmentList();
        pendingDraftPrompt = draftPromptFromAttachments();

        if (payload.clientMesh) {
          try {
            await render3DPreview({ part_type: "uploaded", geometry: {} });
            lastExport.name = exportBaseName({ part_type: payload.filename });
            downloadBtn.disabled = false;
            currentEditIntent = null;
            buildParamControls(null);
          } catch (previewError) {
            cleanupViewer();
          }
        }

        const messageParts = [`Extracted file context from ${payload.filename}.`, payload.summary];
        if (meshNote) {
          messageParts.push(meshNote);
        }
        messageParts.push(`Proposed short CAD prompt:\\n${pendingDraftPrompt}`);
        messageParts.push('Type "proceed" to apply the dimensions and lock in the model, or type corrections/additional dimensions.');
        appendMsg("bot", messageParts.join("\\n\\n"));

        summaryBox.innerHTML = `
          <p><strong>Awaiting approval.</strong> Review the proposed short prompt in the chat. Type "proceed" to generate the structured CAD result and preview, or add corrections.</p>
        `;
        chatInput.value = "proceed";
        autoResizeChatInput();
        contextFile.value = "";
        completeActivity(payload.clientMesh ? "Geometry loaded" : "Draft ready");
      } catch (error) {
        appendMsg("bot err", error.message);
        failActivity("Upload failed");
      } finally {
        uploadContextButton.disabled = false;
        uploadContextButton.textContent = "Attach File";
      }
    }

    function buildFullPrompt() {
      const parts = [];
      if (attachmentContexts.length) {
        parts.push("Use the following uploaded engineering context when interpreting the CAD request.");
        for (const context of attachmentContexts) {
          parts.push(context.prompt_context);
        }
      }
      if (requestContext.length) {
        parts.push("User chat request:");
        parts.push(requestContext.join("\\n"));
      }
      parts.push(
        "Return a CAD intent that can be used to generate the best possible 3D CAD model preview. Do not invent missing critical dimensions."
      );
      return parts.join("\\n\\n");
    }

    function draftPromptFromAttachments() {
      if (!attachmentContexts.length) {
        return "";
      }
      const summaries = attachmentContexts.map((context) => context.summary).join(" ");
      const measured = attachmentContexts
        .map((context) => context.measured)
        .filter(Boolean)
        .join(" ");
      const lines = [
        "Create a CAD model from the uploaded engineering context.",
        summaries,
      ];
      if (measured) {
        lines.push(measured);
      }
      lines.push(
        "Extract the part type, material, dimensions, chamfer/fillet details, and any status or validation hints from the uploaded file.",
        "If critical dimensions are missing, mark them as missing rather than inventing them."
      );
      return lines.join(" ");
    }

    function isApproval(value) {
      return /^(proceed|agree|approved|approve|yes|ok|okay|generate|create|go)$/i.test(value.trim());
    }

    function renderAttachmentList() {
      if (!attachmentContexts.length) {
        attachmentList.innerHTML = "";
        return;
      }
      attachmentList.innerHTML = attachmentContexts.map((context, index) => `
        <div class="attachment-item">
          <div>
            <strong>${escapeHtml(context.filename)}</strong>
            <span>${escapeHtml(context.file_kind)} · ${formatBytes(context.size_bytes)}</span>
          </div>
          <button type="button" data-remove-context="${index}">Remove</button>
        </div>
      `).join("");

      for (const button of attachmentList.querySelectorAll("[data-remove-context]")) {
        button.addEventListener("click", () => {
          const index = Number(button.dataset.removeContext);
          attachmentContexts.splice(index, 1);
          pendingDraftPrompt = attachmentContexts.length ? draftPromptFromAttachments() : "";
          renderAttachmentList();
        });
      }
    }

    function formatBytes(value) {
      const bytes = Number(value) || 0;
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function formatChatSummary(intent) {
      const missing = intent.missing_information || [];
      const summary = [
        `Summary: ${formatPartName(intent.part_type)}${intent.material?.name ? " in " + intent.material.name : ""}.`,
        `Dimensions: ${formatDimensions(intent.geometry || {})}.`,
        "Interactive preview created on the right."
      ];
      if (missing.length) {
        summary.push(`Needs clarification: ${missing.join("; ")}.`);
      }
      return summary.join("\\n");
    }

    function updateSummary(intent) {
      const missing = intent.missing_information || [];
      summaryBox.innerHTML = `
        <dl>
          <dt>Part</dt><dd>${escapeHtml(formatPartName(intent.part_type))}</dd>
          <dt>Material</dt><dd>${escapeHtml(intent.material?.name || "Not specified")}</dd>
          <dt>Dimensions</dt><dd>${escapeHtml(formatDimensions(intent.geometry || {}))}</dd>
          <dt>Output</dt><dd>Interactive preview with live parametric editing</dd>
          <dt>Status</dt><dd>${missing.length ? escapeHtml("Clarification needed: " + missing.join("; ")) : "Ready for engineering review"}</dd>
        </dl>
      `;
    }

    function formatPartName(value) {
      return String(value || "unknown part").replaceAll("_", " ");
    }

    function formatDimensions(geometry) {
      const labels = {
        outer_diameter_mm: "OD",
        inner_diameter_mm: "ID",
        height_mm: "height",
        length_mm: "length",
        width_mm: "width",
        thickness_mm: "thickness",
        chamfer_mm: "chamfer",
        fillet_mm: "fillet",
        coil_count: "coils"
      };
      const parts = [];
      for (const [key, label] of Object.entries(labels)) {
        const value = geometry[key];
        if (value !== null && value !== undefined && value !== "") {
          parts.push(key === "coil_count" ? `${value} ${label}` : `${label} ${value} mm`);
        }
      }
      return parts.length ? parts.join(", ") : "No dimensions specified";
    }

    function appendMsg(role, text) {
      const el = document.createElement("div");
      el.className = `msg ${role}`;
      el.textContent = text;
      chatLog.appendChild(el);
      syncChatState();
      chatLog.scrollTop = chatLog.scrollHeight;
      return el;
    }

    function autoResizeChatInput() {
      chatInput.style.height = "auto";
      const maxHeight = chatShell.classList.contains("idle") ? 280 : 220;
      chatInput.style.height = `${Math.min(chatInput.scrollHeight, maxHeight)}px`;
    }

    function syncChatState() {
      const messageCount = chatLog.querySelectorAll(".msg").length;
      chatShell.classList.toggle("idle", messageCount <= 1);
      autoResizeChatInput();
    }

    function startActivity(label, phases) {
      stopActivityTimer();
      activityPanel.className = "activity-panel active";
      activityLabel.textContent = label;
      activityPhase.textContent = phases[0] || "Working";
      activityFill.style.width = "10%";
      previewProgress.className = "preview-progress active";
      previewProgressLabel.textContent = label;
      previewProgressPhase.textContent = phases[0] || "Working";
      previewProgressFill.style.width = "10%";
      let progress = 10;
      let phaseIndex = 0;
      activityTimer = window.setInterval(() => {
        progress = Math.min(progress + 12, 90);
        activityFill.style.width = `${progress}%`;
        previewProgressFill.style.width = `${progress}%`;
        if (phaseIndex < phases.length - 1 && progress >= (phaseIndex + 1) * 30) {
          phaseIndex += 1;
          activityPhase.textContent = phases[phaseIndex];
          previewProgressPhase.textContent = phases[phaseIndex];
        }
      }, 700);
    }

    function completeActivity(phaseText) {
      stopActivityTimer();
      activityPanel.className = "activity-panel active done";
      activityPhase.textContent = phaseText;
      activityFill.style.width = "100%";
      previewProgress.className = "preview-progress active done";
      previewProgressPhase.textContent = phaseText;
      previewProgressFill.style.width = "100%";
      window.setTimeout(() => {
        activityPanel.className = "activity-panel";
        activityLabel.textContent = "Ready";
        activityPhase.textContent = "Waiting";
        activityFill.style.width = "0%";
        previewProgress.className = "preview-progress";
        previewProgressFill.style.width = "0%";
      }, 1400);
    }

    function failActivity(phaseText) {
      stopActivityTimer();
      activityPanel.className = "activity-panel active error";
      activityPhase.textContent = phaseText;
      activityFill.style.width = "100%";
      previewProgress.className = "preview-progress active error";
      previewProgressPhase.textContent = phaseText;
      previewProgressFill.style.width = "100%";
    }

    function stopActivityTimer() {
      if (activityTimer) {
        window.clearInterval(activityTimer);
        activityTimer = null;
      }
    }

    async function render3DPreview(cadIntent) {
      cleanupViewer();
      const container = document.createElement("div");
      container.className = "viewer3d";
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-label", "Interactive CAD preview");
      container.appendChild(canvas);
      preview.innerHTML = "";
      preview.appendChild(container);

      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Canvas rendering is not available in this browser.");
      }

      const uploaded = (preferParametric || overrideMeshFaces) ? null : pickUploadedMesh();
      const mesh = overrideMeshFaces
        ? { faces: overrideMeshFaces }
        : (uploaded ? { faces: uploaded.faces } : createPreviewMesh(cadIntent || {}));
      lastExport.mesh = mesh;
      lastExport.canvas = canvas;
      lastExport.intent = cadIntent || {};
      const bounds = computeMeshBounds(mesh.faces);
      const size = bounds.size;
      const extent = Math.max(size.x, size.y, size.z, 40);
      const center = bounds.center;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

      const state = {
        width: 0,
        height: 0,
        rotationX: viewerCamera.rotationX,
        rotationY: viewerCamera.rotationY,
        zoom: viewerCamera.zoom,
        dragging: false,
        lastX: 0,
        lastY: 0,
        animationFrameId: 0,
        renderQueued: false,
      };

      const lightDirection = normalizeVector({ x: 0.35, y: 0.85, z: 0.4 });
      const centeredFaces = mesh.faces.map((face) => ({
        color: face.color,
        points: face.points.map((point) => ({
          x: point.x - center.x,
          y: point.y - center.y,
          z: point.z - center.z,
        })),
      }));
      const gridLevel = -size.y * 0.55;

      function scheduleRender() {
        if (state.renderQueued) {
          return;
        }
        state.renderQueued = true;
        state.animationFrameId = requestAnimationFrame(() => {
          state.renderQueued = false;
          drawPreview();
        });
      }

      function resizeCanvas() {
        state.width = Math.max(320, Math.floor(preview.clientWidth || 720));
        state.height = Math.max(320, Math.floor(preview.clientHeight || 470));
        canvas.width = Math.floor(state.width * pixelRatio);
        canvas.height = Math.floor(state.height * pixelRatio);
        canvas.style.width = `${state.width}px`;
        canvas.style.height = `${state.height}px`;
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        scheduleRender();
      }

      function projectWorldPoint(point) {
        return projectRotatedPoint(rotatePoint(point, state.rotationX, state.rotationY));
      }

      function projectRotatedPoint(rotated) {
        const scale = (Math.min(state.width, state.height) / (extent * 2.4)) * state.zoom;
        return {
          x: state.width / 2 + rotated.x * scale,
          y: state.height / 2 - rotated.y * scale,
          z: rotated.z,
        };
      }

      function drawGrid() {
        const spacing = niceGridSpacing(extent);
        const radius = Math.max(extent * 0.9, spacing * 4);
        context.save();
        context.lineWidth = 1;
        context.strokeStyle = "#dbe6f2";
        for (let value = -radius; value <= radius + 0.001; value += spacing) {
          drawLine(
            projectWorldPoint({ x: value, y: gridLevel, z: -radius }),
            projectWorldPoint({ x: value, y: gridLevel, z: radius })
          );
          drawLine(
            projectWorldPoint({ x: -radius, y: gridLevel, z: value }),
            projectWorldPoint({ x: radius, y: gridLevel, z: value })
          );
        }
        context.restore();
      }

      function drawLine(start, end) {
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      }

      function drawFaces() {
        const projectedFaces = centeredFaces
          .map((face) => {
            const rotatedPoints = face.points.map((point) => rotatePoint(point, state.rotationX, state.rotationY));
            const projectedPoints = rotatedPoints.map((point) => projectRotatedPoint(point));
            const normal = computeFaceNormal(rotatedPoints);
            const intensity = clamp(dotProduct(normal, lightDirection) * 0.55 + 0.72, 0.28, 0.95);
            const averageDepth = rotatedPoints.reduce((sum, point) => sum + point.z, 0) / rotatedPoints.length;
            return {
              projectedPoints,
              averageDepth,
              fill: shadeColor(face.color, intensity),
              stroke: shadeColor(face.color, Math.max(intensity - 0.2, 0.18)),
            };
          })
          .sort((left, right) => left.averageDepth - right.averageDepth);

        for (const face of projectedFaces) {
          context.beginPath();
          context.moveTo(face.projectedPoints[0].x, face.projectedPoints[0].y);
          for (let index = 1; index < face.projectedPoints.length; index += 1) {
            context.lineTo(face.projectedPoints[index].x, face.projectedPoints[index].y);
          }
          context.closePath();
          context.fillStyle = face.fill;
          context.strokeStyle = face.stroke;
          context.lineWidth = 1.2;
          context.fill();
          context.stroke();
        }
      }

      function drawPreview() {
        context.clearRect(0, 0, state.width, state.height);
        context.fillStyle = "#f8fbff";
        context.fillRect(0, 0, state.width, state.height);
        drawGrid();
        drawFaces();
        drawAxisIndicator();
      }

      function drawAxisIndicator() {
        // Small orientation triad pinned to the bottom-left corner. It rotates with
        // the camera so the user can always read the part orientation. Viewer world
        // Y is "up" (the part height), so it is labelled Z to match CAD convention.
        const originX = 52;
        const originY = state.height - 52;
        const axisLength = 30;
        const axes = [
          { vector: { x: 1, y: 0, z: 0 }, label: "X", color: "#e23b3b" },
          { vector: { x: 0, y: 0, z: 1 }, label: "Y", color: "#2faa4d" },
          { vector: { x: 0, y: 1, z: 0 }, label: "Z", color: "#2f7bff" },
        ];
        const projected = axes
          .map((axis) => {
            const rotated = rotatePoint(axis.vector, state.rotationX, state.rotationY);
            return {
              label: axis.label,
              color: axis.color,
              tipX: originX + rotated.x * axisLength,
              tipY: originY - rotated.y * axisLength,
              depth: rotated.z,
            };
          })
          .sort((left, right) => left.depth - right.depth);

        context.save();
        context.lineWidth = 2.5;
        context.lineCap = "round";
        context.font = "bold 11px system-ui, -apple-system, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillStyle = "#9aa7b8";
        context.beginPath();
        context.arc(originX, originY, 2.5, 0, Math.PI * 2);
        context.fill();
        for (const axis of projected) {
          context.strokeStyle = axis.color;
          context.beginPath();
          context.moveTo(originX, originY);
          context.lineTo(axis.tipX, axis.tipY);
          context.stroke();
          const labelX = originX + (axis.tipX - originX) * 1.2;
          const labelY = originY + (axis.tipY - originY) * 1.2;
          context.fillStyle = axis.color;
          context.beginPath();
          context.arc(labelX, labelY, 7.5, 0, Math.PI * 2);
          context.fill();
          context.fillStyle = "#ffffff";
          context.fillText(axis.label, labelX, labelY);
        }
        context.restore();
      }

      function onPointerDown(event) {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      }

      function onPointerMove(event) {
        if (!state.dragging) {
          return;
        }
        const deltaX = event.clientX - state.lastX;
        const deltaY = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        state.rotationY += deltaX * 0.01;
        state.rotationX = clamp(state.rotationX + deltaY * 0.01, -1.35, 1.35);
        viewerCamera.rotationX = state.rotationX;
        viewerCamera.rotationY = state.rotationY;
        scheduleRender();
      }

      function onPointerUp(event) {
        state.dragging = false;
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      }

      function onWheel(event) {
        event.preventDefault();
        const nextZoom = state.zoom * (event.deltaY > 0 ? 0.92 : 1.08);
        state.zoom = clamp(nextZoom, 0.55, 2.8);
        viewerCamera.zoom = state.zoom;
        scheduleRender();
      }

      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointerleave", onPointerUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      const resizeObserver = new ResizeObserver(resizeCanvas);
      resizeObserver.observe(preview);
      resizeCanvas();

      activeViewer = {
        resizeObserver,
        dispose() {
          cancelAnimationFrame(state.animationFrameId);
          resizeObserver.disconnect();
          canvas.removeEventListener("pointerdown", onPointerDown);
          canvas.removeEventListener("pointermove", onPointerMove);
          canvas.removeEventListener("pointerup", onPointerUp);
          canvas.removeEventListener("pointerleave", onPointerUp);
          canvas.removeEventListener("wheel", onWheel);
        },
      };
    }

    // ===== Uploaded mesh (real STL geometry) support =====
    const UPLOAD_MESH_COLOR = "#8b929b";
    const MAX_PREVIEW_TRIANGLES = 60000;

    function parseStlMesh(buffer, sourceLabel) {
      if (!buffer || buffer.byteLength < 84) {
        return null;
      }
      const view = new DataView(buffer);
      const headerCount = view.getUint32(80, true);
      if (84 + headerCount * 50 === buffer.byteLength) {
        return finalizeUploadedMesh(parseBinaryStl(view, headerCount), sourceLabel);
      }
      const text = new TextDecoder("utf-8").decode(new Uint8Array(buffer));
      if (/facet\\s+normal/i.test(text)) {
        return finalizeUploadedMesh(parseAsciiStl(text), sourceLabel);
      }
      if (headerCount > 0 && 84 + headerCount * 50 <= buffer.byteLength) {
        return finalizeUploadedMesh(parseBinaryStl(view, headerCount), sourceLabel);
      }
      return null;
    }

    function parseBinaryStl(view, triCount) {
      const faces = [];
      const stride = triCount > MAX_PREVIEW_TRIANGLES ? Math.ceil(triCount / MAX_PREVIEW_TRIANGLES) : 1;
      for (let i = 0; i < triCount; i += stride) {
        const base = 84 + i * 50;
        faces.push(makeStlFace(
          readStlVec(view, base + 12),
          readStlVec(view, base + 24),
          readStlVec(view, base + 36)
        ));
      }
      return { faces, original: triCount };
    }

    function readStlVec(view, offset) {
      return {
        x: view.getFloat32(offset, true),
        y: view.getFloat32(offset + 4, true),
        z: view.getFloat32(offset + 8, true),
      };
    }

    function parseAsciiStl(text) {
      const faces = [];
      const verts = [];
      const tokens = text.split(/\\s+/);
      for (let i = 0; i < tokens.length; i += 1) {
        if (tokens[i] !== "vertex") {
          continue;
        }
        const x = parseFloat(tokens[i + 1]);
        const y = parseFloat(tokens[i + 2]);
        const z = parseFloat(tokens[i + 3]);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          verts.push({ x, y, z });
          if (verts.length === 3) {
            faces.push(makeStlFace(verts[0], verts[1], verts[2]));
            verts.length = 0;
          }
        }
      }
      const original = faces.length;
      let used = faces;
      if (faces.length > MAX_PREVIEW_TRIANGLES) {
        const stride = Math.ceil(faces.length / MAX_PREVIEW_TRIANGLES);
        used = faces.filter((face, index) => index % stride === 0);
      }
      return { faces: used, original };
    }

    function makeStlFace(a, b, c) {
      // Reverse winding to keep outward normals after the Z-up -> Y-up swap below.
      return { color: UPLOAD_MESH_COLOR, points: [stlPoint(a), stlPoint(c), stlPoint(b)] };
    }

    function stlPoint(p) {
      // STL is usually Z-up; the viewer is Y-up, so swap Y and Z.
      return { x: p.x, y: p.z, z: p.y };
    }

    function finalizeUploadedMesh(parsed, sourceLabel) {
      if (!parsed || !parsed.faces || !parsed.faces.length) {
        return null;
      }
      const bounds = computeMeshBounds(parsed.faces);
      return {
        faces: parsed.faces,
        bounds,
        original: parsed.original || parsed.faces.length,
        shown: parsed.faces.length,
        source: sourceLabel || "stl",
      };
    }

    function pickUploadedMesh() {
      for (let i = attachmentContexts.length - 1; i >= 0; i -= 1) {
        if (attachmentContexts[i] && attachmentContexts[i].clientMesh) {
          return attachmentContexts[i].clientMesh;
        }
      }
      return null;
    }

    function measuredSummaryFromMesh(mesh) {
      if (!mesh || !mesh.bounds) {
        return "";
      }
      const s = mesh.bounds.size;
      const dx = Math.round(s.x * 100) / 100;
      const dy = Math.round(s.y * 100) / 100;
      const dz = Math.round(s.z * 100) / 100;
      const envelope = Math.round(Math.max(s.x, s.y, s.z) * 100) / 100;
      return (
        "Measured from the uploaded geometry: bounding box approximately " +
        dx + " x " + dy + " x " + dz + " mm; maximum envelope dimension about " +
        envelope + " mm. Treat these as the real, measured dimensions and do not invent values."
      );
    }

    // ===== Phase 2b: measure a bushing's OD/ID/height from the uploaded mesh =====
    function percentile(sortedValues, fraction) {
      if (!sortedValues.length) return 0;
      const index = Math.min(
        sortedValues.length - 1,
        Math.max(0, Math.round(fraction * (sortedValues.length - 1)))
      );
      return sortedValues[index];
    }

    function measureBushingFromMesh(mesh) {
      if (!mesh || !mesh.faces || !mesh.faces.length || !mesh.bounds) {
        return null;
      }
      // The viewer Y axis is the bushing axis (STL Z-up was swapped to Y-up).
      // Radial distance is measured in the X/Z plane about the part centroid.
      let sumX = 0;
      let sumZ = 0;
      let count = 0;
      for (const face of mesh.faces) {
        for (const p of face.points) {
          sumX += p.x;
          sumZ += p.z;
          count += 1;
        }
      }
      if (!count) return null;
      const centerX = sumX / count;
      const centerZ = sumZ / count;

      const radii = [];
      for (const face of mesh.faces) {
        for (const p of face.points) {
          radii.push(Math.hypot(p.x - centerX, p.z - centerZ));
        }
      }
      radii.sort((a, b) => a - b);

      // Outer radius: high percentile to ignore stray spikes. Inner bore radius:
      // low percentile, which lands on the bore wall (the closest material to the axis).
      const outerRadius = percentile(radii, 0.99);
      let innerRadius = percentile(radii, 0.03);
      if (!(innerRadius > 0) || innerRadius >= outerRadius) {
        innerRadius = outerRadius * 0.4;
      }
      const height = mesh.bounds.size.y;
      if (!(outerRadius > 0) || !(height > 0)) {
        return null;
      }
      const round = (v) => Math.round(v * 2) / 2; // nearest 0.5 mm
      const outer = round(outerRadius * 2);
      let inner = round(innerRadius * 2);
      if (inner >= outer) inner = round(Math.max(1, outer - 1));
      return {
        outer_diameter_mm: outer,
        inner_diameter_mm: Math.max(1, inner),
        height_mm: round(height),
      };
    }

    function convertUploadedToBushing() {
      const uploaded = pickUploadedMesh();
      const dims = measureBushingFromMesh(uploaded);
      if (!dims || !uploaded || !uploaded.faces) {
        appendMsg("bot", "I could not measure clean bushing dimensions from this mesh.");
        return;
      }

      // Keep the REAL geometry: store the original vertices and the part axis so we
      // can warp (stretch) it live instead of replacing it with a plain cylinder.
      let sumX = 0;
      let sumZ = 0;
      let count = 0;
      for (const face of uploaded.faces) {
        for (const p of face.points) {
          sumX += p.x;
          sumZ += p.z;
          count += 1;
        }
      }
      const centerX = sumX / count;
      const centerZ = sumZ / count;
      const centerY = uploaded.bounds ? uploaded.bounds.center.y : 0;

      editableMesh = {
        originalFaces: uploaded.faces,
        centerX: centerX,
        centerZ: centerZ,
        centerY: centerY,
        od0: dims.outer_diameter_mm,
        id0: dims.inner_diameter_mm,
        h0: dims.height_mm,
      };

      const intent = {
        part_type: "bushing",
        material: { name: "rubber" },
        geometry: {
          outer_diameter_mm: dims.outer_diameter_mm,
          inner_diameter_mm: dims.inner_diameter_mm,
          height_mm: dims.height_mm,
          arms: [],
          holes: [],
        },
        simulation_hints: { target_output: "cad" },
        missing_information: [],
      };

      meshEditMode = true;
      preferParametric = false;
      currentEditIntent = intent;
      lastExport.intent = intent;
      lastExport.name = exportBaseName(intent);
      jsonOutput.textContent = JSON.stringify(intent, null, 2);
      downloadBtn.disabled = false;

      // Identity warp to start (keeps the exact uploaded shape on screen).
      overrideMeshFaces = warpEditableMeshFaces(intent.geometry);
      render3DPreview(intent).catch(() => {});
      buildParamControls(intent);
      appendMsg(
        "bot",
        "This bushing is now editable while keeping its real shape: outer diameter " +
        dims.outer_diameter_mm + " mm, inner diameter " + dims.inner_diameter_mm +
        " mm, height " + dims.height_mm + " mm. Drag the sliders to stretch it, then use Download to export."
      );
    }

    function warpEditableMeshFaces(geom) {
      if (!editableMesh) {
        return overrideMeshFaces;
      }
      const newOd = Number(geom.outer_diameter_mm) > 0 ? Number(geom.outer_diameter_mm) : editableMesh.od0;
      const newId = Number(geom.inner_diameter_mm) > 0 ? Number(geom.inner_diameter_mm) : editableMesh.id0;
      const newH = Number(geom.height_mm) > 0 ? Number(geom.height_mm) : editableMesh.h0;

      const ro0 = editableMesh.od0 / 2;
      const ri0 = editableMesh.id0 / 2;
      const ro1 = newOd / 2;
      const ri1 = newId / 2;
      const span0 = ro0 - ri0;
      const heightScale = editableMesh.h0 > 0 ? newH / editableMesh.h0 : 1;
      const cx = editableMesh.centerX;
      const cz = editableMesh.centerZ;
      const cy = editableMesh.centerY;

      return editableMesh.originalFaces.map((face) => ({
        color: face.color,
        points: face.points.map((p) => {
          const dx = p.x - cx;
          const dz = p.z - cz;
          const r = Math.hypot(dx, dz);
          // Linearly remap the annular wall so the bore goes to the new ID and the
          // outer surface to the new OD; everything in between stretches with it.
          let newR;
          if (span0 > 1e-6) {
            newR = ri1 + (r - ri0) * (ro1 - ri1) / span0;
          } else {
            newR = r * (ro1 > 0 ? ro1 / Math.max(ro0, 1e-6) : 1);
          }
          if (newR < 0) newR = 0;
          const scale = r > 1e-6 ? newR / r : 0;
          return {
            x: cx + dx * scale,
            y: cy + (p.y - cy) * heightScale,
            z: cz + dz * scale,
          };
        }),
      }));
    }

    // ===== Parametric editor (Phase 1): live dimension sliders =====
    const PARAM_FIELDS = {
      outer_diameter_mm:   { label: "Outer diameter", min: 5,   max: 400, step: 0.5, fallback: 60 },
      inner_diameter_mm:   { label: "Inner diameter", min: 1,   max: 400, step: 0.5, fallback: 20 },
      height_mm:           { label: "Height",         min: 1,   max: 500, step: 0.5, fallback: 40 },
      length_mm:           { label: "Length",         min: 1,   max: 600, step: 0.5, fallback: 80 },
      width_mm:            { label: "Width",          min: 1,   max: 600, step: 0.5, fallback: 40 },
      thickness_mm:        { label: "Thickness",      min: 0.1, max: 200, step: 0.1, fallback: 5 },
      chamfer_mm:          { label: "Chamfer",        min: 0,   max: 30,  step: 0.1, fallback: 0 },
      fillet_mm:           { label: "Fillet",         min: 0,   max: 30,  step: 0.1, fallback: 0 },
      flange_diameter_mm:  { label: "Flange diameter",min: 5,   max: 500, step: 0.5, fallback: 0 },
      flange_thickness_mm: { label: "Flange thickness",min: 0,  max: 60,  step: 0.1, fallback: 0 },
      coil_count:          { label: "Coil count",     min: 1,   max: 50,  step: 1,   fallback: 8 },
    };
    const PART_FIELD_SETS = {
      bushing:      { core: ["outer_diameter_mm", "inner_diameter_mm", "height_mm"], optional: ["chamfer_mm", "fillet_mm", "flange_diameter_mm", "flange_thickness_mm"] },
      rubber_mount: { core: ["outer_diameter_mm", "inner_diameter_mm", "height_mm"], optional: ["chamfer_mm", "fillet_mm", "flange_diameter_mm", "flange_thickness_mm"] },
      plate:        { core: ["length_mm", "width_mm", "thickness_mm"], optional: ["chamfer_mm", "fillet_mm"] },
      bracket:      { core: ["length_mm", "width_mm", "thickness_mm"], optional: ["fillet_mm"] },
      spring:       { core: ["outer_diameter_mm", "height_mm", "thickness_mm"], optional: ["coil_count"] },
    };

    function roundParam(value, step) {
      const s = Number(step) || 0.1;
      const decimals = (String(s).split(".")[1] || "").length;
      return Number(value).toFixed(decimals);
    }

    // ===== Simulation (Tier 1): live analytical modal estimate =====
    // Linear-elastic isotropic properties in the tonne-mm-s unit system, so
    // frequencies come out directly in Hz. Values mirror simulate/materials.py.
    const SIM_MATERIALS = {
      steel:           { E: 210000, rho: 7.85e-9 },
      stainless_steel: { E: 193000, rho: 8.00e-9 },
      aluminum:        { E: 69000,  rho: 2.70e-9 },
      cast_iron:       { E: 110000, rho: 7.20e-9 },
      rubber:          { E: 10,     rho: 1.10e-9 },
      epdm:            { E: 6,      rho: 1.15e-9 },
      abs:             { E: 2300,   rho: 1.05e-9 },
      generic:         { E: 200000, rho: 7.85e-9 },
    };
    const SIM_ALIASES = {
      metal: "steel", "mild steel": "steel", "carbon steel": "steel",
      ss: "stainless_steel", inox: "stainless_steel",
      alu: "aluminum", aluminium: "aluminum", al: "aluminum",
      iron: "cast_iron", "natural rubber": "rubber", nr: "rubber",
      elastomer: "rubber", plastic: "abs",
    };

    function simResolveMaterial(name) {
      const key = String(name || "").trim().toLowerCase();
      if (SIM_MATERIALS[key]) return { key, ...SIM_MATERIALS[key] };
      if (SIM_ALIASES[key] && SIM_MATERIALS[SIM_ALIASES[key]]) {
        const resolved = SIM_ALIASES[key];
        return { key: resolved, ...SIM_MATERIALS[resolved] };
      }
      return { key: "generic", ...SIM_MATERIALS.generic };
    }

    function estimateBushingModal(geom, materialName) {
      const od = Number(geom && geom.outer_diameter_mm);
      const idRaw = Number(geom && geom.inner_diameter_mm) || 0;
      const length = Number(geom && geom.height_mm);
      if (!(od > 0) || !(length > 0)) return null;
      const id = Math.min(Math.max(idRaw, 0), od - 1e-3);
      const mat = simResolveMaterial(materialName);
      const E = mat.E;       // MPa = N/mm^2
      const rho = mat.rho;   // tonne/mm^3
      const area = Math.PI / 4 * (od * od - id * id);              // mm^2
      const inertia = Math.PI / 64 * (Math.pow(od, 4) - Math.pow(id, 4)); // mm^4
      const kAxial = E * area / length;                           // N/mm
      const kBending = 3 * E * inertia / Math.pow(length, 3);     // N/mm (cantilever tip)
      // Euler-Bernoulli cantilever (fixed-free) bending modes.
      const betaL = [1.875104, 4.694091, 7.854757];
      const c = Math.sqrt((E * inertia) / (rho * area * Math.pow(length, 4)));
      const bending = betaL.map((b) => (b * b / (2 * Math.PI)) * c); // Hz
      const axial1 = (1 / (4 * length)) * Math.sqrt(E / rho);        // Hz (fixed-free rod)
      return { material: mat.key, area, inertia, kAxial, kBending, bending, axial1 };
    }

    function formatHz(value) {
      if (!Number.isFinite(value)) return "\u2013";
      if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + " kHz";
      return value.toFixed(value >= 100 ? 0 : 1) + " Hz";
    }

    function formatStiffness(value) {
      if (!Number.isFinite(value)) return "\u2013";
      if (value >= 1e6) return (value / 1e6).toFixed(2) + " MN/mm";
      if (value >= 1000) return (value / 1000).toFixed(1) + " kN/mm";
      return value.toFixed(value >= 100 ? 0 : 1) + " N/mm";
    }

    function updateSimEstimate(intent) {
      if (!simResults) return;
      const source = intent || currentEditIntent;
      const type = String((source && source.part_type) || "").toLowerCase();
      const geom = source && source.geometry;
      if (!geom || (type !== "bushing" && type !== "rubber_mount")) {
        simResults.innerHTML = "";
        return;
      }
      const materialName = (source.material && source.material.name) || "generic";
      const est = estimateBushingModal(geom, materialName);
      if (!est) {
        simResults.innerHTML = "";
        return;
      }
      simResults.innerHTML =
        '<div class="sim-block">' +
        '<strong>Simulation estimate</strong>' +
        '<p class="muted">First-pass analytical modal estimate \u00b7 fixed-bottom \u00b7 linear-elastic ' +
        est.material.replace("_", " ") + '. Run full FEM for validated results.</p>' +
        '<table class="sim-table">' +
        '<tr><th>Mode</th><th style="text-align:right">Frequency</th></tr>' +
        '<tr><td>1st bending</td><td class="sim-value">' + formatHz(est.bending[0]) + '</td></tr>' +
        '<tr><td>2nd bending</td><td class="sim-value">' + formatHz(est.bending[1]) + '</td></tr>' +
        '<tr><td>3rd bending</td><td class="sim-value">' + formatHz(est.bending[2]) + '</td></tr>' +
        '<tr><td>1st axial</td><td class="sim-value">' + formatHz(est.axial1) + '</td></tr>' +
        '</table>' +
        '<div class="sim-stiff">' +
        '<span>Axial stiffness: <b>' + formatStiffness(est.kAxial) + '</b></span>' +
        '<span>Bending stiffness: <b>' + formatStiffness(est.kBending) + '</b></span>' +
        '</div>' +
        '</div>';
    }

    function buildParamControls(intent) {
      if (!paramControls) {
        return;
      }
      const uploaded = (preferParametric || meshEditMode) ? null : pickUploadedMesh();
      if (uploaded) {
        const dims = measureBushingFromMesh(uploaded);
        if (dims) {
          paramControls.innerHTML =
            '<p class="muted">This is an uploaded mesh, shown exactly as provided. ' +
            'Measured bushing dimensions: OD ' + dims.outer_diameter_mm + ' mm, ID ' +
            dims.inner_diameter_mm + ' mm, height ' + dims.height_mm + ' mm.</p>' +
            '<div class="param-actions"><button type="button" class="param-reset" id="convertBushingBtn">Convert to editable bushing</button></div>';
          const convertBtn = document.getElementById("convertBushingBtn");
          if (convertBtn) convertBtn.addEventListener("click", convertUploadedToBushing);
          if (paramHint) paramHint.textContent = "Convert to edit OD / ID / height live.";
        } else {
          paramControls.innerHTML = '<p class="muted">This is an uploaded mesh, shown exactly as provided. Live parametric editing is available for AI-generated parts.</p>';
          if (paramHint) paramHint.textContent = "Uploaded geometry is read-only.";
        }
        updateSimEstimate(null);
        return;
      }
      const type = String((intent && intent.part_type) || "unknown").toLowerCase();
      const spec = PART_FIELD_SETS[type];
      if (!intent || !spec) {
        paramControls.innerHTML = '<p class="muted">Adjustable dimensions will appear here once a model is generated. Use the Download menu to export the edited part.</p>';
        if (paramHint) paramHint.textContent = "";
        updateSimEstimate(null);
        return;
      }

      currentEditIntent = intent;
      baseGeometry = Object.assign({}, intent.geometry || {});
      const geom = intent.geometry || {};

      const fields = spec.core.slice();
      for (const key of spec.optional) {
        const value = Number(geom[key]);
        if (Number.isFinite(value) && value > 0) {
          fields.push(key);
        }
      }

      const rows = fields.map((key) => {
        const def = PARAM_FIELDS[key];
        if (!def) return "";
        let value = Number(geom[key]);
        if (!Number.isFinite(value) || value <= 0) value = def.fallback;
        const max = Math.max(def.max, value);
        const display = roundParam(value, def.step);
        const unit = key === "coil_count" ? "" : '<span class="param-unit">mm</span>';
        return `
          <div class="param-row" data-key="${key}">
            <label class="param-label" for="slider_${key}">${def.label}</label>
            <span class="param-value">
              <input type="number" class="param-number" id="num_${key}" value="${display}" min="${def.min}" max="${max}" step="${def.step}">
              ${unit}
            </span>
            <input type="range" class="param-slider" id="slider_${key}" value="${value}" min="${def.min}" max="${max}" step="${def.step}">
          </div>`;
      }).join("");

      paramControls.innerHTML = rows + '<div class="param-actions"><button type="button" class="param-reset" id="paramReset">Reset</button></div>';
      bindParamControls();
      if (paramHint) paramHint.textContent = "Drag a slider to resize the model live.";
      updateSimEstimate(intent);
    }

    function bindParamControls() {
      const rows = paramControls.querySelectorAll(".param-row");
      for (const row of rows) {
        const key = row.dataset.key;
        const slider = row.querySelector(".param-slider");
        const number = row.querySelector(".param-number");
        if (!slider || !number) continue;
        const onChange = (raw) => {
          const val = Number(raw);
          if (!Number.isFinite(val)) return;
          slider.value = val;
          number.value = roundParam(val, Number(slider.step) || 0.1);
          applyParamEdit(key, val);
        };
        slider.addEventListener("input", () => onChange(slider.value));
        number.addEventListener("input", () => onChange(number.value));
      }
      const reset = document.getElementById("paramReset");
      if (reset) {
        reset.addEventListener("click", () => {
          if (!currentEditIntent || !baseGeometry) return;
          currentEditIntent.geometry = Object.assign({}, baseGeometry);
          lastExport.intent = currentEditIntent;
          jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
          if (meshEditMode) {
            overrideMeshFaces = warpEditableMeshFaces(currentEditIntent.geometry);
          }
          buildParamControls(currentEditIntent);
          scheduleParamRender();
        });
      }
    }

    function applyParamEdit(key, value) {
      if (!currentEditIntent) return;
      const geom = currentEditIntent.geometry || (currentEditIntent.geometry = {});
      geom[key] = key === "coil_count" ? Math.max(1, Math.round(value)) : value;

      // Keep the inner diameter strictly inside the outer diameter.
      const od = Number(geom.outer_diameter_mm);
      const id = Number(geom.inner_diameter_mm);
      if (Number.isFinite(od) && Number.isFinite(id) && id >= od) {
        geom.inner_diameter_mm = Math.max(1, od - 1);
        const idNum = document.getElementById("num_inner_diameter_mm");
        const idSlider = document.getElementById("slider_inner_diameter_mm");
        if (idNum) idNum.value = roundParam(geom.inner_diameter_mm, 0.5);
        if (idSlider) idSlider.value = geom.inner_diameter_mm;
      }

      lastExport.intent = currentEditIntent;
      jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
      if (meshEditMode) {
        overrideMeshFaces = warpEditableMeshFaces(geom);
      }
      updateSimEstimate(currentEditIntent);
      scheduleParamRender();
    }

    function scheduleParamRender() {
      if (paramRenderQueued) return;
      paramRenderQueued = true;
      requestAnimationFrame(() => {
        paramRenderQueued = false;
        if (currentEditIntent) {
          render3DPreview(currentEditIntent).catch(() => {});
        }
      });
    }

    function createPreviewMesh(cadIntent) {
      const type = String(cadIntent.part_type || "unknown").toLowerCase();
      const geometry = cadIntent.geometry || {};
      if (type === "bushing" || type === "rubber_mount") {
        return createBushingMesh(geometry);
      }
      if (type === "plate") {
        return createPlateMesh(geometry);
      }
      if (type === "bracket") {
        return createBracketMesh(geometry);
      }
      if (type === "spring") {
        return createSpringMesh(geometry);
      }
      return createUnknownMesh(geometry);
    }

    function createBushingMesh(geometry) {
      const outerDiameter = readPositive(geometry.outer_diameter_mm, 60);
      const innerDiameter = readPositive(geometry.inner_diameter_mm, 20);
      const height = readPositive(geometry.height_mm, 40);
      const outerRadius = Math.max(outerDiameter / 2, 5);
      const innerRadius = Math.min(Math.max(innerDiameter / 2, 2), outerRadius - 1);

      const segments = 36;
      const topY = height / 2;
      const bottomY = -height / 2;

      // Optional Vibracoustic-style features.
      const sleeveOuter = Math.max(0, geometry.metal_sleeve_thickness_mm || 0);
      const sleeveInner = Math.max(0, geometry.inner_sleeve_thickness_mm || 0);
      const flangeDiameter = readPositive(geometry.flange_diameter_mm, 0);
      const flangeThickness = readPositive(geometry.flange_thickness_mm, 0);
      const boreOffset = Number(geometry.bore_offset_mm) || 0;
      const chamferSize = Math.max(0, Number(geometry.chamfer_mm) || 0);
      const filletSize = Math.max(0, Number(geometry.fillet_mm) || 0);

      // Clamp shells so they never overlap.
      const safeSleeveOuter = Math.min(sleeveOuter, (outerRadius - innerRadius) * 0.45);
      const safeSleeveInner = Math.min(sleeveInner, (outerRadius - innerRadius) * 0.45);

      // Concentric shell radii (outer -> inner).
      const rOuter = outerRadius;
      const rRubberOuter = outerRadius - safeSleeveOuter;
      const rRubberInner = innerRadius + safeSleeveInner;
      const rInner = innerRadius;
      const offsetX = Math.max(-Math.abs(rRubberInner - rInner) - (rRubberInner - 1), Math.min(boreOffset, rRubberInner - 1));

      // Clamp edge break so it never collapses the rubber wall or eats the height.
      const rubberThickness = Math.max(rRubberOuter - rRubberInner, 0.5);
      const maxEdgeBreak = Math.min(height * 0.45, rubberThickness * 0.45);
      const safeChamfer = Math.min(chamferSize, maxEdgeBreak);
      const safeFillet = Math.min(filletSize, maxEdgeBreak);
      // Fillet wins if both are set; otherwise chamfer.
      const useFillet = safeFillet > 0;
      const useChamfer = !useFillet && safeChamfer > 0;
      const edgeBreak = useFillet ? safeFillet : safeChamfer;
      const filletSteps = 4;

      const colorMetal = "#9aa6b4";
      const colorMetalTop = "#b6c1ce";
      const colorMetalBottom = "#7f8a98";
      const colorRubber = "#3b3a3a";
      const colorRubberTop = "#525151";
      const colorRubberBottom = "#2a2929";

      const faces = [];

      // Add a coaxial annulus shell: side walls + top/bottom rings.
      function addAnnulus(rOut, rIn, centerOutX, centerInX, color, topColor, bottomColor) {
        for (let index = 0; index < segments; index += 1) {
          const thetaA = (index / segments) * Math.PI * 2;
          const thetaB = ((index + 1) / segments) * Math.PI * 2;
          const outerTopA = pointOnRing(rOut, thetaA, topY, centerOutX);
          const outerTopB = pointOnRing(rOut, thetaB, topY, centerOutX);
          const outerBottomA = pointOnRing(rOut, thetaA, bottomY, centerOutX);
          const outerBottomB = pointOnRing(rOut, thetaB, bottomY, centerOutX);
          const innerTopA = pointOnRing(rIn, thetaA, topY, centerInX);
          const innerTopB = pointOnRing(rIn, thetaB, topY, centerInX);
          const innerBottomA = pointOnRing(rIn, thetaA, bottomY, centerInX);
          const innerBottomB = pointOnRing(rIn, thetaB, bottomY, centerInX);
          // Outer wall
          faces.push(makeFace([outerTopA, outerTopB, outerBottomB, outerBottomA], color));
          // Inner wall (reversed winding so the bore looks correct)
          faces.push(makeFace([innerTopB, innerTopA, innerBottomA, innerBottomB], shadeColor(color, 0.85)));
          // Top annular ring
          faces.push(makeFace([outerTopA, innerTopA, innerTopB, outerTopB], topColor));
          // Bottom annular ring
          faces.push(makeFace([outerBottomB, innerBottomB, innerBottomA, outerBottomA], bottomColor));
        }
      }

      // Build the side wall as a stack of horizontal rings so chamfers / fillets
      // can be applied to the top and bottom edges of both the outer and inner walls.
      function buildProfile() {
        if (edgeBreak <= 0) {
          return {
            outer: [
              { y: topY, dr: 0 },
              { y: bottomY, dr: 0 },
            ],
            inner: [
              { y: topY, dr: 0 },
              { y: bottomY, dr: 0 },
            ],
          };
        }
        if (useChamfer) {
          const c = edgeBreak;
          return {
            outer: [
              { y: topY, dr: -c },
              { y: topY - c, dr: 0 },
              { y: bottomY + c, dr: 0 },
              { y: bottomY, dr: -c },
            ],
            inner: [
              { y: topY, dr: c },
              { y: topY - c, dr: 0 },
              { y: bottomY + c, dr: 0 },
              { y: bottomY, dr: c },
            ],
          };
        }
        // Fillet: quarter-circle sweep with `filletSteps` segments at each corner.
        const r = edgeBreak;
        const outer = [];
        const inner = [];
        for (let step = 0; step <= filletSteps; step += 1) {
          const t = step / filletSteps;
          const angle = (Math.PI / 2) * t;
          const dx = r - r * Math.cos(angle);
          const dy = r * Math.sin(angle);
          outer.push({ y: topY - dy, dr: -dx });
          inner.push({ y: topY - dy, dr: dx });
        }
        for (let step = 0; step <= filletSteps; step += 1) {
          const t = step / filletSteps;
          const angle = (Math.PI / 2) * t;
          const dy = r * Math.sin(angle);
          const dx = r - r * Math.cos(angle);
          outer.push({ y: bottomY + dy - 1e-6, dr: -dx });
          inner.push({ y: bottomY + dy - 1e-6, dr: dx });
        }
        // Add the actual bottom edge so the wall closes.
        outer.push({ y: bottomY, dr: -r });
        inner.push({ y: bottomY, dr: r });
        return { outer, inner };
      }

      // Add a rubber annulus whose outer + inner edges can carry a chamfer or fillet.
      function addProfiledAnnulus(rOutBase, rInBase, centerOutX, centerInX, color, topColor, bottomColor) {
        const profile = buildProfile();
        const outerRings = profile.outer.map((row) => ({
          y: row.y,
          radius: Math.max(rOutBase + row.dr, rInBase + 0.4),
        }));
        const innerRings = profile.inner.map((row) => ({
          y: row.y,
          radius: Math.min(rInBase + row.dr, rOutBase - 0.4),
        }));

        function sweepWall(rings, centerX, isOuter) {
          for (let level = 0; level < rings.length - 1; level += 1) {
            const ringA = rings[level];
            const ringB = rings[level + 1];
            for (let index = 0; index < segments; index += 1) {
              const thetaA = (index / segments) * Math.PI * 2;
              const thetaB = ((index + 1) / segments) * Math.PI * 2;
              const a1 = pointOnRing(ringA.radius, thetaA, ringA.y, centerX);
              const a2 = pointOnRing(ringA.radius, thetaB, ringA.y, centerX);
              const b1 = pointOnRing(ringB.radius, thetaA, ringB.y, centerX);
              const b2 = pointOnRing(ringB.radius, thetaB, ringB.y, centerX);
              const shade = level === 0 || level === rings.length - 2 ? topColor : color;
              const face = isOuter
                ? [a1, a2, b2, b1]
                : [a2, a1, b1, b2];
              faces.push(makeFace(face, shade));
            }
          }
        }

        sweepWall(outerRings, centerOutX, true);
        sweepWall(innerRings, centerInX, false);

        // Cap top + bottom annulus rings between the (possibly shrunk) outer/inner edges.
        const topOuter = outerRings[0];
        const topInner = innerRings[0];
        const bottomOuter = outerRings[outerRings.length - 1];
        const bottomInner = innerRings[innerRings.length - 1];
        for (let index = 0; index < segments; index += 1) {
          const thetaA = (index / segments) * Math.PI * 2;
          const thetaB = ((index + 1) / segments) * Math.PI * 2;
          const tOA = pointOnRing(topOuter.radius, thetaA, topOuter.y, centerOutX);
          const tOB = pointOnRing(topOuter.radius, thetaB, topOuter.y, centerOutX);
          const tIA = pointOnRing(topInner.radius, thetaA, topInner.y, centerInX);
          const tIB = pointOnRing(topInner.radius, thetaB, topInner.y, centerInX);
          const bOA = pointOnRing(bottomOuter.radius, thetaA, bottomOuter.y, centerOutX);
          const bOB = pointOnRing(bottomOuter.radius, thetaB, bottomOuter.y, centerOutX);
          const bIA = pointOnRing(bottomInner.radius, thetaA, bottomInner.y, centerInX);
          const bIB = pointOnRing(bottomInner.radius, thetaB, bottomInner.y, centerInX);
          faces.push(makeFace([tOA, tIA, tIB, tOB], topColor));
          faces.push(makeFace([bOB, bIB, bIA, bOA], bottomColor));
        }
      }

      // Outer metal sleeve (bonded).
      if (safeSleeveOuter > 0) {
        addAnnulus(rOuter, rRubberOuter, 0, 0, colorMetal, colorMetalTop, colorMetalBottom);
      }

      // Rubber middle layer (the main visible body for Vibracoustic bushings).
      addProfiledAnnulus(rRubberOuter, rRubberInner, 0, offsetX, colorRubber, colorRubberTop, colorRubberBottom);

      // Inner metal sleeve (bonded around the bore).
      if (safeSleeveInner > 0) {
        addAnnulus(rRubberInner, rInner, offsetX, offsetX, colorMetal, colorMetalTop, colorMetalBottom);
      } else if (safeSleeveOuter > 0) {
        // Bore wall when only outer sleeve is set so the inside still looks like a hole.
        addAnnulus(rRubberInner, Math.max(rInner, 0.5), offsetX, offsetX, colorRubber, colorRubberTop, colorRubberBottom);
      }

      // Optional flange (a thin disk sitting on top of the bushing).
      if (flangeDiameter > outerDiameter && flangeThickness > 0) {
        const flangeOuterRadius = flangeDiameter / 2;
        const flangeTopY = topY + flangeThickness;
        for (let index = 0; index < segments; index += 1) {
          const thetaA = (index / segments) * Math.PI * 2;
          const thetaB = ((index + 1) / segments) * Math.PI * 2;
          const outerTopA = pointOnRing(flangeOuterRadius, thetaA, flangeTopY);
          const outerTopB = pointOnRing(flangeOuterRadius, thetaB, flangeTopY);
          const outerBottomA = pointOnRing(flangeOuterRadius, thetaA, topY);
          const outerBottomB = pointOnRing(flangeOuterRadius, thetaB, topY);
          const innerTopA = pointOnRing(rInner, thetaA, flangeTopY, offsetX);
          const innerTopB = pointOnRing(rInner, thetaB, flangeTopY, offsetX);
          const innerBottomA = pointOnRing(rOuter, thetaA, topY);
          const innerBottomB = pointOnRing(rOuter, thetaB, topY);
          // Outer wall of flange
          faces.push(makeFace([outerTopA, outerTopB, outerBottomB, outerBottomA], colorMetal));
          // Top face (annulus around the bore)
          faces.push(makeFace([outerTopA, innerTopA, innerTopB, outerTopB], colorMetalTop));
          // Underside ring sitting on the bushing's outer wall
          faces.push(makeFace([outerBottomB, innerBottomB, innerBottomA, outerBottomA], colorMetalBottom));
        }
      }

      // Arms / tabs / lugs attached to the outside of the bushing.
      // Prefer the new list (geometry.arms); fall back to the legacy single-arm fields.
      const armColor = safeSleeveOuter > 0 ? colorMetal : colorRubber;
      const armList = Array.isArray(geometry.arms) ? geometry.arms.slice() : [];
      if (armList.length === 0) {
        const legacyLength = readPositive(geometry.arm_length_mm, 0);
        const legacyWidth = readPositive(geometry.arm_width_mm, 0);
        const legacyThickness = readPositive(geometry.arm_thickness_mm, 0);
        if (legacyLength > 0 && legacyWidth > 0 && legacyThickness > 0) {
          armList.push({
            length_mm: legacyLength,
            width_mm: legacyWidth,
            thickness_mm: legacyThickness,
            angle_deg: 0,
            position: geometry.arm_position || "centered",
          });
        }
      }

      for (const arm of armList) {
        const armLength = readPositive(arm.length_mm, 0);
        const armWidth = readPositive(arm.width_mm, 0);
        const armThickness = readPositive(arm.thickness_mm, 0);
        if (armLength <= 0 || armWidth <= 0 || armThickness <= 0) {
          continue;
        }
        const armPosition = arm.position || "centered";
        const halfThickness = armThickness / 2;
        let centerY;
        if (armPosition === "top") {
          centerY = topY - halfThickness;
        } else if (armPosition === "bottom") {
          centerY = bottomY + halfThickness;
        } else {
          centerY = 0;
        }
        const armCenterX = rOuter + armLength / 2;
        const boxFaces = createBoxFaces(
          armLength,
          armThickness,
          armWidth,
          { x: armCenterX, y: centerY, z: 0 },
          armColor
        );
        const angleRad = ((Number(arm.angle_deg) || 0) * Math.PI) / 180;
        if (Math.abs(angleRad) < 1e-6) {
          faces.push(...boxFaces);
        } else {
          const cos = Math.cos(angleRad);
          const sin = Math.sin(angleRad);
          const rotated = boxFaces.map((face) => ({
            color: face.color,
            points: face.points.map((p) => ({
              x: p.x * cos + p.z * sin,
              y: p.y,
              z: -p.x * sin + p.z * cos,
            })),
          }));
          faces.push(...rotated);
        }
      }

      // Bolt holes through the top face (visualized as dark cylindrical pockets).
      const holeList = Array.isArray(geometry.holes) ? geometry.holes : [];
      const topPlateY = flangeDiameter > outerDiameter && flangeThickness > 0
        ? topY + flangeThickness
        : topY;
      const holeColor = "#1b1b1b";
      const holeSides = 16;
      for (const hole of holeList) {
        const holeDiameter = readPositive(hole.diameter_mm, 0);
        const pcd = readPositive(hole.pitch_circle_diameter_mm, 0);
        const count = Math.max(1, Math.round(Number(hole.count) || 1));
        const startAngle = ((Number(hole.start_angle_deg) || 0) * Math.PI) / 180;
        if (holeDiameter <= 0 || pcd <= 0) {
          continue;
        }
        const holeRadius = holeDiameter / 2;
        const pcdRadius = pcd / 2;
        const holeDepth = Math.max(flangeThickness, 1.5);
        const holeTopY = topPlateY + 0.05;
        const holeBottomY = topPlateY - holeDepth;
        for (let pick = 0; pick < count; pick += 1) {
          const angle = startAngle + (pick / count) * Math.PI * 2;
          const cx = Math.cos(angle) * pcdRadius;
          const cz = Math.sin(angle) * pcdRadius;
          for (let side = 0; side < holeSides; side += 1) {
            const a = (side / holeSides) * Math.PI * 2;
            const b = ((side + 1) / holeSides) * Math.PI * 2;
            const ax = cx + Math.cos(a) * holeRadius;
            const az = cz + Math.sin(a) * holeRadius;
            const bx = cx + Math.cos(b) * holeRadius;
            const bz = cz + Math.sin(b) * holeRadius;
            // Pocket side wall
            faces.push(
              makeFace(
                [
                  { x: ax, y: holeTopY, z: az },
                  { x: bx, y: holeTopY, z: bz },
                  { x: bx, y: holeBottomY, z: bz },
                  { x: ax, y: holeBottomY, z: az },
                ],
                holeColor
              )
            );
            // Pocket floor wedge
            faces.push(
              makeFace(
                [
                  { x: cx, y: holeBottomY, z: cz },
                  { x: ax, y: holeBottomY, z: az },
                  { x: bx, y: holeBottomY, z: bz },
                ],
                "#0e0e0e"
              )
            );
          }
        }
      }

      return { faces };
    }

    function createPlateMesh(geometry) {
      const length = readPositive(geometry.length_mm, 120);
      const width = readPositive(geometry.width_mm, 60);
      const thickness = readPositive(geometry.thickness_mm, 8);
      return {
        faces: createBoxFaces(length, thickness, width, { x: 0, y: 0, z: 0 }, "#95afcc"),
      };
    }

    function createBracketMesh(geometry) {
      const length = readPositive(geometry.length_mm, 120);
      const width = readPositive(geometry.width_mm, 60);
      const thickness = readPositive(geometry.thickness_mm, 8);
      const legHeight = Math.max(length * 0.7, thickness * 6);
      return {
        faces: [
          ...createBoxFaces(length, thickness, width, { x: 0, y: thickness / 2, z: 0 }, "#90abc9"),
          ...createBoxFaces(
            thickness,
            legHeight,
            width,
            { x: length / 2 - thickness / 2, y: legHeight / 2, z: 0 },
            "#7f9cc0"
          ),
        ],
      };
    }

    function createUnknownMesh(geometry) {
      const length = readPositive(geometry.length_mm, 90);
      const width = readPositive(geometry.width_mm, 70);
      const thickness = readPositive(geometry.thickness_mm, 50);
      return {
        faces: createBoxFaces(length, thickness, width, { x: 0, y: 0, z: 0 }, "#a0b6d0"),
      };
    }

    function createSpringMesh(geometry) {
      const coilDiameter = readPositive(geometry.outer_diameter_mm, 40);
      const freeLength = readPositive(geometry.height_mm, 50);
      const wireDiameter = readPositive(geometry.thickness_mm, Math.max(coilDiameter * 0.08, 2));
      const coils = Math.max(2, Math.round(readPositive(geometry.coil_count, 6)));

      const coilRadius = Math.max(coilDiameter / 2, wireDiameter);
      const wireRadius = Math.min(Math.max(wireDiameter / 2, 0.8), coilRadius * 0.45);
      const segmentsPerCoil = 28;
      const totalSteps = Math.max(coils * segmentsPerCoil, 32);
      const tubeSides = 8;
      const pitch = freeLength / coils;
      const startY = -freeLength / 2;
      const color = "#7f9bbd";

      const centerline = [];
      for (let step = 0; step <= totalSteps; step += 1) {
        const t = step / segmentsPerCoil;
        const theta = t * Math.PI * 2;
        centerline.push({
          x: coilRadius * Math.cos(theta),
          y: startY + pitch * t,
          z: coilRadius * Math.sin(theta),
        });
      }

      function ringAt(index) {
        const point = centerline[index];
        const next = centerline[Math.min(index + 1, centerline.length - 1)];
        const prev = centerline[Math.max(index - 1, 0)];
        const tangent = normalizeVector({
          x: next.x - prev.x,
          y: next.y - prev.y,
          z: next.z - prev.z,
        });
        let reference = { x: 0, y: 1, z: 0 };
        if (Math.abs(tangent.y) > 0.92) {
          reference = { x: 1, y: 0, z: 0 };
        }
        const normal = normalizeVector(crossProduct(tangent, reference));
        const binormal = normalizeVector(crossProduct(tangent, normal));
        const ring = [];
        for (let side = 0; side < tubeSides; side += 1) {
          const angle = (side / tubeSides) * Math.PI * 2;
          const cos = Math.cos(angle) * wireRadius;
          const sin = Math.sin(angle) * wireRadius;
          ring.push({
            x: point.x + normal.x * cos + binormal.x * sin,
            y: point.y + normal.y * cos + binormal.y * sin,
            z: point.z + normal.z * cos + binormal.z * sin,
          });
        }
        return ring;
      }

      const faces = [];
      let previousRing = ringAt(0);
      for (let index = 1; index < centerline.length; index += 1) {
        const currentRing = ringAt(index);
        for (let side = 0; side < tubeSides; side += 1) {
          const nextSide = (side + 1) % tubeSides;
          faces.push(
            makeFace(
              [previousRing[side], currentRing[side], currentRing[nextSide], previousRing[nextSide]],
              color
            )
          );
        }
        previousRing = currentRing;
      }

      return { faces };
    }

    function crossProduct(a, b) {
      return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
      };
    }

    function createBoxFaces(length, height, width, center, color) {
      const halfLength = length / 2;
      const halfHeight = height / 2;
      const halfWidth = width / 2;
      const vertices = [
        { x: center.x - halfLength, y: center.y - halfHeight, z: center.z - halfWidth },
        { x: center.x + halfLength, y: center.y - halfHeight, z: center.z - halfWidth },
        { x: center.x + halfLength, y: center.y + halfHeight, z: center.z - halfWidth },
        { x: center.x - halfLength, y: center.y + halfHeight, z: center.z - halfWidth },
        { x: center.x - halfLength, y: center.y - halfHeight, z: center.z + halfWidth },
        { x: center.x + halfLength, y: center.y - halfHeight, z: center.z + halfWidth },
        { x: center.x + halfLength, y: center.y + halfHeight, z: center.z + halfWidth },
        { x: center.x - halfLength, y: center.y + halfHeight, z: center.z + halfWidth },
      ];
      return [
        makeFace([vertices[0], vertices[1], vertices[2], vertices[3]], color),
        makeFace([vertices[4], vertices[5], vertices[6], vertices[7]], lightenHex(color, 0.05)),
        makeFace([vertices[0], vertices[4], vertices[7], vertices[3]], darkenHex(color, 0.08)),
        makeFace([vertices[1], vertices[5], vertices[6], vertices[2]], darkenHex(color, 0.16)),
        makeFace([vertices[3], vertices[2], vertices[6], vertices[7]], lightenHex(color, 0.12)),
        makeFace([vertices[0], vertices[1], vertices[5], vertices[4]], darkenHex(color, 0.2)),
      ];
    }

    function makeFace(points, color) {
      return { points, color };
    }

    function pointOnRing(radius, angle, y, centerX) {
      const cx = centerX || 0;
      return {
        x: cx + Math.cos(angle) * radius,
        y,
        z: Math.sin(angle) * radius,
      };
    }

    function computeMeshBounds(faces) {
      const points = faces.flatMap((face) => face.points);
      let minX = Infinity;
      let minY = Infinity;
      let minZ = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      let maxZ = -Infinity;

      for (const point of points) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);
        minZ = Math.min(minZ, point.z);
        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
        maxZ = Math.max(maxZ, point.z);
      }

      return {
        center: {
          x: (minX + maxX) / 2,
          y: (minY + maxY) / 2,
          z: (minZ + maxZ) / 2,
        },
        size: {
          x: maxX - minX,
          y: maxY - minY,
          z: maxZ - minZ,
        },
      };
    }

    function rotatePoint(point, rotationX, rotationY) {
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);
      const x1 = point.x * cosY - point.z * sinY;
      const z1 = point.x * sinY + point.z * cosY;

      const cosX = Math.cos(rotationX);
      const sinX = Math.sin(rotationX);
      const y2 = point.y * cosX - z1 * sinX;
      const z2 = point.y * sinX + z1 * cosX;

      return { x: x1, y: y2, z: z2 };
    }

    function computeFaceNormal(points) {
      const a = points[0];
      const b = points[1];
      const c = points[2];
      const ab = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
      const ac = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
      return normalizeVector({
        x: ab.y * ac.z - ab.z * ac.y,
        y: ab.z * ac.x - ab.x * ac.z,
        z: ab.x * ac.y - ab.y * ac.x,
      });
    }

    function normalizeVector(vector) {
      const length = Math.hypot(vector.x, vector.y, vector.z) || 1;
      return {
        x: vector.x / length,
        y: vector.y / length,
        z: vector.z / length,
      };
    }

    function dotProduct(left, right) {
      return left.x * right.x + left.y * right.y + left.z * right.z;
    }

    function shadeColor(color, intensity) {
      const rgb = hexToRgb(color);
      const base = intensity >= 0.5
        ? mixColor(rgb, { r: 255, g: 255, b: 255 }, (intensity - 0.5) * 0.5)
        : mixColor(rgb, { r: 30, g: 45, b: 76 }, (0.5 - intensity) * 0.9);
      return `rgb(${base.r}, ${base.g}, ${base.b})`;
    }

    function lightenHex(color, amount) {
      return rgbToHex(mixColor(hexToRgb(color), { r: 255, g: 255, b: 255 }, amount));
    }

    function darkenHex(color, amount) {
      return rgbToHex(mixColor(hexToRgb(color), { r: 28, g: 43, b: 70 }, amount));
    }

    function mixColor(source, target, amount) {
      return {
        r: Math.round(source.r + (target.r - source.r) * clamp(amount, 0, 1)),
        g: Math.round(source.g + (target.g - source.g) * clamp(amount, 0, 1)),
        b: Math.round(source.b + (target.b - source.b) * clamp(amount, 0, 1)),
      };
    }

    function hexToRgb(color) {
      const value = color.replace("#", "");
      const normalized = value.length === 3
        ? value.split("").map((char) => char + char).join("")
        : value;
      return {
        r: parseInt(normalized.slice(0, 2), 16),
        g: parseInt(normalized.slice(2, 4), 16),
        b: parseInt(normalized.slice(4, 6), 16),
      };
    }

    function rgbToHex(color) {
      return `#${[color.r, color.g, color.b]
        .map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0"))
        .join("")}`;
    }

    function niceGridSpacing(extent) {
      const roughSpacing = Math.max(extent / 4.5, 10);
      const magnitude = Math.pow(10, Math.floor(Math.log10(roughSpacing)));
      const normalized = roughSpacing / magnitude;
      if (normalized <= 1.5) return magnitude;
      if (normalized <= 3.5) return magnitude * 2;
      if (normalized <= 7.5) return magnitude * 5;
      return magnitude * 10;
    }

    function readPositive(value, fallback) {
      const numeric = Number(value);
      return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
    }

    // ===== Download / export helpers =====

    function exportBaseName(intent) {
      const type = String((intent && intent.part_type) || "model")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
      return type || "model";
    }

    async function handleDownload(format) {
      const name = lastExport.name || "model";
      switch (format) {
        case "png": return downloadBlob(await canvasPngBlob(), name + ".png");
        case "json": return downloadBlob(jsonBlob(), name + ".json");
        case "stl": return downloadBlob(stlBlob(requireMesh(), name), name + ".stl");
        case "glb": return downloadBlob(glbBlob(requireMesh()), name + ".glb");
        case "dxf": return downloadBlob(dxfBlob(lastExport.intent || {}), name + ".dxf");
        case "pdf": return downloadBlob(await pdfBlob(), name + ".pdf");
        case "step": return downloadStep(name);
        case "zip": return downloadAll(name);
        default: throw new Error("Unknown format: " + format);
      }
    }

    function requireMesh() {
      if (!lastExport.mesh) throw new Error("Generate a 3D model first, then download.");
      return lastExport.mesh;
    }

    function downloadBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function exNum(value, fallback) {
      const parsed = Number(value);
      return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
    }

    function concatBytes(arrays) {
      let total = 0;
      for (const part of arrays) total += part.length;
      const out = new Uint8Array(total);
      let offset = 0;
      for (const part of arrays) {
        out.set(part, offset);
        offset += part.length;
      }
      return out;
    }

    function hexToRgb01(hex) {
      const match = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
      if (!match) return [0.6, 0.6, 0.6];
      const value = parseInt(match[1], 16);
      return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
    }

    function triNormal(a, b, c) {
      const ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
      const vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz) || 1;
      return [nx / len, ny / len, nz / len];
    }

    function triangulateMesh(mesh) {
      const positions = [];
      const colors = [];
      if (!mesh || !Array.isArray(mesh.faces)) return { positions, colors };
      for (const face of mesh.faces) {
        const pts = face.points;
        if (!pts || pts.length < 3) continue;
        const [r, g, b] = hexToRgb01(face.color || "#999999");
        for (let i = 1; i < pts.length - 1; i += 1) {
          for (const point of [pts[0], pts[i], pts[i + 1]]) {
            positions.push(point.x, point.y, point.z);
            colors.push(r, g, b);
          }
        }
      }
      return { positions, colors };
    }

    function jsonBlob() {
      const data = JSON.stringify(lastExport.intent || {}, null, 2);
      return new Blob([data], { type: "application/json" });
    }

    function canvasPngBlob() {
      const canvas = lastExport.canvas;
      if (!canvas) throw new Error("Generate a 3D model first, then download.");
      const flat = flattenCanvas(canvas);
      return new Promise((resolve, reject) => {
        flat.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not render PNG."))), "image/png");
      });
    }

    function flattenCanvas(canvas) {
      const off = document.createElement("canvas");
      off.width = canvas.width;
      off.height = canvas.height;
      const ctx = off.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, off.width, off.height);
      ctx.drawImage(canvas, 0, 0);
      return off;
    }

    function stlBlob(mesh, name) {
      const lines = ["solid " + name];
      for (const face of mesh.faces) {
        const pts = face.points;
        if (!pts || pts.length < 3) continue;
        for (let i = 1; i < pts.length - 1; i += 1) {
          const a = pts[0], b = pts[i], c = pts[i + 1];
          const nrm = triNormal(a, b, c);
          lines.push("  facet normal " + nrm[0] + " " + nrm[1] + " " + nrm[2]);
          lines.push("    outer loop");
          for (const point of [a, b, c]) lines.push("      vertex " + point.x + " " + point.y + " " + point.z);
          lines.push("    endloop");
          lines.push("  endfacet");
        }
      }
      lines.push("endsolid " + name);
      return new Blob([lines.join("\\n")], { type: "model/stl" });
    }

    function align4(value) { return (value + 3) & ~3; }

    function glbBlob(mesh) {
      const { positions, colors } = triangulateMesh(mesh);
      if (!positions.length) throw new Error("No geometry to export.");
      const posArray = new Float32Array(positions);
      const colArray = new Float32Array(colors);
      const count = posArray.length / 3;
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < posArray.length; i += 3) {
        for (let k = 0; k < 3; k += 1) {
          const value = posArray[i + k];
          if (value < min[k]) min[k] = value;
          if (value > max[k]) max[k] = value;
        }
      }
      const posBytes = posArray.byteLength;
      const colBytes = colArray.byteLength;
      const colOffset = align4(posBytes);
      const binLength = align4(colOffset + colBytes);
      const bin = new Uint8Array(binLength);
      bin.set(new Uint8Array(posArray.buffer), 0);
      bin.set(new Uint8Array(colArray.buffer), colOffset);

      const gltf = {
        asset: { version: "2.0", generator: "Resonance AI" },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0, name: lastExport.name || "model" }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 }, material: 0, mode: 4 }] }],
        materials: [{ name: "vertexColor", pbrMetallicRoughness: { metallicFactor: 0.1, roughnessFactor: 0.8 } }],
        buffers: [{ byteLength: binLength }],
        bufferViews: [
          { buffer: 0, byteOffset: 0, byteLength: posBytes, target: 34962 },
          { buffer: 0, byteOffset: colOffset, byteLength: colBytes, target: 34962 }
        ],
        accessors: [
          { bufferView: 0, componentType: 5126, count, type: "VEC3", min, max },
          { bufferView: 1, componentType: 5126, count, type: "VEC3" }
        ]
      };

      let jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
      const jsonPad = align4(jsonBytes.length) - jsonBytes.length;
      if (jsonPad) {
        const padded = new Uint8Array(jsonBytes.length + jsonPad);
        padded.set(jsonBytes);
        padded.fill(0x20, jsonBytes.length);
        jsonBytes = padded;
      }
      const totalLength = 12 + 8 + jsonBytes.length + 8 + bin.length;
      const buffer = new ArrayBuffer(totalLength);
      const dv = new DataView(buffer);
      let off = 0;
      dv.setUint32(off, 0x46546c67, true); off += 4;
      dv.setUint32(off, 2, true); off += 4;
      dv.setUint32(off, totalLength, true); off += 4;
      dv.setUint32(off, jsonBytes.length, true); off += 4;
      dv.setUint32(off, 0x4e4f534a, true); off += 4;
      new Uint8Array(buffer, off, jsonBytes.length).set(jsonBytes); off += jsonBytes.length;
      dv.setUint32(off, bin.length, true); off += 4;
      dv.setUint32(off, 0x004e4942, true); off += 4;
      new Uint8Array(buffer, off, bin.length).set(bin);
      return new Blob([buffer], { type: "model/gltf-binary" });
    }

    function dxfBlob(intent) {
      const geometry = (intent && intent.geometry) || {};
      const type = String((intent && intent.part_type) || "").toLowerCase();
      const entities = [];
      const circle = (cx, cy, r) => {
        if (!(r > 0)) return;
        entities.push("0", "CIRCLE", "8", "PROFILE", "10", String(cx), "20", String(cy), "30", "0", "40", String(r));
      };
      const rect = (w, h) => {
        const x = w / 2, y = h / 2;
        entities.push("0", "LWPOLYLINE", "8", "PROFILE", "90", "4", "70", "1");
        for (const [px, py] of [[-x, -y], [x, -y], [x, y], [-x, y]]) entities.push("10", String(px), "20", String(py));
      };
      if (type === "bushing" || type === "rubber_mount") {
        const od = exNum(geometry.outer_diameter_mm, 60);
        const id = exNum(geometry.inner_diameter_mm, 20);
        const fd = exNum(geometry.flange_diameter_mm, 0);
        if (fd > od) circle(0, 0, fd / 2);
        circle(0, 0, od / 2);
        circle(0, 0, id / 2);
      } else if (type === "plate" || type === "bracket") {
        rect(exNum(geometry.length_mm, 120), exNum(geometry.width_mm, 80));
        const holes = Array.isArray(geometry.holes) ? geometry.holes : [];
        for (const hole of holes) circle(0, 0, exNum(hole.diameter_mm, 0) / 2);
      } else if (type === "spring") {
        circle(0, 0, exNum(geometry.outer_diameter_mm, exNum(geometry.coil_diameter_mm, 40)) / 2);
      } else {
        rect(exNum(geometry.length_mm, 60), exNum(geometry.width_mm, 40));
      }
      const dxf = ["0", "SECTION", "2", "ENTITIES", ...entities, "0", "ENDSEC", "0", "EOF"].join("\\n");
      return new Blob([dxf], { type: "application/dxf" });
    }

    function pdfSummaryLines(intent) {
      const geometry = (intent && intent.geometry) || {};
      const lines = [];
      lines.push("Part type: " + ((intent && intent.part_type) || "unknown"));
      const material = (intent && intent.material) || {};
      if (material.name) {
        lines.push("Material: " + material.name + (material.shore_a ? " (Shore A " + material.shore_a + ")" : ""));
      }
      lines.push("");
      lines.push("Dimensions (mm):");
      const dimKeys = [
        ["outer_diameter_mm", "Outer diameter"], ["inner_diameter_mm", "Inner diameter"],
        ["height_mm", "Height"], ["length_mm", "Length"], ["width_mm", "Width"],
        ["thickness_mm", "Thickness"], ["flange_diameter_mm", "Flange diameter"],
        ["flange_thickness_mm", "Flange thickness"], ["chamfer_mm", "Chamfer"],
        ["fillet_mm", "Fillet"], ["coil_count", "Coil count"]
      ];
      for (const [key, label] of dimKeys) {
        if (geometry[key] !== undefined && geometry[key] !== null && geometry[key] !== "" && Number(geometry[key]) > 0) {
          lines.push("  " + label + ": " + geometry[key]);
        }
      }
      const missing = (intent && intent.missing_information) || [];
      if (Array.isArray(missing) && missing.length) {
        lines.push("");
        lines.push("Assumptions / open items:");
        for (const item of missing) lines.push("  - " + item);
      }
      lines.push("");
      lines.push("Generated by Resonance AI on " + new Date().toISOString().slice(0, 10));
      return lines;
    }

    function pdfEscape(text) {
      return String(text).replace(/\\\\/g, "\\\\\\\\").replace(/\\(/g, "\\\\(").replace(/\\)/g, "\\\\)");
    }

    function dataUrlToUint8(dataUrl) {
      const base64 = dataUrl.split(",")[1] || "";
      const binary = atob(base64);
      const out = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
      return out;
    }

    async function pdfBlob() {
      const intent = lastExport.intent || {};
      const textLines = pdfSummaryLines(intent);
      let jpeg = null, imgW = 0, imgH = 0;
      if (lastExport.canvas) {
        const flat = flattenCanvas(lastExport.canvas);
        jpeg = dataUrlToUint8(flat.toDataURL("image/jpeg", 0.92));
        imgW = flat.width;
        imgH = flat.height;
      }
      return buildPdf(textLines, jpeg, imgW, imgH);
    }

    function buildPdf(textLines, jpeg, imgW, imgH) {
      const enc = (text) => new TextEncoder().encode(text);
      const pageWidth = 595.28, pageHeight = 841.89;
      const hasImage = !!(jpeg && imgW && imgH);

      let drawImage = "";
      let textTop = pageHeight - 90;
      if (hasImage) {
        const maxW = pageWidth - 100;
        const maxH = 300;
        const ratio = Math.min(maxW / imgW, maxH / imgH);
        const w = imgW * ratio, h = imgH * ratio;
        const x = (pageWidth - w) / 2;
        const y = pageHeight - 90 - h;
        drawImage = "q " + w.toFixed(2) + " 0 0 " + h.toFixed(2) + " " + x.toFixed(2) + " " + y.toFixed(2) + " cm /Im0 Do Q\\n";
        textTop = y - 28;
      }

      let content = "BT /F1 18 Tf 50 " + (pageHeight - 60).toFixed(2) + " Td (Resonance AI - CAD Technical Summary) Tj ET\\n";
      content += drawImage;
      content += "BT /F1 11 Tf 50 " + textTop.toFixed(2) + " Td 16 TL\\n";
      for (const line of textLines) content += "(" + pdfEscape(line) + ") Tj T*\\n";
      content += "ET\\n";
      const contentBytes = enc(content);

      const objBodies = [];
      objBodies.push(enc("<< /Type /Catalog /Pages 2 0 R >>"));
      objBodies.push(enc("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"));
      let resources = "<< /Font << /F1 5 0 R >>";
      if (hasImage) resources += " /XObject << /Im0 6 0 R >>";
      resources += " >>";
      objBodies.push(enc("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + pageWidth + " " + pageHeight + "] /Resources " + resources + " /Contents 4 0 R >>"));
      objBodies.push(concatBytes([enc("<< /Length " + contentBytes.length + " >>\\nstream\\n"), contentBytes, enc("\\nendstream")]));
      objBodies.push(enc("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"));
      if (hasImage) {
        const header = enc("<< /Type /XObject /Subtype /Image /Width " + imgW + " /Height " + imgH + " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length " + jpeg.length + " >>\\nstream\\n");
        objBodies.push(concatBytes([header, jpeg, enc("\\nendstream")]));
      }

      const chunks = [];
      let length = 0;
      const offsets = [];
      const push = (bytes) => { chunks.push(bytes); length += bytes.length; };
      push(enc("%PDF-1.4\\n"));
      push(new Uint8Array([0x25, 0xe2, 0xe3, 0xcf, 0xd3, 0x0a]));
      for (let i = 0; i < objBodies.length; i += 1) {
        offsets[i + 1] = length;
        push(enc((i + 1) + " 0 obj\\n"));
        push(objBodies[i]);
        push(enc("\\nendobj\\n"));
      }
      const xrefStart = length;
      const total = objBodies.length + 1;
      let xref = "xref\\n0 " + total + "\\n0000000000 65535 f \\n";
      for (let i = 1; i < total; i += 1) xref += String(offsets[i]).padStart(10, "0") + " 00000 n \\n";
      push(enc(xref));
      push(enc("trailer\\n<< /Size " + total + " /Root 1 0 R >>\\nstartxref\\n" + xrefStart + "\\n%%EOF"));
      return new Blob([concatBytes(chunks)], { type: "application/pdf" });
    }

    async function downloadStep(name) {
      startActivity("Generating STEP", ["Sending request", "Running CadQuery", "Packaging STEP"]);
      try {
        const response = await fetch("/export/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: lastExport.prompt || "", intent: lastExport.intent || {}, name })
        });
        if (!response.ok) {
          let detail = "STEP export is not available on this server.";
          try { const payload = await response.json(); detail = payload.detail || detail; } catch (err) {}
          throw new Error(detail);
        }
        downloadBlob(await response.blob(), name + ".step");
        completeActivity("STEP ready");
      } catch (error) {
        completeActivity("STEP unavailable");
        throw error;
      }
    }

    function crc32(bytes) {
      let crc = ~0;
      for (let i = 0; i < bytes.length; i += 1) {
        crc ^= bytes[i];
        for (let j = 0; j < 8; j += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
      }
      return (~crc) >>> 0;
    }

    function zipStore(files) {
      const encoder = new TextEncoder();
      const u16 = (value) => new Uint8Array([value & 255, (value >> 8) & 255]);
      const u32 = (value) => new Uint8Array([value & 255, (value >> 8) & 255, (value >> 16) & 255, (value >>> 24) & 255]);
      const locals = [];
      const central = [];
      let offset = 0;
      for (const file of files) {
        const nameBytes = encoder.encode(file.name);
        const crc = crc32(file.data);
        const size = file.data.length;
        const local = concatBytes([
          u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
          u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0),
          nameBytes, file.data
        ]);
        locals.push(local);
        central.push(concatBytes([
          u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
          u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0), u16(0), u16(0), u16(0), u32(0),
          u32(offset), nameBytes
        ]));
        offset += local.length;
      }
      const centralBytes = concatBytes(central);
      const end = concatBytes([
        u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
        u32(centralBytes.length), u32(offset), u16(0)
      ]);
      return new Blob([concatBytes([...locals, centralBytes, end])], { type: "application/zip" });
    }

    async function downloadAll(name) {
      const files = [];
      const add = async (suffix, blob) => files.push({ name: name + suffix, data: new Uint8Array(await blob.arrayBuffer()) });
      await add(".json", jsonBlob());
      await add(".dxf", dxfBlob(lastExport.intent || {}));
      if (lastExport.mesh) {
        await add(".stl", stlBlob(lastExport.mesh, name));
        await add(".glb", glbBlob(lastExport.mesh));
      }
      if (lastExport.canvas) {
        await add(".png", await canvasPngBlob());
        await add(".pdf", await pdfBlob());
      }
      downloadBlob(zipStore(files), name + ".zip");
    }

    function cleanupViewer() {
      lastExport.mesh = null;
      lastExport.canvas = null;
      if (!activeViewer) {
        return;
      }

      activeViewer.dispose();
      preview.innerHTML = "";
      activeViewer = null;
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    window.addEventListener("beforeunload", cleanupViewer);

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""
