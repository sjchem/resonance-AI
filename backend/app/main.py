"""FastAPI app for OpenAI-powered CAD prompt parsing."""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

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
          <div class="metric"><strong>02</strong><span>Structured CAD JSON</span></div>
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
            <strong>3D CAD Model</strong>
            <span class="muted">Interactive preview</span>
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
            <strong>Structured CAD JSON</strong>
          </div>
          <pre id="jsonOutput">{}</pre>
        </div>
      </section>
    </section>
  </main>

  <script type="module">
    const preview = document.getElementById("preview");
    const jsonOutput = document.getElementById("jsonOutput");
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
    let activeViewer = null;
    let activityTimer = null;
    let requestContext = [];
    let attachmentContexts = [];
    let pendingDraftPrompt = "";
    let chatHistory = [];

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
        if (previewReady) {
          try {
            await render3DPreview(intent);
          } catch (previewError) {
            cleanupViewer();
            preview.innerHTML = '<div class="placeholder"><p class="muted">The CAD intent was parsed, but the interactive preview library could not be loaded.</p></div>';
          }
        } else {
          cleanupViewer();
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
        attachmentContexts.push(payload);
        renderAttachmentList();
        pendingDraftPrompt = draftPromptFromAttachments();
        appendMsg(
          "bot",
          `Extracted file context from ${payload.filename}.\\n${payload.summary}\\n\\nProposed short CAD prompt:\\n${pendingDraftPrompt}\\n\\nType "proceed" to create the final CAD summary and interactive preview, or type corrections/additional dimensions.`
        );
        summaryBox.innerHTML = `
          <p><strong>Awaiting approval.</strong> Review the proposed short prompt in the chat. Type "proceed" to generate the structured CAD result and preview, or add corrections.</p>
        `;
        chatInput.value = "proceed";
        autoResizeChatInput();
        contextFile.value = "";
        completeActivity("Draft ready");
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
      return [
        "Create a CAD model from the uploaded engineering context.",
        summaries,
        "Extract the part type, material, dimensions, chamfer/fillet details, and any status or validation hints from the uploaded file.",
        "If critical dimensions are missing, mark them as missing rather than inventing them."
      ].join(" ");
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
          <dt>Output</dt><dd>Interactive preview and structured CAD JSON</dd>
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

      const mesh = createPreviewMesh(cadIntent || {});
      const bounds = computeMeshBounds(mesh.faces);
      const size = bounds.size;
      const extent = Math.max(size.x, size.y, size.z, 40);
      const center = bounds.center;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

      const state = {
        width: 0,
        height: 0,
        rotationX: -0.55,
        rotationY: 0.78,
        zoom: 1,
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

    function cleanupViewer() {
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
