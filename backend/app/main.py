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
        return await respond_to_cad_chat(request.message, request.prompt)
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
      min-height: 420px;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: linear-gradient(135deg, #f9fbfd, #eef3f8);
      display: flex;
      flex-direction: column;
      overflow: hidden;
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
    .msg.bot.err { border-color: #fecdca; background: #fff7f6; color: var(--danger); }
    .chat-composer {
      display: flex;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      align-items: flex-end;
    }
    .chat-composer textarea {
      flex: 1;
      min-height: 56px;
      max-height: 144px;
      resize: none;
      margin: 0;
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
        <div class="chat-shell">
          <div id="chatLog" class="chat-log">
            <div class="msg bot">Start by adding a PDF, image, JSON status file, or old CAD model. I will extract context and draft a short prompt for approval. Images are reference-only in this POC, so please type the key dimensions in chat.</div>
          </div>
          <form id="chatForm" class="chat-composer">
            <textarea id="chatInput" placeholder="Write your CAD request here..." autocomplete="off"></textarea>
            <button type="submit" id="chatSend" class="primary-action">Send</button>
          </form>
        </div>
        <div class="summary-box" id="summaryBox">
          <p>Upload a file or write a request. I will summarize the proposed CAD intent before generating the model.</p>
        </div>
      </section>

      <section class="stack">
        <div class="panel">
          <div class="section-title">
            <strong>3D CAD Model</strong>
            <span class="muted">Interactive preview</span>
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
    import * as THREE from "https://unpkg.com/three@0.166.1/build/three.module.js";
    import { OrbitControls } from "https://unpkg.com/three@0.166.1/examples/jsm/controls/OrbitControls.js";

    const preview = document.getElementById("preview");
    const jsonOutput = document.getElementById("jsonOutput");
    const summaryBox = document.getElementById("summaryBox");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatLog = document.getElementById("chatLog");
    const chatSend = document.getElementById("chatSend");
    const contextFile = document.getElementById("contextFile");
    const uploadContextButton = document.getElementById("uploadContextButton");
    const attachmentList = document.getElementById("attachmentList");
    let activeViewer = null;
    let requestContext = [];
    let attachmentContexts = [];
    let pendingDraftPrompt = "";

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

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = chatInput.value.trim();
      if (!text) return;
      appendMsg("user", text);
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
      const thinking = appendMsg("bot", "Reviewing your request...");
      try {
        const prompt = buildFullPrompt();
        const response = await fetch("/chat-cad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({message: text, prompt})
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "CAD parsing failed");
        }
        const intent = payload.cad_intent || {};
        const previewReady = Boolean(payload.preview_ready);
        jsonOutput.textContent = JSON.stringify(intent, null, 2);
        if (previewReady) {
          render3DPreview(intent);
        } else {
          cleanupViewer();
          preview.innerHTML = '<div class="placeholder"><p class="muted">I need one more engineering detail before I can show a useful preview.</p></div>';
        }
        updateSummary(intent);
        thinking.textContent = payload.assistant_message || formatChatSummary(intent);
      } catch (error) {
        thinking.classList.add("err");
        thinking.textContent = error.message;
        summaryBox.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
        preview.innerHTML = `<pre class="error-box">${escapeHtml(error.message)}</pre>`;
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
      } catch (error) {
        appendMsg("bot err", error.message);
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
        fillet_mm: "fillet"
      };
      const parts = [];
      for (const [key, label] of Object.entries(labels)) {
        const value = geometry[key];
        if (value !== null && value !== undefined && value !== "") {
          parts.push(`${label} ${value} mm`);
        }
      }
      return parts.length ? parts.join(", ") : "No dimensions specified";
    }

    function appendMsg(role, text) {
      const el = document.createElement("div");
      el.className = `msg ${role}`;
      el.textContent = text;
      chatLog.appendChild(el);
      chatLog.scrollTop = chatLog.scrollHeight;
      return el;
    }

    function autoResizeChatInput() {
      chatInput.style.height = "auto";
      chatInput.style.height = `${Math.min(chatInput.scrollHeight, 144)}px`;
    }

    function render3DPreview(cadIntent) {
      cleanupViewer();

      const container = document.createElement("div");
      container.className = "viewer3d";
      preview.innerHTML = "";
      preview.appendChild(container);

      const initialWidth = Math.max(320, Math.floor(preview.clientWidth || 720));
      const initialHeight = Math.max(320, Math.floor(preview.clientHeight || 470));

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf7fbff);

      const camera = new THREE.PerspectiveCamera(44, initialWidth / initialHeight, 0.1, 10000);
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setSize(initialWidth, initialHeight, false);
      container.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.screenSpacePanning = true;
      controls.minDistance = 5;
      controls.maxDistance = 8000;

      scene.add(new THREE.HemisphereLight(0xffffff, 0xa4bfdc, 0.85));
      const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
      keyLight.position.set(180, 220, 140);
      scene.add(keyLight);

      const fillLight = new THREE.DirectionalLight(0xffffff, 0.45);
      fillLight.position.set(-180, 120, -140);
      scene.add(fillLight);

      const grid = new THREE.GridHelper(800, 40, 0xc3d2e5, 0xdeebf8);
      scene.add(grid);

      const meshGroup = createPartObject(cadIntent || {});
      scene.add(meshGroup);

      const bounds = new THREE.Box3().setFromObject(meshGroup);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      meshGroup.position.sub(center);

      const fit = Math.max(size.x, size.y, size.z, 40);
      camera.position.set(fit * 1.35, fit * 1.1, fit * 1.45);
      controls.target.set(0, 0, 0);
      controls.update();

      const resizeObserver = new ResizeObserver(() => {
        const width = Math.max(280, Math.floor(preview.clientWidth || initialWidth));
        const height = Math.max(280, Math.floor(preview.clientHeight || initialHeight));
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      });
      resizeObserver.observe(preview);

      let animationFrameId = 0;
      const animate = () => {
        animationFrameId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };
      animate();

      activeViewer = {
        scene,
        renderer,
        controls,
        resizeObserver,
        animationFrameId,
      };
    }

    function createPartObject(cadIntent) {
      const type = String(cadIntent.part_type || "unknown").toLowerCase();
      const geometry = cadIntent.geometry || {};
      if (type === "bushing" || type === "rubber_mount") {
        return createBushingObject(geometry);
      }
      if (type === "plate") {
        return createPlateObject(geometry);
      }
      if (type === "bracket") {
        return createBracketObject(geometry);
      }
      return createUnknownObject(geometry);
    }

    function createBushingObject(geometry) {
      const outerDiameter = readPositive(geometry.outer_diameter_mm, 60);
      const innerDiameter = readPositive(geometry.inner_diameter_mm, 20);
      const height = readPositive(geometry.height_mm, 40);
      const outerRadius = Math.max(outerDiameter / 2, 5);
      const innerRadius = Math.min(Math.max(innerDiameter / 2, 2), outerRadius - 1);

      const shape = new THREE.Shape();
      shape.absarc(0, 0, outerRadius, 0, Math.PI * 2, false);
      const hole = new THREE.Path();
      hole.absarc(0, 0, innerRadius, 0, Math.PI * 2, true);
      shape.holes.push(hole);

      const bodyGeometry = new THREE.ExtrudeGeometry(shape, {
        depth: height,
        bevelEnabled: false,
        curveSegments: 80,
      });
      bodyGeometry.translate(0, 0, -height / 2);

      const bodyMaterial = new THREE.MeshStandardMaterial({
        color: 0x7f9bbd,
        roughness: 0.56,
        metalness: 0.18,
      });
      const mesh = new THREE.Mesh(bodyGeometry, bodyMaterial);

      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(bodyGeometry, 30),
        new THREE.LineBasicMaterial({ color: 0x435f81 })
      );

      const group = new THREE.Group();
      group.add(mesh);
      group.add(edges);
      return group;
    }

    function createPlateObject(geometry) {
      const length = readPositive(geometry.length_mm, 120);
      const width = readPositive(geometry.width_mm, 60);
      const thickness = readPositive(geometry.thickness_mm, 8);

      const bodyGeometry = new THREE.BoxGeometry(length, thickness, width);
      const bodyMaterial = new THREE.MeshStandardMaterial({
        color: 0x95afcc,
        roughness: 0.55,
        metalness: 0.25,
      });

      const mesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(bodyGeometry, 25),
        new THREE.LineBasicMaterial({ color: 0x4f6b8d })
      );

      const group = new THREE.Group();
      group.add(mesh);
      group.add(edges);
      return group;
    }

    function createBracketObject(geometry) {
      const length = readPositive(geometry.length_mm, 120);
      const width = readPositive(geometry.width_mm, 60);
      const thickness = readPositive(geometry.thickness_mm, 8);
      const legHeight = Math.max(length * 0.7, thickness * 6);

      const mat = new THREE.MeshStandardMaterial({
        color: 0x90abc9,
        roughness: 0.52,
        metalness: 0.28,
      });
      const edgeMat = new THREE.LineBasicMaterial({ color: 0x4a6688 });

      const baseGeom = new THREE.BoxGeometry(length, thickness, width);
      const baseMesh = new THREE.Mesh(baseGeom, mat);
      baseMesh.position.set(0, thickness / 2, 0);

      const legGeom = new THREE.BoxGeometry(thickness, legHeight, width);
      const legMesh = new THREE.Mesh(legGeom, mat);
      legMesh.position.set(length / 2 - thickness / 2, legHeight / 2, 0);

      const group = new THREE.Group();
      group.add(baseMesh);
      group.add(legMesh);

      const baseEdges = new THREE.LineSegments(new THREE.EdgesGeometry(baseGeom, 25), edgeMat);
      baseEdges.position.copy(baseMesh.position);
      const legEdges = new THREE.LineSegments(new THREE.EdgesGeometry(legGeom, 25), edgeMat);
      legEdges.position.copy(legMesh.position);
      group.add(baseEdges);
      group.add(legEdges);
      return group;
    }

    function createUnknownObject(geometry) {
      const length = readPositive(geometry.length_mm, 90);
      const width = readPositive(geometry.width_mm, 70);
      const thickness = readPositive(geometry.thickness_mm, 50);
      const bodyGeometry = new THREE.BoxGeometry(length, thickness, width);
      const bodyMaterial = new THREE.MeshStandardMaterial({
        color: 0xa0b6d0,
        roughness: 0.6,
        metalness: 0.2,
      });
      const mesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
      const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(bodyGeometry, 20),
        new THREE.LineBasicMaterial({ color: 0x5e7899 })
      );
      const group = new THREE.Group();
      group.add(mesh);
      group.add(edges);
      return group;
    }

    function readPositive(value, fallback) {
      const numeric = Number(value);
      return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
    }

    function cleanupViewer() {
      if (!activeViewer) {
        return;
      }

      cancelAnimationFrame(activeViewer.animationFrameId);
      activeViewer.resizeObserver.disconnect();
      activeViewer.controls.dispose();

      activeViewer.scene.traverse((node) => {
        if (node.geometry) {
          node.geometry.dispose();
        }
        if (node.material) {
          if (Array.isArray(node.material)) {
            node.material.forEach((material) => material.dispose());
          } else {
            node.material.dispose();
          }
        }
      });

      activeViewer.renderer.dispose();
      preview.innerHTML = "";
      activeViewer = null;
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
