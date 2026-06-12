"""FastAPI app for OpenAI-powered CAD prompt parsing."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.cad_preview import build_preview_svg
from app.openai_client import (
    CADValidationError,
    OpenAIConfigurationError,
    OpenAIRequestError,
    openai_status,
    parse_cad_prompt,
)
from app.schemas import CADPromptOutput, CADPromptRequest


app = FastAPI(title="Resonance AI", version="0.1.0")


@app.get("/")
async def health_check() -> dict[str, str]:
    """Simple health check for local development."""

    return {"status": "ok", "service": "openai-cad-prompt-parser"}


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> str:
    """Small local frontend for prompt parsing and CAD-style preview."""

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


@app.post("/generate-cad")
async def generate_cad(request: CADPromptRequest) -> dict:
    """Parse prompt JSON and return a lightweight CAD SVG preview."""

    parsed = await parse_cad(request)
    return {
        "cad_intent": parsed.model_dump(),
        "preview_svg": build_preview_svg(parsed),
        "note": "POC preview only; STEP/STL generation is intentionally not implemented in this backend yet.",
    }


@app.post("/preview-cad")
async def preview_cad(parsed: CADPromptOutput) -> dict:
    """Return a CAD-style preview for already structured step-by-step input."""

    return {
        "cad_intent": parsed.model_dump(),
        "preview_svg": build_preview_svg(parsed),
        "note": "POC preview only; STEP/STL generation is intentionally not implemented in this backend yet.",
    }


UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resonance AI — Vibracoustic</title>
  <style>
    :root {
      --bg: #f4f6f8;
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
      --soft: #eef3f8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
      color: #fff;
      background:
        linear-gradient(110deg, rgba(7, 31, 63, 0.97), rgba(7, 31, 63, 0.82) 52%, rgba(216, 34, 42, 0.82)),
        radial-gradient(circle at 70% 40%, rgba(255, 255, 255, 0.16), transparent 32%),
        var(--brand);
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
      color: #bed0e7;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 1.6px;
      text-transform: uppercase;
    }
    .hero-copy {
      margin: 0;
      max-width: 660px;
      color: #dbe7f5;
      font-size: 16px;
      line-height: 1.6;
    }
    .hero-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1px;
      background: rgba(255, 255, 255, 0.22);
      border: 1px solid rgba(255, 255, 255, 0.22);
    }
    .metric {
      min-height: 92px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.08);
    }
    .metric strong {
      display: block;
      font-size: 21px;
      line-height: 1;
      margin-bottom: 9px;
    }
    .metric span {
      display: block;
      color: #dbe7f5;
      font-size: 12px;
      line-height: 1.35;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(360px, 460px) minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }
    #cadForm, .panel {
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
    #submitButton {
      background: var(--accent);
    }
    #submitButton:hover { background: var(--accent-dark); }
    .chat {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .chat-log {
      min-height: 220px;
      max-height: 360px;
      overflow-y: auto;
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 12px;
      background: linear-gradient(135deg, #f9fbfd, #eef3f8);
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
    .chat-input {
      display: flex;
      gap: 8px;
    }
    .chat-input input {
      flex: 1;
    }
    .chat-input button {
      width: auto;
      margin: 0;
      padding: 10px 18px;
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
      background:
        linear-gradient(90deg, rgba(216, 224, 234, 0.34) 1px, transparent 1px),
        linear-gradient(rgba(216, 224, 234, 0.34) 1px, transparent 1px),
        #fff;
      background-size: 28px 28px;
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
      <form id="cadForm">
        <div class="section-title">
          <strong>CAD Request</strong>
          <span class="status-pill">Model ready</span>
        </div>
        <label for="prompt">CAD prompt</label>
        <textarea id="prompt" name="prompt">Create a rubber bushing with outer diameter 60 mm, inner diameter 20 mm, height 40 mm and chamfer 2 mm.</textarea>
        <button id="submitButton" type="submit">Parse and Preview</button>

        <hr class="divider">

        <label for="partType">Step 1: Part type</label>
        <select id="partType">
          <option value="bushing">Bushing</option>
          <option value="plate">Plate</option>
          <option value="bracket">Bracket</option>
        </select>

        <div id="guidedFields"></div>
        <button id="guidedButton" type="button">Build JSON and Preview</button>
      </form>

      <section class="stack">
        <div class="panel chat">
          <div class="section-title">
            <strong>Engineering Chat</strong>
            <span class="muted">Azure OpenAI</span>
          </div>
          <div id="chatLog" class="chat-log">
            <div class="msg bot">Describe a bushing, bracket, or plate and I will prepare the CAD intent for review.</div>
          </div>
          <form id="chatForm" class="chat-input">
            <input id="chatInput" type="text" placeholder="Ask Resonance AI to design a part..." autocomplete="off">
            <button type="submit" id="chatSend">Send</button>
          </form>
        </div>
        <div class="panel preview" id="preview">
          <div class="placeholder">
            <svg viewBox="0 0 640 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Technical CAD placeholder">
              <rect width="640" height="300" fill="none"/>
              <g fill="none" stroke="#071f3f" stroke-width="3">
                <path d="M110 200 L400 158 L530 210 L235 255 Z"/>
                <path d="M110 200 L110 156 L400 112 L530 164 L530 210"/>
                <path d="M400 112 L400 158"/>
                <circle cx="230" cy="194" r="20"/>
                <circle cx="405" cy="169" r="20"/>
              </g>
              <g stroke="#d8222a" stroke-width="2" fill="none">
                <path d="M92 146 H392"/>
                <path d="M92 146 l18 -10 M92 146 l18 10"/>
                <path d="M392 146 l-18 -10 M392 146 l-18 10"/>
                <path d="M552 165 V230"/>
                <path d="M552 165 l-9 16 M552 165 l9 16"/>
                <path d="M552 230 l-9 -16 M552 230 l9 -16"/>
              </g>
              <text x="244" y="132" text-anchor="middle" font-size="16" fill="#647084">length</text>
              <text x="570" y="203" font-size="16" fill="#647084">width</text>
            </svg>
            <p class="muted">CAD preview will appear here.</p>
          </div>
        </div>
        <div class="panel">
          <div class="section-title">
            <strong>Validated JSON</strong>
          </div>
          <pre id="jsonOutput">{}</pre>
        </div>
      </section>
    </section>
  </main>

  <script>
    const form = document.getElementById("cadForm");
    const button = document.getElementById("submitButton");
    const guidedButton = document.getElementById("guidedButton");
    const partType = document.getElementById("partType");
    const guidedFields = document.getElementById("guidedFields");
    const preview = document.getElementById("preview");
    const jsonOutput = document.getElementById("jsonOutput");

    const fieldTemplates = {
      bushing: [
        ["outer_diameter_mm", "Outer diameter mm", "60"],
        ["inner_diameter_mm", "Inner diameter mm", "20"],
        ["height_mm", "Height mm", "40"],
        ["chamfer_mm", "Chamfer mm", "2"],
        ["material", "Material", "rubber"]
      ],
      plate: [
        ["length_mm", "Length mm", "120"],
        ["width_mm", "Width mm", "60"],
        ["thickness_mm", "Thickness mm", "5"],
        ["chamfer_mm", "Chamfer mm", "0"],
        ["material", "Material", "steel"]
      ],
      bracket: [
        ["length_mm", "Length mm", "120"],
        ["width_mm", "Width mm", "60"],
        ["thickness_mm", "Thickness mm", "5"],
        ["chamfer_mm", "Chamfer mm", "0"],
        ["material", "Material", "steel"]
      ]
    };

    renderGuidedFields();
    partType.addEventListener("change", renderGuidedFields);

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      button.textContent = "Parsing...";
      preview.innerHTML = '<p class="muted">Waiting for the configured AI model...</p>';

      try {
        const response = await fetch("/generate-cad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({prompt: document.getElementById("prompt").value})
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "CAD parsing failed");
        }
        jsonOutput.textContent = JSON.stringify(payload.cad_intent, null, 2);
        preview.innerHTML = payload.preview_svg;
      } catch (error) {
        preview.innerHTML = `<pre class="error-box">${escapeHtml(error.message)}</pre>`;
      } finally {
        button.disabled = false;
        button.textContent = "Parse and Preview";
      }
    });

    guidedButton.addEventListener("click", async () => {
      guidedButton.disabled = true;
      guidedButton.textContent = "Building...";
      preview.innerHTML = '<p class="muted">Building structured CAD JSON...</p>';

      try {
        const cadIntent = buildGuidedIntent();
        const response = await fetch("/preview-cad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(cadIntent)
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "CAD preview failed");
        }
        jsonOutput.textContent = JSON.stringify(payload.cad_intent, null, 2);
        preview.innerHTML = payload.preview_svg;
      } catch (error) {
        preview.innerHTML = `<pre class="error-box">${escapeHtml(error.message)}</pre>`;
      } finally {
        guidedButton.disabled = false;
        guidedButton.textContent = "Build JSON and Preview";
      }
    });

    function renderGuidedFields() {
      const fields = fieldTemplates[partType.value];
      guidedFields.innerHTML = `
        <p class="muted">Step 2: Fill the required fields for ${escapeHtml(partType.value)}.</p>
        <div class="grid">
          ${fields.map(([name, label, value]) => `
            <div class="field">
              <label for="guided_${name}">${label}</label>
              <input id="guided_${name}" name="${name}" value="${value}">
            </div>
          `).join("")}
        </div>
      `;
    }

    function buildGuidedIntent() {
      const type = partType.value;
      const geometry = {
        outer_diameter_mm: null,
        inner_diameter_mm: null,
        height_mm: null,
        length_mm: null,
        width_mm: null,
        thickness_mm: null,
        chamfer_mm: null,
        fillet_mm: 0
      };

      for (const [name] of fieldTemplates[type]) {
        if (name === "material") continue;
        geometry[name] = readNumber(name);
      }

      return {
        part_type: type,
        geometry,
        material: {
          name: readText("material"),
          shore_a: null,
          density_kg_m3: null
        },
        simulation_hints: {
          boundary_condition: null,
          load_direction: null,
          target_output: "cad"
        },
        missing_information: missingInfoFor(type, geometry)
      };
    }

    function missingInfoFor(type, geometry) {
      const missing = [];
      const requiredByType = {
        bushing: ["outer_diameter_mm", "inner_diameter_mm", "height_mm"],
        plate: ["length_mm", "width_mm", "thickness_mm"],
        bracket: ["length_mm", "width_mm", "thickness_mm"]
      };
      for (const field of requiredByType[type]) {
        if (geometry[field] === null) {
          missing.push(`${field} not specified`);
        }
      }
      return missing;
    }

    function readNumber(name) {
      const value = document.getElementById(`guided_${name}`).value.trim();
      if (!value) return null;
      const number = Number(value);
      if (!Number.isFinite(number)) {
        throw new Error(`${name} must be a number`);
      }
      return number;
    }

    function readText(name) {
      const value = document.getElementById(`guided_${name}`).value.trim();
      return value || null;
    }

    // --- Chat prompt ---
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatLog = document.getElementById("chatLog");
    const chatSend = document.getElementById("chatSend");

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = chatInput.value.trim();
      if (!text) return;
      appendMsg("user", text);
      chatInput.value = "";
      chatSend.disabled = true;
      const thinking = appendMsg("bot", "Thinking…");
      try {
        const response = await fetch("/generate-cad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({prompt: text})
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "CAD parsing failed");
        }
        jsonOutput.textContent = JSON.stringify(payload.cad_intent, null, 2);
        preview.innerHTML = payload.preview_svg;
        const intent = payload.cad_intent || {};
        const part = intent.part_type || "part";
        const missing = (intent.missing_information || []).length;
        thinking.textContent = `Generated a ${part} preview. ${missing ? missing + " field(s) need clarification." : "All required fields detected."}`;
      } catch (error) {
        thinking.classList.add("err");
        thinking.textContent = error.message;
      } finally {
        chatSend.disabled = false;
        chatInput.focus();
      }
    });

    function appendMsg(role, text) {
      const el = document.createElement("div");
      el.className = `msg ${role}`;
      el.textContent = text;
      chatLog.appendChild(el);
      chatLog.scrollTop = chatLog.scrollHeight;
      return el;
    }

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
