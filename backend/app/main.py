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


app = FastAPI(title="Local CAD Prompt Parser", version="0.1.0")


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
  <title>Local CAD Prompt Parser</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #182231;
      --muted: #66758a;
      --line: #dbe4ee;
      --accent: #0f766e;
      --accent-dark: #0b5f59;
      --danger: #b42318;
      --cad: #13877e;
      --cad-dark: #0b4f49;
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
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 28px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    main {
      width: min(1240px, calc(100% - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 420px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }
    form, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
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
      border-radius: 6px;
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
      border-radius: 6px;
      background: var(--accent);
      color: white;
      padding: 12px 16px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.62; cursor: wait; }
    .stack {
      display: grid;
      gap: 18px;
    }
    .preview {
      min-height: 420px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    svg {
      width: 100%;
      max-height: 560px;
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
    .error {
      color: var(--danger);
      background: #fff7f6;
      border-color: #fecdca;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Local CAD Prompt Parser</h1>
  </header>
  <main>
    <form id="cadForm">
      <label for="prompt">CAD prompt</label>
      <textarea id="prompt" name="prompt">Create a rubber bushing with outer diameter 60 mm, inner diameter 20 mm, height 40 mm and chamfer 2 mm.</textarea>
      <button id="submitButton" type="submit">Parse and Preview</button>
      <p class="muted">Uses Azure OpenAI or OpenAI structured parsing through /generate-cad.</p>

      <hr class="divider">

      <label for="partType">Step 1: Part type</label>
      <select id="partType">
        <option value="bushing">Bushing</option>
        <option value="plate">Plate</option>
        <option value="bracket">Bracket</option>
      </select>

      <div id="guidedFields"></div>
      <button id="guidedButton" type="button">Build JSON and Preview</button>
      <p class="muted">Guided mode does not call OpenAI. It builds validated JSON from your step-by-step inputs.</p>
    </form>

    <section class="stack">
      <div class="panel preview" id="preview">
        <p class="muted">Parsed CAD preview will appear here.</p>
      </div>
      <div class="panel">
        <label>Validated JSON</label>
        <pre id="jsonOutput">{}</pre>
      </div>
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
        preview.innerHTML = `<div class="panel error"><pre>${escapeHtml(error.message)}</pre></div>`;
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
        preview.innerHTML = `<div class="panel error"><pre>${escapeHtml(error.message)}</pre></div>`;
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
