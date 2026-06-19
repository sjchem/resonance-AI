"""FastAPI web front end for ResonanceAI CAD generation."""

from __future__ import annotations

import html
import os
from pathlib import Path
import re
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


APP_TITLE = "ResonanceAI CAD Generator"
OUTPUT_ROOT = Path(os.getenv("RESONANCE_OUTPUT_DIR", "outputs")).resolve()
WEB_OUTPUT_ROOT = OUTPUT_ROOT / "web"
WEB_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=APP_TITLE)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page()


@app.post("/generate", response_class=HTMLResponse)
def generate_page(
    prompt: str = Form(...),
    name: str = Form(""),
    provider: str = Form("auto"),
) -> str:
    result = _generate(prompt=prompt, name=name, provider=provider)
    if not result["ok"]:
        return _page(prompt=prompt, name=name, provider=provider, error=result["error"])
    return _page(prompt=prompt, name=name, provider=provider, result=result)


@app.post("/api/generate")
async def generate_api(request: Request) -> JSONResponse:
    payload: dict[str, Any] = await request.json()
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    name = str(payload.get("name", "")).strip()
    provider = str(payload.get("provider", "auto"))
    result = _generate(prompt=prompt, name=name, provider=provider)
    status_code = 200 if result["ok"] else 500
    return JSONResponse(result, status_code=status_code)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _generate(prompt: str, name: str, provider: str) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        return {"ok": False, "error": "Prompt is required."}

    output_name = _safe_name(name) if name.strip() else _slug_from_prompt(prompt)
    output_dir = WEB_OUTPUT_ROOT / output_name
    try:
        from text_to_cad.cad_agent import generate_with_agent

        code = generate_with_agent(
            prompt=prompt,
            output_dir=output_dir,
            output_name=output_name,
            provider=provider if provider in {"auto", "azure", "fallback"} else "auto",
            execute=True,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if code != 0:
        error_log = output_dir / "cad_error_attempt_1.txt"
        error = error_log.read_text(encoding="utf-8") if error_log.exists() else "CAD generation failed."
        return {"ok": False, "error": error}

    base = f"/outputs/web/{output_name}"
    return {
        "ok": True,
        "name": output_name,
        "prompt": prompt,
        "provider": provider,
        "viewer_url": f"{base}/viewer.html",
        "preview_url": f"{base}/preview.png",
        "step_url": f"{base}/{output_name}.step",
        "stl_url": f"{base}/{output_name}.stl",
        "document_url": f"{base}/agent_document.json",
    }


def _page(
    prompt: str = "",
    name: str = "",
    provider: str = "auto",
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    escaped_prompt = html.escape(prompt)
    escaped_name = html.escape(name)
    provider_options = "".join(
        f'<option value="{value}"{" selected" if provider == value else ""}>{label}</option>'
        for value, label in [
            ("auto", "Auto: Azure when configured, fallback otherwise"),
            ("azure", "Azure OpenAI"),
            ("fallback", "Deterministic fallback"),
        ]
    )
    result_html = _result_html(result) if result else ""
    error_html = f'<section class="message error"><pre>{html.escape(error)}</pre></section>' if error else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #16202c;
      --muted: #66758a;
      --line: #dbe4ee;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 18px 28px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: 420px minmax(0, 1fr);
      gap: 22px;
      align-items: start;
    }}
    form, .output, .message {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }}
    label {{
      display: block;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }}
    textarea, input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 11px 12px;
      color: var(--ink);
      font: inherit;
      background: #fff;
    }}
    textarea {{
      min-height: 172px;
      resize: vertical;
      line-height: 1.45;
    }}
    .field {{ margin-bottom: 16px; }}
    button {{
      width: 100%;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 12px 16px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .output h2 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    .viewer {{
      width: 100%;
      height: 560px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f9fbfd;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .links a {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--accent-strong);
      text-decoration: none;
      font-weight: 700;
      font-size: 14px;
    }}
    .links a:hover {{ border-color: var(--accent); }}
    .message.error {{
      grid-column: 1 / -1;
      border-color: #fecdca;
      color: var(--danger);
      background: #fff7f6;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      .viewer {{ height: 62vh; }}
    }}
  </style>
</head>
<body>
  <header><h1>{APP_TITLE}</h1></header>
  <main>
    <form action="/generate" method="post">
      <div class="field">
        <label for="prompt">CAD prompt</label>
        <textarea id="prompt" name="prompt" required>{escaped_prompt}</textarea>
      </div>
      <div class="field">
        <label for="name">Output name</label>
        <input id="name" name="name" value="{escaped_name}" placeholder="optional">
      </div>
      <div class="field">
        <label for="provider">Generation mode</label>
        <select id="provider" name="provider">{provider_options}</select>
      </div>
      <button type="submit">Generate CAD</button>
    </form>
    <div>
      {error_html}
      {result_html}
    </div>
  </main>
</body>
</html>
"""


def _result_html(result: dict[str, Any]) -> str:
    viewer = html.escape(result["viewer_url"])
    name = html.escape(result["name"])
    return f"""<section class="output">
  <h2>{name}</h2>
  <iframe class="viewer" src="{viewer}" title="Generated CAD viewer"></iframe>
  <div class="links">
    <a href="{viewer}" target="_blank" rel="noreferrer">Open 3D viewer</a>
    <a href="{html.escape(result["step_url"])}">Download STEP</a>
    <a href="{html.escape(result["stl_url"])}">Download STL</a>
    <a href="{html.escape(result["preview_url"])}" target="_blank" rel="noreferrer">Preview PNG</a>
    <a href="{html.escape(result["document_url"])}" target="_blank" rel="noreferrer">CAD JSON</a>
  </div>
</section>"""


def _slug_from_prompt(prompt: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", prompt.lower())[:5]
    return _safe_name("_".join(words) if words else "generated_part")


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower())
    value = re.sub(r"_+", "_", value).strip("_-")
    return value[:64] or "generated_part"
