# VC.ResonanceAI

Local prototype for a text-to-CAD plus simulation-surrogate workflow.

## Local Ollama CAD Prompt Parser

This POC backend uses local Ollama only. It does not call Azure OpenAI, OpenAI
API, or any remote runtime LLM. The endpoint converts a natural-language CAD
prompt into validated structured JSON.

Start Ollama in one WSL 2 terminal:

```bash
ollama serve
```

Start the FastAPI backend in a second terminal:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/
```

List local Ollama models:

```bash
curl http://localhost:8000/models
```

Parse a CAD prompt:

```bash
curl -X POST http://localhost:8000/parse-cad \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a rubber bushing with outer diameter 60 mm, inner diameter 20 mm, height 40 mm and chamfer 2 mm."
  }'
```

The backend expects this local model to be available:

```bash
ollama pull qwen2.5-coder:7b
```

## Phase A: Text to CAD

Phase A converts a short engineering prompt into a deterministic CadQuery script,
a STEP file, an STL file, a quick 2D preview, and a standalone live 3D CAD
viewer.

Run the bracket example:

```bash
python -m text_to_cad.cad_generator \
  --prompt-file examples/bracket_prompt.txt \
  --output-dir outputs/phase_a/bracket \
  --name bracket
```

Generate files without executing CadQuery:

```bash
python -m text_to_cad.cad_generator \
  --prompt "Create a 100 mm x 50 mm x 4 mm simple rectangular steel plate." \
  --output-dir outputs/phase_a/plate \
  --name plate \
  --dry-run
```

Expected files:

```text
outputs/phase_a/bracket/
├── bracket.step
├── bracket.stl
├── generated_cad.py
├── preview.png
├── prompt.txt
├── spec.json
└── viewer.html
```

Open the live 3D viewer:

```bash
python -m text_to_cad.open_viewer outputs/phase_a/bracket/viewer.html
```

On WSL you can also open it directly through Windows Explorer:

```bash
explorer.exe "$(wslpath -w outputs/phase_a/bracket/viewer.html)"
```

The viewer is self-contained and includes the STL geometry, CAD dimensions, hole
details, mesh triangle count, and generated file names.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Phase B: LLM CAD Agent

Phase B adds a structured CAD-agent path. The agent asks Azure OpenAI for a
validated CAD document JSON, exports it locally with CadQuery, and still falls
back to the deterministic Phase A parser when Azure is not configured.

Run with deterministic fallback:

```bash
python -m text_to_cad.cad_agent \
  --prompt "Create a 120 mm x 60 mm x 5 mm bracket with four bolt holes." \
  --output-dir outputs/phase_b/bracket \
  --name bracket \
  --provider fallback
```

Run with Azure OpenAI:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource-name>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<key>"
export AZURE_OPENAI_DEPLOYMENT="<deployment-name>"
export AZURE_OPENAI_API_VERSION="2024-10-21"

python -m text_to_cad.cad_agent \
  --prompt "Create a 120 mm x 60 mm x 5 mm bracket with four bolt holes and rounded outer edges." \
  --output-dir outputs/phase_b/rounded_bracket \
  --name rounded_bracket \
  --provider azure
```

Run with local Ollama and Qwen2.5-Coder:

```bash
ollama pull qwen2.5-coder:7b
ollama serve

export OLLAMA_MODEL="qwen2.5-coder:7b"

python -m text_to_cad.cad_agent \
  --prompt "Create a 30 mm diameter compression spring, 4 mm wire diameter, 70 mm height, and 8 turns." \
  --output-dir outputs/phase_b/ollama_spring \
  --name ollama_spring \
  --provider ollama
```

Smaller laptop-friendly model options:

```bash
ollama pull qwen2.5-coder:3b
export OLLAMA_MODEL="qwen2.5-coder:3b"
```

Expected Phase B files:

```text
outputs/phase_b/bracket/
├── agent_document.json
├── agent_generated_cad.py
├── agent_source.txt
├── bracket.step
├── bracket.stl
├── preview.png
├── prompt.txt
└── viewer.html
```

Open the Phase B viewer:

```bash
python -m text_to_cad.open_viewer outputs/phase_b/bracket/viewer.html
```

## Resonance CAD MCP Server

The local MCP server exposes renamed Resonance CAD tools over stdio:

```text
create_resonance_cad_document
inspect_resonance_cad
export_resonance_cad
```

MCP server command:

```bash
python -m text_to_cad.mcp_server
```

Example MCP config:

```json
{
  "mcpServers": {
    "resonance-cad": {
      "command": "python",
      "args": ["-m", "text_to_cad.mcp_server"],
      "cwd": "/home/santanujana/code/vibracoustic/vc.resonanceAI"
    }
  }
}
```

## Azure Web App

This repo includes a FastAPI web UI with a prompt box and generated CAD output.
It is Azure App Service-ready through [main.py](main.py), [startup.sh](startup.sh),
and [.github/workflows/azure-webapp.yml](.github/workflows/azure-webapp.yml).

Run locally:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000
```

The configured Azure Web App name in the workflow is:

```text
ext-sjana-vibrac
```

The public URL will be:

```text
https://ext-sjana-vibrac.azurewebsites.net
```

Azure App Service settings to configure:

```text
Runtime stack: Python 3.12 or newer on Linux
Startup command: bash startup.sh
SCM_DO_BUILD_DURING_DEPLOYMENT: true
AZURE_OPENAI_ENDPOINT: https://<resource-name>.openai.azure.com
AZURE_OPENAI_API_KEY: <key>
AZURE_OPENAI_DEPLOYMENT: <deployment-name>
AZURE_OPENAI_API_VERSION: 2024-10-21
OLLAMA_HOST: http://localhost:11434
OLLAMA_MODEL: qwen2.5-coder:7b
```

GitHub Actions deployment uses a publish profile. Add this repository secret:

```text
AZURE_WEBAPP_PUBLISH_PROFILE
```

Paste the publish profile XML downloaded from the Azure Web App Deployment
Center or Overview page.
