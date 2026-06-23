# VC.ResonanceAI

POC for a text-to-CAD plus simulation-surrogate workflow.

## OpenAI CAD Prompt Parser

This POC backend uses Azure OpenAI or the public OpenAI API with Structured
Outputs. It converts a natural-language CAD prompt into validated structured
JSON and returns an interactive preview in the web app.

For Azure deployment, configure:

```text
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_ENDPOINT=https://<resource-name>.services.ai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
```

The app also accepts an Azure AI Foundry project URL such as
`https://<resource-name>.services.ai.azure.com/api/projects/<project-name>`.
It normalizes that to the OpenAI-compatible inference base automatically.

For the public OpenAI API instead, configure:

```text
OPENAI_API_KEY=<your-openai-api-key>
OPENAI_MODEL=gpt-4.1-mini
```

Start the FastAPI backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/
```

Check model configuration:

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

Generate parsed JSON plus CAD preview SVG:

```bash
curl -X POST http://localhost:8000/generate-cad \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a rubber bushing with outer diameter 60 mm, inner diameter 20 mm, height 40 mm and chamfer 2 mm."
  }'
```

Open the UI locally only if you need to test the same flow as the deployed POC:

```text
http://localhost:8000/ui
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

## Phase C: Simulation (STEP to natural frequencies)

Phase C runs a local NVH modal analysis on any generated STEP file using the
open-source toolchain: **Gmsh** for meshing and **CalculiX** for solving. The
full workflow is:

```text
STEP -> tetra mesh (Gmsh) -> clean -> quality check -> CalculiX modal -> frequencies (Hz)
```

### System dependencies

The mesher and solver are native tools, not pip packages:

```bash
sudo apt-get install -y libglu1-mesa calculix-ccx
pip install -r requirements.txt
```

`libglu1-mesa` is required by the Gmsh Python binding; `calculix-ccx` provides
the `ccx` solver.

### One command: full pipeline

```bash
python -m simulate.pipeline outputs/phase_a/bracket/bracket.step \
  --output-dir outputs/simulation/bracket \
  --modes 8 \
  --boundary fixed_bottom
```

The material is auto-detected from a sibling `spec.json` (`material_hint`) when
present, or set it explicitly with `--material steel`. Boundary presets are
`free`, `fixed_bottom`, `fixed_top`, and `encastre`.

Expected output:

```text
[1/5] Meshing bracket.step ...
[2/5] Cleaning mesh ...
[3/5] Checking mesh quality ...
[4/5] Solving modal analysis with CalculiX (steel) ...
[5/5] Extracting natural frequencies ...

Natural frequencies from bracket.dat:
  Mode  1:   3155.654 Hz
  ...
Fundamental flexible frequency: 3155.654 Hz
```

Generated files in the output directory:

```text
outputs/simulation/bracket/
├── bracket.msh          # Gmsh volume mesh
├── bracket_clean.vtk    # cleaned mesh
├── bracket.inp          # CalculiX input deck
├── bracket.dat          # solver eigenvalue output
├── bracket.frd          # mode shapes (open in ParaView / CalculiX cgx)
└── bracket_modal.json   # parsed natural frequencies
```

### Individual stages

Each stage is also runnable on its own:

```bash
# 1. STEP -> volume mesh
python -m geometry.step_to_mesh outputs/phase_a/bracket/bracket.step outputs/simulation/bracket/bracket.msh

# 2. Clean the mesh
python -m geometry.mesh_cleaner outputs/simulation/bracket/bracket.msh outputs/simulation/bracket/bracket_clean.vtk

# 3. Check mesh quality
python -m geometry.mesh_quality outputs/simulation/bracket/bracket_clean.vtk

# 4. Run the modal solver
python -m simulate.modal_solver outputs/simulation/bracket/bracket_clean.vtk --material steel --modes 8

# 5. Parse natural frequencies
python -m simulate.results outputs/simulation/bracket/modal.dat --json report.json
```

### Contour images (von Mises stress / displacement)

The pipeline also renders a coloured contour PNG of the first mode (deformed
mode shape coloured by von Mises stress) using PyVista — no ParaView/cgx GUI
needed. It is written next to the other results, e.g.
`bracket_mode1_mises.png`.

Disable it with `--no-contour`, or pick a different mode with
`--contour-mode 2`.

Render a contour from any existing `.frd` directly:

```bash
# Von Mises stress on the deformed mode shape
python -m simulate.visualize outputs/simulation/bracket/bracket.frd \
  --field mises --mode 1 --warp \
  --output outputs/simulation/bracket/bracket_mode1_mises.png

# Displacement magnitude
python -m simulate.visualize outputs/simulation/bracket/bracket.frd \
  --field disp --mode 2 --warp
```

Options: `--field {mises,disp}`, `--mode N`, `--warp` (deform by the mode
shape), `--warp-scale`, `--cmap` (default `jet`), and `--colors` (number of
discrete contour bands). Modal stress is an eigenvector quantity, so the
contour *pattern* is meaningful while the absolute magnitude is relative.

### Visualize mode shapes

Open the `.frd` file in [ParaView](https://www.paraview.org/) or CalculiX `cgx`:

```bash
cgx -o outputs/simulation/bracket/bracket.frd
```

> Units: geometry is in millimetres, so the solver uses a tonne-mm-s unit
> system and natural frequencies are reported directly in Hz. Rubber/EPDM are
> modelled as soft linear-elastic solids for a first-pass estimate.

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

This repo includes a lightweight FastAPI web UI with a prompt box, validated
OpenAI CAD JSON, and an SVG CAD-style preview. The Azure deployment packages
only [main.py](main.py), [startup.sh](startup.sh), and the `backend/` app so App
Service does not need CadQuery or native OpenCascade libraries.

Set these App Service application settings:

```text
AZURE_OPENAI_API_KEY=<your-azure-openai-key>
AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

Use this startup command:

```bash
bash startup.sh
```

Run locally:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000/generate
```

The same UI is also available at `/ui`. The root `/` remains a JSON health
check endpoint.

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
AZURE_OPENAI_API_KEY: <key>
AZURE_OPENAI_ENDPOINT: https://<resource-name>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT: <deployment-name>
AZURE_OPENAI_API_VERSION: 2024-10-21
```

GitHub Actions deployment uses a publish profile. Add this repository secret:

```text
AZURE_WEBAPP_PUBLISH_PROFILE
```

Paste the publish profile XML downloaded from the Azure Web App Deployment
Center or Overview page.
