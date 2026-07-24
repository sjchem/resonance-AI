# Resonance AI

Resonance AI is a proof-of-concept CAD and FEM assistant for vibroacoustic and
mechanical components. It turns an engineering request into validated CAD JSON,
shows an interactive 3D preview, exports CAD files, generates a Gmsh mesh, and
runs first-pass modal FEM results in the web app.

The current POC is focused on bushings, rubber mounts, brackets, plates, and
springs, with deterministic fallback paths where possible.

## What The App Does

- Chat-style CAD intent capture with Azure OpenAI or OpenAI.
- Upload context from documents, images, JSON, and CAD-like files.
- Validated structured CAD JSON using Pydantic schemas.
- Interactive browser CAD preview with downloadable STL, GLB, DXF, PNG, PDF, and JSON.
- Server-side STEP export through the CadQuery CAD pipeline.
- Gmsh mesh generation with structured/global hexahedral workflows for suitable axisymmetric parts.
- Mesh quality summary and interactive mesh preview.
- FEM modal batch runs through CalculiX with interactive von Mises contour preview and color scale.
- Directional static FEM for bushing `Kx`, `Ky`, and `Kz` in `N/mm`.
- Checkpointed design-of-experiments dataset generation with shared-connectivity shape PCA.
- Lightweight neural geometry-to-stiffness surrogate and target-driven design search.
- Azure App Service deployment through GitHub Actions and GHCR container images.

## Web UI

Run the app and open:

```text
http://localhost:8000/generate
```

The same UI is also available at:

```text
http://localhost:8000/ui
```

The main workflow is:

1. Add a CAD request in Engineering Chat, or attach a supported file.
2. Review the assistant summary and type `proceed` when ready.
3. Inspect the interactive CAD preview.
4. Open the Parametric Editor only when dimensions need live editing.
5. Generate a Gmsh mesh.
6. Run `Static K` to validate directional stiffness or `FEM batch` for modal contours.
7. Use Design Space, Target Stiffness, and the PCA Dataset dashboard for bushing studies.
8. Download CAD, mesh-friendly formats, images, PDF, or JSON.

## Repository Layout

```text
backend/app/             FastAPI app, schemas, OpenAI client, upload handling, UI
text_to_cad/             Prompt-to-CAD agent, deterministic fallback, CadQuery export
geometry/                STEP-to-mesh, mesh cleaning, mesh quality, hex/swept meshing
simulate/                Gmsh/CalculiX modal pipeline and contour visualization
models/stiffness/        Installed, reviewed stiffness-model artifacts for the web API
tests/                   Static FEM and surrogate regression tests
examples/                Example prompts
outputs/                 Local generated outputs
Dockerfile               FEM-capable Azure container image
Dockerfile.web           Lighter web image kept for quick web-only iteration
requirements-web-fem.txt Container Python stack for CAD + FEM
backend/requirements.txt Lightweight web/API requirements
main.py                  Azure/root entrypoint that imports backend/app/main.py
startup.sh               Non-container App Service startup script
```

## Configuration

Create `backend/.env` for local development. Do not commit this file.

Azure OpenAI / Azure AI Foundry:

```text
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<resource-name>.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

Important: `AZURE_OPENAI_DEPLOYMENT` must be the deployment name shown in Azure
AI Foundry or Azure OpenAI, not only the base model name.

The endpoint may also be an Azure AI Foundry project URL such as:

```text
https://<resource-name>.services.ai.azure.com/api/projects/<project-name>
```

The app normalizes supported Foundry/OpenAI endpoint formats internally.

Public OpenAI fallback:

```text
OPENAI_API_KEY=<your-openai-key>
OPENAI_MODEL=gpt-4.1-mini
```

## Run Locally

### Lightweight web/API mode

This is enough for chat, parsing, upload context, and browser-side preview.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/generate
```

### FEM-capable local mode

Use this when testing STEP export, Gmsh, CalculiX, PyVista, and FEM contours.

```bash
sudo apt-get update
sudo apt-get install -y \
  calculix-ccx \
  libgomp1 \
  libglu1-mesa \
  libgl1 \
  libxrender1 \
  libxext6 \
  libsm6 \
  libxt6 \
  xvfb

python -m pip install -r requirements-web-fem.txt
python -m uvicorn main:app --reload --port 8000
```

### Docker FEM container

This matches the production-style FEM image more closely.

```bash
docker build -f Dockerfile -t resonance-ai:fem .
docker run --rm -p 8000:8000 --env-file backend/.env resonance-ai:fem
```

Then open:

```text
http://localhost:8000/generate
```

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/models` | Show configured provider/model without secrets |
| `GET` | `/generate` | Main web UI |
| `GET` | `/ui` | Same web UI |
| `POST` | `/parse-cad` | Natural-language prompt -> validated CAD JSON |
| `POST` | `/chat-cad` | Interactive chat reply plus current CAD state |
| `POST` | `/generate-cad` | Parsed JSON plus lightweight preview payload |
| `POST` | `/preview-cad` | Preview from already-structured CAD JSON |
| `POST` | `/upload-context` | Extract context from uploaded document/image/CAD file |
| `POST` | `/export/step` | Generate a real STEP file through CadQuery |
| `POST` | `/generate-mesh` | Generate and evaluate a Gmsh volume mesh |
| `POST` | `/run-fem` | Run modal FEM batch and return contour data |
| `POST` | `/run-static-stiffness` | Run three directional static FEM load cases |
| `GET` | `/stiffness-model` | Report trained surrogate availability and metadata |
| `GET` | `/stiffness-dashboard-data` | Return shape-PCA/FEM training points |
| `POST` | `/search-stiffness` | Search design bounds with the trained surrogate |

Example prompt parse:

```bash
curl -X POST http://localhost:8000/parse-cad \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a rubber bushing with outer diameter 60 mm, inner diameter 20 mm, height 40 mm and chamfer 2 mm."
  }'
```

Check model configuration:

```bash
curl http://localhost:8000/models
```

## Azure Deployment

The active GitHub Actions workflow builds the heavy FEM container and deploys it
to Azure App Service:

```text
.github/workflows/main_ext-sjana-vibrac.yml
```

Current workflow behavior:

1. Build `Dockerfile`.
2. Push image to GitHub Container Registry:
   `ghcr.io/sjchem/resonance-ai`.
3. Point Azure Web App `ext-sjana-vibrac` at the new image.
4. Set container runtime app settings.
5. Restart the web app.

Required GitHub/Azure setup:

- GitHub Actions workflow permissions must allow package write.
- Azure login secrets must exist for the workflow.
- Azure App Service must have Azure OpenAI settings.
- For private GHCR pulls, App Service must have registry credentials.

Important App Service settings:

```text
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource-name>.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
WEBSITES_PORT=8000
PORT=8000
WEBSITES_CONTAINER_START_TIME_LIMIT=1800
GUNICORN_TIMEOUT=600
```

For private GitHub Container Registry images, also configure:

```text
DOCKER_REGISTRY_SERVER_URL=https://ghcr.io
DOCKER_REGISTRY_SERVER_USERNAME=<github-username>
DOCKER_REGISTRY_SERVER_PASSWORD=<GitHub PAT with read:packages>
```

For the container deployment, leave the Azure Startup Command empty. The
container runs `docker-entrypoint.sh`, which starts Xvfb and gunicorn.

## Text-To-CAD CLI

Deterministic Phase A generator:

```bash
python -m text_to_cad.cad_generator \
  --prompt "Create a 100 mm x 50 mm x 4 mm simple rectangular steel plate." \
  --output-dir outputs/phase_a/plate \
  --name plate
```

LLM CAD agent with deterministic fallback:

```bash
python -m text_to_cad.cad_agent \
  --prompt "Create a 120 mm x 60 mm x 5 mm bracket with four bolt holes." \
  --output-dir outputs/phase_b/bracket \
  --name bracket \
  --provider fallback
```

LLM CAD agent with Azure:

```bash
python -m text_to_cad.cad_agent \
  --prompt "Create a flanged rubber bushing with OD 60 mm, ID 20 mm, height 40 mm, flange diameter 90 mm and flange thickness 5 mm." \
  --output-dir outputs/phase_b/flanged_bushing \
  --name flanged_bushing \
  --provider azure
```

Expected generated files commonly include:

```text
agent_document.json
agent_generated_cad.py
agent_source.txt
<name>.step
<name>.stl
preview.png
prompt.txt
viewer.html
```

Open a generated standalone viewer:

```bash
python -m text_to_cad.open_viewer outputs/phase_b/bracket/viewer.html
```

On WSL, opening through Windows Explorer is often easiest:

```bash
explorer.exe "$(wslpath -w outputs/phase_b/bracket/viewer.html)"
```

## Meshing And FEM CLI

Run the full modal pipeline from a STEP file:

```bash
python -m simulate.pipeline outputs/phase_b/bracket/bracket.step \
  --output-dir outputs/simulation/bracket \
  --modes 8 \
  --boundary fixed_bottom \
  --material rubber
```

Run stages individually:

```bash
# STEP -> Gmsh volume mesh
python -m geometry.step_to_mesh outputs/phase_b/bracket/bracket.step outputs/simulation/bracket/bracket.msh

# Clean mesh
python -m geometry.mesh_cleaner outputs/simulation/bracket/bracket.msh outputs/simulation/bracket/bracket_clean.vtk

# Check quality
python -m geometry.mesh_quality outputs/simulation/bracket/bracket_clean.vtk

# Run CalculiX modal solve
python -m simulate.modal_solver outputs/simulation/bracket/bracket_clean.vtk --material rubber --modes 8

# Render contour image from FRD
python -m simulate.visualize outputs/simulation/bracket/bracket.frd \
  --field mises \
  --mode 1 \
  --warp \
  --output outputs/simulation/bracket/bracket_mode1_mises.png
```

Notes:

- Units are millimetres in CAD.
- The solver uses a tonne-mm-s unit system.
- Natural frequencies are reported in Hz.
- Rubber materials are treated as linear-elastic first-pass approximations.
- Modal stress contours show useful spatial patterns; absolute stress magnitude is relative for eigenmodes.

## Static Stiffness And Training Dataset

The static bushing workflow uses these explicit POC assumptions:

- Client `X` is the bushing centerline.
- The generated mesh centerline is geometric `Z`, so mesh `Z -> Kx`.
- Mesh `X/Y -> Ky/Kz`.
- The outer-core interface is fixed.
- The inner-core interface is translated by `1 mm`.
- Stiffness is the summed interface reaction divided by displacement, in `N/mm`.
- Rubber is currently linear-elastic and isotropic; production use requires material and test calibration.

Run one structured-mesh stiffness validation:

```bash
python -m simulate.static_stiffness \
  outputs/stiffness_dataset/meshes/rb-0001.vtk \
  --material rubber \
  --inner-length 40 \
  --outer-length 40 \
  --output-dir outputs/static_stiffness/rb-0001
```

Build the complete offline design dataset, shape PCA, and neural surrogate:

```bash
python -m simulate.stiffness_dataset \
  --output-dir outputs/stiffness_dataset \
  --samples 200 \
  --material rubber \
  --circumferential 48 \
  --radial 4 \
  --axial 8 \
  --shape-components 6
```

Each design runs three CalculiX solves. The command checkpoints
`stiffness_dataset.json` after every case, writes a CSV summary, fits shape PCA,
and saves validation metrics beside `stiffness_model.npz`.

Review the validation `MAE`, `MAPE`, and `R²` before installing the model. A
successful training run does not by itself establish engineering accuracy.

Install reviewed artifacts for the web application:

```bash
cp outputs/stiffness_dataset/stiffness_model.npz models/stiffness/
cp outputs/stiffness_dataset/stiffness_dataset.json models/stiffness/
```

The web Target Stiffness search uses the installed neural model. When no model
is installed, it clearly reports the analytical screening fallback. The PCA
Dataset dashboard plots the first three geometry shape codes, with solved
training designs in green and target-near designs in red.

Use a mounted artifact location in Azure by setting:

```text
STIFFNESS_MODEL_DIR=/path/to/reviewed/model/artifacts
```

Run focused verification:

```bash
python -m unittest discover -s tests -v
```

## MCP Server

The local MCP server exposes Resonance CAD tools over stdio:

```text
create_resonance_cad_document
inspect_resonance_cad
export_resonance_cad
```

Run it:

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

## Troubleshooting

### Azure OpenAI 404 Resource not found

Check these values in Azure App Service environment variables:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION
```

The deployment value must match the deployment name in Azure AI Foundry or Azure
OpenAI. The endpoint should normally look like:

```text
https://<resource-name>.cognitiveservices.azure.com/
```

### FEM error: `libgomp.so.1` missing

The FEM/Gmsh stack needs OpenMP runtime support. The production `Dockerfile`
installs:

```text
libgomp1
```

Rebuild and redeploy the FEM container after Dockerfile changes.

### Container timeout on Azure

Useful settings:

```text
WEBSITES_CONTAINER_START_TIME_LIMIT=1800
WEBSITES_PORT=8000
PORT=8000
GUNICORN_TIMEOUT=600
```

For private GHCR images, verify the App Service registry credentials and that
the PAT has `read:packages`.

### Do not commit secrets

Keep `backend/.env` local. Azure secrets belong in App Service environment
variables or GitHub Actions secrets.
