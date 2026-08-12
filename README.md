# Resonance AI Product Design Studio

Resonance AI is a proof-of-concept engineering workspace for generating,
optimizing, meshing, simulating, and reviewing Vibracoustic components.

The application combines conversational CAD intent capture, uploaded engineering
context, browser-based 3D visualization, CadQuery/OpenSCAD export, Gmsh meshing,
CalculiX finite-element analysis, and bushing stiffness optimization in one
FastAPI web application.

> **POC status:** Results are intended for engineering exploration and workflow
> demonstration. They are not production design releases. Material calibration,
> boundary conditions, mesh convergence, solver settings, and generated geometry
> must be reviewed by a qualified engineer.

## Current Capabilities

### Product Design Studio

- Product families for **Bushing**, **Mounts**, and **Anonymous** components.
- Bushing starters for Four Arm Bushing, OptiBush, BYD Bush, Rubber bushing,
  and Bonded Bushing.
- Mount starters for engine, hydraulic, and strut mounts.
- Interactive Engineering Chat backed by Azure OpenAI or the public OpenAI API.
- Structured CAD intent validated with Pydantic before it reaches the CAD path.
- Optional KISS Agent and FAIR Explorer source selectors for demonstrating
  source-aware answers.

### Upload And CAD

- Upload PDF, image, JSON, text, STEP/STP, IGES/IGS, STL, OBJ, DXF, SCAD, or
  FreeCAD files up to 8 MB.
- Extract text and metadata from supported files and add it to the chat context.
- Persist STEP/STP and STL uploads for exact mesh and FEM workflows.
- Generate an interactive Three.js CAD preview with orbit, zoom, a view cube,
  floor lighting, and soft shadows.
- Edit supported dimensions through the collapsible CAD Editor.
- Generate CadQuery or OpenSCAD geometry and export STEP/STL where supported.

### Mesh And Simulation

- Generate Gmsh volume meshes from supported generated or uploaded geometry.
- Use structured/global hexahedral bushing templates for compatible parametric
  bushing geometry.
- Use body-fitted uploaded-geometry meshing when the STEP/STL is a valid closed
  solid; invalid or open files remain preview-only.
- Inspect mesh statistics, element edges, quality information, and interactive
  3D mesh views.
- Run **Static K** for directional `Kx`, `Ky`, and `Kz` stiffness in `N/mm`.
- Run a **FEM batch** for up to 100 CalculiX modes and inspect the selected
  interactive von Mises contour.
- Export the best CAD result as STL and simulation results as a multi-sheet
  Excel workbook.

### Design Exploration

- Explore bushing design ranges for inner diameter and inner/outer core lengths.
- Search target stiffness values with an installed geometry-to-stiffness
  surrogate model.
- Generate checkpointed FEM training datasets.
- Fit shared-connectivity Shape PCA and inspect PCA dashboards.
- Train and validate a lightweight neural stiffness surrogate.

## POC Placeholders

The following controls are visible to demonstrate the intended integration
architecture, but they do not call live external systems yet:

- KISS Agent knowledge retrieval.
- FAIR Explorer knowledge retrieval.
- Search from FAIR Engineering Data.
- FAIR Publisher for CAD output.
- Synera Run for simulation workbooks.

The application labels these actions as placeholders and does not claim that a
record, dataset, or workflow was retrieved or published.

Image uploads currently provide filename and image-size context only. Visual
feature extraction/OCR is not enabled, so dimensions must be supplied in chat
or through structured data.

## Web Workflow

1. Open Product Design Studio at `/generate`.
2. Select a product family and starter component, or write a request directly
   in Engineering Chat.
3. Optionally attach an engineering document, image, JSON file, or CAD model.
4. Review and refine the generated structured CAD intent.
5. Inspect the interactive 3D model and use the CAD Editor when needed.
6. Generate and inspect the Gmsh mesh.
7. Run **Static K** for directional stiffness or **FEM batch** for modal results.
8. Download the best CAD as STL or the complete simulation report as Excel.

## Repository Layout

```text
backend/app/                  FastAPI API, chat, schemas, uploads, exports, and UI
backend/app/static/           Three.js CAD viewer and vendored browser modules
cad_backends/                 OpenSCAD generation and export
text_to_cad/                  Phase A/B CAD CLIs, CadQuery renderer, and MCP server
geometry/                     Uploaded-volume, Gmsh, quality, and hex mesh pipelines
simulate/                     CalculiX modal/static FEM, PCA, datasets, and surrogate
models/stiffness/             Reviewed stiffness artifacts loaded by the web app
tests/                        Unit and solver smoke tests
examples/                     Example natural-language prompts
outputs/                      Local generated artifacts; not application source
main.py                       Root ASGI entrypoint
Dockerfile                    Active FEM-capable production container
Dockerfile.web                Alternate web/FEM image for local iteration
docker-entrypoint.sh          Xvfb and gunicorn container startup
startup.sh                    Non-container Azure App Service startup
requirements-web-fem.txt      Production Python dependency set
backend/requirements.txt      Lightweight web/API dependency set
```

## Configuration

Create `backend/.env` for local development. This file is ignored by Git and
must never be committed.

### Azure OpenAI

```dotenv
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<resource-name>.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

`AZURE_OPENAI_DEPLOYMENT` must exactly match the deployment name displayed in
Azure AI Foundry/Azure OpenAI. It is not necessarily the base model name.

The client also accepts supported Azure AI Foundry project endpoints:

```text
https://<resource-name>.services.ai.azure.com/api/projects/<project-name>
```

### Public OpenAI Fallback

```dotenv
OPENAI_API_KEY=<your-key>
OPENAI_MODEL=gpt-4.1-mini
```

Azure configuration takes precedence when both providers are configured.

### Optional Runtime Paths

```dotenv
STIFFNESS_MODEL_DIR=/path/to/reviewed/stiffness/artifacts
RESONANCE_UPLOAD_DIR=/path/to/persistent/upload/storage
```

`RESONANCE_UPLOAD_DIR` should point to persistent writable storage in a
production deployment if uploaded geometry must survive container replacement.

## Run Locally

### 1. Lightweight Web/API Mode

Use this for chat, structured JSON, uploads, browser CAD preview, and report
export development:

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

Native CadQuery, Gmsh, and CalculiX operations require the full environment.

### 2. Full CAD/FEM Mode On WSL/Linux

```bash
sudo apt-get update
sudo apt-get install -y \
  calculix-ccx \
  libgomp1 \
  libglu1-mesa \
  libgl1 \
  libgl1-mesa-dri \
  libxrender1 \
  libxext6 \
  libsm6 \
  libxt6 \
  xvfb

python -m pip install -r requirements-web-fem.txt
python -m uvicorn main:app --reload --port 8000
```

### 3. Production-Style Docker Mode

```bash
docker build -f Dockerfile -t resonance-ai:fem .
docker run --rm \
  -p 8000:8000 \
  --env-file backend/.env \
  -v resonance-uploads:/home/resonance-ai/uploads \
  resonance-ai:fem
```

Open `http://localhost:8000/generate`.

The container starts Xvfb for headless VTK/PyVista rendering and serves the app
with gunicorn on port `8000`.

## API

FastAPI documentation is available locally at:

```text
http://localhost:8000/docs
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/models` | Active provider/deployment status without secrets |
| `GET` | `/generate` | Product Design Studio UI |
| `GET` | `/ui` | UI compatibility route |
| `POST` | `/parse-cad` | Natural-language prompt to validated CAD JSON |
| `POST` | `/chat-cad` | Conversational response plus current CAD intent |
| `POST` | `/generate-cad` | Parse prompt and return lightweight preview data |
| `POST` | `/preview-cad` | Preview an existing structured CAD intent |
| `POST` | `/generate-parametric-cad` | Generate a parametric CAD result |
| `POST` | `/upload-context` | Extract prompt context and persist supported geometry |
| `POST` | `/export/step` | Export CadQuery geometry as STEP |
| `POST` | `/export/openscad` | Generate and export OpenSCAD bushing artifacts |
| `POST` | `/export/simulation-report` | Export simulation/design results as XLSX |
| `POST` | `/generate-mesh` | Generate and evaluate a Gmsh volume mesh |
| `POST` | `/shape-pca` | Encode/reconstruct compatible bushing mesh geometry |
| `POST` | `/run-static-stiffness` | Run three directional CalculiX static cases |
| `POST` | `/run-fem` | Run modal FEM batch and return selected contour data |
| `GET` | `/stiffness-model` | Report installed surrogate status and metadata |
| `GET` | `/stiffness-dashboard-data` | Return reviewed training/PCA dashboard data |
| `POST` | `/search-stiffness` | Search design bounds using the installed surrogate |

### Parse Example

```bash
curl -X POST http://localhost:8000/parse-cad \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a rubber bushing with outer diameter 60 mm, inner diameter 20 mm, height 40 mm and chamfer 2 mm."
  }'
```

Check provider configuration:

```bash
curl http://localhost:8000/models
```

## CAD Command-Line Workflows

### Phase A: Deterministic CAD

```bash
python -m text_to_cad.cad_generator \
  --prompt "Create a 100 mm x 50 mm x 4 mm simple rectangular steel plate." \
  --output-dir outputs/phase_a/plate \
  --name plate
```

### Phase B: Structured CAD Agent

Use Azure when configured, otherwise deterministic fallback:

```bash
python -m text_to_cad.cad_agent \
  --prompt "Create a 120 mm x 60 mm x 5 mm bracket with four bolt holes." \
  --output-dir outputs/phase_b/bracket \
  --name bracket \
  --provider auto
```

Force deterministic fallback:

```bash
python -m text_to_cad.cad_agent \
  --prompt "Create a rubber bushing with OD 60 mm, ID 20 mm and height 40 mm." \
  --output-dir outputs/phase_b/bushing \
  --name bushing \
  --provider fallback
```

Common generated artifacts:

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

Open a generated viewer from WSL:

```bash
explorer.exe "$(wslpath -w outputs/phase_b/bracket/viewer.html)"
```

## Meshing And FEM CLI

Run the modal pipeline from a STEP file:

```bash
python -m simulate.pipeline outputs/phase_b/bracket/bracket.step \
  --output-dir outputs/simulation/bracket \
  --modes 8 \
  --boundary fixed_bottom \
  --material rubber
```

Run individual stages:

```bash
python -m geometry.step_to_mesh \
  outputs/phase_b/bracket/bracket.step \
  outputs/simulation/bracket/bracket.msh

python -m geometry.mesh_cleaner \
  outputs/simulation/bracket/bracket.msh \
  outputs/simulation/bracket/bracket_clean.vtk

python -m geometry.mesh_quality \
  outputs/simulation/bracket/bracket_clean.vtk

python -m simulate.modal_solver \
  outputs/simulation/bracket/bracket_clean.vtk \
  --material rubber \
  --modes 8
```

CAD dimensions use millimetres. The solver uses a tonne-mm-s unit system and
reports natural frequencies in Hz. Modal stress magnitude is relative for
eigenmodes; the contour is primarily useful for comparing spatial patterns.

## Static Stiffness Assumptions

The current structured-bushing Static K POC uses explicit assumptions:

- Client `X` is the bushing centerline.
- Mesh `Z` maps to client `Kx`; mesh `X/Y` map to `Ky/Kz`.
- The outer-core interface is fixed.
- The inner-core interface receives a prescribed `1 mm` translation.
- Stiffness is summed interface reaction divided by displacement in `N/mm`.
- Directional rubber uses a client-calibrated effective modulus of `1.10 MPa`.

Run one structured-mesh validation:

```bash
python -m simulate.static_stiffness \
  outputs/stiffness_dataset/meshes/rb-0001.vtk \
  --material rubber \
  --inner-length 40 \
  --outer-length 40 \
  --output-dir outputs/static_stiffness/rb-0001
```

The calibration aligns this POC with the supplied stiffness scale. Production
release still requires measured material curves, fixture validation, mesh
convergence, and suitable nearly-incompressible or hyperelastic elements.

## Stiffness Dataset, Shape PCA, And Surrogate

Build the offline dataset and model:

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

Each successful design runs three CalculiX static cases. The process
checkpoints after every case and writes:

```text
stiffness_dataset.json
stiffness_dataset.csv
stiffness_model.npz
stiffness_model_metrics.json
shape_pca_model.npz
shape_pca_summary.json
```

Review MAE, MAPE, R-squared, the sampled bounds, failed cases, mesh template,
and calibration metadata before installing artifacts:

```bash
cp outputs/stiffness_dataset/stiffness_model.npz models/stiffness/
cp outputs/stiffness_dataset/stiffness_dataset.json models/stiffness/
```

The web target search uses installed reviewed artifacts. Generated training
data is not automatically trusted as production engineering data.

## MCP Server

The local stdio MCP server exposes:

```text
create_resonance_cad_document
inspect_resonance_cad
export_resonance_cad
```

Run it with:

```bash
python -m text_to_cad.mcp_server
```

Example configuration:

```json
{
  "mcpServers": {
    "resonance-cad": {
      "command": "python",
      "args": ["-m", "text_to_cad.mcp_server"],
      "cwd": "/path/to/resonance-AI"
    }
  }
}
```

## Tests

Run the repository test suite:

```bash
PYTHONPATH=.:backend python -m unittest discover -s tests -v
```

The CalculiX smoke test skips automatically when `ccx` is unavailable.

## Azure App Service Deployment

The workflow `.github/workflows/main_ext-sjana-vibrac.yml` runs on pushes to
`main` and manual dispatches. It:

1. Builds the active `Dockerfile`.
2. Pushes `latest` and commit-SHA images to `ghcr.io/sjchem/resonance-ai`.
3. Authenticates to Azure with GitHub OIDC secrets.
4. Points Web App `ext-sjana-vibrac` to the new image.
5. Clears any App Service Startup Command so the image `CMD` is used.
6. Applies container settings and restarts the app.

Required App Service settings:

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

For private GHCR images:

```text
DOCKER_REGISTRY_SERVER_URL=https://ghcr.io
DOCKER_REGISTRY_SERVER_USERNAME=<github-username>
DOCKER_REGISTRY_SERVER_PASSWORD=<PAT-with-read:packages>
```

GitHub Actions needs `packages: write` for image publishing. Keep the App
Service Startup Command empty for the container deployment; the image starts
`docker-entrypoint.sh` itself.

## Troubleshooting

### Azure OpenAI returns 404 `Resource not found`

Verify all four values:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_API_KEY
```

Use the endpoint and deployment name shown on the Azure deployment's endpoint
page. Do not append a deployment path manually when the SDK expects the base
resource endpoint.

### Uploaded geometry is preview-only

Exact volume FEM requires a closed solid. STL failures commonly indicate open
edges, self-intersections, overlapping internal faces, or a surface-only shell.
Prefer the original STEP solid when available. The application deliberately
does not substitute a simplified cylindrical surrogate for a failed exact
uploaded-geometry mesh.

### `libgomp.so.1` is missing

Install `libgomp1` or rebuild from the active `Dockerfile`, which includes it.

### CalculiX or rendering is unavailable

Confirm `ccx`, Xvfb, Mesa/OpenGL libraries, and the full Python FEM requirements
are installed. In Docker, inspect entrypoint logs for the Python, gunicorn, and
Xvfb diagnostics.

### Azure container does not start

Check image pull credentials, port `8000`, Startup Command, and container logs.
For large FEM images, retain:

```text
WEBSITES_CONTAINER_START_TIME_LIMIT=1800
```

### Secrets

Never commit `backend/.env`, API keys, publish profiles, PATs, or Azure
credentials. Store production secrets in Azure App Service settings and GitHub
Actions secrets.
