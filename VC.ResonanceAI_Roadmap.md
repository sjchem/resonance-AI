# VC.ResonanceAI Roadmap: Text-to-CAD + PhysicsNeMo MeshGraphNet Platform

## 1. Project Goal

Build a local-to-cloud engineering AI platform inspired by Neural Concept.

The platform should take a natural language design prompt, generate a CAD or mesh representation, convert it into a simulation-ready mesh/graph, and use NVIDIA PhysicsNeMo MeshGraphNet to predict physical behavior such as displacement, vibration amplitude, or stress-related fields.

High-level workflow:

```text
Text prompt
   ↓
Text-to-CAD / Text-to-Mesh generator
   ↓
Clean CAD / mesh
   ↓
PhysicsNeMo MeshGraphNet
   ↓
Physical prediction
   ↓
Validation + uncertainty + FastAPI service
```

Initial focus should be a simple structural or NVH-style surrogate model, not the full commercial-level Neural Concept platform.

Recommended first MVP:

```text
Text prompt:
"Create a 120 mm x 60 mm x 5 mm bracket with four bolt holes."

↓

CadQuery generates bracket.step / bracket.stl

↓

Gmsh creates bracket.msh

↓

Graph builder creates node and edge features

↓

PhysicsNeMo MeshGraphNet predicts displacement magnitude

↓

FastAPI returns max displacement, uncertainty, and validation plots
```

---

## 2. Why Separate the Platform into Two Modules

Do not train text-to-CAD and physics prediction together at the beginning.

Keep the system modular:

```text
Module A: Text → CAD / mesh
Module B: Mesh / graph → physics prediction
```

This separation makes debugging much easier.

Text-to-CAD is a generative design problem.
Physics prediction is a surrogate modeling problem.

Later, both modules can be connected into one end-to-end platform.

---

## 3. Local vs Azure Development Strategy

### Local Lenovo T14, 32 GB RAM

Use the laptop for:

```text
- Environment setup
- Text-to-CAD experiments
- CadQuery / Build123d script execution
- STEP / STL export
- Small mesh conversion
- Mesh-to-graph debugging
- FastAPI prototype
- Tiny training or inference tests
```

Avoid heavy model training locally unless you have a capable NVIDIA GPU.

### Azure VM

Use Azure for:

```text
- Real PhysicsNeMo training
- Large FEM / NVH datasets
- Large mesh batches
- Hyperparameter tuning
- Ensemble uncertainty models
- Production-like inference
```

Recommended Azure GPU options:

```text
Small experiments: NCasT4_v3
Better training:   NC A100 v4
Large-scale:       ND A100 v4 or H100-based VM
```

---

## 4. Recommended Technology Stack

### Text-to-CAD

```text
CadQuery
Build123d
FreeCAD
OpenCASCADE / OCCT
Optional: Ollama + Qwen Coder / cloud LLM / Azure-hosted LLM
```

### Mesh Processing

```text
Gmsh
meshio
PyVista
trimesh
FreeCAD
```

### Physics AI

```text
NVIDIA PhysicsNeMo
MeshGraphNet
Hybrid MeshGraphNet
PyTorch
PyTorch Geometric if building custom models
```

### API and Deployment

```text
FastAPI
Uvicorn
Docker
Azure VM
Azure Blob Storage
MLflow or Weights & Biases
```

### Visualization and Validation

```text
Matplotlib
PyVista
NumPy
Pandas
Scikit-learn
```

---

## 5. Recommended Project Folder Structure

```text
ResonanceAI/
│
├── text_to_cad/
│   ├── prompts/
│   ├── cad_generator.py
│   ├── cad_executor.py
│   └── examples/
│
├── geometry/
│   ├── step_to_mesh.py
│   ├── mesh_cleaner.py
│   ├── mesh_to_graph.py
│   └── mesh_quality.py
│
├── data/
│   ├── raw_fem/
│   ├── processed_graphs/
│   └── samples/
│
├── physics_model/
│   ├── train_meshgraphnet.py
│   ├── infer.py
│   ├── dataset.py
│   └── config.yaml
│
├── validation/
│   ├── compare_fem.py
│   ├── plots.py
│   └── uncertainty.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yaml
│
├── notebooks/
│
└── README.md
```

---

## 6. Phase 1: Text-to-CAD Locally

### Goal

Convert natural language into a parametric CAD model.

Example prompt:

```text
Create a rectangular rubber mount bracket with four bolt holes and 5 mm thickness.
```

Expected output:

```text
bracket.step
bracket.stl
preview.png
```

### Why Use CadQuery First

CadQuery creates parametric CAD-style geometry instead of only faceted mesh geometry.

This is important because engineering workflows need editable and precise geometry.

Mesh-only generators are useful for concept visualization, but they are usually weaker for FEM, manufacturing, and design iteration.

### Local Installation

```bash
conda create -n resonanceAI python=3.10 -y
conda activate resonanceAI

pip install cadquery
pip install build123d
pip install trimesh meshio pyvista
pip install gmsh
pip install fastapi uvicorn pydantic
pip install matplotlib numpy pandas scikit-learn
pip install jupyterlab
```

### Example CadQuery Script

```python
import cadquery as cq

length = 120
width = 60
thickness = 5
hole_radius = 4

part = (
    cq.Workplane("XY")
    .box(length, width, thickness)
    .faces(">Z")
    .workplane()
    .pushPoints([
        (-45, -20),
        (45, -20),
        (-45, 20),
        (45, 20),
    ])
    .hole(2 * hole_radius)
)

cq.exporters.export(part, "bracket.step")
cq.exporters.export(part, "bracket.stl")
```

---

## 7. Phase 2: Text-to-CAD Agent

A reliable text-to-CAD system should not use a single-shot generation approach.

Use an iterative agentic loop:

```text
User prompt
   ↓
LLM generates CadQuery / Build123d code
   ↓
Execute generated code locally
   ↓
If code fails, capture error log
   ↓
Send error back to LLM for correction
   ↓
If code works, export STEP / STL
   ↓
Render preview
   ↓
Run basic geometry diagnostics
   ↓
Accept geometry or self-correct
```

### Practical Options

For your laptop:

```text
Option A: Cloud LLM generates CadQuery code, laptop executes CAD code
Option B: Azure-hosted LLM generates CAD code, laptop executes CAD code
Option C: Ollama local model for simple parts only
```

Recommended first approach:

```text
Use a cloud or Azure-hosted LLM for generation
Use local Python environment for CAD execution
```

Reason: 32 GB RAM is enough for CAD scripting and mesh conversion, but not ideal for large local LLM-based mechanical reasoning.

---

## 8. Phase 3: CAD to Mesh Conversion

PhysicsNeMo MeshGraphNet requires mesh or graph data.

Typical conversion:

```text
STEP / STL
   ↓
Gmsh
   ↓
.msh file
   ↓
meshio / PyVista
   ↓
points, cells, connectivity
```

### Example STEP to Mesh with Gmsh

```python
import gmsh

gmsh.initialize()
gmsh.open("bracket.step")

gmsh.model.mesh.generate(3)
gmsh.write("bracket.msh")

gmsh.finalize()
```

### Read Mesh with meshio

```python
import meshio

mesh = meshio.read("bracket.msh")

points = mesh.points
cells = mesh.cells

print(points.shape)
print(cells)
```

---

## 9. Phase 4: Mesh to Graph Conversion

MeshGraphNet treats a mesh as a graph.

```text
Nodes = mesh vertices
Edges = mesh connectivity
Node features = geometry + material + boundary conditions
Edge features = distance + relative direction
Targets = FEM / NVH result fields
```

### Node Features

For each mesh node:

```text
x, y, z
is_fixed
is_loaded
material_youngs_modulus
material_density
material_poisson_ratio
material_damping
excitation_frequency
force_x, force_y, force_z
```

Example node feature vector:

```python
node_features = [
    x,
    y,
    z,
    fixed_flag,
    load_flag,
    youngs_modulus,
    density,
    poisson_ratio,
    damping_ratio,
    excitation_frequency,
    force_x,
    force_y,
    force_z,
]
```

### Edge Features

For each edge between node `i` and node `j`:

```text
dx = x_j - x_i
dy = y_j - y_i
dz = z_j - z_i
distance = ||p_j - p_i||
```

Example:

```python
edge_features = [
    x_j - x_i,
    y_j - y_i,
    z_j - z_i,
    distance_ij,
]
```

### Targets

For displacement prediction:

```text
u_x, u_y, u_z
```

For simpler first MVP:

```text
displacement_magnitude
```

For vibration:

```text
vibration_amplitude_at_frequency
```

For scalar prediction:

```text
max_displacement
first_natural_frequency
peak_FRF
```

---

## 10. Phase 5: Start with PhysicsNeMo MeshGraphNet Deforming Plate

Before using your own dataset, first run the official NVIDIA PhysicsNeMo MeshGraphNet deforming plate example.

Goal:

```text
Understand the data format
Understand graph construction
Understand model input/output
Understand training loop
Understand inference output
```

### Clone PhysicsNeMo

```bash
git clone https://github.com/NVIDIA/physicsnemo.git
cd physicsnemo
```

Inspect:

```text
examples/structural_mechanics/deforming_plate/
```

Run the example without modification first.

Only after it runs successfully, start replacing parts of the dataset pipeline.

---

## 11. Phase 6: Replace Deforming Plate Dataset with FEM/NVH Data

### Original Example Data Concept

```text
mesh
boundary conditions
time step
mesh deformation target
```

### Your Vibracoustic-Style Data Concept

```text
component mesh
rubber / metal material properties
mounting constraints
force / excitation position
excitation frequency
FEM displacement / vibration output
```

### Recommended Dataset Format

```text
sample_0001/
    mesh.msh
    node_features.npy
    edge_index.npy
    edge_features.npy
    target_displacement.npy
    global_params.json

sample_0002/
    mesh.msh
    node_features.npy
    edge_index.npy
    edge_features.npy
    target_displacement.npy
    global_params.json
```

Example `global_params.json`:

```json
{
  "material": "rubber",
  "youngs_modulus": 7000000,
  "density": 1100,
  "poisson_ratio": 0.49,
  "damping_ratio": 0.08,
  "frequency_hz": 120,
  "load_n": 50,
  "temperature_c": 23
}
```

---

## 12. Phase 7: Add Material and Boundary-Condition Features

For NVH, geometry alone is not enough.

Add material properties:

```text
Young's modulus
Density
Poisson ratio
Damping factor
Rubber hardness, if available
Temperature, if relevant
```

Add boundary-condition features:

```text
Fixed node flag
Loaded node flag
Contact region flag
Bolt hole region flag
Force vector
Excitation frequency
Mounting condition
```

Add geometry features:

```text
x, y, z
surface normal
curvature, optional
distance to mounting point
component ID, if assembly
```

For vibration-specific prediction, include:

```text
frequency_hz
load_direction_x
load_direction_y
load_direction_z
damping_ratio
```

---

## 13. Phase 8: Choose the First Prediction Target

Do not start with full NVH behavior.

Recommended progression:

```text
Stage 1: Static displacement magnitude
Stage 2: Vector displacement, u_x/u_y/u_z
Stage 3: Displacement under harmonic load
Stage 4: Vibration amplitude at selected frequency
Stage 5: Modal frequency prediction
Stage 6: Full frequency response function prediction
```

Best first target:

```text
nodal displacement magnitude
```

Reason:

```text
- Easier to visualize
- Easier to validate
- Easier to debug
- Useful for engineering trust
```

---

## 14. Phase 9: Validation Plots

Validation is critical because engineers need to trust the surrogate model.

Create these plots:

```text
1. FEM vs AI displacement scatter plot
2. Error heatmap on mesh
3. Histogram of relative error
4. Predicted vs true max displacement
5. Worst 10 examples
6. Uncertainty vs error plot
```

Recommended metrics:

```text
MAE
RMSE
Relative L2 error
Max displacement error
Peak displacement error
Peak location error
R² for scalar outputs
```

### Example Relative L2 Error

```python
import numpy as np

def relative_l2_error(y_pred, y_true):
    return np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true)
```

---

## 15. Phase 10: Uncertainty Estimation

Start with ensemble uncertainty.

Train 5 MeshGraphNet models:

```text
Same dataset
Different random seeds
Same architecture
```

At inference:

```text
Prediction mean = final prediction
Prediction standard deviation = uncertainty
```

Interpretation:

```text
Low standard deviation  → model is confident
High standard deviation → send design to full FEM simulation
```

Later improvements:

```text
Monte Carlo dropout
Conformal prediction
Out-of-distribution geometry detection
Latent-space distance
Bayesian neural networks
```

For the first platform, ensemble uncertainty is enough.

---

## 16. Phase 11: FastAPI Inference Service

The API should support these endpoints:

```text
POST /generate-cad
POST /mesh
POST /predict
POST /validate
```

### Simple FastAPI Skeleton

```python
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI(title="ResonanceAI API")

class TextCADRequest(BaseModel):
    prompt: str

@app.post("/generate-cad")
def generate_cad(req: TextCADRequest):
    return {
        "status": "ok",
        "step_file": "bracket.step",
        "stl_file": "bracket.stl"
    }

@app.post("/predict")
def predict_physics():
    return {
        "max_displacement": 0.42,
        "unit": "mm",
        "uncertainty": 0.03
    }
```

Later endpoints:

```text
POST /upload-step
POST /run-meshgraphnet
GET  /download-results
GET  /health
```

---

## 17. Local Development Timeline

### Week 1: Text-to-CAD

Build:

```text
prompt → CadQuery code → STEP / STL
```

Test simple parts:

```text
bracket
plate with holes
rubber block
bushing-like cylinder
mount base
```

Deliverable:

```text
Generated STEP/STL files
```

---

### Week 2: Mesh Pipeline

Build:

```text
STEP → Gmsh mesh → graph
```

Deliverable:

```text
node_features.npy
edge_index.npy
edge_features.npy
```

---

### Week 3: PhysicsNeMo Example

Run:

```text
PhysicsNeMo MeshGraphNet deforming plate example
```

Do not modify it first.

Understand:

```text
data loader
graph construction
training loop
inference
output format
```

---

### Week 4: Custom Toy Dataset

Create simple toy data first:

```text
plate geometry
simple fixed boundary
simple load
synthetic displacement target
```

Deliverable:

```text
Your own dataset format works with the model pipeline
```

---

### Week 5-6: Real FEM/NVH Data

Replace toy data with real simulation output:

```text
mesh + boundary conditions + material → displacement / vibration target
```

Deliverable:

```text
First real surrogate model
```

---

## 18. Azure Scale-Up Plan

### Azure Environment

Recommended setup:

```text
Ubuntu 22.04
NVIDIA driver
CUDA
Docker
PhysicsNeMo environment
MLflow or Weights & Biases
Azure Blob Storage
```

### Azure Workflow

```text
Upload FEM/NVH dataset to Azure Blob Storage
   ↓
Start GPU VM
   ↓
Pull Docker image / environment
   ↓
Train MeshGraphNet
   ↓
Save checkpoints and metrics
   ↓
Run validation
   ↓
Export model
   ↓
Deploy FastAPI inference service
```

### Use Azure For

```text
large graph batches
multi-run experiments
hyperparameter tuning
ensemble models
large-scale validation
production-like inference
```

---

## 19. Minimum Files to Build First

Start with these files:

```text
text_to_cad/cad_generator.py
geometry/step_to_mesh.py
geometry/mesh_to_graph.py
physics_model/infer.py
api/main.py
validation/plots.py
```

Recommended order:

```text
1. cad_generator.py
2. step_to_mesh.py
3. mesh_to_graph.py
4. Run PhysicsNeMo example
5. Adapt dataset.py
6. Build infer.py
7. Build api/main.py
8. Add validation plots
9. Add uncertainty
10. Move to Azure training
```

---

## 20. First MVP Definition

The first complete MVP should do this:

```text
Input:
Natural language text prompt

Example:
"Create a 120 mm x 60 mm x 5 mm bracket with four bolt holes."

Output:
1. Generated CAD file
2. Generated mesh file
3. Graph representation
4. Predicted displacement magnitude
5. Uncertainty estimate
6. Validation plot
```

Minimal user-facing output:

```json
{
  "status": "success",
  "cad_file": "bracket.step",
  "mesh_file": "bracket.msh",
  "max_displacement_mm": 0.42,
  "uncertainty": 0.03,
  "recommendation": "Prediction confidence is acceptable. Full FEM validation optional."
}
```

---

## 21. Important Engineering Warnings

### Do Not Start Too Broad

Avoid starting with:

```text
full automotive assembly
full rubber mount behavior
full NVH prediction
full frequency response curves
```

Start with:

```text
simple plate
simple bracket
simple static displacement
```

### Text-to-CAD Can Generate Invalid Geometry

Always validate:

```text
code execution
solid validity
watertightness
mesh quality
minimum wall thickness
hole positions
self-intersections
```

### Physics Surrogate Needs Good FEM Data

A MeshGraphNet model is only as good as the simulation data it learns from.

Prioritize:

```text
consistent simulation setup
consistent boundary conditions
clean material metadata
well-labeled outputs
clear train/validation/test split
```

### Always Keep FEM in the Loop

The AI model should accelerate design exploration.

It should not replace final engineering validation.

Use the surrogate model for:

```text
fast screening
early design feedback
optimization loops
candidate ranking
```

Use FEM for:

```text
final validation
safety-critical confirmation
out-of-distribution designs
high-uncertainty predictions
```

---

## 22. Final Roadmap Summary

```text
1. Build text-to-CAD for simple parts
2. Export STEP/STL
3. Convert CAD to Gmsh mesh
4. Convert mesh to graph
5. Run PhysicsNeMo deforming plate example
6. Replace example data with toy data
7. Replace toy data with FEM/NVH data
8. Add material and boundary-condition features
9. Predict displacement magnitude
10. Add validation plots
11. Add ensemble uncertainty
12. Build FastAPI service
13. Scale training on Azure GPU VM
14. Extend to vibration amplitude and modal/NVH tasks
```

---
