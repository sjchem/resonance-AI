---
name: cadquery-llm-skill
description: >-
  Helps LLMs write correct, idiomatic CadQuery code. Loaded into the Resonance
  AI CAD agent so generated parts use the B-Rep mindset, valid selectors, and
  proven patterns instead of brittle constructive-solid-geometry hacks.
applies_to: CadQuery 2.x (Python), OCCT / OpenCASCADE kernel
license: Apache-2.0 (adapted from github.com/jmwright/cadquery-llm-skill)
---

# CadQuery LLM Skill

A compact, high-signal reference for generating **correct, idiomatic CadQuery**.
Read this before writing or repairing any CadQuery script. The companion files
go deeper:

- `concepts/brep-mindset.md` — think in faces/edges/features, not CSG primitives.
- `concepts/workplanes.md` — where geometry is created and how the stack moves.
- `concepts/selectors.md` — the selector cheat sheet (full table below too).
- `concepts/free-function-api.md` — the `cadquery.func` functional API.
- `patterns/common-patterns.md` — battle-tested recipes.
- `patterns/anti-patterns.md` — mistakes that produce invalid solids.
- `examples/` — runnable, annotated parts (bushing, compression spring).

---

## 1. The one rule that matters most: B-Rep, not CSG

CadQuery is a **boundary-representation (B-Rep)** modeler on the OpenCASCADE
kernel. You build solids by *selecting boundary geometry* (faces, edges, wires,
vertices) and *applying features* to it — exactly like a human in a parametric
CAD package. You are **not** scripting a chain of union/subtract primitives.

Prefer, in order:

1. A feature on selected boundary geometry (`.hole()`, `.fillet()`, `.shell()`,
   `.chamfer()`, `.extrude()` from a sketch on a face).
2. A 2D sketch extruded/revolved into the solid.
3. A boolean (`.cut()`, `.union()`, `.intersect()`) — only when no feature fits.

If you reach for a boolean to do something a feature already does, stop and use
the feature. See `patterns/anti-patterns.md`.

---

## 2. Two APIs — pick one and stay consistent

### Fluent / Workplane API (default; best for parametric parts)

```python
import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(80, 60, 10)            # primitive on the current workplane
    .faces(">Z")                # select the top face ...
    .workplane()                # ... and start a new workplane on it
    .rect(60, 40, forConstruction=True)
    .vertices()                 # 4 corners of that construction rect
    .hole(6)                    # one feature -> 4 holes
    .edges("|Z")                # vertical edges
    .fillet(3)
)
```

### Free Function API (best for direct, math-driven geometry)

```python
from cadquery.func import *

base = box(80, 60, 10)
top = faces(base, ">Z")
result = fillet(edges(base, "|Z"), 3)
```

Do not mix the two styles in one script. The Workplane API keeps an internal
"stack" of selected objects; the functional API passes shapes explicitly.

---

## 3. Workplane stack mental model (Fluent API)

A `Workplane` carries a **stack** of the objects you last selected. Each call
either *replaces* the stack (a selector like `.faces(...)`) or *consumes* it
(a feature like `.hole(...)`). Key moves:

- `.faces("<sel>")`, `.edges("<sel>")`, `.vertices("<sel>")` — select boundary.
- `.workplane()` — create a new sketch plane on the currently selected face.
- `.workplane(offset=z)` — offset along the plane normal.
- `.center(dx, dy)` — move the local origin on the current plane.
- `.tag("name")` / `.workplaneFromTagged("name")` — remember/return to a place.
- `.end()` — pop back up the construction history (NOT to a tag).

When a feature reads "for each item on the stack", selecting 4 vertices then
calling `.hole(6)` drills 4 holes. That is the idiomatic pattern — never loop in
Python to place identical holes.

---

## 4. Selector cheat sheet

| Selector            | Meaning                                              |
| ------------------- | ---------------------------------------------------- |
| `>Z` / `<Z`         | Face/edge furthest in +Z / −Z (max / min)            |
| `>X`, `<Y`, …       | Same idea on other axes                              |
| `|Z`                | Edges **parallel** to Z                              |
| `#Z`                | Faces/edges **perpendicular** to Z                   |
| `+Z` / `-Z`         | Faces whose **normal** points +Z / −Z               |
| `%Plane`            | Faces of type Plane (also `%Cylinder`, `%Circle`)    |
| `>>Z[1]`            | The **second** group sorted along Z (0-indexed)      |
| `>Z and >X`         | Intersection of two selectors                        |
| `>Z or <Z`          | Union of two selectors                               |
| `not >Z`            | Everything except the +Z extreme                     |
| `cq.NearestToPointSelector((x, y, z))` | Closest face/edge to a point       |

Rules of thumb:

- `|`, `#`, `+`, `-` describe **direction**; `>`, `<` describe **position**.
- Combine with `and` / `or` / `not`; group with parentheses when needed.
- After a `.tag()`, use `.workplaneFromTagged("tag")` — do **not** rely on
  `.end()` to get back there.

---

## 5. Feature quick reference

```python
.box(l, w, h)                       # centered primitive
.cylinder(h, r)                     # centered primitive
.sphere(r)
.extrude(dist)                      # extrude the active sketch/wire
.revolve(angleDeg=360)              # revolve sketch about Y of the workplane
.hole(diameter)                     # blind/through hole at each stack point
.cboreHole(d, cbD, cbDepth)         # counterbored hole
.cskHole(d, cskD, cskAngle)         # countersunk hole
.fillet(radius)                     # on selected edges
.chamfer(distance)                  # on selected edges
.shell(thickness)                   # hollow out, removing selected faces
.sweep(path)                        # sweep sketch along a wire path
.loft()                             # loft between successive sketches
```

`radius`/`distance` must be **smaller than the local geometry** or OCCT throws.
A fillet radius must be < half the thinnest adjacent edge; a chamfer distance
likewise. See `patterns/anti-patterns.md`.

---

## 6. Closing wires & validity

- Every profile you extrude/revolve must be a **closed** wire. `.close()` your
  polyline sketches.
- `revolve` spins about the workplane's local Y axis — keep the profile on one
  side of that axis or you get a self-intersecting solid.
- After non-trivial booleans/shells, the result must stay a single valid solid.
  Prefer features that cannot leave dangling faces.

---

## 7. Export

```python
cq.exporters.export(result, "part.step")   # STEP for downstream CAD/CAE
cq.exporters.export(result, "part.stl")     # STL for meshing / preview
```

Always export deterministic, millimeter-unit geometry. No network calls, no
external assets.

---

## 8. Minimal correct template

```python
import cadquery as cq

# 1. base primitive
result = cq.Workplane("XY").box(60, 40, 12)

# 2. feature on selected boundary geometry (NOT a boolean)
result = (
    result.faces(">Z").workplane()
    .rect(40, 20, forConstruction=True)
    .vertices().hole(6)          # 4 mounting holes in one feature
)

# 3. soften edges
result = result.edges("|Z").fillet(3)

# 4. export
cq.exporters.export(result, "part.step")
cq.exporters.export(result, "part.stl")
```

That is the shape of almost every good CadQuery script: primitive → select →
feature → finish → export.
