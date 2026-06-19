# B-Rep mindset (not CSG)

CadQuery sits on the OpenCASCADE (OCCT) kernel, which is a **boundary
representation** modeler. A solid is described by its *boundary*: faces bounded
by edges, edges bounded by vertices. This is the same data model a human uses
inside SolidWorks, Fusion, or FreeCAD.

## CSG thinking (avoid)

> "Make a box, make a smaller box, subtract it; make four cylinders, subtract
> them; make a plate, union it."

This works numerically but is fragile: tiny coincident faces, non-manifold
results, and OCCT boolean failures are common, and the script is hard to edit.

## B-Rep thinking (prefer)

> "Make the base. Select the top face, put 4 holes through it. Select the
> vertical edges, fillet them. Select the bottom face, shell out 2 mm."

You always answer two questions:

1. **Which boundary geometry?** → use a selector (`>Z`, `|Z`, `%Plane`, …).
2. **Which feature?** → `.hole`, `.fillet`, `.chamfer`, `.shell`, `.extrude`.

## Decision order

When you need to add or remove material, choose the **highest** option that fits:

1. A dedicated feature on a selection (`.hole`, `.fillet`, `.chamfer`, `.shell`).
2. A sketch on a selected face, extruded (`+`) or cut (`-`) into the solid.
3. A boolean with another solid (`.cut`, `.union`, `.intersect`) — last resort.

If you typed `.cut(cq.Workplane().cylinder(...))` to make a hole, you used a
boolean where `.hole(d)` was correct. Fix it.

## Why it matters for an LLM

Feature-first scripts:

- execute far more reliably (OCCT booleans are the #1 source of failures),
- stay parametric and editable,
- produce clean, single, valid solids ready for STEP/STL export and meshing.
