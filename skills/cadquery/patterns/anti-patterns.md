# Anti-patterns

These are the mistakes that most often produce invalid solids or OCCT crashes.
Each has the wrong way and the fix.

## 1. Boolean when a feature would do

```python
# WRONG — boolean subtraction to make a hole
hole = cq.Workplane("XY").cylinder(50, 3)
result = plate.cut(hole)

# RIGHT — use the hole feature on a selected face
result = plate.faces(">Z").workplane().hole(6)
```

Booleans are the #1 source of execution failures. Reach for `.hole`, `.fillet`,
`.chamfer`, `.shell`, or a sketch-extrude first.

## 2. Fillet / chamfer larger than the geometry

```python
# WRONG — radius >= half the thinnest adjacent edge -> OCCT error
cq.Workplane("XY").box(10, 10, 2).edges("|Z").fillet(5)

# RIGHT — keep radius < half the thinnest neighbouring dimension
cq.Workplane("XY").box(10, 10, 2).edges("|Z").fillet(1.5)
```

Validate radii against local thickness before emitting them.

## 3. Unclosed wire before extrude/revolve

```python
# WRONG — polyline never closed
.moveTo(0, 0).lineTo(10, 0).lineTo(10, 5).extrude(3)

# RIGHT
.moveTo(0, 0).lineTo(10, 0).lineTo(10, 5).close().extrude(3)
```

## 4. Selector ordering that narrows to nothing

```python
# WRONG — vertical edges *of the top face* may be empty
.faces(">Z").edges("|Z").fillet(2)

# RIGHT — select vertical edges from the solid
.edges("|Z").fillet(2)
```

## 5. `.end()` used as "go back to my tag"

```python
# WRONG — .end() pops construction history, not to a tag
.faces(">Z").workplane().tag("top").hole(6).end().hole(8)

# RIGHT — return explicitly to the tagged plane
.faces(">Z").workplane().tag("top").hole(6)
.workplaneFromTagged("top").hole(8)
```

## 6. `centerOption` mismatch on faces

When sketching on a non-origin face, the default `centerOption` can place
geometry at the projected global origin, not the face center.

```python
# Be explicit about where the new workplane is centered:
.faces(">Z").workplane(centerOption="CenterOfBoundBox")
```

Use `CenterOfBoundBox` to sketch relative to the face's own center;
`ProjectedOrigin` (default) projects the global origin onto the plane.

## 7. Revolve profile crossing the axis

`revolve` spins about the workplane local Y axis. Keep the whole profile on one
side of that axis, or the solid self-intersects and OCCT fails.

## 8. Mixing the two APIs

Do not interleave `cq.Workplane(...)` fluent chains with `cadquery.func`
calls in the same script. Pick one.

## 9. Forgetting deterministic, mm-unit output

No random values, no network/file fetches, no implicit unit assumptions.
Everything in millimeters, every run identical.
