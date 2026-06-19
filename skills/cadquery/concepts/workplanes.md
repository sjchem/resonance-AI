# Workplanes and the selection stack

A `Workplane` is two things at once:

1. A **coordinate frame** (origin + X/Y/normal) where new 2D geometry is drawn.
2. A **stack** of the objects you most recently selected (faces, edges,
   vertices, or wires).

Understanding the stack is the key to fluent CadQuery.

## Creating and moving the plane

```python
cq.Workplane("XY")            # start on the global XY plane
  .box(40, 40, 20)
  .faces(">Z")                # stack now = [top face]
  .workplane()                # new plane sits ON that top face
  .workplane(offset=5)        # push 5 mm along the plane normal
  .center(10, 0)              # move the local origin on the plane
```

- `.workplane()` consumes a selected **face** and becomes a sketch plane there.
- `.workplane(offset=d)` shifts along the normal (useful for stepped features).
- `.center(dx, dy)` repositions the local origin without changing orientation.

## The stack drives features

A feature acts **once per item on the stack**:

```python
.faces(">Z").workplane()
.rect(30, 20, forConstruction=True)
.vertices()                   # stack = 4 corner vertices
.hole(6)                      # -> 4 holes, one feature call
```

`forConstruction=True` means the rectangle is a *layout helper*: it is not turned
into geometry, only its vertices are used to place features. This is the
canonical bolt-pattern idiom — never loop in Python to place identical holes.

## Remembering positions

```python
.faces(">Z").workplane().tag("top")
# ... do other things, stack moves around ...
.workplaneFromTagged("top")   # jump straight back to the tagged plane
```

- Use `.tag("name")` + `.workplaneFromTagged("name")` to return to a place.
- `.end()` walks **back up the construction history** by one step — it does NOT
  return to a tag. Mixing these up is a common bug (see anti-patterns).

## Common mistakes

- Calling `.workplane()` when the stack holds an **edge/vertex** (needs a face).
- Forgetting `.close()` before `.extrude()` on a polyline sketch.
- Expecting `.end()` to undo back to a tagged plane — it does not.
