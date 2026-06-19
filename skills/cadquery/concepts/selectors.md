# Selectors

Selectors are tiny strings that pick boundary geometry. Master these and most of
CadQuery falls into place. Pass them to `.faces(...)`, `.edges(...)`, or
`.vertices(...)`.

## Position vs direction

Two independent ideas:

- **Position** — *where* along an axis: `>` (max) and `<` (min).
  - `>Z` = the topmost face/edge, `<Z` = the bottommost, `>X` = the +X extreme.
- **Direction** — *how* geometry is oriented:
  - `|Z` — edges **parallel** to Z.
  - `#Z` — faces/edges **perpendicular** to Z.
  - `+Z` / `-Z` — faces whose **normal** points along +Z / −Z.

> Mnemonic: `>` `<` ask "which end?"; `|` `#` `+` `-` ask "which way?".

## Type and ordering

- `%Plane` — only planar faces. Also `%Cylinder`, `%Circle`, `%Line`, `%Sphere`.
- `>>Z[0]`, `>>Z[1]`, … — geometry grouped and sorted along Z; index picks the
  Nth group (0-based). `>>Z[-2]` = second from the top.

## Boolean combinations

```python
.faces(">Z and %Plane")     # topmost, and planar
.edges("|Z or |X")          # vertical or X-aligned edges
.faces("not >Z")            # every face except the top
```

Group with parentheses when precedence is ambiguous.

## Point-based selection

```python
.faces(cq.NearestToPointSelector((10, 0, 5)))   # closest face to a point
.edges(cq.NearestToPointSelector((0, 0, 0)))
```

Use these when a directional/positional selector is ambiguous.

## Worked examples

```python
# Fillet only the four vertical edges of a box
.edges("|Z").fillet(2)

# Chamfer the top rim of a cylinder
.faces(">Z").edges("%Circle").chamfer(1)

# Drill the bottom face only
.faces("<Z").workplane().hole(8)

# Shell the part, leaving the top face open
.faces(">Z").shell(-2)
```

## Pitfalls

- Selector **ordering matters**: `.faces(">Z").edges("|Z")` selects vertical
  edges *of the top face*, which may be empty. Select the edges from the solid,
  not from an already-narrowed face, unless that is what you want.
- An empty selection makes the next feature throw. Verify your selector returns
  the geometry you expect.
