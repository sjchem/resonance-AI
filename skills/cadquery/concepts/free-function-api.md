# Free Function API (`cadquery.func`)

CadQuery 2.4+ ships a functional API that is ideal for **direct, math-driven
geometry** — lofts, sweeps, point clouds, helices — where the fluent stack feels
awkward. Import everything and pass shapes explicitly:

```python
from cadquery.func import *
```

## Primitives

```python
b = box(80, 60, 10)
c = cylinder(h=20, r=8)
s = sphere(5)
```

## Selection as functions

Instead of a stack, you call selector functions on a shape:

```python
top   = faces(b, ">Z")
verts = vertices(top)
vedge = edges(b, "|Z")
```

## Features as functions

```python
result = fillet(edges(b, "|Z"), 3)
result = chamfer(edges(top, "%Circle"), 1)
result = shell(b, faces(b, ">Z"), 2)
```

## Booleans as operators

```python
result = b - cylinder(20, 4)     # cut
result = a + c                    # union (fuse)
result = a & c                    # intersect
```

## Construction geometry

```python
w = wire([(0, 0), (10, 0), (10, 5), (0, 5)], close=True)
solid = extrude(w, 12)
```

## When to choose which API

| Use the Fluent / Workplane API when…       | Use the Free Function API when…              |
| ------------------------------------------- | -------------------------------------------- |
| Parametric prismatic parts, bolt patterns   | Lofts/sweeps along computed paths            |
| You want the selection stack to chain holes | You generate points/wires from math (helix)  |
| Readability for "primitive → feature" parts | You need explicit control over every shape   |

Pick **one** API per script. Do not interleave `cq.Workplane` chains with
`cadquery.func` calls.
