# Common patterns

Proven recipes. Copy the shape, swap the numbers.

## Bolt-hole pattern (construction rectangle + vertices)

```python
result = (
    cq.Workplane("XY").box(100, 60, 8)
    .faces(">Z").workplane()
    .rect(80, 40, forConstruction=True)   # layout helper only
    .vertices().hole(6)                   # one feature -> 4 holes
)
```

## Circular bolt pattern (polar layout)

```python
result = (
    cq.Workplane("XY").cylinder(10, 40)
    .faces(">Z").workplane()
    .polarArray(radius=28, startAngle=0, angle=360, count=6)
    .hole(5)
)
```

## Counterbored / countersunk holes

```python
.faces(">Z").workplane().rect(60, 40, forConstruction=True).vertices()
.cboreHole(6, 12, 4)     # through 6 mm, 12 mm counterbore 4 mm deep
# or
.cskHole(6, 12, 82)      # 82-degree countersink
```

## Filleting / chamfering the right edges

```python
.edges("|Z").fillet(3)                       # vertical edges of a box
.faces(">Z").edges("%Circle").chamfer(1)     # top rim of a cylinder
```

## Shelling (hollow part)

```python
result = cq.Workplane("XY").box(60, 40, 30).faces(">Z").shell(-2)
# negative thickness shells inward, removing the selected top face
```

## Revolve a profile (bushings, bosses, turned parts)

```python
result = (
    cq.Workplane("XZ")
    .moveTo(10, 0).lineTo(20, 0).lineTo(20, 30).lineTo(10, 30)
    .close()
    .revolve(360)        # spins about the workplane local Y axis
)
```

## Sweep along a path (helical spring, tubing)

```python
from cadquery.func import *
helix = wire([...computed points...])     # build the centerline
profile = circle(1.5).located(...)         # wire cross-section
spring = sweep(profile, helix)
```

## Always export both formats

```python
cq.exporters.export(result, "part.step")
cq.exporters.export(result, "part.stl")
```
