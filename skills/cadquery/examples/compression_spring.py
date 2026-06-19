"""Example: a helical compression spring.

Shows the Free Function API (`cadquery.func`) for math-driven geometry, where a
helical centerline is swept by a circular wire cross-section. This is the right
tool when the fluent Workplane stack would be awkward.

Run with: python compression_spring.py  (requires `pip install cadquery`)
"""

import math

from cadquery.func import *  # noqa: F403  (functional API, explicit by design)

# --- parameters (mm) ---------------------------------------------------------
coil_radius = 12.0      # radius to the wire centerline
wire_radius = 1.5       # cross-section radius
free_height = 40.0      # uncompressed height
turns = 6
samples_per_turn = 24

# --- build the helical centerline as a wire ----------------------------------
points = []
total_samples = turns * samples_per_turn
for i in range(total_samples + 1):
    t = i / samples_per_turn          # turn fraction
    angle = 2.0 * math.pi * t
    x = coil_radius * math.cos(angle)
    y = coil_radius * math.sin(angle)
    z = free_height * (i / total_samples)
    points.append((x, y, z))

centerline = polyline(points)

# --- circular cross-section, placed at the start of the helix ----------------
start = points[0]
profile = circle(wire_radius).moved(Location(start))

# --- sweep the profile along the centerline ----------------------------------
spring = sweep(profile, centerline)

# --- export ------------------------------------------------------------------
export(spring, "compression_spring.step")
export(spring, "compression_spring.stl")
