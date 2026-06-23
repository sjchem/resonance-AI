"""Local NVH simulation package for Resonance AI.

Pipeline stages:

1. :mod:`geometry.step_to_mesh`  - STEP -> tetrahedral volume mesh
2. :mod:`geometry.mesh_cleaner`  - merge nodes, drop degenerate cells
3. :mod:`geometry.mesh_quality`  - solver-readiness quality check
4. :mod:`simulate.materials`     - material property library
5. :mod:`simulate.modal_solver`  - CalculiX modal (eigenfrequency) analysis
6. :mod:`simulate.results`       - parse and report natural frequencies

The :mod:`simulate.pipeline` module ties these together into one command.
"""

from __future__ import annotations
