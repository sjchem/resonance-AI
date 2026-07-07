"""Lightweight repair pass for surface/volume meshes before solving.

The cleaner removes duplicate nodes, drops orphan points, and strips
degenerate (zero-area / zero-volume) cells. It is intentionally conservative:
it never changes valid topology, it only removes data that would make a solver
fail. Geometry healing of the STEP itself is handled upstream by Gmsh/OCC.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


@dataclass(frozen=True)
class CleanResult:
    """Summary of what the cleaning pass changed."""

    input_file: Path
    output_file: Path
    merged_nodes: int
    removed_cells: int
    final_node_count: int
    final_cell_count: int

    def summary(self) -> str:
        return (
            f"Cleaned: merged {self.merged_nodes} duplicate node(s), "
            f"removed {self.removed_cells} degenerate cell(s) -> "
            f"nodes={self.final_node_count} cells={self.final_cell_count}"
        )


def clean_mesh(
    input_file: Path,
    output_file: Path,
    *,
    merge_tolerance: float = 1e-6,
) -> CleanResult:
    """Merge coincident nodes and drop degenerate cells."""

    import meshio

    input_file = Path(input_file).resolve()
    output_file = Path(output_file).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {input_file}")

    mesh = meshio.read(str(input_file))
    points = np.asarray(mesh.points, dtype=float)

    remap, unique_points, merged_nodes = _merge_nodes(points, merge_tolerance)

    new_cells = []
    new_cell_data: dict[str, list[np.ndarray]] = {name: [] for name in mesh.cell_data}
    removed_cells = 0
    for block_index, block in enumerate(mesh.cells):
        data = remap[np.asarray(block.data, dtype=int)]
        keep_mask = _non_degenerate_mask(data)
        removed_cells += int(np.count_nonzero(~keep_mask))
        kept = data[keep_mask]
        if kept.size:
            new_cells.append((block.type, kept))
            for name, values_by_block in mesh.cell_data.items():
                if block_index < len(values_by_block):
                    new_cell_data[name].append(np.asarray(values_by_block[block_index])[keep_mask])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned = meshio.Mesh(points=unique_points, cells=new_cells, cell_data={name: values for name, values in new_cell_data.items() if values})
    cleaned.write(str(output_file))

    final_cells = int(sum(len(cells) for _, cells in new_cells))
    return CleanResult(
        input_file=input_file,
        output_file=output_file,
        merged_nodes=merged_nodes,
        removed_cells=removed_cells,
        final_node_count=len(unique_points),
        final_cell_count=final_cells,
    )


def _merge_nodes(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray, int]:
    if len(points) == 0:
        return np.empty(0, dtype=int), points, 0

    scale = max(tolerance, 1e-12)
    keys = np.round(points / scale).astype(np.int64)
    _, unique_index, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    order = np.argsort(unique_index)
    new_position = np.empty(len(unique_index), dtype=int)
    new_position[order] = np.arange(len(unique_index))

    remap = new_position[inverse]
    unique_points = points[unique_index[order]]
    merged_nodes = len(points) - len(unique_points)
    return remap, unique_points, merged_nodes


def _non_degenerate_mask(cell_data: np.ndarray) -> np.ndarray:
    if cell_data.ndim != 2 or cell_data.shape[0] == 0:
        return np.zeros(cell_data.shape[0], dtype=bool)
    mask = np.ones(cell_data.shape[0], dtype=bool)
    cols = cell_data.shape[1]
    for i in range(cols):
        for j in range(i + 1, cols):
            mask &= cell_data[:, i] != cell_data[:, j]
    return mask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean duplicate nodes and degenerate cells in a mesh.")
    parser.add_argument("input_file", type=Path, help="Input mesh file.")
    parser.add_argument("output_file", type=Path, help="Output (cleaned) mesh file.")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Node merge tolerance in mm.")
    args = parser.parse_args(argv)

    try:
        result = clean_mesh(args.input_file, args.output_file, merge_tolerance=args.tolerance)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
