"""Quality metrics for tetrahedral and hexahedral volume meshes.

Run after :mod:`geometry.step_to_mesh` to confirm a mesh is good enough to
solve. Poor element quality (slivers, near-degenerate tetrahedra) is the most
common cause of bad or non-converging FE results, so this stage gives a quick,
solver-independent health check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


# A tetrahedron's normalized quality. 1.0 is a perfect regular tetra, values
# near 0.0 are degenerate slivers. Anything below this is flagged.
DEFAULT_MIN_QUALITY = 0.05


@dataclass(frozen=True)
class MeshQualityReport:
    """Aggregate quality statistics for a solid mesh."""

    mesh_file: Path
    node_count: int
    tetra_count: int
    hexa_count: int
    min_quality: float
    mean_quality: float
    min_volume_mm3: float
    inverted_count: int
    poor_count: int
    quality_threshold: float

    @property
    def is_solvable(self) -> bool:
        return (self.tetra_count + self.hexa_count) > 0 and self.inverted_count == 0 and self.poor_count == 0

    def summary(self) -> str:
        verdict = "OK" if self.is_solvable else "NEEDS ATTENTION"
        return (
            f"Quality [{verdict}]: tetra={self.tetra_count} hex={self.hexa_count} "
            f"min={self.min_quality:.3f} mean={self.mean_quality:.3f} "
            f"inverted={self.inverted_count} poor(<{self.quality_threshold:g})={self.poor_count} "
            f"min_volume={self.min_volume_mm3:.4g} mm^3"
        )


def evaluate_mesh(mesh_file: Path, *, quality_threshold: float = DEFAULT_MIN_QUALITY) -> MeshQualityReport:
    """Compute quality statistics for a tetrahedral mesh file."""

    import meshio

    mesh_file = Path(mesh_file).resolve()
    if not mesh_file.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    mesh = meshio.read(str(mesh_file))
    points = np.asarray(mesh.points, dtype=float)

    cells = _collect_tetra_cells(mesh)
    hex_cells = _collect_hex_cells(mesh)
    if cells.size == 0 and hex_cells.size == 0:
        return MeshQualityReport(
            mesh_file=mesh_file,
            node_count=len(points),
            tetra_count=0,
            hexa_count=0,
            min_quality=0.0,
            mean_quality=0.0,
            min_volume_mm3=0.0,
            inverted_count=0,
            poor_count=0,
            quality_threshold=quality_threshold,
        )

    quality_parts = []
    volume_parts = []
    inverted_count = 0

    if cells.size:
        p0 = points[cells[:, 0]]
        p1 = points[cells[:, 1]]
        p2 = points[cells[:, 2]]
        p3 = points[cells[:, 3]]

        signed_volume = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
        volume = np.abs(signed_volume)
        quality = _tetra_quality(p0, p1, p2, p3, volume)
        quality_parts.append(quality)
        volume_parts.append(volume)
        inverted_count += int(np.count_nonzero(signed_volume <= 0.0))

    if hex_cells.size:
        hex_points = points[hex_cells]
        volume = _hex_volume(hex_points)
        quality = _hex_edge_quality(hex_points)
        quality_parts.append(quality)
        volume_parts.append(volume)
        inverted_count += int(np.count_nonzero(volume <= 0.0))

    all_quality = np.concatenate(quality_parts)
    all_volume = np.concatenate(volume_parts)
    poor_count = int(np.count_nonzero(all_quality < quality_threshold))

    return MeshQualityReport(
        mesh_file=mesh_file,
        node_count=len(points),
        tetra_count=int(cells.shape[0]),
        hexa_count=int(hex_cells.shape[0]),
        min_quality=float(all_quality.min()),
        mean_quality=float(all_quality.mean()),
        min_volume_mm3=float(all_volume.min()),
        inverted_count=inverted_count,
        poor_count=poor_count,
        quality_threshold=quality_threshold,
    )


def _collect_tetra_cells(mesh) -> np.ndarray:
    blocks = []
    for block in mesh.cells:
        if block.type in ("tetra", "tetra10"):
            # Only the four corner nodes matter for quality.
            blocks.append(np.asarray(block.data, dtype=int)[:, :4])
    if not blocks:
        return np.empty((0, 4), dtype=int)
    return np.vstack(blocks)


def _collect_hex_cells(mesh) -> np.ndarray:
    blocks = []
    for block in mesh.cells:
        if block.type in ("hexahedron", "hexahedron20", "hexahedron27"):
            # Only the eight corner nodes matter for volume/edge checks.
            blocks.append(np.asarray(block.data, dtype=int)[:, :8])
    if not blocks:
        return np.empty((0, 8), dtype=int)
    return np.vstack(blocks)


def _tetra_quality(p0, p1, p2, p3, volume) -> np.ndarray:
    """Normalized shape quality based on volume vs. mean edge length.

    quality = 12 * (3 * volume)^(2/3) / sum(edge_length^2)

    The constant scales a regular tetrahedron to a quality of 1.0.
    """

    edges = [
        p1 - p0,
        p2 - p0,
        p3 - p0,
        p2 - p1,
        p3 - p1,
        p3 - p2,
    ]
    edge_sq_sum = sum(np.einsum("ij,ij->i", e, e) for e in edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = 12.0 * np.power(3.0 * volume, 2.0 / 3.0)
        quality = np.where(edge_sq_sum > 0.0, numerator / edge_sq_sum, 0.0)
    return np.clip(quality, 0.0, 1.0)


def _hex_volume(points: np.ndarray) -> np.ndarray:
    """Approximate hex volume by decomposing each corner hex into tetrahedra."""

    tet_indices = np.asarray(
        (
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 3, 4, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
        ),
        dtype=int,
    )
    volume = np.zeros(points.shape[0], dtype=float)
    for a, b, c, d in tet_indices:
        signed = np.einsum("ij,ij->i", np.cross(points[:, b] - points[:, a], points[:, c] - points[:, a]), points[:, d] - points[:, a]) / 6.0
        volume += np.abs(signed)
    return volume


def _hex_edge_quality(points: np.ndarray) -> np.ndarray:
    """Simple 0..1 edge-length quality for hexes; 1 means all edges equal."""

    edge_pairs = np.asarray(
        (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ),
        dtype=int,
    )
    lengths = np.linalg.norm(points[:, edge_pairs[:, 0]] - points[:, edge_pairs[:, 1]], axis=2)
    min_edge = lengths.min(axis=1)
    max_edge = lengths.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        quality = np.where(max_edge > 0.0, min_edge / max_edge, 0.0)
    return np.clip(quality, 0.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check tetrahedral mesh quality.")
    parser.add_argument("mesh_file", type=Path, help="Mesh file (.msh, .inp, .vtk).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_MIN_QUALITY,
        help="Minimum acceptable element quality (0-1).",
    )
    args = parser.parse_args(argv)

    try:
        report = evaluate_mesh(args.mesh_file, quality_threshold=args.threshold)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(report.summary())
    return 0 if report.is_solvable else 1


if __name__ == "__main__":
    raise SystemExit(main())
