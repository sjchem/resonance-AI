"""PCA analysis for CalculiX modal displacement results.

The modal solver writes one displacement field per mode into the ``.frd`` file.
This module treats each mode shape as one sample, flattens nodal XYZ
displacements into a feature vector, normalizes the arbitrary modal scale, and
computes a compact PCA summary with NumPy SVD.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

try:
    from simulate.visualize import parse_frd
except ModuleNotFoundError:  # pragma: no cover - allow direct execution
    from visualize import parse_frd


@dataclass(frozen=True)
class PcaComponent:
    """Compact PCA component summary."""

    component: int
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    dominant_mode: int
    dominant_frequency_hz: float
    dominant_axis: str
    axis_energy: dict[str, float]
    characteristic: str
    score_min: float
    score_max: float


@dataclass(frozen=True)
class ModePcaScore:
    """Scores for one FEM mode in PCA space."""

    mode_number: int
    frequency_hz: float
    scores: list[float]


@dataclass(frozen=True)
class ModalPcaResult:
    """PCA summary for a modal ``.frd`` result."""

    frd_file: Path
    mode_count: int
    node_count: int
    feature_count: int
    component_count: int
    components: list[PcaComponent]
    mode_scores: list[ModePcaScore]

    def to_dict(self) -> dict:
        return {
            "frd_file": str(self.frd_file),
            "mode_count": self.mode_count,
            "node_count": self.node_count,
            "feature_count": self.feature_count,
            "component_count": self.component_count,
            "components": [asdict(component) for component in self.components],
            "mode_scores": [asdict(score) for score in self.mode_scores],
        }


def analyze_modal_pca(frd_file: Path, *, max_components: int = 6) -> ModalPcaResult:
    """Compute PCA over modal displacement vectors from a CalculiX ``.frd`` file."""

    frd_file = Path(frd_file).resolve()
    mesh, fields = parse_frd(frd_file)
    disp_fields = sorted((field for field in fields if field.kind == "DISP"), key=lambda field: field.mode)
    if len(disp_fields) < 2:
        raise ValueError("PCA requires at least two displacement mode shapes in the FRD file.")

    matrix = np.vstack([_normalized_mode_vector(field.data) for field in disp_fields])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    component_count = max(1, min(int(max_components), len(disp_fields), vt.shape[0]))
    scores = centered @ vt[:component_count].T

    variances = (singular_values**2) / max(len(disp_fields) - 1, 1)
    total_variance = float(variances.sum())
    ratios = variances / total_variance if total_variance > 0 else np.zeros_like(variances)
    cumulative = np.cumsum(ratios)

    components: list[PcaComponent] = []
    for index in range(component_count):
        component_scores = scores[:, index]
        dominant_index = int(np.argmax(np.abs(component_scores)))
        dominant_field = disp_fields[dominant_index]
        axis_energy = _axis_energy(vt[index], mesh.points.shape[0])
        dominant_axis = max(axis_energy, key=axis_energy.get)
        components.append(
            PcaComponent(
                component=index + 1,
                explained_variance_ratio=float(ratios[index]),
                cumulative_variance_ratio=float(cumulative[index]),
                dominant_mode=int(dominant_field.mode),
                dominant_frequency_hz=float(dominant_field.frequency_hz),
                dominant_axis=dominant_axis,
                axis_energy=axis_energy,
                characteristic=_characteristic_label(index + 1, ratios[index], dominant_axis),
                score_min=float(component_scores.min()),
                score_max=float(component_scores.max()),
            )
        )

    mode_scores = [
        ModePcaScore(
            mode_number=int(field.mode),
            frequency_hz=float(field.frequency_hz),
            scores=[float(value) for value in scores[row_index]],
        )
        for row_index, field in enumerate(disp_fields)
    ]

    return ModalPcaResult(
        frd_file=frd_file,
        mode_count=len(disp_fields),
        node_count=int(mesh.points.shape[0]),
        feature_count=int(matrix.shape[1]),
        component_count=component_count,
        components=components,
        mode_scores=mode_scores,
    )


def write_pca_report(result: ModalPcaResult, output_file: Path) -> Path:
    """Write a PCA summary to JSON."""

    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return output_file


def _normalized_mode_vector(displacements: np.ndarray) -> np.ndarray:
    vector = np.asarray(displacements, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    if vector.size:
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            vector = -vector
    return vector


def _axis_energy(component_vector: np.ndarray, node_count: int) -> dict[str, float]:
    """Return normalized component energy in global X/Y/Z displacement axes."""

    if node_count <= 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    vectors = np.asarray(component_vector, dtype=float).reshape(node_count, 3)
    energy = np.sum(vectors * vectors, axis=0)
    total = float(energy.sum())
    if total <= 0.0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return {"x": float(energy[0] / total), "y": float(energy[1] / total), "z": float(energy[2] / total)}


def _characteristic_label(component: int, variance_ratio: float, dominant_axis: str) -> str:
    axis_label = {"x": "X lateral", "y": "Y lateral", "z": "Z axial"}.get(dominant_axis, dominant_axis.upper())
    if component == 1:
        prefix = "Primary mode-family variation"
    elif variance_ratio >= 0.15:
        prefix = "Secondary mode-family variation"
    else:
        prefix = "Detail/local mode variation"
    return f"{prefix} ({axis_label} dominant)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute PCA over CalculiX modal displacement shapes.")
    parser.add_argument("frd_file", type=Path, help="CalculiX .frd file from a modal solve.")
    parser.add_argument("--components", type=int, default=6, help="Maximum number of PCA components to report.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    try:
        result = analyze_modal_pca(args.frd_file, max_components=args.components)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for component in result.components:
        print(
            f"PC{component.component}: {component.explained_variance_ratio * 100:.1f}% "
            f"(cum {component.cumulative_variance_ratio * 100:.1f}%), "
            f"dominant mode {component.dominant_mode} @ {component.dominant_frequency_hz:.3f} Hz"
        )
    if args.json is not None:
        path = write_pca_report(result, args.json)
        print(f"Wrote PCA report to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
