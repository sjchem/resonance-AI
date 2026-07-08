"""Shape PCA encoding for global-mesh bushing geometry.

This module implements the standard PCA shape model:

    X ~= X_mean + sum(alpha_m * sqrt(lambda_m) * W_m)
    alpha_m = W_m^T (X - X_mean) / sqrt(lambda_m)

Unlike simulate.pca, which analyzes modal displacement shapes, this module
works on global mesh node coordinates. All meshes must share the same node order
and cell connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShapePcaModel:
    mean_shape: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cells: list[tuple[str, np.ndarray]]
    template_id: str
    node_count: int
    sample_count: int

    @property
    def component_count(self) -> int:
        return int(self.components.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.mean_shape.size)


def fit_shape_pca(mesh_files: list[Path | str], *, components: int = 10, template_id: str = "") -> ShapePcaModel:
    """Fit a PCA model to meshes with identical nodes/connectivity."""

    if len(mesh_files) < 2:
        raise ValueError("Shape PCA requires at least two global meshes.")

    vectors: list[np.ndarray] = []
    reference_cells: list[tuple[str, np.ndarray]] | None = None
    reference_signature: tuple[Any, ...] | None = None

    for mesh_file in mesh_files:
        points, cells = _read_mesh(Path(mesh_file))
        signature = _connectivity_signature(cells)
        if reference_signature is None:
            reference_signature = signature
            reference_cells = cells
        elif signature != reference_signature:
            raise ValueError("Shape PCA requires identical mesh connectivity for every sample.")
        vectors.append(points.reshape(-1))

    matrix = np.vstack(vectors)
    mean_shape = matrix.mean(axis=0)
    centered = matrix - mean_shape
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    max_components = max(1, min(int(components), len(mesh_files) - 1, vt.shape[0]))
    eigenvalues = (singular_values[:max_components] ** 2) / max(len(mesh_files) - 1, 1)
    all_eigenvalues = (singular_values**2) / max(len(mesh_files) - 1, 1)
    total = float(all_eigenvalues.sum())
    ratios = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)

    return ShapePcaModel(
        mean_shape=mean_shape,
        components=vt[:max_components],
        eigenvalues=eigenvalues,
        explained_variance_ratio=ratios,
        cells=reference_cells or [],
        template_id=template_id,
        node_count=int(mean_shape.size // 3),
        sample_count=len(mesh_files),
    )


def encode_shape(mesh_file: Path | str, model: ShapePcaModel) -> np.ndarray:
    """Encode one mesh into normalized PCA alpha coefficients."""

    points, cells = _read_mesh(Path(mesh_file))
    if _connectivity_signature(cells) != _connectivity_signature(model.cells):
        raise ValueError("Shape connectivity does not match the PCA model template.")
    vector = points.reshape(-1)
    centered = vector - model.mean_shape
    alpha = []
    for component, eigenvalue in zip(model.components, model.eigenvalues, strict=False):
        scale = float(np.sqrt(max(eigenvalue, 1e-18)))
        alpha.append(float(np.dot(component, centered) / scale))
    return np.asarray(alpha, dtype=float)


def reconstruct_shape(alpha: np.ndarray | list[float], model: ShapePcaModel) -> np.ndarray:
    """Reconstruct node coordinates from normalized PCA alpha coefficients."""

    alpha_array = np.asarray(alpha, dtype=float)
    count = min(alpha_array.size, model.component_count)
    vector = model.mean_shape.copy()
    for index in range(count):
        vector += alpha_array[index] * np.sqrt(max(float(model.eigenvalues[index]), 0.0)) * model.components[index]
    return vector.reshape(model.node_count, 3)


def reconstruction_metrics(original_mesh: Path | str, reconstructed_points: np.ndarray) -> dict[str, float]:
    """Return RMS error and a diagonal-normalized reconstruction precision."""

    original_points, _ = _read_mesh(Path(original_mesh))
    diff = np.asarray(reconstructed_points, dtype=float) - original_points
    rms_error = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    spans = np.ptp(original_points, axis=0)
    diagonal = float(np.linalg.norm(spans)) or 1.0
    precision = max(0.0, min(100.0, 100.0 * (1.0 - rms_error / diagonal)))
    return {
        "rms_error_mm": rms_error,
        "max_error_mm": float(np.sqrt(np.sum(diff * diff, axis=1)).max()),
        "precision_percent": precision,
    }


def write_reconstructed_mesh(points: np.ndarray, model: ShapePcaModel, output_file: Path | str) -> Path:
    """Write reconstructed points using the model's shared connectivity."""

    import meshio

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    meshio.write(str(output), meshio.Mesh(np.asarray(points, dtype=float), model.cells))
    return output


def write_shape_pca_model(model: ShapePcaModel, output_file: Path | str) -> Path:
    """Save the fitted shape PCA model as a compact NPZ artifact."""

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    cell_types = np.asarray([cell_type for cell_type, _ in model.cells], dtype=object)
    np.savez_compressed(
        output,
        mean_shape=model.mean_shape,
        components=model.components,
        eigenvalues=model.eigenvalues,
        explained_variance_ratio=model.explained_variance_ratio,
        cell_types=cell_types,
        **{f"cell_data_{index}": data for index, (_, data) in enumerate(model.cells)},
        template_id=np.asarray(model.template_id),
        node_count=np.asarray(model.node_count),
        sample_count=np.asarray(model.sample_count),
    )
    return output


def shape_pca_summary(model: ShapePcaModel, alpha: np.ndarray) -> dict[str, Any]:
    cumulative = np.cumsum(model.explained_variance_ratio)
    return {
        "template_id": model.template_id,
        "sample_count": model.sample_count,
        "node_count": model.node_count,
        "feature_count": model.feature_count,
        "component_count": model.component_count,
        "components": [
            {
                "component": index + 1,
                "eigenvalue": float(model.eigenvalues[index]),
                "explained_variance_ratio": float(model.explained_variance_ratio[index]),
                "cumulative_variance_ratio": float(cumulative[index]),
                "alpha": float(alpha[index]) if index < alpha.size else 0.0,
            }
            for index in range(model.component_count)
        ],
        "alpha": [float(value) for value in alpha[: model.component_count]],
        "formula": "X ~= X_mean + sum(alpha_m * sqrt(lambda_m) * W_m)",
        "encoding_formula": "alpha_m = W_m^T (X - X_mean) / sqrt(lambda_m)",
    }


def _read_mesh(mesh_file: Path) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    import meshio

    mesh = meshio.read(str(mesh_file))
    points = np.asarray(mesh.points, dtype=float)
    cells = [(block.type, np.asarray(block.data, dtype=int).copy()) for block in mesh.cells]
    return points, cells


def _connectivity_signature(cells: list[tuple[str, np.ndarray]]) -> tuple[Any, ...]:
    return tuple((cell_type, data.shape, data.tobytes()) for cell_type, data in cells)
