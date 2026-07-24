"""Small NumPy neural surrogate for geometry-to-stiffness prediction.

The model is intentionally dependency-light so the trained artifact can be
loaded by the Azure web container without a separate machine-learning runtime.
Inputs and outputs are standardized, then mapped by a two-hidden-layer tanh
network to Kx/Ky/Kz in N/mm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = ("inner_diameter_mm", "inner_core_length_mm", "outer_core_length_mm")
TARGET_NAMES = ("kx_n_per_mm", "ky_n_per_mm", "kz_n_per_mm")


@dataclass(frozen=True)
class StiffnessSurrogate:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    weights_1: np.ndarray
    bias_1: np.ndarray
    weights_2: np.ndarray
    bias_2: np.ndarray
    weights_3: np.ndarray
    bias_3: np.ndarray
    metadata: dict[str, Any]

    def predict(self, features: np.ndarray | list[list[float]] | list[float]) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} stiffness features, got {values.shape[1]}.")
        normalized = (values - self.feature_mean) / self.feature_scale
        hidden_1 = np.tanh(normalized @ self.weights_1 + self.bias_1)
        hidden_2 = np.tanh(hidden_1 @ self.weights_2 + self.bias_2)
        predicted = hidden_2 @ self.weights_3 + self.bias_3
        return predicted * self.target_scale + self.target_mean


def train_stiffness_surrogate(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_sizes: tuple[int, int] = (24, 12),
    epochs: int = 4000,
    learning_rate: float = 0.01,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> tuple[StiffnessSurrogate, dict[str, Any]]:
    """Train a compact MLP with Adam and deterministic validation splitting."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Features must have shape (samples, {len(FEATURE_NAMES)}).")
    if y.ndim != 2 or y.shape != (x.shape[0], len(TARGET_NAMES)):
        raise ValueError(f"Targets must have shape (samples, {len(TARGET_NAMES)}).")
    if x.shape[0] < 8:
        raise ValueError("At least 8 solved designs are required to train the stiffness surrogate.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Stiffness training data contains non-finite values.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(x.shape[0])
    validation_count = max(1, min(x.shape[0] - 4, int(round(x.shape[0] * validation_fraction))))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    x_train, y_train = x[training_indices], y[training_indices]
    x_validation, y_validation = x[validation_indices], y[validation_indices]

    feature_mean = x_train.mean(axis=0)
    feature_scale = _safe_scale(x_train.std(axis=0))
    target_mean = y_train.mean(axis=0)
    target_scale = _safe_scale(y_train.std(axis=0))
    x_train_n = (x_train - feature_mean) / feature_scale
    y_train_n = (y_train - target_mean) / target_scale
    x_validation_n = (x_validation - feature_mean) / feature_scale
    y_validation_n = (y_validation - target_mean) / target_scale

    h1, h2 = hidden_sizes
    parameters = {
        "w1": rng.normal(0.0, np.sqrt(2.0 / (x.shape[1] + h1)), size=(x.shape[1], h1)),
        "b1": np.zeros(h1),
        "w2": rng.normal(0.0, np.sqrt(2.0 / (h1 + h2)), size=(h1, h2)),
        "b2": np.zeros(h2),
        "w3": rng.normal(0.0, np.sqrt(2.0 / (h2 + y.shape[1])), size=(h2, y.shape[1])),
        "b3": np.zeros(y.shape[1]),
    }
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}

    best_parameters = {name: value.copy() for name, value in parameters.items()}
    best_validation_loss = float("inf")
    best_epoch = 0
    patience = max(250, epochs // 8)
    beta_1, beta_2, epsilon = 0.9, 0.999, 1e-8
    regularization = 1e-5

    for epoch in range(1, max(1, epochs) + 1):
        prediction, cache = _forward(x_train_n, parameters)
        error = prediction - y_train_n
        gradients = _backward(x_train_n, error, cache, parameters, regularization)
        for name, value in parameters.items():
            gradient = gradients[name]
            first_moment[name] = beta_1 * first_moment[name] + (1.0 - beta_1) * gradient
            second_moment[name] = beta_2 * second_moment[name] + (1.0 - beta_2) * gradient * gradient
            corrected_first = first_moment[name] / (1.0 - beta_1**epoch)
            corrected_second = second_moment[name] / (1.0 - beta_2**epoch)
            value -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)

        validation_prediction, _ = _forward(x_validation_n, parameters)
        validation_loss = float(np.mean((validation_prediction - y_validation_n) ** 2))
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_parameters = {name: value.copy() for name, value in parameters.items()}
        elif epoch - best_epoch >= patience:
            break

    metadata = {
        "model_type": "numpy_mlp",
        "feature_names": list(FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "hidden_sizes": list(hidden_sizes),
        "sample_count": int(x.shape[0]),
        "training_count": int(training_indices.size),
        "validation_count": int(validation_indices.size),
        "best_epoch": int(best_epoch),
        "validation_loss_standardized": float(best_validation_loss),
        "seed": seed,
    }
    model = StiffnessSurrogate(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        weights_1=best_parameters["w1"],
        bias_1=best_parameters["b1"],
        weights_2=best_parameters["w2"],
        bias_2=best_parameters["b2"],
        weights_3=best_parameters["w3"],
        bias_3=best_parameters["b3"],
        metadata=metadata,
    )
    metrics = regression_metrics(y_validation, model.predict(x_validation))
    metrics["training"] = metadata
    return model, metrics


def save_stiffness_surrogate(model: StiffnessSurrogate, output_file: Path | str) -> Path:
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        target_mean=model.target_mean,
        target_scale=model.target_scale,
        weights_1=model.weights_1,
        bias_1=model.bias_1,
        weights_2=model.weights_2,
        bias_2=model.bias_2,
        weights_3=model.weights_3,
        bias_3=model.bias_3,
        metadata=np.asarray(json.dumps(model.metadata)),
    )
    return output


def load_stiffness_surrogate(model_file: Path | str) -> StiffnessSurrogate:
    with np.load(Path(model_file), allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"].item()))
        return StiffnessSurrogate(
            feature_mean=artifact["feature_mean"],
            feature_scale=artifact["feature_scale"],
            target_mean=artifact["target_mean"],
            target_scale=artifact["target_scale"],
            weights_1=artifact["weights_1"],
            bias_1=artifact["bias_1"],
            weights_2=artifact["weights_2"],
            bias_2=artifact["bias_2"],
            weights_3=artifact["weights_3"],
            bias_3=artifact["bias_3"],
            metadata=metadata,
        )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute = np.abs(error)
    safe_actual = np.maximum(np.abs(actual), 1e-9)
    residual_sum = np.sum(error * error, axis=0)
    total_sum = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
    r_squared = np.where(total_sum > 1e-12, 1.0 - residual_sum / total_sum, 0.0)
    return {
        "mae_n_per_mm": {name: float(value) for name, value in zip(TARGET_NAMES, absolute.mean(axis=0), strict=True)},
        "mape_percent": {
            name: float(value)
            for name, value in zip(TARGET_NAMES, (absolute / safe_actual).mean(axis=0) * 100.0, strict=True)
        },
        "r_squared": {name: float(value) for name, value in zip(TARGET_NAMES, r_squared, strict=True)},
    }


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(scale, dtype=float) > 1e-12, scale, 1.0)


def _forward(x: np.ndarray, parameters: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    hidden_1 = np.tanh(x @ parameters["w1"] + parameters["b1"])
    hidden_2 = np.tanh(hidden_1 @ parameters["w2"] + parameters["b2"])
    return hidden_2 @ parameters["w3"] + parameters["b3"], (hidden_1, hidden_2)


def _backward(
    x: np.ndarray,
    error: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray],
    parameters: dict[str, np.ndarray],
    regularization: float,
) -> dict[str, np.ndarray]:
    hidden_1, hidden_2 = cache
    scale = 2.0 / max(x.shape[0] * error.shape[1], 1)
    output_gradient = error * scale
    gradients: dict[str, np.ndarray] = {}
    gradients["w3"] = hidden_2.T @ output_gradient + regularization * parameters["w3"]
    gradients["b3"] = output_gradient.sum(axis=0)
    hidden_2_gradient = (output_gradient @ parameters["w3"].T) * (1.0 - hidden_2 * hidden_2)
    gradients["w2"] = hidden_1.T @ hidden_2_gradient + regularization * parameters["w2"]
    gradients["b2"] = hidden_2_gradient.sum(axis=0)
    hidden_1_gradient = (hidden_2_gradient @ parameters["w2"].T) * (1.0 - hidden_1 * hidden_1)
    gradients["w1"] = x.T @ hidden_1_gradient + regularization * parameters["w1"]
    gradients["b1"] = hidden_1_gradient.sum(axis=0)
    return gradients
