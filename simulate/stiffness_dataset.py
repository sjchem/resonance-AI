"""Generate a bushing design/FEM dataset and train the stiffness surrogate.

This is an offline workflow by design: each sample requires three CalculiX
static solves. The JSON checkpoint is updated after every case so a long run
can be inspected or resumed after interruption.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from geometry.bushing_hex_mesh import generate_bushing_hex_mesh
    from simulate.materials import resolve_material
    from simulate.shape_pca import encode_shape, fit_shape_pca, shape_pca_summary, write_shape_pca_model
    from simulate.static_stiffness import (
        StaticStiffnessSetup,
        effective_static_youngs_modulus_mpa,
        run_static_stiffness,
        static_stiffness_calibration,
    )
    from simulate.stiffness_surrogate import (
        FEATURE_NAMES,
        TARGET_NAMES,
        save_stiffness_surrogate,
        train_stiffness_surrogate,
    )
except ModuleNotFoundError:  # pragma: no cover - allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geometry.bushing_hex_mesh import generate_bushing_hex_mesh
    from simulate.materials import resolve_material
    from simulate.shape_pca import encode_shape, fit_shape_pca, shape_pca_summary, write_shape_pca_model
    from simulate.static_stiffness import (
        StaticStiffnessSetup,
        effective_static_youngs_modulus_mpa,
        run_static_stiffness,
        static_stiffness_calibration,
    )
    from simulate.stiffness_surrogate import (
        FEATURE_NAMES,
        TARGET_NAMES,
        save_stiffness_surrogate,
        train_stiffness_surrogate,
    )


@dataclass(frozen=True)
class DesignBounds:
    inner_diameter_min_mm: float = 21.0
    inner_diameter_max_mm: float = 35.0
    inner_core_length_min_mm: float = 20.0
    inner_core_length_max_mm: float = 71.0
    outer_core_length_min_mm: float = 20.0
    outer_core_length_max_mm: float = 55.0
    outer_diameter_mm: float = 76.0
    swaging_value_mm: float = 3.0
    decking_value_mm: float = 0.0


@dataclass(frozen=True)
class DatasetConfig:
    output_dir: Path
    sample_count: int = 200
    material: str = "rubber"
    displacement_mm: float = 1.0
    circumferential_divisions: int = 48
    radial_divisions: int = 4
    axial_divisions: int = 8
    shape_components: int = 6
    bounds: DesignBounds = DesignBounds()


def build_stiffness_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Generate compatible meshes, solve stiffness, fit PCA, and train MLP."""

    output_dir = Path(config.output_dir).resolve()
    mesh_dir = output_dir / "meshes"
    solve_dir = output_dir / "solves"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    solve_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "stiffness_dataset.json"
    template = {
        "circumferential_divisions": config.circumferential_divisions,
        "radial_divisions": config.radial_divisions,
        "axial_divisions": config.axial_divisions,
    }
    samples = design_samples(config.sample_count, config.bounds)
    existing = _load_checkpoint(checkpoint_path)
    material = resolve_material(config.material)
    calibration = static_stiffness_calibration(
        material.name,
        effective_static_youngs_modulus_mpa(
            StaticStiffnessSetup(mesh_file=Path("calibration.vtk"), material=material)
        ),
    )
    compatible_checkpoint = existing.get("stiffness_calibration") == calibration
    completed = {
        str(item.get("case_id")): item
        for item in existing.get("samples", [])
        if compatible_checkpoint and item.get("status") == "ok"
    }
    records: list[dict[str, Any]] = []

    for index, design in enumerate(samples, start=1):
        case_id = f"RB-{index:04d}"
        if case_id in completed:
            records.append(completed[case_id])
            continue
        intent = design_intent(design, config.bounds)
        mesh_file = mesh_dir / f"{case_id.lower()}.vtk"
        record: dict[str, Any] = {"case_id": case_id, "design": design, "status": "running"}
        try:
            mesh_result = generate_bushing_hex_mesh(intent, mesh_file, mesh_mode="global", template=template)
            stiffness = run_static_stiffness(
                StaticStiffnessSetup(
                    mesh_file=mesh_file,
                    material=material,
                    displacement_mm=config.displacement_mm,
                    inner_interface_length_mm=design["inner_core_length_mm"],
                    outer_interface_length_mm=design["outer_core_length_mm"],
                ),
                solve_dir / case_id.lower(),
                job_name=case_id.lower(),
            )
            record.update(
                {
                    "status": "ok",
                    "mesh_file": str(mesh_file.relative_to(output_dir)),
                    "template_id": mesh_result.template_id,
                    "node_count": mesh_result.node_count,
                    "hexahedra": mesh_result.hex_count,
                    "stiffness": {
                        "kx_n_per_mm": stiffness.kx_n_per_mm,
                        "ky_n_per_mm": stiffness.ky_n_per_mm,
                        "kz_n_per_mm": stiffness.kz_n_per_mm,
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve failures in long offline batches
            record.update({"status": "failed", "error": str(exc)})
        records.append(record)
        _write_checkpoint(checkpoint_path, config, records, template)

    successful = [item for item in records if item.get("status") == "ok"]
    if len(successful) < 8:
        final = _checkpoint_payload(config, records, template)
        final["error"] = "At least 8 successful FEM cases are required for surrogate training."
        checkpoint_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        return final

    _attach_shape_codes(successful, output_dir, config)
    features, targets = dataset_arrays(successful)
    model, metrics = train_stiffness_surrogate(features, targets)
    model.metadata["stiffness_calibration"] = calibration
    model_path = save_stiffness_surrogate(model, output_dir / "stiffness_model.npz")
    metrics_path = output_dir / "stiffness_model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "stiffness_dataset.csv", successful)

    final = _checkpoint_payload(config, records, template)
    final.update(
        {
            "successful_count": len(successful),
            "failed_count": len(records) - len(successful),
            "surrogate_model": str(model_path.relative_to(output_dir)),
            "surrogate_metrics": metrics,
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "axis_assumption": "Client X is bushing centerline; mesh Z maps to Kx.",
            "boundary_assumption": "Outer core fixed; inner core translated by 1 mm.",
            "stiffness_calibration": calibration,
        }
    )
    checkpoint_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    return final


def design_samples(count: int, bounds: DesignBounds) -> list[dict[str, float]]:
    samples = []
    for index in range(1, max(1, count) + 1):
        samples.append(
            {
                "outer_diameter_mm": bounds.outer_diameter_mm,
                "inner_diameter_mm": _lerp(
                    bounds.inner_diameter_min_mm, bounds.inner_diameter_max_mm, _halton(index, 2)
                ),
                "inner_core_length_mm": _lerp(
                    bounds.inner_core_length_min_mm, bounds.inner_core_length_max_mm, _halton(index, 3)
                ),
                "outer_core_length_mm": _lerp(
                    bounds.outer_core_length_min_mm, bounds.outer_core_length_max_mm, _halton(index, 5)
                ),
            }
        )
    return samples


def design_intent(design: dict[str, float], bounds: DesignBounds) -> dict[str, Any]:
    inner_length = float(design["inner_core_length_mm"])
    outer_length = float(design["outer_core_length_mm"])
    return {
        "part_type": "bushing",
        "material": {"name": "rubber"},
        "geometry": {
            "outer_diameter_mm": float(design["outer_diameter_mm"]),
            "inner_diameter_mm": float(design["inner_diameter_mm"]),
            "height_mm": max(inner_length, outer_length),
            "inner_core_length_mm": inner_length,
            "outer_core_length_mm": outer_length,
            "swaging_value_mm": bounds.swaging_value_mm,
            "decking_value_mm": bounds.decking_value_mm,
            "internal_teeth": False,
            "slot_count": 0,
            "bore_shape": "round",
        },
    }


def dataset_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(
        [[float(item["design"][name]) for name in FEATURE_NAMES] for item in records],
        dtype=float,
    )
    targets = np.asarray(
        [[float(item["stiffness"][name]) for name in TARGET_NAMES] for item in records],
        dtype=float,
    )
    return features, targets


def _attach_shape_codes(records: list[dict[str, Any]], output_dir: Path, config: DatasetConfig) -> None:
    mesh_files = [output_dir / str(item["mesh_file"]) for item in records]
    template_id = str(records[0].get("template_id") or "")
    model = fit_shape_pca(mesh_files, components=config.shape_components, template_id=template_id)
    write_shape_pca_model(model, output_dir / "shape_pca_model.npz")
    for item, mesh_file in zip(records, mesh_files, strict=True):
        alpha = encode_shape(mesh_file, model)
        item["shape_codes"] = [float(value) for value in alpha]
    summary = shape_pca_summary(model, np.zeros(model.component_count))
    (output_dir / "shape_pca_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _write_csv(output_file: Path, records: list[dict[str, Any]]) -> None:
    max_components = max((len(item.get("shape_codes", [])) for item in records), default=0)
    fieldnames = [
        "case_id",
        *FEATURE_NAMES,
        *TARGET_NAMES,
        *[f"pc{index + 1}" for index in range(max_components)],
    ]
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            row: dict[str, Any] = {"case_id": item["case_id"]}
            row.update({name: item["design"][name] for name in FEATURE_NAMES})
            row.update({name: item["stiffness"][name] for name in TARGET_NAMES})
            row.update({f"pc{index + 1}": value for index, value in enumerate(item.get("shape_codes", []))})
            writer.writerow(row)


def _checkpoint_payload(
    config: DatasetConfig,
    records: list[dict[str, Any]],
    template: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "global_mesh_template": template,
        "stiffness_calibration": _dataset_calibration(config.material),
        "samples": records,
    }


def _dataset_calibration(material_name: str) -> dict[str, Any]:
    material = resolve_material(material_name)
    setup = StaticStiffnessSetup(mesh_file=Path("calibration.vtk"), material=material)
    return static_stiffness_calibration(material.name, effective_static_youngs_modulus_mpa(setup))


def _write_checkpoint(
    output_file: Path,
    config: DatasetConfig,
    records: list[dict[str, Any]],
    template: dict[str, int],
) -> None:
    output_file.write_text(
        json.dumps(_checkpoint_payload(config, records, template), indent=2) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(output_file: Path) -> dict[str, Any]:
    if not output_file.exists():
        return {}
    try:
        return json.loads(output_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _halton(index: int, base: int) -> float:
    fraction = 1.0
    result = 0.0
    value = max(1, int(index))
    while value:
        fraction /= base
        result += fraction * (value % base)
        value //= base
    return result


def _lerp(start: float, end: float, fraction: float) -> float:
    return float(start + (end - start) * fraction)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Resonance AI bushing stiffness dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "stiffness_dataset")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--material", default="rubber")
    parser.add_argument("--circumferential", type=int, default=48)
    parser.add_argument("--radial", type=int, default=4)
    parser.add_argument("--axial", type=int, default=8)
    parser.add_argument("--shape-components", type=int, default=6)
    args = parser.parse_args(argv)
    config = DatasetConfig(
        output_dir=args.output_dir,
        sample_count=max(8, args.samples),
        material=args.material,
        circumferential_divisions=args.circumferential,
        radial_divisions=args.radial,
        axial_divisions=args.axial,
        shape_components=max(1, args.shape_components),
    )
    result = build_stiffness_dataset(config)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    return 0 if result.get("successful_count", 0) >= 8 else 2


if __name__ == "__main__":
    raise SystemExit(main())
