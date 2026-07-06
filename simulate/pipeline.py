"""End-to-end local simulation pipeline: STEP -> natural frequencies.

This single command runs the full "best practical workflow":

    1. STEP  -> structured/swept hex volume mesh (geometry.hex_swept_mesh)
    2. clean -> merge nodes, drop bad cells     (geometry.mesh_cleaner)
    3. check -> mesh quality / solver readiness  (geometry.mesh_quality)
    4. solve -> CalculiX modal analysis          (simulate.modal_solver)
    5. report-> natural frequencies (Hz) + JSON  (simulate.results)

Material is taken from ``--material`` or auto-detected from a sibling
``spec.json`` produced by the Text-to-CAD generator.

Example
-------
    python -m simulate.pipeline outputs/phase_a/bracket/bracket.step \
        --output-dir outputs/simulation/bracket --modes 8 --boundary fixed_bottom
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

try:
    from geometry.hex_swept_mesh import StructuredHexUnavailable, step_to_swept_hex_mesh
    from geometry.mesh_cleaner import clean_mesh
    from geometry.mesh_quality import evaluate_mesh
    from simulate.materials import Material, resolve_material
    from simulate.modal_solver import ModalSetup, run_modal
    from simulate.results import parse_dat, write_report
except ModuleNotFoundError:  # pragma: no cover - allow `python simulate/pipeline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from geometry.hex_swept_mesh import StructuredHexUnavailable, step_to_swept_hex_mesh
    from geometry.mesh_cleaner import clean_mesh
    from geometry.mesh_quality import evaluate_mesh
    from simulate.materials import Material, resolve_material
    from simulate.modal_solver import ModalSetup, run_modal
    from simulate.results import parse_dat, write_report


@dataclass(frozen=True)
class PipelineConfig:
    step_file: Path
    output_dir: Path
    material: Material
    num_modes: int = 10
    boundary: str = "fixed_bottom"
    element_size_mm: float | None = None
    mesh_strategy: str = "hex"
    name: str = "part"
    contour_image: bool = True
    contour_mode: int = 1


def run_pipeline(config: PipelineConfig) -> int:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_mesh = config.output_dir / f"{config.name}.msh"
    clean_inp = config.output_dir / f"{config.name}_clean.vtk"

    print(f"[1/5] Meshing {config.step_file.name} ...")
    mesh_result, mesh_kind = _generate_volume_mesh(config, raw_mesh)
    generated_mesh = Path(mesh_result.mesh_file)
    print(f"      {_mesh_summary(mesh_result, mesh_kind)}")

    print("[2/5] Cleaning mesh ...")
    clean_result = clean_mesh(generated_mesh, clean_inp)
    print(f"      {clean_result.summary()}")

    print("[3/5] Checking mesh quality ...")
    quality = evaluate_mesh(clean_inp)
    print(f"      {quality.summary()}")
    if not quality.is_solvable:
        print(
            "      Warning: mesh has degenerate/inverted elements; "
            "results may be unreliable. Try a smaller --element-size.",
            file=sys.stderr,
        )

    print(f"[4/5] Solving modal analysis with CalculiX ({config.material.name}) ...")
    setup = ModalSetup(
        mesh_file=clean_inp,
        material=config.material,
        num_modes=config.num_modes,
        boundary=config.boundary,
    )
    run = run_modal(setup, config.output_dir, job_name=config.name)
    if not run.ok:
        if run.stderr:
            print(run.stderr, end="", file=sys.stderr)
        print("      CalculiX solve failed.", file=sys.stderr)
        return run.returncode or 1

    print("[5/5] Extracting natural frequencies ...")
    results = parse_dat(run.dat_file)
    report_path = write_report(results, config.output_dir / f"{config.name}_modal.json")
    pca_path = _try_pca(run.frd_file, config)

    contour_path = None
    if config.contour_image:
        contour_path = _try_contour(run.frd_file, config)

    print()
    print(results.summary())
    print()
    print(f"Mesh:    {clean_inp}")
    print(f"Deck:    {run.inp_file}")
    print(f"Results: {run.dat_file}")
    print(f"Report:  {report_path}")
    if pca_path is not None:
        print(f"PCA:     {pca_path}")
    if contour_path is not None:
        print(f"Contour: {contour_path}")
    print(f"View:    open {run.frd_file} in ParaView or CalculiX cgx")
    return 0


def _try_contour(frd_file: Path, config: "PipelineConfig") -> Path | None:
    """Render a von Mises contour PNG of the first mode; never fail the pipeline."""

    try:
        from simulate.visualize import render_contour
    except ImportError:
        from visualize import render_contour  # pragma: no cover

    output = config.output_dir / f"{config.name}_mode{config.contour_mode}_mises.png"
    try:
        return render_contour(
            frd_file,
            output,
            field_name="mises",
            mode=config.contour_mode,
            warp=True,
        )
    except Exception as exc:  # noqa: BLE001 - visualization is best-effort
        print(f"      Contour image skipped: {exc}", file=sys.stderr)
        return None


def _try_pca(frd_file: Path, config: "PipelineConfig") -> Path | None:
    """Write a modal displacement PCA JSON report; never fail the pipeline."""

    try:
        from simulate.pca import analyze_modal_pca, write_pca_report
    except ImportError:
        from pca import analyze_modal_pca, write_pca_report  # pragma: no cover

    output = config.output_dir / f"{config.name}_pca.json"
    try:
        result = analyze_modal_pca(frd_file, max_components=min(6, config.num_modes))
        return write_pca_report(result, output)
    except Exception as exc:  # noqa: BLE001 - PCA is best-effort post-processing
        print(f"      PCA skipped: {exc}", file=sys.stderr)
        return None


def _generate_volume_mesh(config: PipelineConfig, raw_mesh: Path):
    """Generate a structured/swept hexahedral volume mesh only."""

    strategy = (config.mesh_strategy or "hex").lower()
    if strategy != "hex":
        raise ValueError("This simulation pipeline is configured for hex-only meshing. Use mesh_strategy='hex'.")

    hex_mesh = raw_mesh.with_name(f"{raw_mesh.stem}_hex{raw_mesh.suffix}")
    try:
        return (
            step_to_swept_hex_mesh(
                step_file=config.step_file,
                output_file=hex_mesh,
                target_size_mm=config.element_size_mm,
            ),
            "structured_hex",
        )
    except StructuredHexUnavailable as exc:
        raise RuntimeError(_hex_only_error(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 - Gmsh can reject non-sweepable imported topology
        raise RuntimeError(_hex_only_error(str(exc))) from exc


def _hex_only_error(reason: str) -> str:
    return (
        "Hex-only meshing failed. Gmsh could not create hexahedral cells for this CAD topology. "
        "Use a sweepable/block-decomposed shape, simplify small fillets/branches, or split the part into mappable volumes. "
        f"Gmsh detail: {reason}"
    )


def _mesh_summary(mesh_result, mesh_kind: str) -> str:
    if hasattr(mesh_result, "summary"):
        return f"{mesh_kind}: {mesh_result.summary()}"
    element_counts = ", ".join(f"{name}={count}" for name, count in mesh_result.element_counts.items())
    return (
        f"{mesh_kind}: Mesh: {mesh_result.mesh_file.name} | nodes={mesh_result.node_count} "
        f"{element_counts} | element size {mesh_result.min_edge_mm:.3f}-{mesh_result.max_edge_mm:.3f} mm"
    )


def _material_from_spec(step_file: Path, explicit: str | None) -> Material:
    if explicit:
        return resolve_material(explicit)
    spec_path = step_file.parent / "spec.json"
    if spec_path.exists():
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            hint = spec.get("material_hint")
            if hint:
                return resolve_material(hint)
        except (json.JSONDecodeError, OSError):
            pass
    return resolve_material(None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full STEP-to-frequencies simulation pipeline.")
    parser.add_argument("step_file", type=Path, help="Input STEP/STP solid model.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write mesh/results.")
    parser.add_argument("--material", default=None, help="Material name/hint (else from spec.json).")
    parser.add_argument("--modes", type=int, default=10, help="Number of modes to extract.")
    parser.add_argument(
        "--boundary",
        default="fixed_bottom",
        choices=("free", "fixed_bottom", "fixed_top", "encastre"),
        help="Boundary condition preset.",
    )
    parser.add_argument("--element-size", type=float, default=None, help="Target element size in mm.")
    parser.add_argument(
        "--mesh-strategy",
        choices=("hex",),
        default="hex",
        help="Volume meshing strategy. The production pipeline is hex-only and fails if hex cells are unavailable.",
    )
    parser.add_argument("--name", default=None, help="Job basename (default: STEP stem).")
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="Skip the von Mises contour PNG (requires pyvista).",
    )
    parser.add_argument(
        "--contour-mode",
        type=int,
        default=1,
        help="Mode number to render in the contour image.",
    )
    args = parser.parse_args(argv)

    step_file = args.step_file.resolve()
    if not step_file.exists():
        print(f"STEP file not found: {step_file}", file=sys.stderr)
        return 2

    name = args.name or step_file.stem
    output_dir = (args.output_dir or Path("outputs") / "simulation" / name).resolve()
    material = _material_from_spec(step_file, args.material)

    config = PipelineConfig(
        step_file=step_file,
        output_dir=output_dir,
        material=material,
        num_modes=args.modes,
        boundary=args.boundary,
        element_size_mm=args.element_size,
        mesh_strategy=args.mesh_strategy,
        name=name,
        contour_image=not args.no_contour,
        contour_mode=args.contour_mode,
    )

    try:
        return run_pipeline(config)
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Install requirements.txt.", file=sys.stderr)
        return 3
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
