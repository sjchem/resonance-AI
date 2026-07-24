"""FastAPI app for OpenAI-powered CAD prompt parsing."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.cad_preview import build_preview_svg
from app.chat_logic import respond_to_cad_chat
from app.openai_client import (
    CADValidationError,
    OpenAIConfigurationError,
    OpenAIRequestError,
    openai_status,
    parse_cad_prompt,
)
from app.schemas import CADChatRequest, CADChatResponse, CADPromptOutput, CADPromptRequest
from app.upload_context import UploadContext, build_upload_context, persist_uploaded_geometry_bytes, uploaded_geometry_path
from cad_backends.openscad_backend import (
    OpenScadExportError,
    OpenScadUnavailable,
    generate_openscad_bushing,
    run_openscad_export,
    write_parameters_json,
)


app = FastAPI(title="Resonance AI", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MAX_FEM_MODES = 100


@app.get("/")
async def health_check() -> dict[str, str]:
    """Simple health check for the deployed web app."""

    return {"status": "ok", "service": "openai-cad-prompt-parser"}


@app.get("/ui", response_class=HTMLResponse)
async def ui() -> str:
    """Frontend for prompt parsing and CAD-style preview."""

    return UI_HTML


@app.get("/generate", response_class=HTMLResponse)
async def generate_ui() -> str:
    """Compatibility route for the existing Azure Web App URL."""

    return UI_HTML


@app.get("/models")
async def models() -> dict:
    """Return OpenAI provider configuration without exposing secrets."""

    return openai_status()


@app.post("/parse-cad", response_model=CADPromptOutput)
async def parse_cad(request: CADPromptRequest) -> CADPromptOutput:
    """Parse a natural-language CAD prompt into validated structured JSON."""

    try:
        return await parse_cad_prompt(request.prompt)
    except OpenAIConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CADValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/chat-cad", response_model=CADChatResponse)
async def chat_cad(request: CADChatRequest) -> CADChatResponse:
    """Return an interactive chat reply plus the current CAD state."""

    try:
        return await respond_to_cad_chat(
            request.message,
            request.prompt,
            request.history,
            request.knowledge_sources,
        )
    except OpenAIConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except CADValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/generate-cad")
async def generate_cad(request: CADPromptRequest) -> dict:
    """Parse prompt JSON and return a lightweight preview payload."""

    parsed = await parse_cad(request)
    return {
        "cad_intent": parsed.model_dump(),
        "preview_svg": build_preview_svg(parsed),
        "note": "Interactive preview only for the current web POC. STEP/STL export is not implemented in this deployed flow yet.",
    }


@app.post("/preview-cad")
async def preview_cad(parsed: CADPromptOutput) -> dict:
    """Return a CAD-style preview for already structured input."""

    return {
        "cad_intent": parsed.model_dump(),
        "preview_svg": build_preview_svg(parsed),
        "note": "Interactive preview only for the current web POC. STEP/STL export is not implemented in this deployed flow yet.",
    }


@app.post("/generate-parametric-cad")
async def generate_parametric_cad(payload: dict) -> dict:
  """Accept structured parametric CAD JSON and return a preview-ready payload."""

  cad_engine = str(payload.get("cad_engine", "cadquery")).strip().lower()
  if cad_engine not in {"cadquery", "openscad"}:
    raise HTTPException(status_code=400, detail="CAD engine must be cadquery or openscad.")

  intent = payload.get("intent")
  if not isinstance(intent, dict):
    raise HTTPException(status_code=400, detail="A structured CAD intent JSON object is required.")

  part_type = str(intent.get("part_type", "")).strip().lower()
  if part_type not in {"bushing", "rubber_mount"}:
    raise HTTPException(status_code=422, detail="Parametric CAD generation currently supports rubber bushing geometry.")

  geometry = intent.get("geometry") if isinstance(intent.get("geometry"), dict) else {}
  try:
    outer = float(geometry.get("outer_diameter_mm") or 0)
    inner = float(geometry.get("inner_diameter_mm") or 0)
    height = float(geometry.get("height_mm") or 0)
  except (TypeError, ValueError) as exc:
    raise HTTPException(status_code=422, detail="Outer diameter, inner diameter, and height must be numeric.") from exc
  if outer <= 0 or inner <= 0 or height <= 0:
    raise HTTPException(status_code=422, detail="Outer diameter, inner diameter, and height must be positive.")
  if inner >= outer:
    raise HTTPException(status_code=422, detail="Inner diameter must be smaller than outer diameter.")

  return {
    "cad_engine": cad_engine,
    "cad_intent": intent,
    "preview_ready": True,
    "download_formats": ["step", "stl", "png", "json"]
    if cad_engine == "cadquery"
    else ["scad", "stl", "png", "json"],
  }


@app.post("/upload-context", response_model=UploadContext)
async def upload_context(file: UploadFile = File(...)) -> UploadContext:
    """Extract prompt context from an uploaded document, image, or CAD file."""

    try:
        return await build_upload_context(file)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read uploaded file: {exc}") from exc


def _safe_export_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "model").strip()).strip("_")
    return cleaned[:60] or "model"


@app.post("/export/step")
async def export_step(payload: dict) -> FileResponse:
    """Generate a real STEP file from the CAD prompt via the CadQuery pipeline.

    STEP is a B-Rep exchange format that cannot be produced from the browser
    preview mesh, so it is generated server-side. This requires the optional
    CadQuery dependency; when it is not installed the endpoint returns 501 and
    the UI falls back to the client-side formats (STL, GLB, DXF, PNG, PDF, JSON).
    """

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="A prompt is required to generate a STEP file.")

    name = _safe_export_name(str(payload.get("name", "model")))

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from text_to_cad.cad_agent import generate_with_agent
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                "STEP export needs the CadQuery pipeline, which is not installed in this "
                "deployment. Use STL, GLB, DXF, PNG, PDF, or JSON instead."
            ),
        ) from exc

    output_dir = Path(tempfile.mkdtemp(prefix="resonance_step_"))
    try:
        code = generate_with_agent(
            prompt=prompt,
            output_dir=output_dir,
            output_name=name,
            provider="auto",
            execute=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure to the client
        raise HTTPException(status_code=500, detail=f"STEP generation failed: {exc}") from exc

    step_path = output_dir / f"{name}.step"
    if code != 0 or not step_path.exists():
        raise HTTPException(status_code=500, detail="STEP generation did not produce a file.")

    return FileResponse(step_path, media_type="application/step", filename=f"{name}.step")


@app.post("/export/simulation-report")
async def export_simulation_report(payload: dict) -> StreamingResponse:
    """Create an Excel workbook containing compact CAD and simulation results."""

    name = _safe_export_name(str(payload.get("name", "model")))
    try:
        from app.simulation_export import build_simulation_workbook
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail="Excel report export requires the openpyxl package in the deployed web image.",
        ) from exc

    try:
        workbook = build_simulation_workbook(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - return a useful POC export error
        raise HTTPException(status_code=500, detail=f"Simulation report generation failed: {exc}") from exc

    filename = f"{name}_simulation_report.xlsx"
    return StreamingResponse(
        workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@app.post("/export/openscad")
async def export_openscad(payload: dict) -> FileResponse:
    """Generate OpenSCAD bushing artifacts from the current parameter JSON."""

    export_format = str(payload.get("format", "scad")).strip().lower().lstrip(".")
    if export_format not in {"scad", "stl", "png", "json"}:
        raise HTTPException(status_code=400, detail="OpenSCAD export format must be scad, stl, png, or json.")

    intent = payload.get("intent")
    if not isinstance(intent, dict):
        raise HTTPException(status_code=400, detail="OpenSCAD export requires the current bushing parameter JSON.")

    name = _safe_export_name(str(payload.get("name", "bushing")))
    output_dir = Path(tempfile.mkdtemp(prefix="resonance_openscad_"))
    scad_path = output_dir / f"{name}.scad"
    parameters_path = output_dir / "parameters.json"

    try:
        bushing = generate_openscad_bushing(intent)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    scad_path.write_text(bushing.scad_text, encoding="utf-8")
    write_parameters_json(parameters_path, bushing.parameters)

    if export_format == "scad":
        return FileResponse(scad_path, media_type="text/plain", filename=f"{name}.scad")
    if export_format == "json":
        return FileResponse(parameters_path, media_type="application/json", filename="parameters.json")

    try:
        result = run_openscad_export(scad_path, output_dir)
    except OpenScadUnavailable as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except OpenScadExportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if export_format == "stl" and result.stl_path:
        return FileResponse(result.stl_path, media_type="model/stl", filename=f"{name}.stl")
    if export_format == "png" and result.png_path:
        return FileResponse(result.png_path, media_type="image/png", filename=f"{name}.png")

    warning = result.warnings[-1] if result.warnings else "OpenSCAD did not produce the requested output file."
    raise HTTPException(status_code=500, detail=warning)


@app.post("/generate-mesh")
async def generate_mesh(payload: dict) -> dict:
    """Generate a Gmsh volume mesh and return quality/readiness stats."""

    prompt = str(payload.get("prompt", "")).strip()
    name = _safe_export_name(str(payload.get("name", "model")))
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    geometry_source = str(payload.get("geometry_source") or "").strip().lower()
    # An uploaded STEP/STL always wins over a parametric mesh hint. This also
    # protects requests sent by an older cached frontend that still includes
    # the former bushing surrogate flag.
    uploaded_geometry = _uploaded_geometry_from_payload(payload)
    if not uploaded_geometry and not prompt and not _is_structured_bushing_intent(intent):
        raise HTTPException(status_code=400, detail="A prompt is required to generate a mesh.")

    element_size = payload.get("element_size_mm")
    try:
        element_size_mm = float(element_size) if element_size not in (None, "") else None
    except (TypeError, ValueError):
        element_size_mm = None
    if element_size_mm is not None and element_size_mm <= 0:
        element_size_mm = None
    mesh_mode = _mesh_mode_from_payload(payload)
    global_template = payload.get("global_template") if isinstance(payload.get("global_template"), dict) else {}

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from geometry.bushing_hex_mesh import generate_bushing_hex_mesh  # noqa: WPS433
        from geometry.hex_swept_mesh import (  # noqa: WPS433
            StructuredHexUnavailable,
            step_to_swept_hex_mesh,
        )
        from geometry.mesh_cleaner import clean_mesh  # noqa: WPS433
        from geometry.mesh_quality import evaluate_mesh  # noqa: WPS433
        from geometry.uploaded_volume_mesh import uploaded_geometry_to_volume_mesh  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail="Gmsh meshing requires the deployed CAD/FEM container dependencies.",
        ) from exc

    work_dir = Path(tempfile.mkdtemp(prefix="resonance_mesh_"))
    try:
        raw_mesh = work_dir / f"{name}_hex.msh"
        step_path: Path | None = None
        fallback_reason = None
        if uploaded_geometry:
            raw_mesh = work_dir / f"{name}_uploaded_hex.msh"
            try:
                mesh_result = uploaded_geometry_to_volume_mesh(
                    uploaded_geometry,
                    raw_mesh,
                    target_size_mm=element_size_mm,
                    mesh_mode=mesh_mode,
                    template=global_template,
                )
            except Exception as exc:  # noqa: BLE001 - exact FEM requires a real closed volume
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Uploaded STL/STEP all-hexa meshing failed after the Gmsh repair and "
                        f"reparametrization stages. Gmsh detail: {exc}"
                    ),
                ) from exc
            step_path = uploaded_geometry
            mesh_strategy = mesh_result.mesh_kind
            mode_label = "global-density" if mesh_mode == "global" else "automatic-density"
            if mesh_result.mesh_kind.startswith("uploaded_geometry_voxel"):
                mesh_format = f"Uploaded {mesh_result.source_format} voxel-repaired all-hexa mesh ({mode_label})"
                fallback_reason = (
                    "Body-fitted Gmsh meshing could not create solver-safe hexahedra, so the uploaded "
                    f"surface was reconstructed as a {mesh_result.voxel_pitch_mm:.2f} mm voxel C3D8 mesh. "
                    "This follows the uploaded shape approximately and does not use OD/ID/height parameters."
                )
            else:
                mesh_format = f"Uploaded {mesh_result.source_format} body-fitted all-hexa mesh ({mode_label})"
        elif _is_structured_bushing_intent(intent):
            raw_mesh = work_dir / f"{name}_hex.vtk"
            mesh_result = generate_bushing_hex_mesh(
                intent,
                raw_mesh,
                target_size_mm=element_size_mm,
                mesh_mode=mesh_mode,
                template=global_template,
            )
            mesh_strategy = mesh_result.mesh_kind
            if mesh_result.global_compatible:
                mesh_format = "Global dataset-compatible hexahedral bushing mesh"
            else:
                mesh_format = "Mapped structured hexahedral slotted-bushing mesh" if mesh_result.mesh_kind == "mapped_slotted_bushing_hex" else "Mapped structured hexahedral bushing mesh"
        else:
            try:
                from text_to_cad.cad_agent import generate_with_agent  # noqa: WPS433
            except ModuleNotFoundError as exc:
                raise HTTPException(
                    status_code=501,
                    detail="Prompt-based meshing requires the CadQuery/text-to-CAD pipeline. Upload STEP/STL for exact uploaded-geometry meshing.",
                ) from exc

            cad_code = -1
            last_exc: Exception | None = None
            for provider in ("auto", "fallback"):
                try:
                    cad_code = generate_with_agent(
                        prompt=prompt,
                        output_dir=work_dir,
                        output_name=name,
                        provider=provider,
                        execute=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    continue
                step_path_attempt = work_dir / f"{name}.step"
                if cad_code == 0 and step_path_attempt.exists():
                    break

            step_path = work_dir / f"{name}.step"
            if cad_code != 0 or not step_path.exists():
                detail = "STEP generation did not produce a file for meshing."
                if last_exc is not None:
                    detail += f" Last error: {last_exc}"
                raise HTTPException(status_code=500, detail=detail)

            mesh_format = "Gmsh structured hex/swept volume mesh"
            mesh_strategy = "structured_hex"
            try:
                mesh_result = step_to_swept_hex_mesh(
                    step_file=step_path,
                    output_file=raw_mesh,
                    target_size_mm=element_size_mm,
                )
            except (StructuredHexUnavailable, Exception) as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Hex-only meshing failed. Gmsh could not create hexahedral cells for this CAD topology. "
                        "Use a sweepable/block-decomposed shape, simplify small fillets/branches, or split the part into mappable volumes. "
                        f"Gmsh detail: {exc}"
                    ),
                ) from exc

        clean_mesh_path = work_dir / f"{name}_clean.vtk"
        clean_result = clean_mesh(raw_mesh, clean_mesh_path)
        cell_counts = _mesh_cell_counts(clean_mesh_path)
        quality = evaluate_mesh(clean_mesh_path)
        quality_payload = _quality_payload(quality, cell_counts)
        surface_mesh = _mesh_surface_preview(clean_mesh_path)
        hexahedra = _count_hex_cells(cell_counts)
        tetrahedra = _count_tetra_cells(cell_counts) or getattr(mesh_result, "tetra_count", 0)

        return {
            "status": "ok",
            "mesh_format": mesh_format,
            "mesh_strategy": mesh_strategy,
            "mesh_source": "exact_uploaded_geometry" if uploaded_geometry else ("bushing_poc_hex" if geometry_source == "bushing_poc_hex" else "generated_or_structured_geometry"),
            "fallback_reason": fallback_reason,
            "step_file": step_path.name if step_path else "structured_bushing_parameters",
            "mesh_file": raw_mesh.name,
            "clean_mesh_file": clean_mesh_path.name,
            "nodes": mesh_result.node_count,
            "tetrahedra": tetrahedra,
            "hexahedra": hexahedra,
            "cell_counts": cell_counts,
            "global_mesh": _global_mesh_payload(mesh_result),
            "min_edge_mm": mesh_result.min_edge_mm,
            "max_edge_mm": mesh_result.max_edge_mm,
            "voxel_pitch_mm": getattr(mesh_result, "voxel_pitch_mm", 0.0),
            "cleaning": {
                "merged_nodes": clean_result.merged_nodes,
                "removed_cells": clean_result.removed_cells,
                "final_node_count": clean_result.final_node_count,
                "final_cell_count": clean_result.final_cell_count,
            },
            "quality": {
                "ready_for_fem": quality_payload["ready_for_fem"],
                "min_quality": quality_payload["min_quality"],
                "mean_quality": quality_payload["mean_quality"],
                "inverted_count": quality_payload["inverted_count"],
                "poor_count": quality_payload["poor_count"],
                "min_volume_mm3": quality_payload["min_volume_mm3"],
            },
            "surface_mesh": surface_mesh,
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Gmsh mesh generation failed: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/shape-pca")
async def run_shape_pca(payload: dict) -> dict:
    """Fit/encode/reconstruct geometry PCA from global bushing mesh nodes."""

    name = _safe_export_name(str(payload.get("name", "bushing_shape")))
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    if not _is_structured_bushing_intent(intent):
        raise HTTPException(status_code=400, detail="Shape PCA requires structured Rubber bushing geometry.")

    try:
        sample_count = max(4, min(40, int(payload.get("samples", 12) or 12)))
    except (TypeError, ValueError):
        sample_count = 12
    try:
        component_count = max(1, min(10, int(payload.get("components", 10) or 10)))
    except (TypeError, ValueError):
        component_count = 10
    global_template = payload.get("global_template") if isinstance(payload.get("global_template"), dict) else {}

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from geometry.bushing_hex_mesh import generate_bushing_hex_mesh  # noqa: WPS433
        from simulate.shape_pca import (  # noqa: WPS433
            encode_shape,
            fit_shape_pca,
            reconstruction_metrics,
            reconstruct_shape,
            shape_pca_summary,
            write_reconstructed_mesh,
            write_shape_pca_model,
        )
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=501, detail="Shape PCA requires meshio and NumPy dependencies.") from exc

    work_dir = Path(tempfile.mkdtemp(prefix="resonance_shape_pca_"))
    try:
        variants = _shape_pca_variants(intent, sample_count)
        mesh_files: list[Path] = []
        base_mesh = work_dir / "shape_base.vtk"
        base_result = generate_bushing_hex_mesh(
            intent,
            base_mesh,
            mesh_mode="global",
            template=global_template,
        )
        template_id = base_result.template_id

        for index, variant in enumerate(variants):
            mesh_path = work_dir / f"shape_sample_{index:03d}.vtk"
            result = generate_bushing_hex_mesh(
                variant,
                mesh_path,
                mesh_mode="global",
                template=global_template,
            )
            if result.template_id != template_id:
                raise HTTPException(status_code=422, detail="Generated shape PCA samples did not share one global mesh template.")
            mesh_files.append(mesh_path)

        model = fit_shape_pca(mesh_files, components=component_count, template_id=template_id)
        alpha = encode_shape(base_mesh, model)
        reconstructed = reconstruct_shape(alpha, model)
        metrics = reconstruction_metrics(base_mesh, reconstructed)

        artifact_dir = repo_root / "outputs" / "shape_pca" / name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = write_shape_pca_model(model, artifact_dir / f"{name}_shape_pca.npz")
        reconstructed_path = write_reconstructed_mesh(reconstructed, model, artifact_dir / f"{name}_reconstructed.vtk")
        summary = shape_pca_summary(model, alpha)
        summary.update(
            {
                "reconstruction": metrics,
                "model_file": str(model_path.relative_to(repo_root)),
                "reconstructed_mesh_file": str(reconstructed_path.relative_to(repo_root)),
                "global_mesh": _global_mesh_payload(base_result),
            }
        )
        (artifact_dir / f"{name}_shape_pca.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Shape PCA failed: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/run-static-stiffness")
async def run_static_stiffness_api(payload: dict) -> dict:
    """Run directional Kx/Ky/Kz static FEM for a structured rubber bushing."""

    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    if not _is_structured_bushing_intent(intent):
        raise HTTPException(
            status_code=400,
            detail="Directional stiffness currently requires structured Rubber bushing geometry.",
        )
    geometry = intent.get("geometry") if isinstance(intent.get("geometry"), dict) else {}
    material_value = payload.get("material")
    if not material_value:
        intent_material = intent.get("material")
        material_value = intent_material.get("name") if isinstance(intent_material, dict) else intent_material
    material_name = str(material_value or "rubber")
    try:
        displacement_mm = float(payload.get("displacement_mm", 1.0) or 1.0)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Displacement must be numeric.") from exc
    if displacement_mm <= 0 or displacement_mm > 5:
        raise HTTPException(status_code=422, detail="Displacement must be greater than 0 and no more than 5 mm.")

    inner_length = _float_value(
        geometry.get("inner_core_length_mm"),
        _float_value(geometry.get("height_mm"), 40.0),
    )
    outer_length = _float_value(
        geometry.get("outer_core_length_mm"),
        _float_value(geometry.get("height_mm"), 40.0),
    )
    global_template = payload.get("global_template") if isinstance(payload.get("global_template"), dict) else {}
    name = _safe_export_name(str(payload.get("name", "bushing")))
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from geometry.bushing_hex_mesh import generate_bushing_hex_mesh  # noqa: WPS433
        from simulate.materials import resolve_material  # noqa: WPS433
        from simulate.static_stiffness import StaticStiffnessSetup, run_static_stiffness  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail="Static stiffness requires the deployed meshio, NumPy, and CalculiX dependencies.",
        ) from exc
    if shutil.which("ccx") is None:
        raise HTTPException(status_code=501, detail="Static stiffness requires CalculiX ('ccx').")

    work_dir = Path(tempfile.mkdtemp(prefix="resonance_static_stiffness_"))
    try:
        mesh_file = work_dir / f"{name}_global.vtk"
        mesh_result = generate_bushing_hex_mesh(
            intent,
            mesh_file,
            mesh_mode="global",
            template=global_template,
        )
        result = run_static_stiffness(
            StaticStiffnessSetup(
                mesh_file=mesh_file,
                material=resolve_material(material_name),
                displacement_mm=displacement_mm,
                inner_interface_length_mm=inner_length,
                outer_interface_length_mm=outer_length,
            ),
            work_dir / "solve",
            job_name=name,
        )
        response = result.as_dict()
        response.update(
            {
                "status": "ok",
                "mesh": _global_mesh_payload(mesh_result),
                "design": {
                    "outer_diameter_mm": _float_value(geometry.get("outer_diameter_mm"), 76.0),
                    "inner_diameter_mm": _float_value(geometry.get("inner_diameter_mm"), 28.0),
                    "inner_core_length_mm": inner_length,
                    "outer_core_length_mm": outer_length,
                },
                "model_limitations": (
                    "Client-calibrated linear-elastic isotropic POC. The effective rubber modulus matches the "
                    "supplied stiffness scale; validate interface assumptions and large-strain behavior before "
                    "engineering release."
                ),
            }
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Static stiffness FEM failed: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/stiffness-model")
async def stiffness_model_status() -> dict:
    """Report whether a trained geometry-to-stiffness surrogate is available."""

    model_path = _stiffness_model_path()
    if not model_path.exists():
        return {
            "status": "not_trained",
            "model_file": str(model_path),
            "message": "Run python -m simulate.stiffness_dataset to build the FEM dataset and model.",
        }
    try:
        from simulate.stiffness_surrogate import load_stiffness_surrogate  # noqa: WPS433

        model = load_stiffness_surrogate(model_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not load stiffness model: {exc}") from exc
    metadata = dict(model.metadata)
    metadata.setdefault("stiffness_calibration", _legacy_stiffness_calibration())
    return {"status": "ready", "model_file": str(model_path), "metadata": metadata}


@app.get("/stiffness-dashboard-data")
async def stiffness_dashboard_data() -> dict:
    """Return PCA/design/stiffness points from the generated training dataset."""

    dataset_path = _stiffness_dataset_path()
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="No stiffness training dataset is installed.")
    try:
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read stiffness dataset: {exc}") from exc
    artifact_scale = _stiffness_artifact_scale(payload.get("stiffness_calibration"))
    samples = [
        {
            "case_id": item.get("case_id"),
            "design": item.get("design"),
            "stiffness": _scaled_stiffness(item.get("stiffness"), artifact_scale),
            "shape_codes": item.get("shape_codes", []),
        }
        for item in payload.get("samples", [])
        if item.get("status") == "ok"
    ]
    return {
        "status": "ready",
        "sample_count": len(samples),
        "samples": samples[:5000],
        "surrogate_metrics": _scaled_surrogate_metrics(payload.get("surrogate_metrics"), artifact_scale),
        "axis_assumption": payload.get("axis_assumption"),
        "boundary_assumption": payload.get("boundary_assumption"),
        "stiffness_calibration": payload.get("stiffness_calibration") or _legacy_stiffness_calibration(),
    }


@app.post("/search-stiffness")
async def search_stiffness(payload: dict) -> dict:
    """Search client design bounds with the trained neural surrogate."""

    model_path = _stiffness_model_path()
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No trained stiffness model is installed. Build the offline FEM dataset first.",
        )
    try:
        from simulate.stiffness_dataset import DesignBounds, design_intent, design_samples  # noqa: WPS433
        from simulate.stiffness_surrogate import FEATURE_NAMES, load_stiffness_surrogate  # noqa: WPS433

        model = load_stiffness_surrogate(model_path)
        targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
        target = [
            float(targets.get("kx_n_per_mm", 88.4)),
            float(targets.get("ky_n_per_mm", 294.5)),
            float(targets.get("kz_n_per_mm", 294.5)),
        ]
        bounds_payload = payload.get("bounds") if isinstance(payload.get("bounds"), dict) else {}
        bounds = DesignBounds(
            inner_diameter_min_mm=float(bounds_payload.get("inner_diameter_min_mm", 21.0)),
            inner_diameter_max_mm=float(bounds_payload.get("inner_diameter_max_mm", 35.0)),
            inner_core_length_min_mm=float(bounds_payload.get("inner_core_length_min_mm", 20.0)),
            inner_core_length_max_mm=float(bounds_payload.get("inner_core_length_max_mm", 71.0)),
            outer_core_length_min_mm=float(bounds_payload.get("outer_core_length_min_mm", 20.0)),
            outer_core_length_max_mm=float(bounds_payload.get("outer_core_length_max_mm", 55.0)),
            outer_diameter_mm=76.0,
            swaging_value_mm=3.0,
            decking_value_mm=0.0,
        )
        sample_count = max(50, min(5000, int(payload.get("samples", 2000) or 2000)))
        candidates = design_samples(sample_count, bounds)
        import numpy as np

        feature_values = np.asarray(
            [[float(candidate[name]) for name in FEATURE_NAMES] for candidate in candidates],
            dtype=float,
        )
        predictions = model.predict(feature_values)
        artifact_scale = _stiffness_artifact_scale(model.metadata.get("stiffness_calibration"))
        predictions = predictions * artifact_scale
        target_values = np.asarray(target, dtype=float)
        relative_errors = np.abs(predictions - target_values) / np.maximum(np.abs(target_values), 1.0)
        scores = np.mean(relative_errors * relative_errors, axis=1)
        best_index = int(np.argmin(scores))
        best_design = candidates[best_index]
        best_prediction = predictions[best_index]
        best_errors = relative_errors[best_index]
        best_intent = design_intent(best_design, bounds)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid stiffness search input: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Stiffness model search failed: {exc}") from exc

    return {
        "status": "ok",
        "source": "trained_static_fem_surrogate",
        "case_id": f"NN-{best_index + 1:04d}",
        "intent": best_intent,
        "design": best_design,
        "predicted_stiffness": {
            "kx_n_per_mm": float(best_prediction[0]),
            "ky_n_per_mm": float(best_prediction[1]),
            "kz_n_per_mm": float(best_prediction[2]),
        },
        "max_relative_error": float(best_errors.max()),
        "rms_relative_error": float(np.sqrt(np.mean(best_errors * best_errors))),
        "within_tolerance": bool(float(best_errors.max()) <= 0.1),
        "sample_count": sample_count,
        "model_metadata": {
            **model.metadata,
            "stiffness_calibration": (
                model.metadata.get("stiffness_calibration") or _legacy_stiffness_calibration()
            ),
        },
        "stiffness_calibration": model.metadata.get("stiffness_calibration") or _legacy_stiffness_calibration(),
    }


def _stiffness_artifact_dir() -> Path:
    configured = os.getenv("STIFFNESS_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "models" / "stiffness"


def _stiffness_model_path() -> Path:
    return _stiffness_artifact_dir() / "stiffness_model.npz"


def _stiffness_dataset_path() -> Path:
    return _stiffness_artifact_dir() / "stiffness_dataset.json"


def _legacy_stiffness_calibration() -> dict:
    from simulate.static_stiffness import (  # noqa: WPS433
        CLIENT_CALIBRATED_RUBBER_E_MPA,
        CLIENT_CALIBRATION_ID,
        CLIENT_REFERENCE_TARGETS_N_PER_MM,
    )

    return {
        "id": CLIENT_CALIBRATION_ID,
        "status": "legacy_artifact_scaled",
        "youngs_modulus_mpa": CLIENT_CALIBRATED_RUBBER_E_MPA,
        "reference_targets_n_per_mm": CLIENT_REFERENCE_TARGETS_N_PER_MM,
        "note": "Legacy 10 MPa artifact scaled to the client-calibrated 1.10 MPa stiffness basis.",
    }


def _stiffness_artifact_scale(calibration: object) -> float:
    from simulate.static_stiffness import (  # noqa: WPS433
        CLIENT_CALIBRATED_RUBBER_E_MPA,
        CLIENT_CALIBRATION_ID,
        LEGACY_RUBBER_E_MPA,
    )

    if isinstance(calibration, dict) and calibration.get("id") == CLIENT_CALIBRATION_ID:
        return 1.0
    return CLIENT_CALIBRATED_RUBBER_E_MPA / LEGACY_RUBBER_E_MPA


def _scaled_stiffness(stiffness: object, scale: float) -> object:
    if not isinstance(stiffness, dict) or scale == 1.0:
        return stiffness
    return {
        key: float(value) * scale if key in {"kx_n_per_mm", "ky_n_per_mm", "kz_n_per_mm"} else value
        for key, value in stiffness.items()
    }


def _scaled_surrogate_metrics(metrics: object, scale: float) -> object:
    if not isinstance(metrics, dict) or scale == 1.0:
        return metrics
    scaled = dict(metrics)
    mae = metrics.get("mae_n_per_mm")
    if isinstance(mae, dict):
        scaled["mae_n_per_mm"] = {key: float(value) * scale for key, value in mae.items()}
    return scaled


def _should_try_structured_hex(prompt: str, intent: dict) -> bool:
    """Use hex-first only for shapes that are likely sweepable/axisymmetric."""

    part_type = str(intent.get("part_type") or "").lower()
    text = f"{prompt} {part_type}".lower()
    axisymmetric_tokens = (
        "bushing",
        "rubber_mount",
        "rubber mount",
        "spring",
        "air spring",
        "coil spring",
        "cylindrical",
        "axisymmetric",
    )
    return any(token in text for token in axisymmetric_tokens)


def _is_structured_bushing_intent(intent: dict) -> bool:
    """True when the frontend sent editable rubber-bushing parameter JSON."""

    if not isinstance(intent, dict):
        return False
    part_type = str(intent.get("part_type") or "").lower()
    geometry = intent.get("geometry") if isinstance(intent.get("geometry"), dict) else {}
    try:
        outer_diameter = float(geometry.get("outer_diameter_mm") or 0)
        inner_diameter = float(geometry.get("inner_diameter_mm") or 0)
        height = float(geometry.get("height_mm") or 0)
    except (TypeError, ValueError):
        return False
    return part_type in {"bushing", "rubber_mount"} and outer_diameter > 0 and inner_diameter > 0 and height > 0


def _mesh_mode_from_payload(payload: dict) -> str:
    mode = str(payload.get("mesh_mode") or "structured").strip().lower()
    return "global" if mode in {"global", "global_template", "dataset"} else "structured"


def _uploaded_geometry_from_payload(payload: dict) -> Path | None:
  upload_id = str(payload.get("upload_id") or "").strip()
  path = uploaded_geometry_path(upload_id) if upload_id else None
  if path is None:
    path = _uploaded_geometry_from_inline_payload(payload)
  if path is None:
    return None if not upload_id else _raise_missing_uploaded_geometry()
  if path.suffix.lower() not in {".step", ".stp", ".stl"}:
    raise HTTPException(status_code=422, detail="Exact uploaded-geometry FEM currently supports STEP/STP and watertight STL files.")
  return path


def _uploaded_geometry_from_inline_payload(payload: dict) -> Path | None:
  encoded = str(payload.get("upload_data_base64") or "").strip()
  if not encoded:
    return None
  filename = str(payload.get("upload_filename") or "uploaded_geometry.stl").strip() or "uploaded_geometry.stl"
  content_type = str(payload.get("upload_content_type") or "application/octet-stream").strip() or None
  try:
    raw = base64.b64decode(encoded, validate=True)
    return persist_uploaded_geometry_bytes(filename, content_type, raw)
  except Exception as exc:  # noqa: BLE001
    raise HTTPException(status_code=422, detail=f"Uploaded geometry fallback bytes could not be read: {exc}") from exc


def _raise_missing_uploaded_geometry() -> None:
  raise HTTPException(
    status_code=404,
    detail=(
      "Uploaded geometry was not found on this server worker. The request may have reached a different Azure "
      "container instance than the upload request. Refresh and upload the STEP/STL again; if this continues, "
      "set RESONANCE_UPLOAD_DIR to a shared writable path such as /home/resonance-ai/uploads."
    ),
  )


def _global_mesh_payload(mesh_result) -> dict:
    mesh_kind = str(getattr(mesh_result, "mesh_kind", "") or "")
    settings_only = mesh_kind in {
        "uploaded_geometry_global_hex",
        "uploaded_geometry_voxel_global_hex",
    }
    template_id = getattr(mesh_result, "template_id", "") or None
    return {
        "enabled": bool(getattr(mesh_result, "global_compatible", False) or settings_only),
        "template_id": template_id,
        "circumferential_divisions": getattr(mesh_result, "circumferential_divisions", 0),
        "radial_divisions": getattr(mesh_result, "radial_divisions", 0),
        "axial_divisions": getattr(mesh_result, "axial_divisions", 0),
        "shared_connectivity": bool(getattr(mesh_result, "global_compatible", False)),
        "settings_only": settings_only,
    }


def _shape_pca_variants(intent: dict, sample_count: int) -> list[dict]:
    """Create topology-compatible design variants for global-mesh Shape PCA."""

    base = json.loads(json.dumps(intent))
    geometry = base.get("geometry") if isinstance(base.get("geometry"), dict) else {}
    outer = _float_value(geometry.get("outer_diameter_mm"), 76.0)
    inner = min(_float_value(geometry.get("inner_diameter_mm"), 28.0), outer - 1.0)
    height = _float_value(geometry.get("height_mm"), 40.0)
    slot_depth = _float_value(geometry.get("slot_depth_mm"), max(1.0, (outer - inner) * 0.25))
    slot_width = _float_value(geometry.get("slot_width_deg"), 18.0)
    corner = _float_value(geometry.get("bore_corner_radius_mm"), 4.0)

    variants: list[dict] = []
    count = max(4, sample_count)
    for index in range(count):
        phase = index / max(count - 1, 1)
        wave = ((index * 37) % count) / max(count - 1, 1)
        variant = json.loads(json.dumps(base))
        geom = variant.setdefault("geometry", {})
        geom["outer_diameter_mm"] = round(outer * (0.92 + 0.16 * phase), 4)
        geom["inner_diameter_mm"] = round(max(1.0, inner * (0.94 + 0.12 * wave)), 4)
        geom["height_mm"] = round(height * (0.88 + 0.24 * ((phase + wave) % 1.0)), 4)
        if int(geom.get("slot_count") or 0) > 0:
            geom["slot_depth_mm"] = round(max(0.1, slot_depth * (0.70 + 0.60 * wave)), 4)
            geom["slot_width_deg"] = round(max(1.0, slot_width * (0.75 + 0.50 * phase)), 4)
        if str(geom.get("bore_shape") or "round") == "rounded_square":
            geom["bore_corner_radius_mm"] = round(max(0.0, corner * (0.75 + 0.50 * ((phase * 0.6 + wave * 0.4) % 1.0))), 4)
        if geom["inner_diameter_mm"] >= geom["outer_diameter_mm"]:
            geom["inner_diameter_mm"] = max(1.0, geom["outer_diameter_mm"] - 1.0)
        variants.append(variant)
    return variants


def _float_value(value, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed == parsed else fallback


def _mesh_cell_counts(mesh_path: Path) -> dict[str, int]:
    """Return mesh cell counts by meshio cell type."""

    import meshio

    mesh = meshio.read(str(mesh_path))
    counts: dict[str, int] = {}
    for block in mesh.cells:
        counts[block.type] = counts.get(block.type, 0) + len(block.data)
    return counts


def _count_hex_cells(cell_counts: dict[str, int]) -> int:
    return sum(count for name, count in cell_counts.items() if name.startswith("hexahedron"))


def _count_tetra_cells(cell_counts: dict[str, int]) -> int:
    return sum(count for name, count in cell_counts.items() if name.startswith("tetra"))


def _quality_payload(quality, cell_counts: dict[str, int]) -> dict:
    """Normalize quality output for tetrahedral or hexahedral volume meshes."""

    hex_count = _count_hex_cells(cell_counts)
    tetra_count = _count_tetra_cells(cell_counts)
    if tetra_count > 0 or hex_count > 0:
        return {
            "ready_for_fem": quality.is_solvable,
            "min_quality": quality.min_quality,
            "mean_quality": quality.mean_quality,
            "inverted_count": quality.inverted_count,
            "poor_count": quality.poor_count,
            "min_volume_mm3": quality.min_volume_mm3,
        }
    return {
        "ready_for_fem": False,
        "min_quality": None,
        "mean_quality": None,
        "inverted_count": 0,
        "poor_count": 0,
        "min_volume_mm3": None,
    }


def _mesh_surface_preview(mesh_path: Path, *, max_faces: int = 6000) -> dict:
    """Extract exterior mesh triangles for the browser preview."""

    import meshio
    import numpy as np

    mesh = meshio.read(str(mesh_path))
    points = np.asarray(mesh.points, dtype=float)
    face_map: dict[tuple[int, ...], dict] = {}
    volumes: list[float] = []

    for block in mesh.cells:
        spec = _cell_preview_spec(block.type)
        if spec is None:
            continue
        cells = np.asarray(block.data, dtype=int)
        if cells.size == 0:
            continue
        corners = cells[:, : spec["corner_count"]]
        for cell in corners:
            volume = _cell_volume(points, cell, spec["volume_tets"])
            volumes.append(volume)
            for local_face in spec["faces"]:
                face = tuple(int(cell[i]) for i in local_face)
                key = tuple(sorted(face))
                if key in face_map:
                    face_map[key]["count"] += 1
                else:
                    face_map[key] = {"count": 1, "face": face, "value": volume}

    exterior = [item for item in face_map.values() if item["count"] == 1]
    if max_faces > 0 and len(exterior) > max_faces:
        stride = max(1, len(exterior) // max_faces)
        exterior = exterior[::stride][:max_faces]

    vmin = min(volumes) if volumes else 0.0
    vmax = max(volumes) if volumes else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    faces = []
    for item in exterior:
        value = float(item["value"])
        color = _contour_hex(value, vmin, vmax)
        face = item["face"]
        triangles = (face,) if len(face) == 3 else ((face[0], face[1], face[2]), (face[0], face[2], face[3]))
        for tri in triangles:
            faces.append(
                {
                    "color": color,
                    "value": value,
                    "points": [
                        {"x": float(points[node_id][0]), "y": float(points[node_id][1]), "z": float(points[node_id][2])}
                        for node_id in tri
                    ],
                }
            )

    return {
        "faces": faces,
        "field": "Volume",
        "unit": "mm3",
        "scalar_min": float(vmin),
        "scalar_max": float(vmax),
        "face_count": len(faces),
    }


def _cell_preview_spec(cell_type: str) -> dict | None:
    """Return corner count, exterior faces and volume decomposition for a cell."""

    if cell_type in ("tetra", "tetra10"):
        return {
            "corner_count": 4,
            "faces": ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
            "volume_tets": ((0, 1, 2, 3),),
        }
    if cell_type.startswith("hexahedron"):
        return {
            "corner_count": 8,
            "faces": ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
            "volume_tets": ((0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6), (1, 5, 4, 6), (3, 7, 4, 6)),
        }
    if cell_type.startswith("wedge"):
        return {
            "corner_count": 6,
            "faces": ((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),
            "volume_tets": ((0, 1, 2, 3), (1, 2, 4, 3), (2, 4, 5, 3)),
        }
    return None


def _cell_volume(points, cell, volume_tets) -> float:
    """Approximate a cell volume by tetrahedral decomposition."""

    import numpy as np

    volume = 0.0
    for a, b, c, d in volume_tets:
        p0 = points[int(cell[a])]
        p1 = points[int(cell[b])]
        p2 = points[int(cell[c])]
        p3 = points[int(cell[d])]
        volume += abs(float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0))
    return volume


def _contour_hex(value: float, vmin: float, vmax: float) -> str:
    """Rainbow contour colour used by the mesh and FEM previews."""

    span = vmax - vmin or 1.0
    t = max(0.0, min(1.0, (value - vmin) / span))
    stops = [
        (0.00, (32, 25, 156)),
        (0.18, (0, 91, 255)),
        (0.36, (0, 200, 255)),
        (0.54, (64, 220, 104)),
        (0.70, (255, 235, 59)),
        (0.84, (255, 135, 0)),
        (1.00, (204, 0, 0)),
    ]
    for idx in range(1, len(stops)):
        left_t, left_rgb = stops[idx - 1]
        right_t, right_rgb = stops[idx]
        if t <= right_t:
            local = (t - left_t) / (right_t - left_t or 1.0)
            rgb = tuple(round(left_rgb[i] + (right_rgb[i] - left_rgb[i]) * local) for i in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#{:02x}{:02x}{:02x}".format(*stops[-1][1])


@app.post("/run-fem")
async def run_fem(payload: dict) -> dict:
    """Run the full FEM modal pipeline and return a von Mises contour PNG.

    Requires the CadQuery + gmsh + PyVista Python stack plus the native
    CalculiX 'ccx' solver. When any of these is missing in a deployment, the
    endpoint returns 501 and the UI keeps the analytical estimate.
    """

    prompt = str(payload.get("prompt", "")).strip()
    name = _safe_export_name(str(payload.get("name", "model")))
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    geometry_source = str(payload.get("geometry_source") or "").strip().lower()
    uploaded_geometry = _uploaded_geometry_from_payload(payload)
    if not uploaded_geometry and not prompt and not _is_structured_bushing_intent(intent):
        raise HTTPException(status_code=400, detail="A prompt is required to run the FEM pipeline.")

    try:
        num_modes = int(payload.get("num_modes", 6) or 6)
    except (TypeError, ValueError):
        num_modes = 6
    num_modes = max(1, min(num_modes, MAX_FEM_MODES))
    try:
        mode = int(payload.get("mode", 1) or 1)
    except (TypeError, ValueError):
        mode = 1
    mode = max(1, min(mode, num_modes))
    material = str(payload.get("material", "")).strip() or None
    mesh_mode = _mesh_mode_from_payload(payload)
    global_template = payload.get("global_template") if isinstance(payload.get("global_template"), dict) else {}

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from geometry.bushing_hex_mesh import generate_bushing_hex_mesh  # noqa: WPS433
        from geometry.mesh_cleaner import clean_mesh  # noqa: WPS433
        from geometry.mesh_quality import evaluate_mesh  # noqa: WPS433
        from geometry.uploaded_volume_mesh import uploaded_geometry_to_volume_mesh  # noqa: WPS433
        from simulate.pipeline import PipelineConfig, run_pipeline  # noqa: WPS433
        from simulate.materials import resolve_material  # noqa: WPS433
        from simulate.modal_solver import ModalSetup, run_modal  # noqa: WPS433
        from simulate.results import parse_dat, write_report  # noqa: WPS433
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=501,
            detail=(
                "FEM requires gmsh and PyVista, which are not installed in this deployment. "
                "The analytical estimate above is still available."
            ),
        ) from exc

    if shutil.which("ccx") is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "FEM requires the CalculiX solver ('ccx'), which is not installed on this "
                "server. The analytical estimate above is still available."
            ),
        )

    work_dir = Path(tempfile.mkdtemp(prefix="resonance_fem_"))
    try:
      if uploaded_geometry:
        sim_dir = work_dir / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        raw_mesh = sim_dir / f"{name}_uploaded_hex.msh"
        clean_mesh_path = sim_dir / f"{name}_clean.vtk"
        try:
          uploaded_geometry_to_volume_mesh(
            uploaded_geometry,
            raw_mesh,
            mesh_mode=mesh_mode,
            template=global_template,
          )
        except Exception as exc:  # noqa: BLE001 - exact FEM requires a valid closed volume
          raise HTTPException(
            status_code=422,
            detail=(
              "FEM batch requires a successful uploaded-geometry all-hexa mesh. "
              f"Gmsh could not repair and volume-mesh this file. Detail: {exc}"
            ),
          ) from exc
        clean_mesh(raw_mesh, clean_mesh_path)
        quality = evaluate_mesh(clean_mesh_path)
        if quality.inverted_count:
          raise HTTPException(
            status_code=422,
            detail=(
              "The uploaded geometry produced "
              f"{quality.inverted_count} folded or inverted hexahedral element(s). "
              "Repair the source solid or reduce its geometric defects before FEM."
            ),
          )
        setup = ModalSetup(
          mesh_file=clean_mesh_path,
          material=resolve_material(material),
          num_modes=num_modes,
          boundary="fixed_bottom",
        )
        run = run_modal(setup, sim_dir, job_name=name)
        if not run.ok:
          raise HTTPException(
            status_code=500,
            detail=f"Uploaded-geometry FEM solve failed: {run.failure_summary()}",
          )
        results = parse_dat(run.dat_file)
        write_report(results, sim_dir / f"{name}_modal.json")
        _try_modal_pca(run.frd_file, sim_dir / f"{name}_pca.json", num_modes)
        _render_modal_contour(run.frd_file, sim_dir / f"{name}_mode{mode}_mises.png", mode)
        response = _fem_response_payload(sim_dir, name, mode, num_modes, material)
        response["fem_source"] = "exact_uploaded_geometry"
        response["source_file"] = uploaded_geometry.name
        return response

      if _is_structured_bushing_intent(intent):
        sim_dir = work_dir / "sim"
        sim_dir.mkdir(parents=True, exist_ok=True)
        raw_mesh = sim_dir / f"{name}.vtk"
        clean_mesh_path = sim_dir / f"{name}_clean.vtk"
        generate_bushing_hex_mesh(intent, raw_mesh, mesh_mode=mesh_mode, template=global_template)
        clean_mesh(raw_mesh, clean_mesh_path)
        setup = ModalSetup(
          mesh_file=clean_mesh_path,
          material=resolve_material(material),
          num_modes=num_modes,
          boundary="fixed_bottom",
        )
        run = run_modal(setup, sim_dir, job_name=name)
        if not run.ok:
          raise HTTPException(status_code=500, detail=f"FEM solve failed: {run.failure_summary()}")
        results = parse_dat(run.dat_file)
        write_report(results, sim_dir / f"{name}_modal.json")
        _try_modal_pca(run.frd_file, sim_dir / f"{name}_pca.json", num_modes)
        response = _structured_fem_response_payload(run.frd_file, sim_dir, name, mode, num_modes, material)
        response["fem_source"] = "bushing_poc_hex" if geometry_source == "bushing_poc_hex" else "structured_bushing_hex"
        return response

        # Try the configured provider ("auto" prefers Azure when available),
        # but fall back to the deterministic parser if the LLM path errors out.
        # FEM only needs valid geometry; we never want a transient LLM error to
        # block the user from seeing a contour image.
        cad_code = -1
        last_exc: Exception | None = None
        try:
            from text_to_cad.cad_agent import generate_with_agent  # noqa: WPS433
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=501,
                detail=(
                    "FEM requires the CadQuery pipeline, which is not installed in this "
                    "deployment. The analytical estimate above is still available."
                ),
            ) from exc

        for provider in ("auto", "fallback"):
            try:
                cad_code = generate_with_agent(
                    prompt=prompt,
                    output_dir=work_dir,
                    output_name=name,
                    provider=provider,
                    execute=True,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
            step_path_attempt = work_dir / f"{name}.step"
            if cad_code == 0 and step_path_attempt.exists():
                break

        step_path = work_dir / f"{name}.step"
        if cad_code != 0 or not step_path.exists():
            detail = "STEP generation did not produce a file."
            if last_exc is not None:
                detail += f" Last error: {last_exc}"
            raise HTTPException(status_code=500, detail=detail)

        sim_dir = work_dir / "sim"
        try:
            config = PipelineConfig(
                step_file=step_path,
                output_dir=sim_dir,
                material=resolve_material(material),
                num_modes=num_modes,
                mesh_strategy="hex",
                name=name,
                contour_image=True,
                contour_mode=mode,
            )
            rc = run_pipeline(config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"FEM pipeline failed: {exc}") from exc
        if rc != 0:
            raise HTTPException(status_code=500, detail="FEM pipeline returned a non-zero status.")

        return _fem_response_payload(sim_dir, name, mode, num_modes, material)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _try_modal_pca(frd_file: Path, output_file: Path, num_modes: int) -> None:
    try:
        from simulate.pca import analyze_modal_pca, write_pca_report

        result = analyze_modal_pca(frd_file, max_components=min(6, num_modes))
        write_pca_report(result, output_file)
    except Exception:
        pass


def _render_modal_contour(frd_file: Path, output_file: Path, mode: int) -> None:
    try:
        from simulate.visualize import render_contour

        render_contour(frd_file, output_file, field_name="mises", mode=mode, warp=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"FEM finished but contour rendering failed: {exc}") from exc


def _fem_response_payload(sim_dir: Path, name: str, mode: int, num_modes: int, material: str | None) -> dict:
    contour_path = sim_dir / f"{name}_mode{mode}_mises.png"
    if not contour_path.exists():
        raise HTTPException(status_code=500, detail="FEM finished but no contour image was produced.")

    fem_mesh = None
    try:
        from simulate.visualize import export_contour_surface_mesh  # noqa: WPS433

        fem_mesh = export_contour_surface_mesh(
            sim_dir / f"{name}.frd",
            field_name="mises",
            mode=mode,
            warp=True,
        )
    except Exception:
        fem_mesh = None

    modal_path = sim_dir / f"{name}_modal.json"
    pca_path = sim_dir / f"{name}_pca.json"
    modes_payload: list[dict] = []
    pca_payload: dict | None = None
    fundamental_hz = None
    if modal_path.exists():
        try:
            modal = json.loads(modal_path.read_text(encoding="utf-8"))
            fundamental_hz = modal.get("fundamental_hz")
            modes_payload = modal.get("modes", [])
        except (OSError, json.JSONDecodeError):
            pass
    if pca_path.exists():
        try:
            pca_payload = json.loads(pca_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pca_payload = None

    image_b64 = base64.b64encode(contour_path.read_bytes()).decode("ascii")
    return {
        "contour_png_base64": image_b64,
        "mode": mode,
        "num_modes": num_modes,
        "material": material or "generic",
        "fundamental_hz": fundamental_hz,
        "modes": modes_payload,
        "pca": pca_payload,
        "field": "mises",
        "fem_mesh": fem_mesh,
    }

def _structured_fem_response_payload(
    frd_file: Path,
    sim_dir: Path,
    name: str,
    mode: int,
    num_modes: int,
    material: str | None,
) -> dict:
    modal = _read_json_file(sim_dir / f"{name}_modal.json") or {}
    pca_payload = _read_json_file(sim_dir / f"{name}_pca.json")
    fem_mesh = _export_frd_hex_surface_mesh(frd_file, mode=mode, warp=True)
    return {
        "contour_png_base64": _transparent_png_base64(),
        "mode": mode,
        "num_modes": num_modes,
        "material": material or "generic",
        "fundamental_hz": modal.get("fundamental_hz"),
        "modes": modal.get("modes", []),
        "pca": pca_payload,
        "field": "mises",
        "fem_mesh": fem_mesh,
    }


def _read_json_file(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _export_frd_hex_surface_mesh(frd_file: Path, *, mode: int, warp: bool, max_faces: int = 12000) -> dict:
    import numpy as np
    from simulate.visualize import _select_field, parse_frd, von_mises

    mesh, fields = parse_frd(frd_file)
    disp = _select_field(fields, "DISP", mode)
    stress = _select_field(fields, "STRESS", mode)
    if stress is None:
        raise HTTPException(status_code=500, detail=f"No STRESS field found for mode {mode} in FEM results.")

    points = np.asarray(mesh.points, dtype=float).copy()
    values = von_mises(stress.data)
    freq = stress.frequency_hz
    if warp and disp is not None:
        points = points + disp.data * _frd_warp_scale(points, disp.data)

    face_defs = (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    )
    face_map: dict[tuple[int, ...], dict] = {}
    for vtk_type, rows in mesh.cells:
        if vtk_type != 12 or len(rows) < 8:
            continue
        corners = rows[:8]
        for face_def in face_defs:
            face = tuple(int(corners[index]) for index in face_def)
            key = tuple(sorted(face))
            if key in face_map:
                face_map[key]["count"] += 1
            else:
                face_map[key] = {"count": 1, "face": face}

    exterior = [item["face"] for item in face_map.values() if item["count"] == 1]
    if max_faces > 0 and len(exterior) * 2 > max_faces:
        stride = max(1, (len(exterior) * 2) // max_faces)
        exterior = exterior[::stride]

    vmin = float(np.nanmin(values)) if values.size else 0.0
    vmax = float(np.nanmax(values)) if values.size else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    faces = []
    for face in exterior:
        avg = float(np.mean(values[list(face)])) if values.size else vmin
        color = _contour_hex(avg, vmin, vmax)
        for tri in ((face[0], face[1], face[2]), (face[0], face[2], face[3])):
            faces.append(
                {
                    "color": color,
                    "points": [
                        {
                            "x": float(points[node_id][0]),
                            "y": float(points[node_id][1]),
                            "z": float(points[node_id][2]),
                        }
                        for node_id in tri
                    ],
                }
            )

    return {
        "faces": faces,
        "field": "S, Mises",
        "mode": mode,
        "frequency_hz": freq,
        "scalar_min": vmin,
        "scalar_max": vmax,
        "face_count": len(faces),
    }


def _frd_warp_scale(points, disp) -> float:
    import numpy as np

    max_disp = float(np.linalg.norm(disp, axis=1).max()) if disp.size else 0.0
    if max_disp <= 0.0:
        return 0.0
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    return 0.1 * diagonal / max_disp


def _transparent_png_base64() -> str:
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/lQn9WQAAAABJRU5ErkJggg=="


UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Product Design Studio — Vibracoustic</title>
  <link rel="stylesheet" href="/static/cad-viewer.css">
  <style>
    :root {
      --bg: #eef5ff;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #647084;
      --line: #d8e0ea;
      --brand: #071f3f;
      --brand-2: #0b2f5f;
      --accent: #d8222a;
      --accent-dark: #ad1720;
      --danger: #b42318;
      --cad: #0f766e;
      --cad-dark: #0b4f49;
      --soft: #e7f0fb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", "Segoe UI Variable", "Aptos", "Noto Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: #fff;
      border-bottom: 1px solid var(--line);
      box-shadow: 0 1px 0 rgba(17, 24, 39, 0.04);
    }
    .topbar {
      width: min(1440px, calc(100% - 48px));
      margin: 0 auto;
      min-height: 72px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }
    .brand-lockup {
      display: flex;
      align-items: center;
      gap: 20px;
      min-width: 0;
    }
    .logo {
      display: inline-flex;
      align-items: center;
      white-space: nowrap;
    }
    .vc-logo {
      display: block;
      width: 245px;
      height: auto;
      max-height: 58px;
      flex: 0 0 auto;
      object-fit: contain;
    }
    .product {
      padding-left: 18px;
      border-left: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      line-height: 1.1;
    }
    .product .name {
      font-size: 17px;
      font-weight: 700;
      color: var(--brand);
      white-space: nowrap;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(30px, 4vw, 54px);
      line-height: 0.98;
      letter-spacing: 0;
      max-width: 720px;
    }
    main {
      width: min(1720px, calc(100% - 32px));
      margin: 0 auto 32px;
    }
    .hero {
      min-height: 220px;
      margin: 0 calc(50% - 50vw) 28px;
      padding: 32px 0;
      color: #0f2a49;
      background:
        linear-gradient(110deg, rgba(222, 236, 255, 0.96), rgba(201, 226, 255, 0.93) 55%, rgba(186, 216, 255, 0.9)),
        radial-gradient(circle at 72% 38%, rgba(255, 255, 255, 0.62), transparent 34%),
        #d9ecff;
      display: grid;
      align-items: end;
      overflow: hidden;
    }
    .hero-inner {
      width: min(1440px, calc(100% - 48px));
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.7fr);
      gap: 28px;
      align-items: end;
    }
    .eyebrow {
      margin: 0 0 14px;
      color: #3d658f;
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 1.6px;
      text-transform: uppercase;
    }
    .hero-copy {
      margin: 0;
      max-width: 660px;
      color: #33587f;
      font-size: 16px;
      line-height: 1.6;
    }
    .hero-metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1px;
      background: rgba(83, 128, 178, 0.24);
      border: 1px solid rgba(83, 128, 178, 0.28);
    }
    .metric {
      min-height: 92px;
      padding: 16px;
      background: rgba(245, 250, 255, 0.82);
    }
    .metric strong {
      display: block;
      font-size: 21px;
      line-height: 1;
      margin-bottom: 9px;
    }
    .metric span {
      display: block;
      color: #4c6f96;
      font-size: 12px;
      line-height: 1.35;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(300px, 340px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .workbench, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 22px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }
    .workbench {
      padding: 18px;
    }
    .left-rail {
      display: grid;
      gap: 14px;
      padding: 0;
      border: 0;
      background: transparent;
      box-shadow: none;
    }
    .rail-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }
    .rail-card-header {
      width: 100%;
      margin: 0;
      padding: 0;
      border: 0;
      background: transparent;
      color: var(--brand);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      text-align: left;
      cursor: pointer;
    }
    .rail-card-title {
      min-width: 0;
      color: var(--brand);
      font-size: 16px;
      font-weight: 800;
    }
    .rail-card-header::after {
      content: "";
      width: 9px;
      height: 9px;
      flex: 0 0 auto;
      border-right: 2px solid currentColor;
      border-bottom: 2px solid currentColor;
      transform: rotate(45deg);
      transition: transform 160ms ease;
    }
    .rail-card.open .rail-card-header::after {
      transform: rotate(-135deg);
    }
    .rail-card-body {
      display: grid;
      gap: 14px;
      margin-top: 16px;
    }
    .rail-card:not(.open) .rail-card-body {
      display: none;
    }
    .section-title {
      margin: 0 0 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .section-title strong {
      color: var(--brand);
      font-size: 16px;
    }
    .title-head {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .download {
      position: relative;
    }
    .download-btn {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line, #d8e0ea);
      background: #fff;
      color: var(--brand);
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      padding: 7px 12px;
      border-radius: 9px;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s;
    }
    .download-btn:hover { background: #f3f6fb; }
    .download-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .download-btn .chev { font-size: 10px; }
    .download-menu {
      position: absolute;
      top: calc(100% + 6px);
      right: 0;
      z-index: 30;
      width: 290px;
      background: #fff;
      border: 1px solid var(--line, #d8e0ea);
      border-radius: 12px;
      box-shadow: 0 18px 40px rgba(15, 36, 64, 0.18);
      padding: 8px;
      display: none;
    }
    .download-menu.open { display: block; }
    .download-group + .download-group {
      margin-top: 6px;
      border-top: 1px solid #eef2f7;
      padding-top: 6px;
    }
    .download-group-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      padding: 4px 8px;
    }
    .download-item {
      display: flex;
      align-items: baseline;
      gap: 8px;
      width: 100%;
      text-align: left;
      border: none;
      background: none;
      font: inherit;
      padding: 8px;
      border-radius: 8px;
      cursor: pointer;
      color: var(--ink, #1c2733);
    }
    .download-item:hover { background: #f3f6fb; }
    .download-item .fmt { font-weight: 700; font-size: 13px; min-width: 44px; }
    .download-item .desc { font-size: 12px; color: var(--muted); }
    .download-item:disabled { opacity: 0.5; cursor: not-allowed; }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      color: #0f5132;
      background: #e7f5ed;
      border: 1px solid #bde3cb;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 800;
    }
    .status-pill::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #16a34a;
    }
    label {
      display: block;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
    }
    textarea, input, select {
      width: 100%;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 3px;
      padding: 12px;
      color: var(--ink);
      font: inherit;
      line-height: 1.45;
      background: #fff;
    }
    textarea { min-height: 130px; }
    input, select { resize: none; }
    .field { margin-bottom: 12px; }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .divider {
      margin: 20px 0;
      border: 0;
      border-top: 1px solid var(--line);
    }
    button {
      width: 100%;
      margin-top: 14px;
      border: 0;
      border-radius: 3px;
      background: var(--brand);
      color: white;
      padding: 12px 16px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }
    button:hover { background: var(--brand-2); }
    button:disabled { opacity: 0.62; cursor: wait; }
    .rail-card-header:hover {
      background: transparent;
      color: var(--brand-2);
    }
    .primary-action {
      background: var(--accent);
    }
    .primary-action:hover { background: var(--accent-dark); }
    .chat {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .chat-shell {
      min-height: 460px;
      max-height: 560px;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: linear-gradient(135deg, #f9fbfd, #eef3f8);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .chat-shell.idle {
      min-height: 500px;
      max-height: none;
    }
    .chat-log {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .chat-shell.idle .chat-log {
      flex: 0 0 auto;
      justify-content: flex-start;
      padding: 16px;
    }
    .msg {
      padding: 10px 12px;
      border-radius: 3px;
      max-width: 85%;
      font-size: 14px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--brand);
      color: #fff;
    }
    .msg.bot {
      align-self: flex-start;
      background: #fff;
      border: 1px solid var(--line);
      color: var(--ink);
    }
    .msg.intro {
      max-width: 100%;
      padding: 14px 16px;
      font-size: 14px;
      line-height: 1.55;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }
    .msg.bot.err { border-color: #fecdca; background: #fff7f6; color: var(--danger); }
    .knowledge-sources {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #f8fbff;
    }
    .knowledge-sources-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .knowledge-sources-head strong {
      color: var(--ink);
      font-size: 13px;
    }
    .knowledge-source-options {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .knowledge-source {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      padding: 9px 10px;
      border: 1px solid var(--line);
      background: #fff;
    }
    .knowledge-source.selected {
      border-color: #79b7aa;
      background: #eef8f5;
    }
    .knowledge-source input {
      width: 16px;
      height: 16px;
      margin: 0;
      accent-color: var(--cad);
      flex: 0 0 auto;
    }
    .knowledge-source-name {
      min-width: 0;
      flex: 1;
      color: var(--ink);
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }
    .knowledge-source .info-dot {
      width: 17px;
      height: 17px;
      flex: 0 0 auto;
    }
    .consulted-sources {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-top: 9px;
      padding-top: 8px;
      border-top: 1px solid var(--line);
    }
    .consulted-source {
      padding: 3px 7px;
      border: 1px solid #b6ddd6;
      background: #eef8f5;
      color: #17695e;
      font-size: 10px;
      font-weight: 800;
    }
    .chat-composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      align-items: end;
    }
    .chat-composer textarea {
      flex: 1;
      min-height: 96px;
      max-height: 220px;
      resize: none;
      margin: 0;
    }
    .chat-shell.idle .chat-composer {
      padding: 16px;
      grid-template-columns: 1fr;
      gap: 12px;
      border-top: 0;
      background: transparent;
    }
    .chat-shell.idle .chat-composer textarea {
      min-height: 200px;
      max-height: 240px;
      padding: 16px;
      font-size: 15px;
      line-height: 1.6;
    }
    .chat-actions {
      display: flex;
      align-items: end;
    }
    .chat-shell.idle .chat-actions {
      justify-content: flex-end;
    }
    .chat-composer button {
      width: auto;
      margin: 0;
      min-width: 104px;
      padding: 12px 18px;
    }
    .attachment-tools {
      display: grid;
      gap: 10px;
      padding: 12px;
      border: 1px dashed #b8c8da;
      background: #f8fbff;
    }
    .upload-context-card .rail-card-body {
      display: grid;
      gap: 10px;
    }
    .upload-pill {
      background: #eef8f5;
      color: var(--cad-dark);
      border-color: #b6ddd6;
    }
    .upload-context-note {
      margin: 0;
      font-size: 12px;
      line-height: 1.45;
    }
    .fair-search-button {
      width: 100%;
      margin: 0;
      padding: 10px 12px;
      border: 1px solid #79b7aa;
      background: #eef8f5;
      color: #17695e;
    }
    .fair-search-button:hover,
    .fair-search-button:focus {
      background: #dff2ed;
      color: #10564e;
    }
    .fair-search-status {
      margin: -2px 0 0;
      padding: 8px 10px;
      border-left: 3px solid var(--cad);
      background: #f2f7fb;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }
    .fair-search-status[hidden] {
      display: none;
    }
    .label-with-info {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 0;
    }
    .info-dot {
      position: relative;
      display: inline-grid;
      place-items: center;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      color: #fff;
      background: var(--brand);
      font-size: 12px;
      font-weight: 800;
      cursor: help;
    }
    .info-dot:hover::after,
    .info-dot:focus::after {
      content: attr(data-tooltip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 9px);
      transform: translateX(-50%);
      z-index: 10;
      width: min(320px, 82vw);
      padding: 10px 12px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
      font-size: 12px;
      font-weight: 650;
      line-height: 1.45;
      text-align: left;
      white-space: normal;
    }
    .info-dot:hover::before,
    .info-dot:focus::before {
      content: "";
      position: absolute;
      left: 50%;
      bottom: calc(100% + 3px);
      transform: translateX(-50%);
      border: 6px solid transparent;
      border-top: 0;
      border-bottom-color: #fff;
      z-index: 11;
    }
    .attachment-row {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .attachment-row input {
      flex: 1;
      padding: 9px;
      background: #fff;
    }
    .attachment-row button {
      flex: 0 0 auto;
      width: auto;
      margin: 0;
      padding: 9px 14px;
      background: #385a7e;
    }
    .attachment-row button:hover {
      background: #2a4562;
    }
    .attachment-list {
      display: grid;
      gap: 8px;
    }
    .attachment-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 9px 10px;
      border: 1px solid var(--line);
      background: #fff;
      font-size: 13px;
    }
    .attachment-item strong {
      color: var(--brand);
      overflow-wrap: anywhere;
    }
    .attachment-item span {
      color: var(--muted);
      display: block;
      margin-top: 2px;
    }
    .attachment-item button {
      width: auto;
      margin: 0;
      padding: 6px 9px;
      color: var(--danger);
      background: #fff1f0;
    }
    .activity-panel {
      display: none;
      gap: 8px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #f8fbff;
    }
    .activity-panel.active {
      display: grid;
    }
    .activity-status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--brand);
      font-size: 13px;
      font-weight: 700;
    }
    .activity-phase {
      color: var(--muted);
      font-weight: 600;
      text-align: right;
    }
    .activity-track {
      width: 100%;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dbe6f2;
    }
    .activity-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #0f766e, #24a39a);
      transition: width 260ms ease;
    }
    .activity-panel.done .activity-fill {
      background: linear-gradient(90deg, #11895d, #34c77b);
    }
    .activity-panel.error .activity-fill {
      background: linear-gradient(90deg, #d8222a, #ef6a6f);
    }
    .preview-progress {
      display: none;
      gap: 8px;
      margin-bottom: 14px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: #f4f9ff;
      border-radius: 3px;
    }
    .preview-progress.active {
      display: grid;
    }
    .preview-progress-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--brand);
      font-size: 13px;
      font-weight: 700;
    }
    .preview-progress-phase {
      color: var(--muted);
      font-weight: 600;
    }
    .preview-progress-track {
      width: 100%;
      height: 8px;
      overflow: hidden;
      border-radius: 999px;
      background: #dbe6f2;
    }
    .preview-progress-fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, #0f766e, #24a39a);
      transition: width 260ms ease;
    }
    .preview-progress.done .preview-progress-fill {
      background: linear-gradient(90deg, #11895d, #34c77b);
    }
    .preview-progress.error .preview-progress-fill {
      background: linear-gradient(90deg, #d8222a, #ef6a6f);
    }
    .category-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
    }
    .category-family-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .category-chip {
      width: auto;
      margin: 0;
      padding: 12px 14px;
      display: grid;
      gap: 4px;
      text-align: left;
      background: #f4f9ff;
      color: var(--brand);
      border: 1px solid var(--line);
      border-radius: 4px;
      cursor: pointer;
      transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
    }
    .family-chip {
      min-height: 68px;
      background: #fff;
    }
    .category-chip strong {
      font-size: 14px;
      font-weight: 750;
      color: var(--brand);
    }
    .category-chip span {
      font-size: 12px;
      color: var(--muted);
      font-weight: 500;
    }
    .category-chip:hover {
      background: #e7f0fb;
      border-color: #b8c8da;
    }
    .category-chip.active {
      background: var(--brand);
      border-color: var(--brand);
    }
    .category-chip.active strong,
    .category-chip.active span {
      color: #fff;
    }
    .category-detail {
      display: none;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }
    .category-detail.open {
      display: block;
    }
    .category-subhead {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin: 4px 0 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .summary-box {
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent);
      background: #fbfdff;
      padding: 12px;
      min-height: 96px;
    }
    .summary-box p {
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.5;
    }
    .summary-box dl {
      margin: 0;
      display: grid;
      grid-template-columns: max-content minmax(0, 1fr);
      gap: 6px 10px;
      font-size: 13px;
    }
    .summary-box dt {
      color: var(--muted);
      font-weight: 700;
    }
    .summary-box dd {
      margin: 0;
      color: var(--ink);
      overflow-wrap: anywhere;
    }
    .stack {
      display: grid;
      gap: 18px;
    }
    .preview {
      min-height: 470px;
      display: grid;
      place-items: center;
      overflow: hidden;
      position: relative;
      background:
        linear-gradient(90deg, rgba(216, 224, 234, 0.34) 1px, transparent 1px),
        linear-gradient(rgba(216, 224, 234, 0.34) 1px, transparent 1px),
        #fff;
      background-size: 28px 28px;
    }
    .viewer3d {
      width: 100%;
      height: 100%;
      min-height: 470px;
    }
    .viewer3d canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    svg {
      width: 100%;
      max-height: 560px;
    }
    .placeholder {
      width: min(560px, 90%);
      display: grid;
      gap: 14px;
      color: var(--muted);
      text-align: center;
    }
    .placeholder svg {
      width: 100%;
      max-height: 300px;
    }
    .muted {
      color: var(--muted);
      font-size: 14px;
    }
    pre {
      margin: 0;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .param-controls {
      display: flex;
      flex-direction: column;
      gap: 16px;
      max-height: 420px;
      overflow: auto;
    }
    .param-panel.collapsed .param-controls {
      display: none;
    }
    .param-title {
      margin-bottom: 12px;
    }
    .param-panel.collapsed .param-title {
      margin-bottom: 0;
    }
    .param-toggle {
      width: auto;
      margin: 0;
      padding: 7px 12px;
      background: #fff;
      color: var(--brand);
      border: 1px solid var(--line);
      border-radius: 3px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }
    .param-toggle:hover {
      background: #eef3f9;
      color: var(--brand);
    }
    .cad-engine-row {
      display: grid;
      gap: 6px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fbfd;
    }
    .cad-engine-row label {
      color: var(--brand);
      font-size: 13px;
      font-weight: 700;
    }
    .cad-engine-row select {
      width: 100%;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--brand);
      font-size: 13px;
    }
    .rubber-workflow {
      display: grid;
      gap: 12px;
    }
    .param-tabs {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
    }
    .param-tab {
      margin: 0;
      padding: 8px 6px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--brand);
      font-size: 12px;
      font-weight: 750;
      line-height: 1.2;
    }
    .param-tab.active {
      background: var(--brand);
      border-color: var(--brand);
      color: #fff;
    }
    .param-section {
      display: grid;
      gap: 10px;
      padding-top: 2px;
    }
    .param-section-title {
      margin: 4px 0 0;
      color: var(--brand);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .param-form-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .param-form-field {
      display: grid;
      gap: 4px;
      min-width: 0;
    }
    .param-form-field.full {
      grid-column: 1 / -1;
    }
    .param-form-field label {
      color: var(--brand);
      font-size: 12px;
      font-weight: 700;
    }
    .param-form-field input,
    .param-form-field select {
      width: 100%;
      min-width: 0;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--ink);
      font-size: 13px;
    }
    .param-primary {
      width: auto;
      margin: 0;
      padding: 8px 14px;
      border-radius: 4px;
      font-size: 13px;
      font-weight: 750;
    }
    .variant-output {
      display: grid;
      gap: 8px;
    }
    .variant-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .variant-table th,
    .variant-table td {
      border-bottom: 1px solid var(--line);
      padding: 6px 4px;
      text-align: left;
      vertical-align: middle;
    }
    .variant-table th {
      color: var(--brand);
      font-weight: 800;
    }
    .variant-table button {
      width: auto;
      margin: 0;
      padding: 5px 8px;
      border-radius: 4px;
      font-size: 12px;
    }
    .best-geometry {
      display: grid;
      gap: 6px;
      padding: 10px;
      border: 1px solid #b7d6ca;
      border-radius: 6px;
      background: #f4fbf7;
      font-size: 12px;
      color: var(--ink);
    }
    .param-row {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 6px 12px;
    }
    .param-row .param-label {
      font-size: 13px;
      font-weight: 600;
      color: var(--brand);
    }
    .param-row .param-value {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      justify-self: end;
    }
    .param-row .param-number {
      width: 76px;
      padding: 5px 7px;
      border: 1px solid var(--line);
      border-radius: 4px;
      font-size: 13px;
      color: var(--brand);
      text-align: right;
      background: #fff;
    }
    .param-row .param-unit {
      font-size: 12px;
      color: var(--muted);
    }
    .param-row .param-slider {
      grid-column: 1 / -1;
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }
    .param-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      padding-top: 4px;
      border-top: 1px solid var(--line);
    }
    .param-reset {
      background: #fff;
      color: var(--brand);
      border: 1px solid var(--line);
      padding: 6px 14px;
      font-size: 13px;
    }
    .param-reset:hover {
      background: #eef3f9;
    }
    .sim-block {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sim-block strong {
      font-size: 13px;
      color: var(--brand);
    }
    .analysis-results {
      margin-top: 14px;
      display: grid;
      gap: 14px;
    }
    .analysis-results[hidden] {
      display: none;
    }
    .analysis-result-block {
      margin-top: 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }
    .sim-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .sim-btn {
      width: auto;
      margin-top: 0;
      background: var(--brand);
      color: #fff;
      border: none;
      border-radius: 3px;
      padding: 6px 16px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    .sim-btn:hover {
      background: var(--brand-2);
    }
    .sim-btn.secondary {
      background: var(--cad);
    }
    .sim-btn.secondary:hover {
      background: var(--cad-dark);
    }
    .sim-btn[disabled] {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .sim-head-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .sim-info-btn {
      width: 26px;
      height: 26px;
      margin: 0;
      padding: 0;
      border-radius: 50%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--brand);
      font-size: 13px;
      font-weight: 800;
      line-height: 1;
      cursor: help;
    }
    .sim-info-btn:hover,
    .sim-info-btn:focus {
      background: #eef3f9;
      outline: none;
    }
    .sim-batch-controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      align-items: end;
      padding: 10px;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 3px;
    }
    .sim-batch-field label {
      margin-bottom: 4px;
      font-size: 12px;
    }
    .sim-batch-field input {
      padding: 7px 9px;
      font-size: 13px;
    }
    .sim-pca {
      border: 1px solid var(--line);
      border-radius: 3px;
      background: #fff;
      overflow: hidden;
    }
    .sim-pca-summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 14px;
      background: #f8fbff;
      color: var(--brand);
      cursor: pointer;
      list-style: none;
      user-select: none;
    }
    .sim-pca-summary::-webkit-details-marker {
      display: none;
    }
    .sim-pca-summary-copy {
      display: flex;
      flex-direction: column;
      gap: 3px;
      min-width: 0;
    }
    .sim-pca-summary-copy strong {
      font-size: 14px;
    }
    .sim-pca-summary-copy small,
    .sim-pca-summary-meta {
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .sim-pca-summary-meta {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      white-space: nowrap;
    }
    .sim-pca-summary-meta::after {
      content: "";
      width: 8px;
      height: 8px;
      border-right: 2px solid currentColor;
      border-bottom: 2px solid currentColor;
      transform: rotate(45deg);
      transition: transform 160ms ease;
    }
    .sim-pca[open] .sim-pca-summary-meta::after {
      transform: rotate(225deg);
    }
    .sim-pca-summary:hover,
    .sim-pca-summary:focus-visible {
      background: #eef4fb;
      outline: none;
    }
    .sim-pca-content {
      padding: 12px;
      border-top: 1px solid var(--line);
    }
    .sim-pca-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 10px;
      align-items: start;
      margin-top: 10px;
    }
    .sim-pca-plot {
      position: relative;
      min-height: 340px;
      border: 1px solid var(--line);
      border-radius: 3px;
      background: #fbfdff;
      overflow: hidden;
    }
    .sim-pca-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      font-size: 12px;
      color: var(--muted);
    }
    .sim-pca-toolbar-copy {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .sim-pca-toolbar-copy strong {
      color: var(--brand);
      font-size: 12px;
    }
    .sim-pca-toolbar-copy small {
      color: var(--muted);
      font-size: 11px;
    }
    .sim-pca-toggle {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 4px;
      overflow: hidden;
      background: #fff;
    }
    .sim-pca-toggle button {
      width: auto;
      margin: 0;
      padding: 5px 10px;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: var(--brand);
      font-size: 12px;
      font-weight: 700;
    }
    .sim-pca-toggle button.active {
      background: var(--brand);
      color: #fff;
    }
    .sim-pca-canvas {
      width: 100%;
      height: 320px;
      display: block;
    }
    .sim-pca-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 14px;
      padding: 8px 12px 10px;
      border-top: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      font-size: 11px;
    }
    .sim-pca-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .sim-pca-legend-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      box-shadow: 0 0 0 2px #fff, 0 0 0 3px rgba(16, 36, 63, 0.12);
    }
    .sim-pca-tooltip {
      position: absolute;
      z-index: 2;
      display: none;
      min-width: 150px;
      padding: 8px 10px;
      border: 1px solid #cbd7e5;
      border-radius: 3px;
      background: rgba(16, 36, 63, 0.96);
      color: #fff;
      box-shadow: 0 6px 18px rgba(16, 36, 63, 0.16);
      font-size: 11px;
      line-height: 1.45;
      pointer-events: none;
    }
    .sim-pca-tooltip strong {
      display: block;
      margin-bottom: 2px;
      font-size: 12px;
    }
    .sim-pca-features {
      min-width: 0;
    }
    .sim-pca .axis-pill {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 24px;
      padding: 2px 6px;
      border-radius: 999px;
      background: #e8f0fb;
      color: var(--brand);
      font-size: 11px;
      font-weight: 800;
    }
    @media (max-width: 980px) {
      .sim-pca-layout {
        grid-template-columns: minmax(0, 1fr);
      }
      .sim-pca-summary {
        align-items: flex-start;
      }
    }
    .mesh-block {
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .mesh-controls {
      display: grid;
      gap: 10px;
      padding: 10px;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 3px;
    }
    .mesh-template-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .mesh-control-field label {
      margin-bottom: 4px;
      font-size: 12px;
    }
    .mesh-control-field input,
    .mesh-control-field select {
      padding: 7px 9px;
      font-size: 13px;
    }
    .mesh-global-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .mesh-global-summary span {
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      font-size: 12px;
      color: var(--muted);
    }
    .shape-pca-formula {
      display: grid;
      gap: 4px;
      margin-bottom: 10px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--brand);
      font-size: 13px;
    }
    .shape-pca-formula span {
      color: var(--muted);
      font-size: 12px;
    }
    .mesh-output {
      padding: 10px;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 3px;
      font-size: 12px;
      color: var(--muted);
    }
    .mesh-output strong {
      color: var(--brand);
    }
    .mesh-output.err {
      background: #fff7f6;
      border-color: #fecdca;
      color: var(--danger);
    }
    .mesh-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 14px;
      margin-top: 8px;
      font-variant-numeric: tabular-nums;
    }
    .mesh-viewer-wrap {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: stretch;
      margin-top: 10px;
    }
    .mesh-viewer {
      height: clamp(520px, 58vh, 760px);
      min-height: 520px;
      border-radius: 3px;
      overflow: hidden;
      background: #ffffff;
      border: 1px solid var(--line);
      touch-action: none;
    }
    .mesh-viewer canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }
    .mesh-viewer canvas:active {
      cursor: grabbing;
    }
    .mesh-legend {
      min-width: 68px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 6px;
      justify-items: center;
      color: var(--brand);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    .mesh-legend strong {
      font-size: 13px;
    }
    .mesh-legend-bar {
      width: 18px;
      min-height: 230px;
      border: 1px solid rgba(10, 30, 63, 0.15);
      background: linear-gradient(
        to top,
        #20199c 0%,
        #005bff 18%,
        #00c8ff 36%,
        #40dc68 54%,
        #ffeb3b 70%,
        #ff8700 84%,
        #cc0000 100%
      );
    }
    .mesh-compare,
    .fem-compare {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
      margin-top: 10px;
    }
    .mesh-compare-card {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
    }
    .mesh-compare-card-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
    }
    .mesh-compare-card-head strong {
      color: var(--brand);
      font-size: 13px;
    }
    .mesh-compare-card-head span {
      font-size: 12px;
      text-align: right;
    }
    .mesh-compare-card .mesh-viewer,
    .fem-compare .sim-fem-viewer {
      height: clamp(430px, 48vh, 650px);
      min-height: 430px;
    }
    .mesh-compare-card .mesh-viewer-wrap {
      margin-top: 0;
      grid-template-columns: minmax(0, 1fr) 58px;
    }
    .mesh-compare-card .mesh-legend {
      min-width: 58px;
      font-size: 11px;
    }
    .sim-compare {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .sim-compare > div {
      min-width: 0;
    }
    .sim-estimate-card {
      width: 100%;
      margin: 0 0 12px;
      padding: 10px 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }
    .sim-estimate-card:hover,
    .sim-estimate-card:focus {
      border-color: #b8c7d9;
      background: #f8fbff;
      outline: none;
    }
    .sim-estimate-card small {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
    }
    .sim-estimate-kpis {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .sim-estimate-kpis span {
      padding: 5px 8px;
      border-radius: 4px;
      background: #eef3f9;
    }
    .sim-modal-backdrop {
      position: fixed;
      inset: 0;
      z-index: 9998;
      display: grid;
      place-items: center;
      padding: 24px;
      background: rgba(7, 31, 63, 0.42);
    }
    .sim-modal {
      width: min(980px, 96vw);
      max-height: 88vh;
      display: flex;
      flex-direction: column;
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 22px 70px rgba(7, 31, 63, 0.28);
      overflow: hidden;
    }
    .sim-modal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #f8fbff;
    }
    .sim-modal-close {
      width: 32px;
      height: 32px;
      margin: 0;
      padding: 0;
      border: 1px solid var(--line);
      border-radius: 50%;
      background: #fff;
      color: var(--brand);
      font-size: 22px;
      line-height: 1;
    }
    .sim-modal-body {
      padding: 16px;
      overflow: auto;
    }
    .stiffness-pca-layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 300px;
      gap: 16px;
      align-items: stretch;
    }
    .stiffness-pca-plot {
      min-height: 480px;
      border: 1px solid var(--line);
      background: #f3f7fc;
    }
    .stiffness-pca-plot canvas {
      width: 100%;
      height: 100%;
      min-height: 480px;
      display: block;
    }
    .stiffness-pca-conditions {
      padding: 14px;
      border-left: 3px solid var(--cad);
      background: #f8fbff;
    }
    .stiffness-pca-conditions strong {
      display: block;
      margin-bottom: 10px;
    }
    .stiffness-pca-conditions span {
      display: block;
      margin: 7px 0;
      color: var(--muted);
      font-size: 13px;
    }
    .stiffness-pca-legend {
      display: flex;
      gap: 16px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .stiffness-pca-legend i {
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 5px;
      border-radius: 50%;
    }
    .sim-contour {
      min-height: 100%;
      padding: 10px;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 3px;
    }
    .sim-contour img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 2px;
      background: #ffffff;
    }
    .sim-fem-frame {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: stretch;
    }
    .sim-fem-viewer {
      width: 100%;
      height: clamp(500px, 56vh, 740px);
      min-height: 500px;
      border-radius: 2px;
      overflow: hidden;
      background: #f8fbff;
      border: 1px solid var(--line);
      touch-action: none;
    }
    .sim-fem-viewer canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }
    .sim-fem-viewer canvas:active {
      cursor: grabbing;
    }
    .sim-contour .sim-fem-msg {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
    }
    .sim-contour .sim-fem-error {
      color: var(--danger);
      font-weight: 500;
    }
    .sim-fem-progress {
      height: 8px;
      margin-top: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #dbe6f2;
    }
    .sim-fem-progress span {
      display: block;
      width: 42%;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--brand), var(--cad));
      animation: sim-progress-slide 1.4s ease-in-out infinite;
    }
    @keyframes sim-progress-slide {
      0% { transform: translateX(-110%); }
      50% { transform: translateX(70%); }
      100% { transform: translateX(260%); }
    }
    .sim-table-rows tr[data-mode] {
      cursor: pointer;
    }
    .sim-table-rows tr[data-mode]:hover {
      background: #eef3f9;
    }
    .sim-row-active {
      background: #f0f5fb;
    }
    .sim-row-active td {
      font-weight: 600;
    }
    .sim-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .sim-table-scroll {
      width: 100%;
      overflow-x: auto;
    }
    .sim-table th, .sim-table td {
      text-align: left;
      padding: 4px 6px;
      border-bottom: 1px solid var(--line);
    }
    .sim-table th {
      color: var(--muted);
      font-weight: 600;
    }
    .sim-table td.sim-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      color: var(--brand);
    }
    .stiffness-error-cell {
      text-align: right !important;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }
    .stiffness-error-good {
      background: #e9f7ef;
      color: #137a45;
    }
    .stiffness-error-warning {
      background: #fff7d6;
      color: #8a6500;
    }
    .stiffness-error-bad {
      background: #fff0ef;
      color: #b42318;
    }
    .sim-dashboard {
      margin-top: 10px;
      display: grid;
      gap: 10px;
    }
    .sim-kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .sim-kpi,
    .sim-frequency-bars,
    .sim-dashboard-detail {
      padding: 10px;
      background: #f8fbff;
      border: 1px solid var(--line);
      border-radius: 3px;
    }
    .sim-kpi span,
    .sim-detail-item span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }
    .sim-kpi strong,
    .sim-detail-item strong {
      display: block;
      margin-top: 3px;
      color: var(--brand);
      font-size: 16px;
      font-variant-numeric: tabular-nums;
    }
    .sim-frequency-bars {
      display: grid;
      gap: 6px;
    }
    .sim-bar-row {
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr) 74px;
      gap: 8px;
      align-items: center;
      padding: 5px 6px;
      border-radius: 2px;
      cursor: pointer;
      font-size: 12px;
      color: var(--brand);
    }
    .sim-bar-row:hover,
    .sim-bar-row.active {
      background: #eef3f9;
    }
    .sim-bar-track {
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: #dbe6f2;
    }
    .sim-bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--brand), var(--cad));
    }
    .sim-bar-value {
      text-align: right;
      color: var(--brand);
      font-variant-numeric: tabular-nums;
    }
    .sim-dashboard-detail {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .sim-stiff {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 18px;
      font-size: 13px;
      color: var(--muted);
    }
    .sim-stiff b {
      color: var(--brand);
      font-variant-numeric: tabular-nums;
    }
    .error-box {
      width: min(860px, 92%);
      margin: 0;
      padding: 18px;
      color: var(--danger);
      background: #fff7f6;
      border: 1px solid #fecdca;
      border-radius: 3px;
    }
    @media (max-width: 1080px) {
      .hero-inner, .workspace { grid-template-columns: 1fr; }
      .hero-metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .sim-compare { grid-template-columns: 1fr; }
      .mesh-compare, .fem-compare { grid-template-columns: 1fr; }
      .stiffness-pca-layout { grid-template-columns: 1fr; }
      .stiffness-pca-conditions {
        border-left-width: 1px;
        border-top: 3px solid var(--cad);
      }
      .mesh-viewer, .sim-fem-viewer {
        height: 440px;
        min-height: 440px;
      }
    }
    @media (max-width: 720px) {
      .topbar, main, .hero-inner { width: min(100% - 28px, 1440px); }
      .brand-lockup { align-items: flex-start; gap: 12px; flex-direction: column; }
      .product { padding-left: 0; border-left: 0; }
      .hero { padding-block: 26px; }
      .hero-metrics { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr; }
      .knowledge-source-options { grid-template-columns: 1fr; }
      .category-family-grid { grid-template-columns: 1fr; }
      .mesh-viewer-wrap, .sim-fem-frame { grid-template-columns: 1fr; }
      .mesh-compare-card .mesh-viewer-wrap { grid-template-columns: 1fr; }
      .mesh-legend {
        grid-template-columns: auto auto minmax(120px, 1fr) auto;
        grid-template-rows: auto;
        justify-items: start;
        align-items: center;
      }
      .mesh-legend-bar {
        width: auto;
        min-height: 16px;
        height: 16px;
        background: linear-gradient(
          to right,
          #20199c 0%,
          #005bff 18%,
          #00c8ff 36%,
          #40dc68 54%,
          #ffeb3b 70%,
          #ff8700 84%,
          #cc0000 100%
        );
      }
      .mesh-viewer, .sim-fem-viewer {
        height: 340px;
        min-height: 340px;
      }
      .sim-kpi-grid, .sim-dashboard-detail { grid-template-columns: 1fr 1fr; }
      .sim-bar-row { grid-template-columns: 86px minmax(0, 1fr) 68px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand-lockup">
        <div class="logo">
          <img
            src="https://vibracoustic.com/wp-content/uploads/2023/08/vibracoustic.webp"
            alt="Vibracoustic"
            class="vc-logo"
          >
        </div>
        <div class="product">
          <span class="name">Product Design Studio</span>
        </div>
      </div>
    </div>
  </header>
  <main>
    <section class="hero">
      <div class="hero-inner">
        <div>
          <p class="eyebrow">NVH Product Engineering</p>
          <h1>Product Design Studio</h1>
          <p class="hero-copy">Design, refine, mesh, and validate vibroacoustic components in one AI-assisted engineering workspace.</p>
        </div>
        <div class="hero-metrics" aria-label="Workflow summary">
          <div class="metric"><strong>01</strong><span>Prompt intake</span></div>
          <div class="metric"><strong>02</strong><span>Parametric editor</span></div>
          <div class="metric"><strong>03</strong><span>Preview review</span></div>
          <div class="metric"><strong>POC</strong><span>Engineering demonstrator</span></div>
        </div>
      </div>
    </section>

    <section class="workspace">
      <section class="workbench left-rail">
        <section class="rail-card engineering-chat-card" id="engineeringChatPanel">
          <button type="button" class="rail-card-header" id="engineeringChatToggle" aria-expanded="false" aria-controls="engineeringChatBody">
            <span class="rail-card-title">Engineering chat</span>
            <span class="status-pill">Model ready</span>
          </button>
          <div class="rail-card-body" id="engineeringChatBody">
            <div class="knowledge-sources" aria-label="Engineering knowledge sources">
              <div class="knowledge-sources-head">
                <strong>Agent knowledge</strong>
                <span>Select sources for this answer</span>
              </div>
              <div class="knowledge-source-options">
                <div class="knowledge-source">
                  <input id="kissAgentSource" type="checkbox" name="knowledgeSource" value="kiss_agent">
                  <label class="knowledge-source-name" for="kissAgentSource">KISS Agent</label>
                  <span
                    class="info-dot"
                    tabindex="0"
                    aria-label="KISS Agent information"
                    data-tooltip="POC source for engineering calculation guidance, sizing rules, and design assumptions."
                  >i</span>
                </div>
                <div class="knowledge-source">
                  <input id="fairExplorerSource" type="checkbox" name="knowledgeSource" value="fair_explorer">
                  <label class="knowledge-source-name" for="fairExplorerSource">FAIR Explorer</label>
                  <span
                    class="info-dot"
                    tabindex="0"
                    aria-label="FAIR Explorer information"
                    data-tooltip="POC source for traceable engineering datasets, metadata, and prior design evidence."
                  >i</span>
                </div>
              </div>
            </div>
            <div id="chatShell" class="chat-shell idle">
              <div id="chatLog" class="chat-log">
                <div class="msg bot intro">Use chat for text instructions, corrections, and engineering decisions. Upload images, documents, or CAD models in the separate Upload Context card.</div>
              </div>
              <form id="chatForm" class="chat-composer">
                <textarea id="chatInput" placeholder="Describe the part, main dimensions, material, and anything important for the CAD model..." autocomplete="off"></textarea>
                <div class="chat-actions">
                  <button type="submit" id="chatSend" class="primary-action">Send</button>
                </div>
              </form>
            </div>
            <div id="activityPanel" class="activity-panel" aria-live="polite">
              <div class="activity-status">
                <span id="activityLabel">Ready</span>
                <span id="activityPhase" class="activity-phase">Waiting</span>
              </div>
              <div class="activity-track" aria-hidden="true">
                <div id="activityFill" class="activity-fill"></div>
              </div>
            </div>
            <div class="summary-box" id="summaryBox">
              <p>Upload a file or write a request. I will summarize the proposed CAD intent before generating the model.</p>
            </div>
          </div>
        </section>

        <section class="rail-card upload-context-card collapsed" id="uploadContextPanel">
          <button type="button" class="rail-card-header" id="uploadContextToggle" aria-expanded="false" aria-controls="uploadContextBody">
            <span class="rail-card-title">Upload context</span>
            <span class="status-pill upload-pill">Image / CAD</span>
          </button>
          <div class="rail-card-body" id="uploadContextBody">
            <div class="attachment-tools">
              <label for="contextFile" class="label-with-info">
                Add document, image, or old CAD model
                <span
                  class="info-dot"
                  tabindex="0"
                  aria-label="Supported file types: PDF, image jpg, JSON, STEP/STP/IGES/STL/OBJ/DXF/SCAD/FCStd"
                  data-tooltip="Supported file types: PDF, image (jpg), JSON, STEP/STP/IGES/STL/OBJ/DXF/SCAD/FCStd"
                >i</span>
              </label>
              <div class="attachment-row">
                <input
                  id="contextFile"
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.step,.stp,.iges,.igs,.stl,.obj,.dxf,.scad,.fcstd,.txt,.md,.json,.xml,.csv"
                >
                <button id="uploadContextButton" type="button">Attach File</button>
              </div>
              <button id="fairEngineeringSearchButton" class="fair-search-button" type="button">
                Search from FAIR Engineering Data
              </button>
              <p id="fairEngineeringSearchStatus" class="fair-search-status" aria-live="polite" hidden>
                FAIR Engineering Data search is a POC placeholder. The live data connection will be added later.
              </p>
              <div id="attachmentList" class="attachment-list"></div>
              <p class="muted upload-context-note">Rubber bushing uploads can drive OpenSCAD CAD, Design Space, Target Stiffness, mesh, and FEM without using chat.</p>
            </div>
          </div>
        </section>

        <section class="rail-card param-panel collapsed" id="paramPanel">
          <div class="section-title param-title">
            <div class="title-head">
              <strong>Parametric input</strong>
              <span id="paramHint" class="muted">Open after a model is generated.</span>
            </div>
            <button type="button" class="param-toggle" id="paramToggle" aria-expanded="false" aria-controls="paramControls">Open input</button>
          </div>
          <div id="paramControls" class="param-controls">
            <div class="cad-engine-row">
              <label for="cadEngineSelect">CAD Engine</label>
              <select id="cadEngineSelect">
                <option value="cadquery" selected>CadQuery</option>
                <option value="openscad">OpenSCAD</option>
              </select>
            </div>
            <p class="muted">Adjustable dimensions will appear here once a model is generated. Use the Download menu to export the edited part.</p>
          </div>
          <div id="meshResults"></div>
          <div id="simResults"></div>
          <pre id="jsonOutput" hidden>{}</pre>
        </section>
      </section>

      <section class="stack">
        <div class="panel" id="categoryPanel">
          <div class="section-title">
            <strong>Product family</strong>
            <span class="muted">Pick a family, then a component</span>
          </div>
          <div class="category-family-grid" id="familyGrid">
            <button type="button" class="category-chip family-chip" data-family="bushing">
              <strong>Bushing</strong>
              <span>Rubber and bonded bushings</span>
            </button>
            <button type="button" class="category-chip family-chip" data-family="mounts">
              <strong>Mounts</strong>
              <span>Powertrain and chassis mounts</span>
            </button>
            <button type="button" class="category-chip family-chip" data-family="anonymous">
              <strong>Anonymous</strong>
              <span>Other NVH components</span>
            </button>
          </div>
          <div class="category-detail" id="categoryDetail">
            <div class="category-subhead">
              <strong id="categorySubhead">Component types</strong>
              <span id="categoryHint">Select a component type</span>
            </div>
            <div class="category-grid" id="categoryGrid">
            </div>
          </div>
        </div>
        <div class="panel">
          <div class="section-title">
            <div class="title-head">
              <strong>3D CAD Model</strong>
              <span class="muted">Interactive preview</span>
            </div>
            <div class="download">
              <button type="button" id="downloadBtn" class="download-btn" aria-haspopup="true" aria-expanded="false" disabled>
                Download <span class="chev">&#9662;</span>
              </button>
              <div id="downloadMenu" class="download-menu" role="menu">
                <div class="download-group">
                  <div class="download-group-label">CAD model</div>
                  <button type="button" class="download-item" data-format="step" data-engine-scope="cadquery" role="menuitem"><span class="fmt">STEP</span><span class="desc">Engineering CAD exchange (.step)</span></button>
                  <button type="button" class="download-item" data-format="scad" data-engine-scope="openscad" role="menuitem"><span class="fmt">SCAD</span><span class="desc">OpenSCAD source model (.scad)</span></button>
                  <button type="button" class="download-item" data-format="stl" role="menuitem"><span class="fmt">STL</span><span class="desc">3D printing / mesh (.stl)</span></button>
                  <button type="button" class="download-item" data-format="glb" data-engine-scope="cadquery" role="menuitem"><span class="fmt">GLB</span><span class="desc">Interactive 3D sharing (.glb)</span></button>
                  <button type="button" class="download-item" data-format="dxf" data-engine-scope="cadquery" role="menuitem"><span class="fmt">DXF</span><span class="desc">2D profile / drawing (.dxf)</span></button>
                </div>
                <div class="download-group">
                  <div class="download-group-label">Documentation</div>
                  <button type="button" class="download-item" data-format="png" role="menuitem"><span class="fmt">PNG</span><span class="desc">CAD preview image (.png)</span></button>
                  <button type="button" class="download-item" data-format="pdf" data-engine-scope="cadquery" role="menuitem"><span class="fmt">PDF</span><span class="desc">Technical summary (.pdf)</span></button>
                  <button type="button" class="download-item" data-format="json" role="menuitem"><span class="fmt">JSON</span><span class="desc">CAD parameters (.json)</span></button>
                </div>
                <div class="download-group">
                  <div class="download-group-label">Convenience</div>
                  <button type="button" class="download-item" data-format="zip" data-engine-scope="cadquery" role="menuitem"><span class="fmt">ZIP</span><span class="desc">Download all files (.zip)</span></button>
                </div>
              </div>
            </div>
          </div>
          <div id="previewProgress" class="preview-progress" aria-live="polite">
            <div class="preview-progress-head">
              <span id="previewProgressLabel">Generating CAD model</span>
              <span id="previewProgressPhase" class="preview-progress-phase">Working</span>
            </div>
            <div class="preview-progress-track" aria-hidden="true">
              <div id="previewProgressFill" class="preview-progress-fill"></div>
            </div>
          </div>
          <div class="preview" id="preview">
            <div class="placeholder">
              <p class="muted">Interactive 3D preview will appear here after the CAD intent is parsed.</p>
            </div>
          </div>
          <div class="analysis-results" id="analysisResults" hidden>
            <div id="meshOutputPanel"></div>
            <div id="shapePcaOutputPanel"></div>
            <div id="simOutputPanel"></div>
          </div>
        </div>
      </section>
    </section>
  </main>

  <script type="module">
    // === Visible error reporter ===
    // If anything in this module throws during init, surface it as a banner so
    // we can see why the chat / preview is silent. Tracked because a single
    // exception in a <script type="module"> aborts the whole module silently.
    function __showInitError(prefix, msg) {
      try {
        const existing = document.getElementById("__initErrorBanner");
        const el = existing || document.createElement("div");
        el.id = "__initErrorBanner";
        el.style.cssText = "position:fixed;top:0;left:0;right:0;background:#7a1414;color:#fff;padding:10px 14px;font:13px/1.4 ui-monospace,Menlo,monospace;z-index:99999;white-space:pre-wrap;box-shadow:0 2px 10px rgba(0,0,0,.3)";
        el.textContent = prefix + ": " + msg;
        if (!existing) document.body.appendChild(el);
      } catch (_) { /* ignore */ }
    }
    window.addEventListener("error", (e) => {
      __showInitError("JS error", (e && e.message ? e.message : String(e)) + " @ " + (e.filename || "") + ":" + (e.lineno || ""));
    });
    window.addEventListener("unhandledrejection", (e) => {
      const r = e && e.reason;
      __showInitError("Unhandled promise rejection", r && (r.stack || r.message) ? (r.stack || r.message) : String(r));
    });

    const preview = document.getElementById("preview");
    const jsonOutput = document.getElementById("jsonOutput");
    const paramPanel = document.getElementById("paramPanel");
    const paramToggle = document.getElementById("paramToggle");
    const paramControls = document.getElementById("paramControls");
    const paramHint = document.getElementById("paramHint");
    const meshResults = document.getElementById("meshResults");
    const simResults = document.getElementById("simResults");
    const analysisResults = document.getElementById("analysisResults");
    const meshOutputPanel = document.getElementById("meshOutputPanel");
    const shapePcaOutputPanel = document.getElementById("shapePcaOutputPanel");
    const simOutputPanel = document.getElementById("simOutputPanel");
    const summaryBox = document.getElementById("summaryBox");
    const engineeringChatPanel = document.getElementById("engineeringChatPanel");
    const engineeringChatToggle = document.getElementById("engineeringChatToggle");
    const uploadContextPanel = document.getElementById("uploadContextPanel");
    const uploadContextToggle = document.getElementById("uploadContextToggle");
    const chatShell = document.getElementById("chatShell");
    const chatForm = document.getElementById("chatForm");
    const chatInput = document.getElementById("chatInput");
    const chatLog = document.getElementById("chatLog");
    const chatSend = document.getElementById("chatSend");
    const knowledgeSourceInputs = Array.from(document.querySelectorAll('input[name="knowledgeSource"]'));
    const activityPanel = document.getElementById("activityPanel");
    const activityLabel = document.getElementById("activityLabel");
    const activityPhase = document.getElementById("activityPhase");
    const activityFill = document.getElementById("activityFill");
    const previewProgress = document.getElementById("previewProgress");
    const previewProgressLabel = document.getElementById("previewProgressLabel");
    const previewProgressPhase = document.getElementById("previewProgressPhase");
    const previewProgressFill = document.getElementById("previewProgressFill");
    const contextFile = document.getElementById("contextFile");
    const uploadContextButton = document.getElementById("uploadContextButton");
    const fairEngineeringSearchButton = document.getElementById("fairEngineeringSearchButton");
    const fairEngineeringSearchStatus = document.getElementById("fairEngineeringSearchStatus");
    const attachmentList = document.getElementById("attachmentList");
    const familyGrid = document.getElementById("familyGrid");
    const categoryDetail = document.getElementById("categoryDetail");
    const categoryGrid = document.getElementById("categoryGrid");
    const categorySubhead = document.getElementById("categorySubhead");
    const categoryHint = document.getElementById("categoryHint");
    const downloadBtn = document.getElementById("downloadBtn");
    const downloadMenu = document.getElementById("downloadMenu");
    let activeViewer = null;
    let activeMeshViewer = null;
    let activeFemViewer = null;
    let activityTimer = null;
    let requestContext = [];
    let attachmentContexts = [];
    let pendingDraftPrompt = "";
    let chatHistory = [];
    let lastExport = { mesh: null, canvas: null, intent: null, prompt: "", name: "model" };
    let lastMeshResult = null;
    let lastShapePcaResult = null;
    let meshMode = "structured";
    let globalMeshTemplate = { circumferential_divisions: 96, radial_divisions: 8, axial_divisions: 16 };
    // Persistent camera so live parametric edits keep the same view angle.
    let viewerCamera = { rotationX: -0.55, rotationY: 0.78, zoom: 1 };
    let meshCamera = { rotationX: -0.55, rotationY: 0.78, zoom: 1 };
    let femCamera = { rotationX: -0.55, rotationY: 0.78, zoom: 1 };
    // Parametric editor state.
    let currentEditIntent = null;
    let baseGeometry = null;
    let engineeringChatOpen = true;
    let uploadContextOpen = false;
    let paramEditorOpen = false;
    let selectedCadEngine = "cadquery";
    let rubberBushingWorkflowActive = false;
    let rubberBushingTab = "space";
    let designSpaceCases = [];
    let targetStiffnessResult = null;
    const CLIENT_BUSHING_SPEC = Object.freeze({
      target_kx_n_mm: 88.4,
      target_ky_n_mm: 294.5,
      target_kz_n_mm: 294.5,
      inner_diameter_min_mm: 21,
      inner_diameter_max_mm: 35,
      inner_core_length_min_mm: 20,
      inner_core_length_max_mm: 71,
      outer_core_length_min_mm: 20,
      outer_core_length_max_mm: 55,
      outer_diameter_mm: 76,
      swaging_value_mm: 3,
      decking_value_mm: 0,
      internal_teeth: false,
      match_tolerance: 0.1,
    });
    let targetSearchInputs = {
      kx: CLIENT_BUSHING_SPEC.target_kx_n_mm,
      ky: CLIENT_BUSHING_SPEC.target_ky_n_mm,
      kz: CLIENT_BUSHING_SPEC.target_kz_n_mm,
      idMin: CLIENT_BUSHING_SPEC.inner_diameter_min_mm,
      idMax: CLIENT_BUSHING_SPEC.inner_diameter_max_mm,
      innerLengthMin: CLIENT_BUSHING_SPEC.inner_core_length_min_mm,
      innerLengthMax: CLIENT_BUSHING_SPEC.inner_core_length_max_mm,
      outerLengthMin: CLIENT_BUSHING_SPEC.outer_core_length_min_mm,
      outerLengthMax: CLIENT_BUSHING_SPEC.outer_core_length_max_mm,
      samples: 200,
    };
    let paramRenderQueued = false;
    let uploadNeedsParametricConfirmation = false;
    // When true, ignore the uploaded mesh and render the parametric model instead
    // (set after "Convert to editable bushing").
    let preferParametric = false;
    // Mesh-warp editing: keep the real uploaded geometry but stretch OD/ID/height.
    let meshEditMode = false;
    let overrideMeshFaces = null;
    let editableMesh = null;
    // Simulation estimate stays hidden until the user presses Simulate.
    let simShown = false;
    let simSelectedMode = "b1";
    let simAnimHandle = 0;
    let lastFemContour = null;
    let femBatchCount = 6;
    let femContourMode = 1;
    let lastStaticStiffness = null;

    if (engineeringChatToggle) {
      engineeringChatToggle.addEventListener("click", () => {
        setEngineeringChatOpen(!engineeringChatOpen);
      });
      setEngineeringChatOpen(false);
    }

    for (const input of knowledgeSourceInputs) {
      const source = input.closest(".knowledge-source");
      source?.classList.toggle("selected", input.checked);
      input.addEventListener("change", () => {
        source?.classList.toggle("selected", input.checked);
      });
    }

    if (uploadContextToggle) {
      uploadContextToggle.addEventListener("click", () => {
        setUploadContextOpen(!uploadContextOpen);
      });
      setUploadContextOpen(false);
    }

    if (paramToggle) {
      paramToggle.addEventListener("click", () => {
        setParamEditorOpen(!paramEditorOpen);
      });
      setParamEditorOpen(false);
    }

    function setEngineeringChatOpen(open) {
      engineeringChatOpen = Boolean(open);
      if (engineeringChatPanel) {
        engineeringChatPanel.classList.toggle("open", engineeringChatOpen);
      }
      if (engineeringChatToggle) {
        engineeringChatToggle.setAttribute("aria-expanded", engineeringChatOpen ? "true" : "false");
      }
      if (engineeringChatOpen) {
        autoResizeChatInput();
      }
    }

    function setUploadContextOpen(open) {
      uploadContextOpen = Boolean(open);
      if (uploadContextPanel) {
        uploadContextPanel.classList.toggle("collapsed", !uploadContextOpen);
        uploadContextPanel.classList.toggle("open", uploadContextOpen);
      }
      if (uploadContextToggle) {
        uploadContextToggle.setAttribute("aria-expanded", uploadContextOpen ? "true" : "false");
      }
    }

    function setParamEditorOpen(open) {
      paramEditorOpen = Boolean(open);
      if (paramPanel) {
        paramPanel.classList.toggle("collapsed", !paramEditorOpen);
      }
      if (paramToggle) {
        paramToggle.textContent = paramEditorOpen ? "Hide input" : "Open input";
        paramToggle.setAttribute("aria-expanded", paramEditorOpen ? "true" : "false");
      }
    }

    // Defensive: if any of these early listeners blow up (e.g. an element id
    // was renamed), do not block the chat-submit listener registered below.
    try {
      uploadContextButton.addEventListener("click", uploadContextFile);
      fairEngineeringSearchButton.addEventListener("click", () => {
        fairEngineeringSearchStatus.hidden = false;
      });
      contextFile.addEventListener("change", () => {
        if (contextFile.files && contextFile.files.length) {
          uploadContextFile();
        }
      });
    } catch (e) {
      console.error("[init] upload-context listeners failed:", e);
    }
    try {
      chatInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          chatForm.requestSubmit();
        }
      });
      chatInput.addEventListener("input", autoResizeChatInput);
      autoResizeChatInput();
      syncChatState();
    } catch (e) {
      console.error("[init] chat-input setup failed:", e);
    }

    const categoryFamilies = {
      bushing: {
        title: "Bushing types",
        hint: "Select a bushing type",
        items: [
          {
            key: "four-arm-bushing",
            title: "Four Arm Bushing",
            desc: "Four-arm voided rubber bushing",
            prompt: "I want a four arm rubber bushing for NVH isolation. Start with outer diameter 76 mm, inner-core diameter 28 mm, inner-core length 45 mm, and outer-core length 40 mm. Use four evenly spaced rubber arms and ask me for target Kx, Ky, Kz, arm thickness, sleeve details, and any missing geometry.",
          },
          {
            key: "optibush",
            title: "OptiBush",
            desc: "Target-driven optimized bushing",
            prompt: "I want an OptiBush design optimized for directional stiffness. Use outer diameter 76 mm and ask me for target Kx, Ky, Kz, inner-core diameter range, inner-core length range, outer-core length range, material, and manufacturing constraints before creating the design.",
          },
          {
            key: "rubber-bushing",
            title: "Rubber bushing",
            desc: "Suspension & chassis bushings",
            prompt: "I want a rubber suspension bushing. Outer diameter 60 mm, inner diameter 20 mm, height 40 mm. Material: rubber, Shore A 55. Please ask me anything else needed (chamfer, fillet, load direction).",
          },
          {
            key: "bonded-bushing",
            title: "Bonded Bushing",
            desc: "Rubber-metal bonded sleeve",
            prompt: "I want a rubber-metal bonded bushing. Outer diameter 60 mm with a 2 mm outer steel sleeve, inner diameter 20 mm with a 1.5 mm inner steel sleeve, height 40 mm. Rubber Shore A 55 in between. Ask me for any missing details.",
          },
        ],
      },
      mounts: {
        title: "Mount types",
        hint: "Select a mount type",
        items: [
          {
            key: "engine-mount",
            title: "Engine mount",
            desc: "Elastomeric powertrain mount",
            prompt: "I want an elastomeric engine mount for powertrain isolation. Ask me for the installation envelope, mounting interfaces, supported mass, target stiffness in X, Y, and Z, material hardness, preload, and allowable displacement.",
          },
          {
            key: "hydraulic-mount",
            title: "Hydraulic mount",
            desc: "Fluid-filled NVH mount",
            prompt: "I want a hydraulic powertrain mount. Ask me for the outer envelope, mounting interfaces, rubber stiffness targets, fluid chamber and inertia-track requirements, supported mass, preload, and operating frequency range.",
          },
          {
            key: "strut-mount",
            title: "Strut mount",
            desc: "Suspension top mount",
            prompt: "I want a suspension strut top mount. Ask me for the body and bearing diameters, total height, mounting-hole pattern, rubber hardness, axial and radial stiffness targets, maximum load, and allowable travel.",
          },
        ],
      },
      anonymous: {
        title: "Anonymous component types",
        hint: "Select another component type",
        items: [
          {
            key: "damper",
            title: "Damper / decoupler",
            desc: "Vibration damper element",
            prompt: "I want a hydraulic damper / decoupler mount. Body diameter 50 mm, height 60 mm, rubber Shore A 50, with an inner steel sleeve 12 mm bore. Ask me for any missing details.",
          },
        ],
      },
    };

    let activeFamily = "";

    function renderCategoryFamily(familyKey) {
      const resolvedKey = categoryFamilies[familyKey] ? familyKey : "";
      if (!resolvedKey) {
        activeFamily = "";
        if (categoryDetail) categoryDetail.classList.remove("open");
        if (categoryGrid) categoryGrid.innerHTML = "";
        for (const chip of familyGrid.querySelectorAll("[data-family]")) {
          chip.classList.remove("active");
        }
        return;
      }
      const family = categoryFamilies[resolvedKey];
      activeFamily = resolvedKey;
      if (categoryDetail) categoryDetail.classList.add("open");
      if (categorySubhead) categorySubhead.textContent = family.title;
      if (categoryHint) categoryHint.textContent = family.hint;
      for (const chip of familyGrid.querySelectorAll("[data-family]")) {
        chip.classList.toggle("active", chip.dataset.family === resolvedKey);
      }
      categoryGrid.innerHTML = family.items.map((item) => (
        '<button type="button" class="category-chip" data-category="' + escapeHtml(item.key) + '">' +
        '<strong>' + escapeHtml(item.title) + '</strong>' +
        '<span>' + escapeHtml(item.desc) + '</span>' +
        '</button>'
      )).join("");
    }

    function findCategoryItem(key) {
      const family = categoryFamilies[activeFamily] || categoryFamilies.bushing;
      return family.items.find((item) => item.key === key) || null;
    }

    familyGrid.addEventListener("click", (event) => {
      const button = event.target.closest("[data-family]");
      if (!button) return;
      renderCategoryFamily(button.dataset.family);
    });

    categoryGrid.addEventListener("click", (event) => {
      const button = event.target.closest(".category-chip");
      if (!button) return;
      const key = button.dataset.category;
      const item = findCategoryItem(key);
      if (!item) return;
      for (const chip of categoryGrid.querySelectorAll(".category-chip")) {
        chip.classList.toggle("active", chip === button);
      }
      chatInput.value = item.prompt;
      autoResizeChatInput();
      chatInput.focus();
      chatInput.setSelectionRange(chatInput.value.length, chatInput.value.length);
    });

    renderCategoryFamily(activeFamily);
    bindCadEngineSelector();

    downloadBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      if (downloadBtn.disabled) return;
      const open = downloadMenu.classList.toggle("open");
      downloadBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.addEventListener("click", (event) => {
      if (!downloadMenu.contains(event.target) && event.target !== downloadBtn) {
        downloadMenu.classList.remove("open");
        downloadBtn.setAttribute("aria-expanded", "false");
      }
    });
    downloadMenu.addEventListener("click", (event) => {
      const item = event.target.closest(".download-item");
      if (!item) return;
      downloadMenu.classList.remove("open");
      downloadBtn.setAttribute("aria-expanded", "false");
      Promise.resolve(handleDownload(item.dataset.format)).catch((error) => {
        appendMsg("bot", "Download failed: " + (error && error.message ? error.message : error));
      });
    });

    chatForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = chatInput.value.trim();
      if (!text) return;
      appendMsg("user", text);
      chatHistory.push({role: "user", content: text});
      const approvedDraft = pendingDraftPrompt && isApproval(text);
      const requestText = approvedDraft ? pendingDraftPrompt : text;
      requestContext.push(requestText);
      if (!approvedDraft) {
        pendingDraftPrompt = "";
      }
      chatInput.value = "";
      autoResizeChatInput();
      chatSend.disabled = true;
      chatSend.textContent = "Working...";
      startActivity("Processing CAD request", [
        "Reading request",
        "Parsing CAD intent",
        "Preparing preview"
      ]);
      const thinking = appendMsg("bot", "Reviewing your request...");
      try {
        const prompt = buildFullPrompt();
        const response = await fetch("/chat-cad", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            message: text,
            prompt,
            history: chatHistory.slice(0, -1),
            knowledge_sources: selectedKnowledgeSources(),
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "CAD parsing failed");
        }
        const intent = payload.cad_intent || {};
        const previewReady = Boolean(payload.preview_ready);
        jsonOutput.textContent = JSON.stringify(intent, null, 2);
        lastExport.intent = intent;
        lastExport.prompt = prompt;
        lastExport.name = exportBaseName(intent);
        downloadBtn.disabled = !intent || Object.keys(intent).length === 0;
        // A fresh chat generation renders the parsed intent, not a prior mesh-warp edit.
        preferParametric = false;
        meshEditMode = false;
        overrideMeshFaces = null;
        rubberBushingWorkflowActive = false;
        lastMeshResult = null;
        lastStaticStiffness = null;
        simShown = false;
        simSelectedMode = "b1";
        stopSimAnimation();
        if (previewReady) {
          try {
            await render3DPreview(intent);
            setParamEditorOpen(false);
            buildParamControls(intent);
          } catch (previewError) {
            cleanupViewer();
            preview.innerHTML = '<div class="placeholder"><p class="muted">The CAD intent was parsed, but the interactive preview library could not be loaded.</p></div>';
          }
        } else {
          cleanupViewer();
          currentEditIntent = null;
          setParamEditorOpen(false);
          buildParamControls(null);
          preview.innerHTML = '<div class="placeholder"><p class="muted">I need one more engineering detail before I can show a useful preview.</p></div>';
        }
        updateSummary(intent);
        thinking.textContent = payload.assistant_message || formatChatSummary(intent);
        renderConsultedSources(thinking, payload.consulted_sources || []);
        chatHistory.push({role: "assistant", content: thinking.textContent});
        completeActivity(previewReady ? "Preview updated" : "More detail needed");
      } catch (error) {
        thinking.classList.add("err");
        thinking.textContent = error.message;
        summaryBox.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
        preview.innerHTML = `<pre class="error-box">${escapeHtml(error.message)}</pre>`;
        failActivity("Request failed");
      } finally {
        chatSend.disabled = false;
        chatSend.textContent = "Send";
        chatInput.focus();
      }
    });

    async function uploadContextFile() {
      if (uploadContextButton.disabled) {
        return;
      }
      const file = contextFile.files && contextFile.files[0];
      if (!file) {
        appendMsg("bot", "Choose a PDF, image, or CAD file first.");
        return;
      }

      // A new upload starts from its exact mesh again (undo any prior convert).
      preferParametric = false;
      meshEditMode = false;
      overrideMeshFaces = null;
      editableMesh = null;
      currentEditIntent = null;
      baseGeometry = null;
      lastExport.intent = null;
      lastExport.prompt = "";
      lastExport.name = "model";
      lastMeshResult = null;
      lastShapePcaResult = null;
      lastStaticStiffness = null;
      designSpaceCases = [];
      targetStiffnessResult = null;
      uploadNeedsParametricConfirmation = false;
      simShown = false;
      simSelectedMode = "b1";
      stopSimAnimation();

      uploadContextButton.disabled = true;
      uploadContextButton.textContent = "Reading...";
      startActivity("Reading uploaded file", [
        "Uploading file",
        "Extracting context",
        "Drafting prompt"
      ]);
      try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/upload-context", {
          method: "POST",
          body: formData
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "File upload failed");
        }
        // Parse real geometry from STL uploads so the 3D viewer shows the actual part.
        let meshNote = "";
        const lowerName = (file.name || "").toLowerCase();
        let uploadBuffer = null;
        if (lowerName.endsWith(".stl")) {
          try {
            uploadBuffer = await file.arrayBuffer();
            const mesh = parseStlMesh(uploadBuffer, payload.filename);
            if (mesh) {
              payload.clientMesh = mesh;
              payload.measured = measuredSummaryFromMesh(mesh);
              const triNote = mesh.shown < mesh.original
                ? ` (preview simplified to ${mesh.shown.toLocaleString()} of ${mesh.original.toLocaleString()} triangles)`
                : ` (${mesh.original.toLocaleString()} triangles)`;
              meshNote = `Loaded the actual STL geometry into the 3D viewer${triNote}.`;
            } else {
              meshNote = "The STL could not be parsed for an exact preview, so I will show a sized model instead.";
            }
          } catch (meshError) {
            meshNote = "The STL could not be parsed for an exact preview, so I will show a sized model instead.";
          }
        } else if (lowerName.endsWith(".step") || lowerName.endsWith(".stp")) {
          uploadBuffer = await file.arrayBuffer();
          meshNote = "STEP file received. I can read its dimensions, but an exact surface preview needs the geometry kernel, so I will show a sized model for now.";
        }
        if (uploadBuffer && payload.exact_fem && payload.exact_fem.supported) {
          payload.upload_data_base64 = arrayBufferToBase64(uploadBuffer);
          payload.upload_filename = payload.filename || file.name || "uploaded_geometry.stl";
          payload.upload_content_type = file.type || "application/octet-stream";
        }

        if (rubberBushingWorkflowActive) {
          attachmentContexts = [];
        }
        attachmentContexts.push(payload);
        renderAttachmentList();
        pendingDraftPrompt = draftPromptFromAttachments();

        if (payload.clientMesh) {
          try {
            await render3DPreview({ part_type: "uploaded", geometry: {} });
            lastExport.name = exportBaseName({ part_type: payload.filename });
            downloadBtn.disabled = false;
            currentEditIntent = null;
            setParamEditorOpen(false);
            buildParamControls(null);
          } catch (previewError) {
            cleanupViewer();
          }
        }

        const autoBushingDims = payload.clientMesh ? measureBushingFromMesh(payload.clientMesh) : null;
        if (autoBushingDims && !rubberBushingWorkflowActive) {
          selectedCadEngine = "openscad";
          meshMode = "structured";
          pendingDraftPrompt = pendingDraftPrompt || "Rubber bushing from uploaded STL geometry";
          convertUploadedToBushing();
          renderMeshPanel();
          renderSimPanel();
          summaryBox.innerHTML = `
            <p><strong>Upload connected to Rubber bushing.</strong> The uploaded STL stays as the front-end geometry example. Exact mesh/FEM will run only when the STL is a watertight solid; no surrogate cylinder is generated for failed exact meshing.</p>
          `;
          appendMsg("bot", `Extracted file context from ${payload.filename}.\n\n${payload.summary}\n\n${meshNote || "Loaded uploaded STL geometry."}`);
          contextFile.value = "";
          completeActivity("Rubber bushing loaded");
          return;
        }
        if (!rubberBushingWorkflowActive) {
          rubberBushingWorkflowActive = true;
          rubberBushingTab = "space";
          designSpaceCases = [];
          targetStiffnessResult = null;
        }

        const messageParts = [`Extracted file context from ${payload.filename}.`, payload.summary];
        if (meshNote) {
          messageParts.push(meshNote);
        }
        if (payload.exact_fem && payload.exact_fem.supported) {
          messageParts.push(payload.exact_fem.message || "Uploaded geometry is stored for exact mesh/FEM.");
        }
        messageParts.push(`Proposed short CAD prompt:\\n${pendingDraftPrompt}`);
        if (rubberBushingWorkflowActive) {
          selectedCadEngine = "openscad";
          meshMode = "structured";
          const baseIntent = defaultRubberBushingIntent();
          if (payload.clientMesh) {
            const measuredDims = measureBushingFromMesh(payload.clientMesh);
            if (measuredDims) {
              baseIntent.geometry.outer_diameter_mm = measuredDims.outer_diameter_mm;
              baseIntent.geometry.inner_diameter_mm = measuredDims.inner_diameter_mm;
              baseIntent.geometry.height_mm = measuredDims.height_mm;
            }
          }
          uploadNeedsParametricConfirmation = !payload.clientMesh;
          lastExport.prompt = pendingDraftPrompt || "Rubber bushing structured parametric JSON";
          lastExport.intent = applyUploadedBushingPocTopology(baseIntent);
          currentEditIntent = lastExport.intent;
          lastExport.cadEngine = selectedCadEngine;
          jsonOutput.textContent = JSON.stringify(lastExport.intent, null, 2);
          syncDownloadItems();
          setParamEditorOpen(true);
          buildParamControls(lastExport.intent);
          if (payload.clientMesh) {
            await generateRubberParametricCad(lastExport.intent);
            await runGmshMesh();
            summaryBox.innerHTML = `
              <p><strong>Upload context meshed.</strong> OpenSCAD CAD and global mesh now use the same rounded-square / windowed bushing schema for this POC. Use Design Space or Target Stiffness to refine the result.</p>
            `;
          } else {
            cleanupViewer();
            preview.innerHTML = '<div class="placeholder"><p class="muted">Image/document upload is loaded as reference context. Confirm or edit OD / ID / height in Parametric input, then Generate CAD or Generate mesh.</p></div>';
            renderMeshPanel();
            renderSimPanel();
            summaryBox.innerHTML = `
              <p><strong>Upload context loaded.</strong> Image and document uploads do not contain mesh geometry. Confirm the Parametric input values before generating mesh so an earlier bushing is not reused.</p>
            `;
          }
        } else {
          messageParts.push('Type "proceed" to apply the dimensions and lock in the model, or type corrections/additional dimensions.');
          chatInput.value = "proceed";
          autoResizeChatInput();
          summaryBox.innerHTML = `
            <p><strong>Awaiting approval.</strong> Review the proposed short prompt in the chat. Type "proceed" to generate the structured CAD result and preview, or add corrections.</p>
          `;
        }
        appendMsg("bot", messageParts.join("\\n\\n"));
        contextFile.value = "";
        completeActivity(payload.clientMesh ? "Geometry loaded" : "Draft ready");
      } catch (error) {
        appendMsg("bot err", error.message);
        failActivity("Upload failed");
      } finally {
        uploadContextButton.disabled = false;
        uploadContextButton.textContent = "Attach File";
      }
    }

    function buildFullPrompt() {
      const parts = [];
      if (attachmentContexts.length) {
        parts.push("Use the following uploaded engineering context when interpreting the CAD request.");
        for (const context of attachmentContexts) {
          parts.push(context.prompt_context);
        }
      }
      if (requestContext.length) {
        parts.push("User chat request:");
        parts.push(requestContext.join("\\n"));
      }
      parts.push(
        "Return a CAD intent that can be used to generate the best possible 3D CAD model preview. Do not invent missing critical dimensions."
      );
      return parts.join("\\n\\n");
    }

    function draftPromptFromAttachments() {
      if (!attachmentContexts.length) {
        return "";
      }
      const summaries = attachmentContexts.map((context) => context.summary).join(" ");
      const measured = attachmentContexts
        .map((context) => context.measured)
        .filter(Boolean)
        .join(" ");
      const lines = [
        "Create a CAD model from the uploaded engineering context.",
        summaries,
      ];
      if (measured) {
        lines.push(measured);
      }
      lines.push(
        "Extract the part type, material, dimensions, chamfer/fillet details, and any status or validation hints from the uploaded file.",
        "If critical dimensions are missing, mark them as missing rather than inventing them."
      );
      return lines.join(" ");
    }

    function isApproval(value) {
      return /^(proceed|agree|approved|approve|yes|ok|okay|generate|create|go)$/i.test(value.trim());
    }

    function renderAttachmentList() {
      if (!attachmentContexts.length) {
        attachmentList.innerHTML = "";
        return;
      }
      attachmentList.innerHTML = attachmentContexts.map((context, index) => `
        <div class="attachment-item">
          <div>
            <strong>${escapeHtml(context.filename)}</strong>
            <span>${escapeHtml(context.file_kind)} · ${formatBytes(context.size_bytes)}</span>
          </div>
          <button type="button" data-remove-context="${index}">Remove</button>
        </div>
      `).join("");

      for (const button of attachmentList.querySelectorAll("[data-remove-context]")) {
        button.addEventListener("click", () => {
          const index = Number(button.dataset.removeContext);
          attachmentContexts.splice(index, 1);
          pendingDraftPrompt = attachmentContexts.length ? draftPromptFromAttachments() : "";
          renderAttachmentList();
        });
      }
    }

    function formatBytes(value) {
      const bytes = Number(value) || 0;
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function formatChatSummary(intent) {
      const missing = intent.missing_information || [];
      const summary = [
        `Summary: ${formatPartName(intent.part_type)}${intent.material?.name ? " in " + intent.material.name : ""}.`,
        `Dimensions: ${formatDimensions(intent.geometry || {})}.`,
        "Interactive preview created on the right."
      ];
      if (missing.length) {
        summary.push(`Needs clarification: ${missing.join("; ")}.`);
      }
      return summary.join("\\n");
    }

    function updateSummary(intent) {
      const missing = intent.missing_information || [];
      summaryBox.innerHTML = `
        <dl>
          <dt>Part</dt><dd>${escapeHtml(formatPartName(intent.part_type))}</dd>
          <dt>Material</dt><dd>${escapeHtml(intent.material?.name || "Not specified")}</dd>
          <dt>Dimensions</dt><dd>${escapeHtml(formatDimensions(intent.geometry || {}))}</dd>
          <dt>Output</dt><dd>Interactive preview with live parametric editing</dd>
          <dt>Status</dt><dd>${missing.length ? escapeHtml("Clarification needed: " + missing.join("; ")) : "Ready for engineering review"}</dd>
        </dl>
      `;
    }

    function formatPartName(value) {
      return String(value || "unknown part").replaceAll("_", " ");
    }

    function formatDimensions(geometry) {
      const labels = {
        outer_diameter_mm: "OD",
        inner_diameter_mm: "ID",
        height_mm: "height",
        length_mm: "length",
        width_mm: "width",
        thickness_mm: "thickness",
        chamfer_mm: "chamfer",
        fillet_mm: "fillet",
        coil_count: "coils"
      };
      const parts = [];
      for (const [key, label] of Object.entries(labels)) {
        const value = geometry[key];
        if (value !== null && value !== undefined && value !== "") {
          parts.push(key === "coil_count" ? `${value} ${label}` : `${label} ${value} mm`);
        }
      }
      return parts.length ? parts.join(", ") : "No dimensions specified";
    }

    function appendMsg(role, text) {
      const el = document.createElement("div");
      el.className = `msg ${role}`;
      el.textContent = text;
      chatLog.appendChild(el);
      syncChatState();
      chatLog.scrollTop = chatLog.scrollHeight;
      return el;
    }

    function selectedKnowledgeSources() {
      return knowledgeSourceInputs
        .filter((input) => input.checked)
        .map((input) => input.value);
    }

    function renderConsultedSources(message, sourceNames) {
      if (!message || !Array.isArray(sourceNames) || !sourceNames.length) {
        return;
      }
      const container = document.createElement("div");
      container.className = "consulted-sources";
      container.setAttribute("aria-label", "Knowledge sources used for this answer");
      for (const sourceName of sourceNames) {
        const badge = document.createElement("span");
        badge.className = "consulted-source";
        badge.textContent = `Source: ${sourceName}`;
        container.appendChild(badge);
      }
      message.appendChild(container);
    }

    function autoResizeChatInput() {
      chatInput.style.height = "auto";
      const maxHeight = chatShell.classList.contains("idle") ? 280 : 220;
      chatInput.style.height = `${Math.min(chatInput.scrollHeight, maxHeight)}px`;
    }

    function syncChatState() {
      const messageCount = chatLog.querySelectorAll(".msg").length;
      chatShell.classList.toggle("idle", messageCount <= 1);
      autoResizeChatInput();
    }

    function startActivity(label, phases) {
      stopActivityTimer();
      activityPanel.className = "activity-panel active";
      activityLabel.textContent = label;
      activityPhase.textContent = phases[0] || "Working";
      activityFill.style.width = "10%";
      previewProgress.className = "preview-progress active";
      previewProgressLabel.textContent = label;
      previewProgressPhase.textContent = phases[0] || "Working";
      previewProgressFill.style.width = "10%";
      let progress = 10;
      let phaseIndex = 0;
      activityTimer = window.setInterval(() => {
        progress = Math.min(progress + 12, 90);
        activityFill.style.width = `${progress}%`;
        previewProgressFill.style.width = `${progress}%`;
        if (phaseIndex < phases.length - 1 && progress >= (phaseIndex + 1) * 30) {
          phaseIndex += 1;
          activityPhase.textContent = phases[phaseIndex];
          previewProgressPhase.textContent = phases[phaseIndex];
        }
      }, 700);
    }

    function completeActivity(phaseText) {
      stopActivityTimer();
      activityPanel.className = "activity-panel active done";
      activityPhase.textContent = phaseText;
      activityFill.style.width = "100%";
      previewProgress.className = "preview-progress active done";
      previewProgressPhase.textContent = phaseText;
      previewProgressFill.style.width = "100%";
      window.setTimeout(() => {
        activityPanel.className = "activity-panel";
        activityLabel.textContent = "Ready";
        activityPhase.textContent = "Waiting";
        activityFill.style.width = "0%";
        previewProgress.className = "preview-progress";
        previewProgressFill.style.width = "0%";
      }, 1400);
    }

    function failActivity(phaseText) {
      stopActivityTimer();
      activityPanel.className = "activity-panel active error";
      activityPhase.textContent = phaseText;
      activityFill.style.width = "100%";
      previewProgress.className = "preview-progress active error";
      previewProgressPhase.textContent = phaseText;
      previewProgressFill.style.width = "100%";
    }

    function stopActivityTimer() {
      if (activityTimer) {
        window.clearInterval(activityTimer);
        activityTimer = null;
      }
    }

    async function render3DPreview(cadIntent) {
      cleanupViewer();
      const uploaded = (preferParametric || overrideMeshFaces) ? null : pickUploadedMesh();
      const mesh = overrideMeshFaces
        ? { faces: overrideMeshFaces }
        : (uploaded ? { faces: uploaded.faces } : createPreviewMesh(cadIntent || {}));
      lastExport.mesh = mesh;
      lastExport.intent = cadIntent || {};

      const container = document.createElement("div");
      container.className = "viewer3d";
      preview.innerHTML = "";
      preview.appendChild(container);

      try {
        const viewerModule = await import("/static/cad-viewer.js");
        const material = cadIntent && cadIntent.material;
        activeViewer = viewerModule.createCadViewer({
          container,
          faces: mesh.faces,
          cameraState: viewerCamera,
          materialName: typeof material === "string" ? material : (material && material.name),
        });
        lastExport.canvas = activeViewer.canvas;
        return;
      } catch (error) {
        console.warn("Three.js CAD preview unavailable; using Canvas fallback.", error);
        container.remove();
        renderCanvas3DPreview(cadIntent, mesh);
      }
    }

    function renderCanvas3DPreview(cadIntent, mesh) {
      const container = document.createElement("div");
      container.className = "viewer3d";
      container.dataset.renderer = "canvas";
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-label", "Interactive CAD preview");
      container.appendChild(canvas);
      preview.innerHTML = "";
      preview.appendChild(container);

      const context = canvas.getContext("2d");
      if (!context) {
        throw new Error("Canvas rendering is not available in this browser.");
      }

      lastExport.canvas = canvas;
      const bounds = computeMeshBounds(mesh.faces);
      const size = bounds.size;
      const extent = Math.max(size.x, size.y, size.z, 40);
      const center = bounds.center;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

      const state = {
        width: 0,
        height: 0,
        rotationX: viewerCamera.rotationX,
        rotationY: viewerCamera.rotationY,
        zoom: viewerCamera.zoom,
        dragging: false,
        lastX: 0,
        lastY: 0,
        animationFrameId: 0,
        renderQueued: false,
      };

      const lightDirection = normalizeVector({ x: 0.35, y: 0.85, z: 0.4 });
      const centeredFaces = mesh.faces.map((face) => ({
        color: face.color,
        smoothPreview: Boolean(face.smoothPreview),
        points: face.points.map((point) => ({
          x: point.x - center.x,
          y: point.y - center.y,
          z: point.z - center.z,
        })),
      }));
      const gridLevel = -size.y * 0.55;

      function scheduleRender() {
        if (state.renderQueued) {
          return;
        }
        state.renderQueued = true;
        state.animationFrameId = requestAnimationFrame(() => {
          state.renderQueued = false;
          drawPreview();
        });
      }

      function resizeCanvas() {
        state.width = Math.max(320, Math.floor(preview.clientWidth || 720));
        state.height = Math.max(320, Math.floor(preview.clientHeight || 470));
        canvas.width = Math.floor(state.width * pixelRatio);
        canvas.height = Math.floor(state.height * pixelRatio);
        canvas.style.width = `${state.width}px`;
        canvas.style.height = `${state.height}px`;
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        scheduleRender();
      }

      function projectWorldPoint(point) {
        return projectRotatedPoint(rotatePoint(point, state.rotationX, state.rotationY));
      }

      function projectRotatedPoint(rotated) {
        const scale = (Math.min(state.width, state.height) / (extent * 2.4)) * state.zoom;
        return {
          x: state.width / 2 + rotated.x * scale,
          y: state.height / 2 - rotated.y * scale,
          z: rotated.z,
        };
      }

      function drawGrid() {
        const spacing = niceGridSpacing(extent);
        const radius = Math.max(extent * 0.9, spacing * 4);
        context.save();
        context.lineWidth = 1;
        context.strokeStyle = "#dbe6f2";
        for (let value = -radius; value <= radius + 0.001; value += spacing) {
          drawLine(
            projectWorldPoint({ x: value, y: gridLevel, z: -radius }),
            projectWorldPoint({ x: value, y: gridLevel, z: radius })
          );
          drawLine(
            projectWorldPoint({ x: -radius, y: gridLevel, z: value }),
            projectWorldPoint({ x: radius, y: gridLevel, z: value })
          );
        }
        context.restore();
      }

      function drawLine(start, end) {
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      }

      function drawFaces() {
        const projectedFaces = centeredFaces
          .map((face) => {
            const rotatedPoints = face.points.map((point) => rotatePoint(point, state.rotationX, state.rotationY));
            const projectedPoints = rotatedPoints.map((point) => projectRotatedPoint(point));
            const normal = computeFaceNormal(rotatedPoints);
            const intensity = clamp(dotProduct(normal, lightDirection) * 0.55 + 0.72, 0.28, 0.95);
            const fillIntensity = face.smoothPreview ? 0.78 : intensity;
            const averageDepth = rotatedPoints.reduce((sum, point) => sum + point.z, 0) / rotatedPoints.length;
            return {
              projectedPoints,
              averageDepth,
              fill: shadeColor(face.color, fillIntensity),
              stroke: shadeColor(face.color, Math.max(intensity - 0.2, 0.18)),
              smoothPreview: face.smoothPreview,
            };
          })
          .sort((left, right) => left.averageDepth - right.averageDepth);

        for (const face of projectedFaces) {
          context.beginPath();
          context.moveTo(face.projectedPoints[0].x, face.projectedPoints[0].y);
          for (let index = 1; index < face.projectedPoints.length; index += 1) {
            context.lineTo(face.projectedPoints[index].x, face.projectedPoints[index].y);
          }
          context.closePath();
          context.fillStyle = face.fill;
          context.fill();
          if (!face.smoothPreview) {
            context.strokeStyle = face.stroke;
            context.lineWidth = 1.2;
            context.stroke();
          }
        }
      }

      function drawPreview() {
        context.clearRect(0, 0, state.width, state.height);
        context.fillStyle = "#f8fbff";
        context.fillRect(0, 0, state.width, state.height);
        drawGrid();
        drawFaces();
        drawAxisIndicator();
      }

      function drawAxisIndicator() {
        // Small orientation triad pinned to the bottom-left corner. It rotates with
        // the camera so the user can always read the part orientation. Viewer world
        // Y is "up" (the part height), so it is labelled Z to match CAD convention.
        const originX = 52;
        const originY = state.height - 52;
        const axisLength = 30;
        const axes = [
          { vector: { x: 1, y: 0, z: 0 }, label: "X", color: "#e23b3b" },
          { vector: { x: 0, y: 0, z: 1 }, label: "Y", color: "#2faa4d" },
          { vector: { x: 0, y: 1, z: 0 }, label: "Z", color: "#2f7bff" },
        ];
        const projected = axes
          .map((axis) => {
            const rotated = rotatePoint(axis.vector, state.rotationX, state.rotationY);
            return {
              label: axis.label,
              color: axis.color,
              tipX: originX + rotated.x * axisLength,
              tipY: originY - rotated.y * axisLength,
              depth: rotated.z,
            };
          })
          .sort((left, right) => left.depth - right.depth);

        context.save();
        context.lineWidth = 2.5;
        context.lineCap = "round";
        context.font = "bold 11px system-ui, -apple-system, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillStyle = "#9aa7b8";
        context.beginPath();
        context.arc(originX, originY, 2.5, 0, Math.PI * 2);
        context.fill();
        for (const axis of projected) {
          context.strokeStyle = axis.color;
          context.beginPath();
          context.moveTo(originX, originY);
          context.lineTo(axis.tipX, axis.tipY);
          context.stroke();
          const labelX = originX + (axis.tipX - originX) * 1.2;
          const labelY = originY + (axis.tipY - originY) * 1.2;
          context.fillStyle = axis.color;
          context.beginPath();
          context.arc(labelX, labelY, 7.5, 0, Math.PI * 2);
          context.fill();
          context.fillStyle = "#ffffff";
          context.fillText(axis.label, labelX, labelY);
        }
        context.restore();
      }

      function onPointerDown(event) {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      }

      function onPointerMove(event) {
        if (!state.dragging) {
          return;
        }
        const deltaX = event.clientX - state.lastX;
        const deltaY = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        state.rotationY += deltaX * 0.01;
        state.rotationX = clamp(state.rotationX + deltaY * 0.01, -1.35, 1.35);
        viewerCamera.rotationX = state.rotationX;
        viewerCamera.rotationY = state.rotationY;
        scheduleRender();
      }

      function onPointerUp(event) {
        state.dragging = false;
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      }

      function onWheel(event) {
        event.preventDefault();
        const nextZoom = state.zoom * (event.deltaY > 0 ? 0.92 : 1.08);
        state.zoom = clamp(nextZoom, 0.55, 2.8);
        viewerCamera.zoom = state.zoom;
        scheduleRender();
      }

      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointerleave", onPointerUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      const resizeObserver = new ResizeObserver(resizeCanvas);
      resizeObserver.observe(preview);
      resizeCanvas();

      activeViewer = {
        resizeObserver,
        dispose() {
          cancelAnimationFrame(state.animationFrameId);
          resizeObserver.disconnect();
          canvas.removeEventListener("pointerdown", onPointerDown);
          canvas.removeEventListener("pointermove", onPointerMove);
          canvas.removeEventListener("pointerup", onPointerUp);
          canvas.removeEventListener("pointerleave", onPointerUp);
          canvas.removeEventListener("wheel", onWheel);
        },
      };
    }

    // ===== Uploaded mesh (real STL geometry) support =====
    const UPLOAD_MESH_COLOR = "#8b929b";
    const MAX_PREVIEW_TRIANGLES = 60000;

    function parseStlMesh(buffer, sourceLabel) {
      if (!buffer || buffer.byteLength < 84) {
        return null;
      }
      const view = new DataView(buffer);
      const headerCount = view.getUint32(80, true);
      if (84 + headerCount * 50 === buffer.byteLength) {
        return finalizeUploadedMesh(parseBinaryStl(view, headerCount), sourceLabel);
      }
      const text = new TextDecoder("utf-8").decode(new Uint8Array(buffer));
      if (/facet\\s+normal/i.test(text)) {
        return finalizeUploadedMesh(parseAsciiStl(text), sourceLabel);
      }
      if (headerCount > 0 && 84 + headerCount * 50 <= buffer.byteLength) {
        return finalizeUploadedMesh(parseBinaryStl(view, headerCount), sourceLabel);
      }
      return null;
    }

    function parseBinaryStl(view, triCount) {
      const faces = [];
      const stride = triCount > MAX_PREVIEW_TRIANGLES ? Math.ceil(triCount / MAX_PREVIEW_TRIANGLES) : 1;
      for (let i = 0; i < triCount; i += stride) {
        const base = 84 + i * 50;
        faces.push(makeStlFace(
          readStlVec(view, base + 12),
          readStlVec(view, base + 24),
          readStlVec(view, base + 36)
        ));
      }
      return { faces, original: triCount };
    }

    function readStlVec(view, offset) {
      return {
        x: view.getFloat32(offset, true),
        y: view.getFloat32(offset + 4, true),
        z: view.getFloat32(offset + 8, true),
      };
    }

    function parseAsciiStl(text) {
      const faces = [];
      const verts = [];
      const tokens = text.split(/\\s+/);
      for (let i = 0; i < tokens.length; i += 1) {
        if (tokens[i] !== "vertex") {
          continue;
        }
        const x = parseFloat(tokens[i + 1]);
        const y = parseFloat(tokens[i + 2]);
        const z = parseFloat(tokens[i + 3]);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          verts.push({ x, y, z });
          if (verts.length === 3) {
            faces.push(makeStlFace(verts[0], verts[1], verts[2]));
            verts.length = 0;
          }
        }
      }
      const original = faces.length;
      let used = faces;
      if (faces.length > MAX_PREVIEW_TRIANGLES) {
        const stride = Math.ceil(faces.length / MAX_PREVIEW_TRIANGLES);
        used = faces.filter((face, index) => index % stride === 0);
      }
      return { faces: used, original };
    }

    function makeStlFace(a, b, c) {
      // Reverse winding to keep outward normals after the Z-up -> Y-up swap below.
      return { color: UPLOAD_MESH_COLOR, smoothPreview: true, points: [stlPoint(a), stlPoint(c), stlPoint(b)] };
    }

    function stlPoint(p) {
      // STL is usually Z-up; the viewer is Y-up, so swap Y and Z.
      return { x: p.x, y: p.z, z: p.y };
    }

    function finalizeUploadedMesh(parsed, sourceLabel) {
      if (!parsed || !parsed.faces || !parsed.faces.length) {
        return null;
      }
      const bounds = computeMeshBounds(parsed.faces);
      return {
        faces: parsed.faces,
        bounds,
        original: parsed.original || parsed.faces.length,
        shown: parsed.faces.length,
        source: sourceLabel || "stl",
      };
    }

    function pickUploadedMesh() {
      for (let i = attachmentContexts.length - 1; i >= 0; i -= 1) {
        if (attachmentContexts[i] && attachmentContexts[i].clientMesh) {
          return attachmentContexts[i].clientMesh;
        }
      }
      return null;
    }

    function measuredSummaryFromMesh(mesh) {
      if (!mesh || !mesh.bounds) {
        return "";
      }
      const s = mesh.bounds.size;
      const dx = Math.round(s.x * 100) / 100;
      const dy = Math.round(s.y * 100) / 100;
      const dz = Math.round(s.z * 100) / 100;
      const envelope = Math.round(Math.max(s.x, s.y, s.z) * 100) / 100;
      return (
        "Measured from the uploaded geometry: bounding box approximately " +
        dx + " x " + dy + " x " + dz + " mm; maximum envelope dimension about " +
        envelope + " mm. Treat these as the real, measured dimensions and do not invent values."
      );
    }

    // ===== Phase 2b: measure a bushing's OD/ID/height from the uploaded mesh =====
    function percentile(sortedValues, fraction) {
      if (!sortedValues.length) return 0;
      const index = Math.min(
        sortedValues.length - 1,
        Math.max(0, Math.round(fraction * (sortedValues.length - 1)))
      );
      return sortedValues[index];
    }

    function measureBushingFromMesh(mesh) {
      if (!mesh || !mesh.faces || !mesh.faces.length || !mesh.bounds) {
        return null;
      }
      // The viewer Y axis is the bushing axis (STL Z-up was swapped to Y-up).
      // Radial distance is measured in the X/Z plane about the part centroid.
      let sumX = 0;
      let sumZ = 0;
      let count = 0;
      for (const face of mesh.faces) {
        for (const p of face.points) {
          sumX += p.x;
          sumZ += p.z;
          count += 1;
        }
      }
      if (!count) return null;
      const centerX = sumX / count;
      const centerZ = sumZ / count;

      const radii = [];
      for (const face of mesh.faces) {
        for (const p of face.points) {
          radii.push(Math.hypot(p.x - centerX, p.z - centerZ));
        }
      }
      radii.sort((a, b) => a - b);

      // Outer radius: high percentile to ignore stray spikes. Inner bore radius:
      // low percentile, which lands on the bore wall (the closest material to the axis).
      const outerRadius = percentile(radii, 0.99);
      let innerRadius = percentile(radii, 0.03);
      if (!(innerRadius > 0) || innerRadius >= outerRadius) {
        innerRadius = outerRadius * 0.4;
      }
      const height = mesh.bounds.size.y;
      if (!(outerRadius > 0) || !(height > 0)) {
        return null;
      }
      const round = (v) => Math.round(v * 2) / 2; // nearest 0.5 mm
      const outer = round(outerRadius * 2);
      let inner = round(innerRadius * 2);
      if (inner >= outer) inner = round(Math.max(1, outer - 1));
      return {
        outer_diameter_mm: outer,
        inner_diameter_mm: Math.max(1, inner),
        height_mm: round(height),
      };
    }

    function convertUploadedToBushing() {
      const uploaded = pickUploadedMesh();
      const dims = measureBushingFromMesh(uploaded);
      if (!dims || !uploaded || !uploaded.faces) {
        appendMsg("bot", "I could not measure clean bushing dimensions from this mesh.");
        return;
      }

      // Keep the REAL geometry: store the original vertices and the part axis so we
      // can warp (stretch) it live instead of replacing it with a plain cylinder.
      let sumX = 0;
      let sumZ = 0;
      let count = 0;
      for (const face of uploaded.faces) {
        for (const p of face.points) {
          sumX += p.x;
          sumZ += p.z;
          count += 1;
        }
      }
      const centerX = sumX / count;
      const centerZ = sumZ / count;
      const centerY = uploaded.bounds ? uploaded.bounds.center.y : 0;

      editableMesh = {
        originalFaces: uploaded.faces,
        centerX: centerX,
        centerZ: centerZ,
        centerY: centerY,
        od0: dims.outer_diameter_mm,
        id0: dims.inner_diameter_mm,
        h0: dims.height_mm,
      };

      const intent = normalizeRubberBushingIntent({
        part_type: "bushing",
        material: { name: "rubber" },
        geometry: {
          outer_diameter_mm: dims.outer_diameter_mm,
          inner_diameter_mm: dims.inner_diameter_mm,
          height_mm: dims.height_mm,
          inner_core_length_mm: Math.max(20, Math.min(71, dims.height_mm)),
          outer_core_length_mm: Math.max(20, Math.min(55, dims.height_mm)),
          swaging_value_mm: 3.0,
          decking_value_mm: 0.0,
          internal_teeth: false,
          arms: [],
          holes: [],
        },
        simulation_hints: { target_output: "cad" },
        missing_information: [],
        ui_workflow: { product_family: "bushing", bushing_type: "rubber-bushing" },
      });

      meshEditMode = true;
      preferParametric = false;
      uploadNeedsParametricConfirmation = false;
      rubberBushingWorkflowActive = true;
      rubberBushingTab = "space";
      targetStiffnessResult = null;
      currentEditIntent = intent;
      lastExport.intent = intent;
      lastExport.name = exportBaseName(intent);
      lastExport.prompt = lastExport.prompt || pendingDraftPrompt || "Rubber bushing structured parametric JSON";
      jsonOutput.textContent = JSON.stringify(intent, null, 2);
      downloadBtn.disabled = false;

      // Identity warp to start (keeps the exact uploaded shape on screen).
      overrideMeshFaces = warpEditableMeshFaces(intent.geometry);
      render3DPreview(intent).catch(() => {});
      setParamEditorOpen(true);
      buildParamControls(intent);
      appendMsg(
        "bot",
        "This bushing is now editable while keeping its real shape: outer diameter " +
        dims.outer_diameter_mm + " mm, inner diameter " + dims.inner_diameter_mm +
        " mm, height " + dims.height_mm + " mm. Design Space and Target Stiffness are now available in the Parametric input tabs."
      );
    }

    function warpEditableMeshFaces(geom) {
      if (!editableMesh) {
        return overrideMeshFaces;
      }
      const newOd = Number(geom.outer_diameter_mm) > 0 ? Number(geom.outer_diameter_mm) : editableMesh.od0;
      const newId = Number(geom.inner_diameter_mm) > 0 ? Number(geom.inner_diameter_mm) : editableMesh.id0;
      const newH = Number(geom.height_mm) > 0 ? Number(geom.height_mm) : editableMesh.h0;

      const ro0 = editableMesh.od0 / 2;
      const ri0 = editableMesh.id0 / 2;
      const ro1 = newOd / 2;
      const ri1 = newId / 2;
      const span0 = ro0 - ri0;
      const heightScale = editableMesh.h0 > 0 ? newH / editableMesh.h0 : 1;
      const cx = editableMesh.centerX;
      const cz = editableMesh.centerZ;
      const cy = editableMesh.centerY;

      return editableMesh.originalFaces.map((face) => ({
        color: face.color,
        smoothPreview: Boolean(face.smoothPreview),
        points: face.points.map((p) => {
          const dx = p.x - cx;
          const dz = p.z - cz;
          const r = Math.hypot(dx, dz);
          // Linearly remap the annular wall so the bore goes to the new ID and the
          // outer surface to the new OD; everything in between stretches with it.
          let newR;
          if (span0 > 1e-6) {
            newR = ri1 + (r - ri0) * (ro1 - ri1) / span0;
          } else {
            newR = r * (ro1 > 0 ? ro1 / Math.max(ro0, 1e-6) : 1);
          }
          if (newR < 0) newR = 0;
          const scale = r > 1e-6 ? newR / r : 0;
          return {
            x: cx + dx * scale,
            y: cy + (p.y - cy) * heightScale,
            z: cz + dz * scale,
          };
        }),
      }));
    }

    // ===== Parametric editor (Phase 1): live dimension sliders =====
    const PARAM_FIELDS = {
      outer_diameter_mm:   { label: "Outer diameter", min: 5,   max: 400, step: 0.5, fallback: 60 },
      inner_diameter_mm:   { label: "Inner diameter", min: 1,   max: 400, step: 0.5, fallback: 20 },
      height_mm:           { label: "Height",         min: 1,   max: 500, step: 0.5, fallback: 40 },
      length_mm:           { label: "Length",         min: 1,   max: 600, step: 0.5, fallback: 80 },
      width_mm:            { label: "Width",          min: 1,   max: 600, step: 0.5, fallback: 40 },
      thickness_mm:        { label: "Thickness",      min: 0.1, max: 200, step: 0.1, fallback: 5 },
      chamfer_mm:          { label: "Chamfer",        min: 0,   max: 30,  step: 0.1, fallback: 0 },
      fillet_mm:           { label: "Fillet",         min: 0,   max: 30,  step: 0.1, fallback: 0 },
      flange_diameter_mm:  { label: "Flange diameter",min: 5,   max: 500, step: 0.5, fallback: 0 },
      flange_thickness_mm: { label: "Flange thickness",min: 0,  max: 60,  step: 0.1, fallback: 0 },
      metal_sleeve_thickness_mm: { label: "Outer sleeve", min: 0, max: 80, step: 0.1, fallback: 0 },
      inner_sleeve_thickness_mm: { label: "Inner sleeve", min: 0, max: 80, step: 0.1, fallback: 0 },
      coil_count:          { label: "Coil count",     min: 1,   max: 50,  step: 1,   fallback: 8 },
    };
    const PART_FIELD_SETS = {
      bushing:      { core: ["outer_diameter_mm", "inner_diameter_mm", "height_mm"], optional: ["chamfer_mm", "fillet_mm", "metal_sleeve_thickness_mm", "inner_sleeve_thickness_mm", "flange_diameter_mm", "flange_thickness_mm"] },
      rubber_mount: { core: ["outer_diameter_mm", "inner_diameter_mm", "height_mm"], optional: ["chamfer_mm", "fillet_mm", "metal_sleeve_thickness_mm", "inner_sleeve_thickness_mm", "flange_diameter_mm", "flange_thickness_mm"] },
      plate:        { core: ["length_mm", "width_mm", "thickness_mm"], optional: ["chamfer_mm", "fillet_mm"] },
      bracket:      { core: ["length_mm", "width_mm", "thickness_mm"], optional: ["fillet_mm"] },
      spring:       { core: ["outer_diameter_mm", "height_mm", "thickness_mm"], optional: ["coil_count"] },
    };

    function roundParam(value, step) {
      const s = Number(step) || 0.1;
      const decimals = (String(s).split(".")[1] || "").length;
      return Number(value).toFixed(decimals);
    }

    // ===== Simulation (Tier 1): live analytical modal estimate =====
    // Linear-elastic isotropic properties in the tonne-mm-s unit system, so
    // frequencies come out directly in Hz. Values mirror simulate/materials.py.
    const SIM_MATERIALS = {
      steel:           { E: 210000, rho: 7.85e-9 },
      stainless_steel: { E: 193000, rho: 8.00e-9 },
      aluminum:        { E: 69000,  rho: 2.70e-9 },
      cast_iron:       { E: 110000, rho: 7.20e-9 },
      rubber:          { E: 10,     rho: 1.10e-9 },
      epdm:            { E: 6,      rho: 1.15e-9 },
      abs:             { E: 2300,   rho: 1.05e-9 },
      generic:         { E: 200000, rho: 7.85e-9 },
    };
    const SIM_ALIASES = {
      metal: "steel", "mild steel": "steel", "carbon steel": "steel",
      ss: "stainless_steel", inox: "stainless_steel",
      alu: "aluminum", aluminium: "aluminum", al: "aluminum",
      iron: "cast_iron", "natural rubber": "rubber", nr: "rubber",
      elastomer: "rubber", plastic: "abs",
    };

    function simResolveMaterial(name) {
      const key = String(name || "").trim().toLowerCase();
      if (SIM_MATERIALS[key]) return { key, ...SIM_MATERIALS[key] };
      if (SIM_ALIASES[key] && SIM_MATERIALS[SIM_ALIASES[key]]) {
        const resolved = SIM_ALIASES[key];
        return { key: resolved, ...SIM_MATERIALS[resolved] };
      }
      return { key: "generic", ...SIM_MATERIALS.generic };
    }

    function estimateBushingModal(geom, materialName) {
      const od = Number(geom && geom.outer_diameter_mm);
      const idRaw = Number(geom && geom.inner_diameter_mm) || 0;
      const length = Number(geom && geom.height_mm);
      if (!(od > 0) || !(length > 0)) return null;
      const id = Math.min(Math.max(idRaw, 0), od - 1e-3);
      const mat = simResolveMaterial(materialName);
      const E = mat.E;       // MPa = N/mm^2
      const rho = mat.rho;   // tonne/mm^3
      const area = Math.PI / 4 * (od * od - id * id);              // mm^2
      const inertia = Math.PI / 64 * (Math.pow(od, 4) - Math.pow(id, 4)); // mm^4
      const kAxial = E * area / length;                           // N/mm
      const kBending = 3 * E * inertia / Math.pow(length, 3);     // N/mm (cantilever tip)
      // Euler-Bernoulli cantilever (fixed-free) bending modes.
      const betaL = [1.875104, 4.694091, 7.854757];
      const c = Math.sqrt((E * inertia) / (rho * area * Math.pow(length, 4)));
      const bending = betaL.map((b) => (b * b / (2 * Math.PI)) * c); // Hz
      const axial1 = (1 / (4 * length)) * Math.sqrt(E / rho);        // Hz (fixed-free rod)
      return { material: mat.key, area, inertia, kAxial, kBending, bending, axial1 };
    }

    function formatHz(value) {
      if (!Number.isFinite(value)) return "\u2013";
      if (value >= 1000) return (value / 1000).toFixed(value >= 10000 ? 0 : 1) + " kHz";
      return value.toFixed(value >= 100 ? 0 : 1) + " Hz";
    }

    function formatStiffness(value) {
      if (!Number.isFinite(value)) return "\u2013";
      if (value >= 1e6) return (value / 1e6).toFixed(2) + " MN/mm";
      if (value >= 1000) return (value / 1000).toFixed(1) + " kN/mm";
      return value.toFixed(value >= 100 ? 0 : 1) + " N/mm";
    }

    // Find the best available bushing dimensions + material for an estimate:
    // a parametric bushing intent, or a raw uploaded mesh measured on the fly.
    function simSourceDims() {
      const intentType = String((currentEditIntent && currentEditIntent.part_type) || "").toLowerCase();
      if (currentEditIntent && currentEditIntent.geometry && (intentType === "bushing" || intentType === "rubber_mount")) {
        return {
          geom: currentEditIntent.geometry,
          material: (currentEditIntent.material && currentEditIntent.material.name) || "generic",
        };
      }
      if (!preferParametric && !meshEditMode) {
        const uploaded = pickUploadedMesh();
        if (uploaded) {
          const dims = measureBushingFromMesh(uploaded);
          if (dims) {
            return { geom: dims, material: "generic" };
          }
        }
      }
      return null;
    }

    function updateAnalysisResultsVisibility() {
      if (!analysisResults) return;
      const hasMesh = meshOutputPanel && meshOutputPanel.innerHTML.trim();
      const hasShapePca = shapePcaOutputPanel && shapePcaOutputPanel.innerHTML.trim();
      const hasSim = simOutputPanel && simOutputPanel.innerHTML.trim();
      analysisResults.hidden = !(hasMesh || hasShapePca || hasSim);
    }

    function ensureSimOutputPanel() {
      if (!simOutputPanel) return null;
      if (!simOutputPanel.querySelector("#simOutput")) {
        simOutputPanel.innerHTML =
          '<div class="sim-block analysis-result-block">' +
          '<div class="sim-head"><strong>Simulation result</strong>' +
          '<div class="sim-head-actions">' +
          '<button type="button" class="sim-btn" id="downloadBestCadStlBtn" disabled>Best CAD (.STL)</button>' +
          '<button type="button" class="sim-btn secondary" id="downloadSimulationExcelBtn" disabled>Simulation Excel</button>' +
          '</div></div>' +
          '<div id="simOutput"></div>' +
          '<div id="staticStiffnessContainer"></div>' +
          '<div id="simFemContainer"></div>' +
          '</div>';
      }
      bindSimulationDownloadActions();
      updateSimulationDownloadActions();
      updateAnalysisResultsVisibility();
      return document.getElementById("simOutput");
    }

    function bindSimulationDownloadActions() {
      const stlButton = document.getElementById("downloadBestCadStlBtn");
      const excelButton = document.getElementById("downloadSimulationExcelBtn");
      if (stlButton && !stlButton.dataset.bound) {
        stlButton.dataset.bound = "true";
        stlButton.addEventListener("click", downloadBestSimulationCad);
      }
      if (excelButton && !excelButton.dataset.bound) {
        excelButton.dataset.bound = "true";
        excelButton.addEventListener("click", downloadSimulationExcel);
      }
    }

    function updateSimulationDownloadActions() {
      const stlButton = document.getElementById("downloadBestCadStlBtn");
      const excelButton = document.getElementById("downloadSimulationExcelBtn");
      const intent = simulationDesignIntent();
      const hasCad = Boolean((lastExport && lastExport.mesh) || (intent && intent.geometry));
      const hasResults = Boolean(
        (lastStaticStiffness && lastStaticStiffness.status === "ok") ||
        (lastFemContour && lastFemContour.status === "ok")
      );
      if (stlButton) stlButton.disabled = !hasCad || !hasResults;
      if (excelButton) excelButton.disabled = !hasResults;
    }

    function simulationDesignIntent() {
      return (targetStiffnessResult && targetStiffnessResult.intent) ||
        (lastExport && lastExport.intent) ||
        currentEditIntent ||
        null;
    }

    function downloadBestSimulationCad() {
      try {
        const intent = simulationDesignIntent();
        let mesh = null;
        if (targetStiffnessResult && targetStiffnessResult.intent) {
          mesh = createPreviewMesh(targetStiffnessResult.intent);
        } else if (lastExport && lastExport.mesh) {
          mesh = lastExport.mesh;
        } else if (intent) {
          mesh = createPreviewMesh(intent);
        }
        if (!mesh || !Array.isArray(mesh.faces) || !mesh.faces.length) {
          throw new Error("No CAD surface is available for STL export.");
        }
        const caseName = targetStiffnessResult && targetStiffnessResult.case_id
          ? String(targetStiffnessResult.case_id).toLowerCase().replace(/[^a-z0-9_-]+/g, "_")
          : ((lastExport && lastExport.name) || exportBaseName(intent || {}) || "best_model");
        downloadBlob(stlBlob(mesh, caseName), caseName + "_best.stl");
      } catch (error) {
        appendMsg("bot", "STL export failed: " + ((error && error.message) || error));
      }
    }

    async function downloadSimulationExcel() {
      const button = document.getElementById("downloadSimulationExcelBtn");
      if (button) {
        button.disabled = true;
        button.textContent = "Preparing Excel...";
      }
      try {
        const payload = simulationReportPayload();
        const response = await fetch("/export/simulation-report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          let detail = "Excel report export failed (HTTP " + response.status + ").";
          try {
            const body = await response.json();
            detail = body.detail || detail;
          } catch (error) {}
          throw new Error(detail);
        }
        const blob = await response.blob();
        downloadBlob(blob, payload.name + "_simulation_report.xlsx");
      } catch (error) {
        appendMsg("bot", "Excel export failed: " + ((error && error.message) || error));
      } finally {
        if (button) button.textContent = "Simulation Excel";
        updateSimulationDownloadActions();
      }
    }

    function simulationReportPayload() {
      const intent = simulationDesignIntent() || {};
      const src = simSourceDims();
      const analytical = src ? estimateBushingModal(src.geom, src.material) : null;
      const nameSource = (targetStiffnessResult && targetStiffnessResult.case_id) ||
        (lastExport && lastExport.name) ||
        exportBaseName(intent) ||
        "model";
      const name = String(nameSource).toLowerCase().replace(/[^a-z0-9_-]+/g, "_") || "model";
      return {
        name,
        generated_at: new Date().toISOString(),
        design_intent: compactReportValue(intent),
        best_design: compactReportValue(targetStiffnessResult),
        client_targets: {
          target_kx_n_mm: targetSearchInputs.kx,
          target_ky_n_mm: targetSearchInputs.ky,
          target_kz_n_mm: targetSearchInputs.kz,
          tolerance_fraction: CLIENT_BUSHING_SPEC.match_tolerance,
          inner_diameter_min_mm: targetSearchInputs.idMin,
          inner_diameter_max_mm: targetSearchInputs.idMax,
          inner_core_length_min_mm: targetSearchInputs.innerLengthMin,
          inner_core_length_max_mm: targetSearchInputs.innerLengthMax,
          outer_core_length_min_mm: targetSearchInputs.outerLengthMin,
          outer_core_length_max_mm: targetSearchInputs.outerLengthMax,
          outer_diameter_mm: CLIENT_BUSHING_SPEC.outer_diameter_mm,
          swaging_value_mm: CLIENT_BUSHING_SPEC.swaging_value_mm,
          decking_value_mm: CLIENT_BUSHING_SPEC.decking_value_mm,
          internal_teeth: CLIENT_BUSHING_SPEC.internal_teeth,
        },
        simulation_settings: {
          requested_modes: femBatchCount,
          selected_contour_mode: femContourMode,
          mesh_mode: meshMode,
          centerline_axis: "X",
          fixed_interface: "outer core",
          prescribed_displacement_mm: 1,
        },
        global_mesh_template: compactReportValue(globalMeshTemplate),
        mesh_summary: compactReportValue(lastMeshResult),
        static_stiffness: lastStaticStiffness && lastStaticStiffness.status === "ok"
          ? compactReportValue(lastStaticStiffness.data)
          : null,
        modal_fem: lastFemContour && lastFemContour.status === "ok"
          ? compactReportValue(lastFemContour.data)
          : null,
        shape_pca: compactReportValue(lastShapePcaResult),
        analytical_estimate: compactReportValue(analytical),
        design_space_cases: designSpaceCases.map((item) => ({
          case_id: item.case_id,
          geometry: compactReportValue(item.geometry),
        })),
      };
    }

    function compactReportValue(value, depth = 0) {
      if (value == null || depth > 7) return value == null ? null : "[nested data omitted]";
      if (Array.isArray(value)) {
        return value.slice(0, 5000).map((item) => compactReportValue(item, depth + 1));
      }
      if (typeof value !== "object") return value;
      const excluded = new Set([
        "contour_png_base64",
        "fem_mesh",
        "surface_mesh",
        "comparison",
        "uploaded_surface",
        "faces",
        "points",
      ]);
      const compact = {};
      Object.entries(value).forEach(([key, item]) => {
        if (!excluded.has(key)) compact[key] = compactReportValue(item, depth + 1);
      });
      return compact;
    }

    function clearSimOutputPanel() {
      stopSimAnimation();
      cleanupFemViewer();
      if (simOutputPanel) simOutputPanel.innerHTML = "";
      updateAnalysisResultsVisibility();
    }

    function renderSimOutput() {
      const out = ensureSimOutputPanel();
      if (!out) return;
      stopSimAnimation();
      const src = simSourceDims();
      const est = src ? estimateBushingModal(src.geom, src.material) : null;
      if (!est) {
        out.innerHTML = '<p class="muted">Not enough bushing dimensions to estimate.</p>';
        return;
      }
      const modes = [
        { key: "b1", label: "1st bending", hz: est.bending[0] },
        { key: "b2", label: "2nd bending", hz: est.bending[1] },
        { key: "b3", label: "3rd bending", hz: est.bending[2] },
        { key: "a1", label: "1st axial",   hz: est.axial1   },
      ];
      const selected = modes.find((m) => m.key === simSelectedMode) || modes[0];
      const maxHz = Math.max(...modes.map((m) => Number(m.hz) || 0), 1);
      const rows = modes.map((m) =>
        '<tr data-mode="' + m.key + '"' + (m.key === simSelectedMode ? ' class="sim-row-active"' : '') + '>' +
        '<td>' + m.label + '</td>' +
        '<td class="sim-value">' + formatHz(m.hz) + '</td>' +
        '</tr>'
      ).join("");
      const bars = modes.map((m) => {
        const pct = clamp(((Number(m.hz) || 0) / maxHz) * 100, 3, 100);
        return (
          '<div class="sim-bar-row' + (m.key === simSelectedMode ? ' active' : '') + '" data-mode="' + m.key + '">' +
          '<strong>' + m.label + '</strong>' +
          '<div class="sim-bar-track"><div class="sim-bar-fill" style="width:' + pct.toFixed(1) + '%"></div></div>' +
          '<span class="sim-bar-value">' + formatHz(m.hz) + '</span>' +
          '</div>'
        );
      }).join("");
      const dashboardHtml =
        '<p class="muted">First-pass analytical estimate \u00b7 fixed-bottom \u00b7 linear-elastic ' +
        est.material.replace("_", " ") + '. Dashboard view. Run full FEM for validated results.</p>' +
        '<div class="sim-dashboard">' +
        '<div class="sim-kpi-grid">' +
        '<div class="sim-kpi"><span>Selected mode</span><strong>' + selected.label + '</strong></div>' +
        '<div class="sim-kpi"><span>Frequency</span><strong>' + formatHz(selected.hz) + '</strong></div>' +
        '<div class="sim-kpi"><span>Axial stiffness</span><strong>' + formatStiffness(est.kAxial) + '</strong></div>' +
        '<div class="sim-kpi"><span>Bending stiffness</span><strong>' + formatStiffness(est.kBending) + '</strong></div>' +
        '</div>' +
        '<div class="sim-frequency-bars">' +
        '<div class="category-subhead" style="margin:0 0 2px"><strong>Frequency dashboard</strong><span>Click a mode to select FEM contour</span></div>' +
        bars +
        '</div>' +
        '<div class="sim-dashboard-detail">' +
        '<div class="sim-detail-item"><span>Material model</span><strong>' + escapeHtml(est.material.replace("_", " ")) + '</strong></div>' +
        '<div class="sim-detail-item"><span>Area</span><strong>' + formatNumber(est.area, 1) + ' mm\u00b2</strong></div>' +
        '<div class="sim-detail-item"><span>Second moment</span><strong>' + formatEngineering(est.inertia, "mm4") + '</strong></div>' +
        '</div>' +
        '</div>' +
        '<table class="sim-table sim-table-rows">' +
        '<tr><th>Mode</th><th style="text-align:right">Frequency</th></tr>' +
        rows +
        '</table>';
      out.innerHTML =
        '<button type="button" class="sim-estimate-card" id="openSimEstimateBtn">' +
        '<span><strong>Analytical estimate</strong><small>Open stiffness and modal dashboard</small></span>' +
        '<span class="sim-estimate-kpis">' +
        '<span>Mode <strong>' + escapeHtml(selected.label) + '</strong></span>' +
        '<span>Frequency <strong>' + formatHz(selected.hz) + '</strong></span>' +
        '<span>Axial <strong>' + formatStiffness(est.kAxial) + '</strong></span>' +
        '</span>' +
        '</button>';
      const openBtn = document.getElementById("openSimEstimateBtn");
      if (openBtn) {
        openBtn.addEventListener("click", () => openSimulationEstimateWindow(dashboardHtml));
      }
    }

    function openSimulationEstimateWindow(contentHtml) {
      const existing = document.getElementById("simEstimateModal");
      if (existing) existing.remove();
      const overlay = document.createElement("div");
      overlay.id = "simEstimateModal";
      overlay.className = "sim-modal-backdrop";
      overlay.innerHTML =
        '<div class="sim-modal" role="dialog" aria-modal="true" aria-label="Analytical simulation estimate">' +
        '<div class="sim-modal-head"><strong>Analytical simulation estimate</strong><button type="button" class="sim-modal-close" aria-label="Close">×</button></div>' +
        '<div class="sim-modal-body">' + contentHtml + '</div>' +
        '</div>';
      document.body.appendChild(overlay);
      const close = () => overlay.remove();
      overlay.addEventListener("click", (event) => {
        if (event.target === overlay) close();
      });
      const closeBtn = overlay.querySelector(".sim-modal-close");
      if (closeBtn) closeBtn.addEventListener("click", close);
      overlay.querySelectorAll("[data-mode]").forEach((el) => {
        el.addEventListener("click", () => {
          simSelectedMode = el.getAttribute("data-mode");
          femContourMode = clampInt(selectedFemMode(), 1, femBatchCount || 6, 1);
          const contourInput = document.getElementById("simContourMode");
          if (contourInput) contourInput.value = String(femContourMode);
          renderSimOutput();
          close();
        });
      });
    }

    function stopSimAnimation() {
      if (simAnimHandle) {
        cancelAnimationFrame(simAnimHandle);
        simAnimHandle = 0;
      }
    }


    function renderSimPanel() {
      if (!simResults) return;
      const src = simSourceDims();
      if (!src) {
        clearSimOutputPanel();
        simResults.innerHTML = "";
        return;
      }
      simResults.innerHTML =
        '<div class="sim-block">' +
        '<div class="sim-head"><strong>Simulation</strong>' +
        '<span class="sim-head-actions">' +
        '<button type="button" class="sim-btn" id="simRunBtn">Simulate</button>' +
        '<button type="button" class="sim-btn secondary" id="simStaticBtn" title="Run three static CalculiX load cases and calculate Kx, Ky, and Kz">Static K</button>' +
        '<button type="button" class="sim-btn secondary" id="simFemBtn" title="Run one multi-mode FEM batch and render the selected contour">FEM batch</button>' +
        '<button type="button" class="sim-info-btn" aria-label="Simulation batch information" title="One CAD mesh is generated, then CalculiX solves up to 100 modal results. Larger batches can take several minutes on Azure.">i</button>' +
        '</span></div>' +
        simBatchControlsHtml() +
        '<p class="muted" style="margin:10px 0 0">Results appear below the CAD model preview.</p>' +
        '</div>';
      const btn = document.getElementById("simRunBtn");
      if (btn) {
        btn.addEventListener("click", () => {
          simShown = true;
          renderSimOutput();
        });
      }
      const femBtn = document.getElementById("simFemBtn");
      if (femBtn) {
        femBtn.addEventListener("click", runFemContour);
      }
      const staticBtn = document.getElementById("simStaticBtn");
      if (staticBtn) {
        staticBtn.addEventListener("click", runStaticStiffness);
      }
      bindSimBatchControls();
      if (simShown) {
        renderSimOutput();
      }
      if (lastFemContour) {
        renderFemContour(lastFemContour);
      }
      if (lastStaticStiffness) {
        renderStaticStiffness(lastStaticStiffness);
      }
    }

    function simBatchControlsHtml() {
      const maxFemModes = 100;
      const count = clampInt(femBatchCount, 1, maxFemModes, 6);
      const contour = clampInt(femContourMode || selectedFemMode(), 1, count, 1);
      return (
        '<div class="sim-batch-controls">' +
        '<div class="sim-batch-field">' +
        '<label for="simBatchCount">Number of simulations</label>' +
        '<input id="simBatchCount" type="number" min="1" max="' + maxFemModes + '" step="1" value="' + count + '">' +
        '</div>' +
        '<div class="sim-batch-field">' +
        '<label for="simContourMode">Contour mode</label>' +
        '<input id="simContourMode" type="number" min="1" max="' + count + '" step="1" value="' + contour + '">' +
        '</div>'
      );
    }

    function bindSimBatchControls() {
      const countInput = document.getElementById("simBatchCount");
      const contourInput = document.getElementById("simContourMode");
      if (countInput) {
        countInput.addEventListener("input", () => {
          femBatchCount = readClampedInput(countInput, 1, 100, femBatchCount || 6);
          if (contourInput) {
            contourInput.max = String(femBatchCount);
            femContourMode = readClampedInput(contourInput, 1, femBatchCount, femContourMode || 1);
            contourInput.value = String(femContourMode);
          }
        });
      }
      if (contourInput) {
        contourInput.addEventListener("input", () => {
          femContourMode = readClampedInput(contourInput, 1, femBatchCount || 6, femContourMode || 1);
        });
      }
    }

    function readClampedInput(input, min, max, fallback) {
      if (!input) return fallback;
      const value = Number(input && input.value);
      return clampInt(value, min, max, fallback);
    }

    function clampInt(value, min, max, fallback) {
      const n = Math.round(Number(value));
      if (!Number.isFinite(n)) return fallback;
      return Math.max(min, Math.min(max, n));
    }

    function renderMeshPanel() {
      if (!meshResults) return;
      const src = simSourceDims();
      if (!src) {
        cleanupMeshViewer();
        meshResults.innerHTML = "";
        if (meshOutputPanel) meshOutputPanel.innerHTML = "";
        if (shapePcaOutputPanel) shapePcaOutputPanel.innerHTML = "";
        updateAnalysisResultsVisibility();
        return;
      }
      meshResults.innerHTML =
        '<div class="mesh-block">' +
        '<div class="sim-head"><strong>Gmsh mesh</strong>' +
        '<span class="sim-head-actions">' +
        '<button type="button" class="sim-btn" id="meshGenerateBtn">Generate mesh</button>' +
        '<button type="button" class="sim-btn secondary" id="shapePcaBtn">Shape PCA</button>' +
        '</span></div>' +
        meshControlsHtml() +
        '<p class="muted" style="margin:10px 0 0">Mesh result appears below the CAD model preview.</p>' +
        '</div>';
      const btn = document.getElementById("meshGenerateBtn");
      if (btn) btn.addEventListener("click", runGmshMesh);
      const shapeBtn = document.getElementById("shapePcaBtn");
      if (shapeBtn) shapeBtn.addEventListener("click", runShapePca);
      bindMeshControls();
      renderMeshOutputPanel();
      renderShapePcaOutputPanel();
    }

    function meshControlsHtml() {
      if (exactUploadedGeometryContext()) {
        return (
          '<div class="mesh-controls">' +
          '<div class="mesh-control-field full"><label for="meshModeSelect">Mesh mode</label>' +
          '<select id="meshModeSelect"><option value="structured"' + (meshMode === "structured" ? " selected" : "") + '>Uploaded geometry - Gmsh all-hexa</option><option value="global"' + (meshMode === "global" ? " selected" : "") + '>Uploaded geometry - global-density all-hexa</option></select></div>' +
          '<div class="mesh-template-grid">' +
          '<div class="mesh-control-field"><label for="globalCircumDivisions">Circum.</label><input id="globalCircumDivisions" type="number" min="24" max="192" step="4" value="' + globalMeshTemplate.circumferential_divisions + '"></div>' +
          '<div class="mesh-control-field"><label for="globalRadialDivisions">Radial</label><input id="globalRadialDivisions" type="number" min="2" max="32" step="1" value="' + globalMeshTemplate.radial_divisions + '"></div>' +
          '<div class="mesh-control-field"><label for="globalAxialDivisions">Axial</label><input id="globalAxialDivisions" type="number" min="3" max="64" step="1" value="' + globalMeshTemplate.axial_divisions + '"></div>' +
          '</div>' +
          '<p class="muted" style="margin:0">Both modes follow only the uploaded STEP/STL shape. Gmsh repairs and volume-meshes the uploaded solid, then converts its volume elements to all-hexa cells. Global mode uses the fixed density settings above for repeatable runs of the same geometry.</p>' +
          '</div>'
        );
      }
      return (
        '<div class="mesh-controls">' +
        '<div class="mesh-control-field full"><label for="meshModeSelect">Mesh mode</label>' +
        '<select id="meshModeSelect"><option value="structured"' + (meshMode === "structured" ? " selected" : "") + '>Structured hex</option><option value="global"' + (meshMode === "global" ? " selected" : "") + '>Global dataset mesh</option></select></div>' +
        '<div class="mesh-template-grid">' +
        '<div class="mesh-control-field"><label for="globalCircumDivisions">Circum.</label><input id="globalCircumDivisions" type="number" min="24" max="192" step="4" value="' + globalMeshTemplate.circumferential_divisions + '"></div>' +
        '<div class="mesh-control-field"><label for="globalRadialDivisions">Radial</label><input id="globalRadialDivisions" type="number" min="2" max="32" step="1" value="' + globalMeshTemplate.radial_divisions + '"></div>' +
        '<div class="mesh-control-field"><label for="globalAxialDivisions">Axial</label><input id="globalAxialDivisions" type="number" min="3" max="64" step="1" value="' + globalMeshTemplate.axial_divisions + '"></div>' +
        '</div>' +
        '<p class="muted" style="margin:0">Global mode keeps node IDs and element connectivity shared across the dataset.</p>' +
        '</div>'
      );
    }

    function bindMeshControls() {
      const modeSelect = document.getElementById("meshModeSelect");
      if (modeSelect) {
        modeSelect.addEventListener("change", () => {
          meshMode = modeSelect.value === "global" ? "global" : "structured";
          lastMeshResult = null;
          lastShapePcaResult = null;
          lastStaticStiffness = null;
          renderMeshPanel();
        });
      }
      const fields = [
        ["globalCircumDivisions", "circumferential_divisions", 24, 192, 96],
        ["globalRadialDivisions", "radial_divisions", 2, 32, 8],
        ["globalAxialDivisions", "axial_divisions", 3, 64, 16],
      ];
      fields.forEach(([id, key, min, max, fallback]) => {
        const input = document.getElementById(id);
        if (!input) return;
        input.addEventListener("input", () => {
          globalMeshTemplate[key] = readClampedInput(input, min, max, fallback);
          lastMeshResult = null;
          lastShapePcaResult = null;
          lastStaticStiffness = null;
          renderMeshOutputPanel();
          renderShapePcaOutputPanel();
        });
      });
    }

    function meshRequestOptions(intent) {
      const exactUpload = exactUploadedGeometryContext();
      const options = {
        mesh_mode: meshMode,
        global_template: Object.assign({}, globalMeshTemplate),
      };
      if (exactUpload && exactUpload.upload_id) {
        options.upload_id = exactUpload.upload_id;
      }
      if (exactUpload && exactUpload.upload_data_base64) {
        options.upload_data_base64 = exactUpload.upload_data_base64;
        options.upload_filename = exactUpload.upload_filename || exactUpload.filename || "uploaded_geometry.stl";
        options.upload_content_type = exactUpload.upload_content_type || "application/octet-stream";
      }
      return options;
    }

    function renderMeshOutputPanel() {
      if (!meshOutputPanel) return;
      cleanupMeshViewer();
      if (!lastMeshResult) {
        meshOutputPanel.innerHTML = "";
        updateAnalysisResultsVisibility();
        return;
      }
      const displayMesh = meshSurfaceForDisplay(lastMeshResult);
      meshOutputPanel.innerHTML =
        '<div class="mesh-block analysis-result-block">' +
        '<div class="sim-head"><strong>Mesh quality preview</strong><span class="muted">Below CAD preview</span></div>' +
        '<div id="meshOutput">' + meshResultHtml(lastMeshResult, displayMesh) + '</div>' +
        '</div>';
      updateAnalysisResultsVisibility();
      renderMeshViewers(lastMeshResult, displayMesh);
    }

    function renderMeshViewers(result, displayMesh) {
      const comparison = meshComparisonMeshes(result);
      if (comparison.uploaded && comparison.surrogate) {
        renderGmshMeshCanvas(document.getElementById("meshUploadedViewer"), comparison.uploaded, {
          edgeColor: "rgba(14, 165, 233, 0.92)",
          wireframe: true,
        });
        renderGmshMeshCanvas(document.getElementById("meshSurrogateViewer"), comparison.surrogate, {
          edgeColor: "rgba(8, 145, 178, 0.98)",
          wireframe: true,
        });
        return;
      }
      if (displayMesh) {
        renderGmshMeshCanvas(document.getElementById("meshViewer"), displayMesh, {
          edgeColor: "rgba(14, 165, 233, 0.92)",
          wireframe: true,
        });
      }
    }

    function meshSurfaceForDisplay(result) {
      if (result && result.status === "preview_only") {
        return uploadedMeshSurfaceForMeshResult(result);
      }
      if (result && result.mesh_source === "exact_uploaded_geometry" && result.surface_mesh) {
        return result.surface_mesh || null;
      }
      if (result && result.mesh_source === "bushing_poc_hex") {
        const uploadedPocSurface = uploadedMeshSurfaceForMeshResult(result);
        if (uploadedPocSurface) return uploadedPocSurface;
      }
      if (result && result.fallback_reason) {
        const uploadedFallbackSurface = uploadedMeshSurfaceForMeshResult(result);
        if (uploadedFallbackSurface) return uploadedFallbackSurface;
      }
      return (result && result.surface_mesh) || uploadedMeshSurfaceForMeshResult(result) || null;
    }

    function renderShapePcaOutputPanel() {
      if (!shapePcaOutputPanel) return;
      if (!lastShapePcaResult) {
        shapePcaOutputPanel.innerHTML = "";
        updateAnalysisResultsVisibility();
        return;
      }
      if (lastShapePcaResult.status === "ok") {
        const reconstruction = lastShapePcaResult.reconstruction || {};
        shapePcaOutputPanel.innerHTML =
          '<div class="mesh-block analysis-result-block shape-pca-block">' +
          '<button type="button" class="sim-estimate-card" id="openShapePcaResultBtn">' +
          '<span><strong>Shape PCA complete</strong><small>Open the encoding and reconstruction report</small></span>' +
          '<span class="sim-estimate-kpis">' +
          '<span>Samples <strong>' + formatInt(lastShapePcaResult.sample_count) + '</strong></span>' +
          '<span>Components <strong>' + formatInt(lastShapePcaResult.component_count) + '</strong></span>' +
          '<span>Precision <strong>' + formatNumber(reconstruction.precision_percent, 2) + '%</strong></span>' +
          '</span>' +
          '</button>' +
          '</div>';
        const openButton = document.getElementById("openShapePcaResultBtn");
        if (openButton) {
          openButton.addEventListener("click", () => openShapePcaResultWindow(lastShapePcaResult));
        }
      } else {
        shapePcaOutputPanel.innerHTML =
          '<div class="mesh-block analysis-result-block shape-pca-block">' +
          '<div class="sim-head"><strong>Shape PCA encoding</strong><span class="muted">Global mesh geometry</span></div>' +
          shapePcaHtml(lastShapePcaResult) +
          '</div>';
      }
      updateAnalysisResultsVisibility();
    }

    function shapePcaHtml(result) {
      if (result.status === "loading") {
        return '<div class="mesh-output">Fitting global mesh Shape PCA, encoding alpha parameters, and reconstructing the current bushing...</div>';
      }
      if (result.status === "error") {
        return '<div class="mesh-output err">Shape PCA failed: ' + escapeHtml(result.message) + '</div>';
      }
      const reconstruction = result.reconstruction || {};
      const components = Array.isArray(result.components) ? result.components : [];
      const rows = components.map((pc) => (
        '<tr>' +
        '<td>PC' + pc.component + '</td>' +
        '<td class="sim-value">' + formatNumber(Number(pc.alpha), 4) + '</td>' +
        '<td class="sim-value">' + formatNumber(Number(pc.eigenvalue), 4) + '</td>' +
        '<td class="sim-value">' + percent(pc.explained_variance_ratio) + '</td>' +
        '<td class="sim-value">' + percent(pc.cumulative_variance_ratio) + '</td>' +
        '</tr>'
      )).join("");
      return (
        '<div class="mesh-output">' +
        '<div class="shape-pca-formula"><strong>X ~= X_mean + sum(alpha_m * sqrt(lambda_m) * W_m)</strong><span>alpha_m = W_m^T (X - X_mean) / sqrt(lambda_m)</span></div>' +
        '<div class="mesh-stats">' +
        '<span>Template: <strong>' + escapeHtml(result.template_id || "global") + '</strong></span>' +
        '<span>Samples: <strong>' + formatInt(result.sample_count) + '</strong></span>' +
        '<span>Nodes: <strong>' + formatInt(result.node_count) + '</strong></span>' +
        '<span>Components: <strong>' + formatInt(result.component_count) + '</strong></span>' +
        '<span>Precision: <strong>' + formatNumber(reconstruction.precision_percent, 2) + '%</strong></span>' +
        '<span>RMS error: <strong>' + formatNumber(reconstruction.rms_error_mm, 4) + ' mm</strong></span>' +
        '</div>' +
        '<table class="sim-table sim-table-rows" style="margin-top:10px"><tr><th>Component</th><th style="text-align:right">alpha</th><th style="text-align:right">lambda</th><th style="text-align:right">Variance</th><th style="text-align:right">Cumulative</th></tr>' + rows + '</table>' +
        '<p class="muted" style="margin:8px 0 0">Model artifact: ' + escapeHtml(result.model_file || "") + ' · Reconstruction: ' + escapeHtml(result.reconstructed_mesh_file || "") + '</p>' +
        '</div>'
      );
    }

    function openShapePcaResultWindow(result) {
      if (!result || result.status !== "ok") return;
      const existing = document.getElementById("shapePcaResultModal");
      if (existing) existing.remove();
      const overlay = document.createElement("div");
      overlay.id = "shapePcaResultModal";
      overlay.className = "sim-modal-backdrop";
      overlay.innerHTML =
        '<div class="sim-modal" role="dialog" aria-modal="true" aria-labelledby="shapePcaResultTitle">' +
        '<div class="sim-modal-head"><strong id="shapePcaResultTitle">Shape PCA encoding</strong>' +
        '<button type="button" class="sim-modal-close" aria-label="Close Shape PCA result">×</button></div>' +
        '<div class="sim-modal-body">' +
        '<div class="sim-head" style="margin-bottom:10px"><strong>Global mesh geometry</strong><span class="muted">Encoding and reconstruction report</span></div>' +
        shapePcaHtml(result) +
        '</div></div>';
      document.body.appendChild(overlay);
      const closeButton = overlay.querySelector(".sim-modal-close");
      const close = () => {
        document.removeEventListener("keydown", onKeyDown);
        overlay.remove();
      };
      const onKeyDown = (event) => {
        if (event.key === "Escape") close();
      };
      overlay.addEventListener("click", (event) => {
        if (event.target === overlay) close();
      });
      if (closeButton) {
        closeButton.addEventListener("click", close);
        closeButton.focus();
      }
      document.addEventListener("keydown", onKeyDown);
    }

    function meshResultHtml(result, displayMesh) {
      if (!result) return "";
      if (result.status === "loading") {
        return '<div class="mesh-output">Generating STEP, creating pure structured hex/swept mesh, then checking mesh quality...</div>';
      }
      if (result.status === "error") {
        return '<div class="mesh-output err">Mesh generation failed: ' + escapeHtml(result.message) + '</div>';
      }
      if (result.status === "preview_only") {
        return (
          '<div class="mesh-output">' +
          '<strong>Uploaded geometry preview only</strong>' +
          '<div class="muted" style="margin-top:6px">' + escapeHtml(result.message || "Exact volume mesh was not generated for this STL.") + '</div>' +
          meshViewerHtml(displayMesh) +
          '</div>'
        );
      }
      const q = result.quality || {};
      const cleaning = result.cleaning || {};
      const ready = q.ready_for_fem ? "Ready for FEM" : "Needs attention";
      const strategyNote = meshStrategyNote(result);
      const globalHtml = globalMeshHtml(result.global_mesh);
      const exactUploadNote = result.mesh_source === "exact_uploaded_geometry"
        ? '<div class="muted" style="margin-top:6px">Uploaded-geometry all-hexa mesh: statistics, FEM readiness, and preview come from the repaired STEP/STL solid. The OD/ID/height bushing generator is not used.</div>'
        : "";
      const bushingPocNote = result.mesh_source === "bushing_poc_hex"
        ? '<div class="muted" style="margin-top:6px">Rubber bushing hex surrogate: generated from measured/confirmed OD, ID, height, and editable parameters. It is not the exact uploaded STL topology; the uploaded STL remains the front-end geometry reference.</div>'
        : "";
      const fallbackNote = result.fallback_reason
        ? '<div class="muted" style="margin-top:6px"><strong>Fallback:</strong> ' + escapeHtml(result.fallback_reason) + '</div>'
        : "";
      const comparison = meshComparisonMeshes(result);
      const uploadPreviewNote = comparison.uploaded && comparison.surrogate
        ? '<div class="muted" style="margin-top:6px">Left view is the uploaded STL reference; right view is the generated dimension-based hex surrogate used for mesh quality and FEM readiness.</div>'
        : (result.mesh_source !== "exact_uploaded_geometry" && displayMesh && displayMesh.source === "uploaded_stl"
        ? '<div class="muted" style="margin-top:6px">Preview surface follows the uploaded STL geometry; mesh statistics/FEM readiness use the generated structured hex mesh.</div>'
        : "");
      const meshPreviewHtml = comparison.uploaded && comparison.surrogate
        ? meshComparisonHtml(comparison.uploaded, comparison.surrogate)
        : meshViewerHtml(displayMesh || result.surface_mesh);
      return (
        '<div class="mesh-output">' +
        '<strong>' + ready + '</strong> · ' + escapeHtml(result.mesh_format || "Gmsh mesh") +
        strategyNote +
        globalHtml +
        exactUploadNote +
        bushingPocNote +
        fallbackNote +
        uploadPreviewNote +
        '<div class="mesh-stats">' +
        '<span>Nodes: <strong>' + formatInt(result.nodes) + '</strong></span>' +
        '<span>Hexahedra: <strong>' + formatInt(result.hexahedra) + '</strong></span>' +
        '<span>Tetrahedra: <strong>' + formatInt(result.tetrahedra) + '</strong></span>' +
        '<span>Element edge: <strong>' + formatRange(result.min_edge_mm, result.max_edge_mm, " mm") + '</strong></span>' +
        '<span>Mean quality: <strong>' + formatNumber(q.mean_quality, 3) + '</strong></span>' +
        '<span>Poor elements: <strong>' + formatInt(q.poor_count) + '</strong></span>' +
        '<span>Inverted: <strong>' + formatInt(q.inverted_count) + '</strong></span>' +
        '<span>Merged nodes: <strong>' + formatInt(cleaning.merged_nodes) + '</strong></span>' +
        '<span>Removed cells: <strong>' + formatInt(cleaning.removed_cells) + '</strong></span>' +
        '</div>' +
        meshPreviewHtml +
        '</div>'
      );
    }

    function meshComparisonMeshes(result) {
      const surrogate = result && result.surface_mesh && Array.isArray(result.surface_mesh.faces) && result.surface_mesh.faces.length
        ? Object.assign({}, result.surface_mesh, { source: result.surface_mesh.source || "structured_hex_surrogate" })
        : null;
      const uploaded = uploadedMeshSurfaceForMeshResult(result);
      if (result && result.mesh_source === "bushing_poc_hex" && uploaded && surrogate) {
        return { uploaded, surrogate };
      }
      return { uploaded: null, surrogate: null };
    }

    function meshComparisonHtml(uploadedMesh, surrogateMesh) {
      return (
        '<div class="mesh-compare" aria-label="Uploaded STL reference and structured hex surrogate mesh comparison">' +
        meshPreviewCardHtml("meshUploadedViewer", "Uploaded STL reference", "front-end geometry reference", uploadedMesh) +
        meshPreviewCardHtml("meshSurrogateViewer", "Mesh quality preview", "dimension-based hex surrogate", surrogateMesh) +
        '</div>'
      );
    }

    function meshPreviewCardHtml(viewerId, title, subtitle, mesh) {
      return (
        '<div class="mesh-compare-card">' +
        '<div class="mesh-compare-card-head"><strong>' + escapeHtml(title) + '</strong><span>' + escapeHtml(subtitle) + '</span></div>' +
        meshViewerHtml(mesh, viewerId, title) +
        '</div>'
      );
    }

    function uploadedMeshSurfaceForMeshResult(result) {
      const faces = meshEditMode && editableMesh
        ? (overrideMeshFaces || warpEditableMeshFaces((currentEditIntent && currentEditIntent.geometry) || {}))
        : (pickUploadedMesh() ? pickUploadedMesh().faces : null);
      if (!Array.isArray(faces) || !faces.length) return null;
      const limitedFaces = faces.length > 12000
        ? faces.filter((face, index) => index % Math.ceil(faces.length / 12000) === 0)
        : faces;
      const scalarMesh = result && result.surface_mesh ? result.surface_mesh : null;
      const scalarMin = scalarMesh && Number.isFinite(Number(scalarMesh.scalar_min)) ? Number(scalarMesh.scalar_min) : 0;
      const scalarMax = scalarMesh && Number.isFinite(Number(scalarMesh.scalar_max)) && Number(scalarMesh.scalar_max) > scalarMin ? Number(scalarMesh.scalar_max) : scalarMin + 1;
      const bounds = computeMeshBounds(limitedFaces);
      const axis = bounds.size.y >= bounds.size.x && bounds.size.y >= bounds.size.z
        ? "y"
        : (bounds.size.x >= bounds.size.z ? "x" : "z");
      const axisMin = bounds.center[axis] - bounds.size[axis] / 2;
      const axisSpan = Math.max(bounds.size[axis], 1e-6);
      return {
        faces: limitedFaces.map((face) => {
          const average = face.points.reduce((sum, point) => sum + Number(point[axis] || 0), 0) / Math.max(face.points.length, 1);
          const t = clamp((average - axisMin) / axisSpan, 0, 1);
          const value = scalarMin + t * (scalarMax - scalarMin);
          return {
            color: scalarMesh ? contourPreviewColor(t) : (face.color || UPLOAD_MESH_COLOR),
            smoothPreview: Boolean(face.smoothPreview),
            value,
            points: face.points,
          };
        }),
        field: scalarMesh && scalarMesh.field ? scalarMesh.field + " mapped to uploaded STL" : "Uploaded STL preview",
        unit: scalarMesh && scalarMesh.unit ? scalarMesh.unit : "",
        scalar_min: scalarMin,
        scalar_max: scalarMax,
        face_count: limitedFaces.length,
        source: "uploaded_stl",
      };
    }

    function meshStrategyNote(result) {
      if (!result || !result.mesh_strategy) return "";
      if (result.mesh_strategy === "global_bushing_hex" || result.mesh_strategy === "global_slotted_bushing_hex") {
        return '<div class="muted" style="margin-top:6px">Global dataset mesh: node count and hex connectivity are fixed by the selected template.</div>';
      }
      if (result.mesh_strategy === "mapped_bushing_hex" || result.mesh_strategy === "mapped_slotted_bushing_hex") {
        return '<div class="muted" style="margin-top:6px">Structured bushing surrogate: divisions are generated from the confirmed bushing dimensions and editable parameters.</div>';
      }
      if (result.mesh_strategy === "structured_hex") {
        return '<div class="muted" style="margin-top:6px">Pure structured hex/swept mesh succeeded for this geometry.</div>';
      }
      if (result.mesh_strategy === "uploaded_geometry_tetra") {
        return '<div class="muted" style="margin-top:6px">Tetrahedral volume mesh generated directly from uploaded geometry. This is the exact FEM path for arbitrary STEP/STL uploads.</div>';
      }
      if (result.mesh_strategy === "uploaded_geometry_subdivided_hex") {
        return '<div class="muted" style="margin-top:6px">Gmsh body-fitted all-hexa mesh generated from the repaired uploaded STEP/STL volume.</div>';
      }
      if (result.mesh_strategy === "uploaded_geometry_global_hex") {
        return '<div class="muted" style="margin-top:6px">Gmsh body-fitted all-hexa mesh generated with fixed global density settings. Connectivity is repeatable for the same uploaded geometry, but is not shared across unrelated CAD models.</div>';
      }
      if (result.mesh_strategy === "uploaded_geometry_voxel_hex" || result.mesh_strategy === "uploaded_geometry_voxel_global_hex") {
        return '<div class="muted" style="margin-top:6px">Uploaded-derived voxel all-hexa fallback: each occupied voxel is a valid C3D8 element. Approximation pitch: <strong>' + formatNumber(result.voxel_pitch_mm, 2) + ' mm</strong>.</div>';
      }
      return '<div class="muted" style="margin-top:6px">Hex-only meshing is required for this workflow.</div>';
    }

    function globalMeshHtml(globalMesh) {
      if (!globalMesh || !globalMesh.enabled) return "";
      return (
        '<div class="mesh-global-summary">' +
        '<span>Template: <strong>' + escapeHtml(globalMesh.template_id || "global") + '</strong></span>' +
        '<span>Connectivity: <strong>' + (globalMesh.shared_connectivity ? "shared" : "unique") + '</strong></span>' +
        '<span>' + (globalMesh.settings_only ? "Density settings" : "Divisions") + ': <strong>C' + formatInt(globalMesh.circumferential_divisions) + ' / R' + formatInt(globalMesh.radial_divisions) + ' / A' + formatInt(globalMesh.axial_divisions) + '</strong></span>' +
        '</div>'
      );
    }

    function meshViewerHtml(mesh, viewerId, ariaLabel) {
      if (!mesh || !Array.isArray(mesh.faces) || !mesh.faces.length) {
        return '<div class="mesh-output err" style="margin-top:10px">Mesh was generated, but no exterior surface triangles were available for preview.</div>';
      }
      const minLabel = formatEngineering(mesh.scalar_min, mesh.unit || "");
      const maxLabel = formatEngineering(mesh.scalar_max, mesh.unit || "");
      const id = viewerId || "meshViewer";
      const label = ariaLabel || "Interactive Gmsh mesh preview";
      return (
        '<div class="mesh-viewer-wrap">' +
        '<div class="mesh-viewer" id="' + escapeHtml(id) + '" aria-label="' + escapeHtml(label) + '"></div>' +
        '<div class="mesh-legend" aria-label="' + escapeHtml(mesh.field || "Volume") + ' legend">' +
        '<strong>' + escapeHtml(mesh.field || "Volume") + '</strong>' +
        '<span>' + escapeHtml(maxLabel) + '</span>' +
        '<div class="mesh-legend-bar"></div>' +
        '<span>' + escapeHtml(minLabel) + '</span>' +
        '</div></div>'
      );
    }

    function formatInt(value) {
      const n = Number(value);
      return Number.isFinite(n) ? Math.round(n).toLocaleString() : "-";
    }

    function formatNumber(value, digits) {
      if (value === null || value === undefined || value === "") return "-";
      const n = Number(value);
      return Number.isFinite(n) ? n.toFixed(digits) : "-";
    }

    function formatRange(min, max, unit) {
      const a = Number(min);
      const b = Number(max);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return "-";
      return a.toFixed(2) + "-" + b.toFixed(2) + unit;
    }

    function formatEngineering(value, unit) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "-";
      const abs = Math.abs(n);
      const text = abs > 0 && (abs < 0.01 || abs >= 10000) ? n.toExponential(2) : n.toFixed(abs >= 100 ? 1 : 3);
      return unit ? text + " " + unit : text;
    }

    function isStructuredBushingIntentClient(intent) {
      const type = String((intent && intent.part_type) || "").toLowerCase();
      const geom = intent && intent.geometry ? intent.geometry : null;
      if (!(type === "bushing" || type === "rubber_mount") || !geom) return false;
      const outer = Number(geom.outer_diameter_mm);
      const inner = Number(geom.inner_diameter_mm);
      const height = Number(geom.height_mm);
      return Number.isFinite(outer) && outer > 0 && Number.isFinite(inner) && inner > 0 && Number.isFinite(height) && height > 0;
    }

    function meshIntentForRequest() {
      const candidates = [lastExport && lastExport.intent, currentEditIntent];
      for (const candidate of candidates) {
        if (isStructuredBushingIntentClient(candidate)) {
          const normalized = normalizeRubberBushingIntent(candidate);
          currentEditIntent = normalized;
          lastExport.intent = normalized;
          return normalized;
        }
      }

      const uploaded = pickUploadedMesh();
      const dims = measureBushingFromMesh(uploaded);
      if (!dims) return null;

      const measuredIntent = defaultRubberBushingIntent();
      measuredIntent.geometry.outer_diameter_mm = dims.outer_diameter_mm;
      measuredIntent.geometry.inner_diameter_mm = dims.inner_diameter_mm;
      measuredIntent.geometry.height_mm = dims.height_mm;
      const connectedIntent = applyUploadedBushingPocTopology(measuredIntent);
      currentEditIntent = connectedIntent;
      lastExport.intent = connectedIntent;
      lastExport.name = exportBaseName(connectedIntent);
      lastExport.prompt = lastExport.prompt || pendingDraftPrompt || "Rubber bushing structured parametric JSON";
      jsonOutput.textContent = JSON.stringify(connectedIntent, null, 2);
      downloadBtn.disabled = false;
      buildParamControls(connectedIntent);
      updateSummary(connectedIntent);
      return connectedIntent;
    }

    function exactUploadedGeometryContext() {
      for (let i = attachmentContexts.length - 1; i >= 0; i -= 1) {
        const context = attachmentContexts[i];
        if (context && context.exact_fem && context.exact_fem.supported && (context.upload_id || context.upload_data_base64)) {
          return context;
        }
      }
      return null;
    }

    async function runGmshMesh() {
      const btn = document.getElementById("meshGenerateBtn");
      if (uploadNeedsParametricConfirmation) {
        lastMeshResult = { status: "error", message: "This image/document upload is reference-only. Confirm or edit the Parametric input values, then generate the mesh; upload STL for exact geometry." };
        renderMeshPanel();
        return;
      }
      const intent = meshIntentForRequest();
      const exactUpload = exactUploadedGeometryContext();
      const prompt = (lastExport && lastExport.prompt) || pendingDraftPrompt || (intent ? "Rubber bushing structured parametric JSON" : (exactUpload ? "Exact uploaded geometry mesh" : ""));
      if (!exactUpload && !prompt.trim() && !isStructuredBushingIntentClient(intent)) {
        lastMeshResult = { status: "error", message: "Generate or upload a Rubber bushing first, or send a chat prompt for a general CAD model." };
        renderMeshPanel();
        return;
      }
      if (btn) btn.disabled = true;
      lastMeshResult = { status: "loading" };
      renderMeshPanel();
      try {
        const response = await fetch("/generate-mesh", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt,
            name: (lastExport && lastExport.name) || exportBaseName(intent || {}) || "model",
            intent: intent || {},
            ...meshRequestOptions(intent),
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || "Mesh generation failed (HTTP " + response.status + ").");
        }
        lastMeshResult = payload;
        renderMeshPanel();
      } catch (error) {
        const exactUpload = exactUploadedGeometryContext();
        if (exactUpload) {
          lastMeshResult = {
            status: "preview_only",
            message: (error.message || String(error)) + " The uploaded geometry remains available for preview; repair the source STL or upload STEP to generate the body-fitted all-hexa volume mesh.",
          };
        } else {
          lastMeshResult = { status: "error", message: error.message || String(error) };
        }
        renderMeshPanel();
      } finally {
        const nextBtn = document.getElementById("meshGenerateBtn");
        if (nextBtn) nextBtn.disabled = false;
      }
    }

    async function runShapePca() {
      const btn = document.getElementById("shapePcaBtn");
      const prompt = (lastExport && lastExport.prompt) || "Rubber bushing structured parametric JSON";
      const intent = (lastExport && lastExport.intent) || currentEditIntent || null;
      if (!intent || !intent.geometry) {
        lastShapePcaResult = { status: "error", message: "Generate or upload a Rubber bushing first." };
        renderShapePcaOutputPanel();
        return;
      }
      if (btn) btn.disabled = true;
      const priorMode = meshMode;
      meshMode = "global";
      lastShapePcaResult = { status: "loading" };
      renderShapePcaOutputPanel();
      try {
        const response = await fetch("/shape-pca", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt,
            name: (lastExport && lastExport.name) || "bushing_shape",
            intent: intent,
            samples: 12,
            components: 10,
            ...meshRequestOptions(intent),
          }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || "Shape PCA failed (HTTP " + response.status + ").");
        }
        lastShapePcaResult = Object.assign({ status: "ok" }, payload);
        renderShapePcaOutputPanel();
      } catch (error) {
        lastShapePcaResult = { status: "error", message: error.message || String(error) };
        renderShapePcaOutputPanel();
      } finally {
        meshMode = priorMode;
        if (priorMode !== "global") renderMeshPanel();
        const nextBtn = document.getElementById("shapePcaBtn");
        if (nextBtn) nextBtn.disabled = false;
      }
    }

    function cleanupMeshViewer() {
      const viewers = Array.isArray(activeMeshViewer) ? activeMeshViewer : (activeMeshViewer ? [activeMeshViewer] : []);
      viewers.forEach((viewer) => {
        if (viewer && typeof viewer.dispose === "function") {
          viewer.dispose();
        }
      });
      activeMeshViewer = null;
    }

    function trackMeshViewer(viewer) {
      if (!viewer) return;
      if (!activeMeshViewer) activeMeshViewer = [];
      if (!Array.isArray(activeMeshViewer)) activeMeshViewer = [activeMeshViewer];
      activeMeshViewer.push(viewer);
    }

    function arrayBufferToBase64(buffer) {
      const bytes = new Uint8Array(buffer);
      const chunkSize = 0x8000;
      let binary = "";
      for (let index = 0; index < bytes.length; index += chunkSize) {
        const chunk = bytes.subarray(index, index + chunkSize);
        binary += String.fromCharCode.apply(null, chunk);
      }
      return btoa(binary);
    }

    function renderGmshMeshCanvas(host, mesh, options) {
      if (!host || !mesh || !Array.isArray(mesh.faces) || !mesh.faces.length) return;
      const renderOptions = options || {};
      const canvas = document.createElement("canvas");
      host.innerHTML = "";
      host.appendChild(canvas);
      const context = canvas.getContext("2d");
      if (!context) return;

      const bounds = computeMeshBounds(mesh.faces);
      const size = bounds.size;
      const extent = Math.max(size.x, size.y, size.z, 1);
      const center = bounds.center;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const lightDirection = normalizeVector({ x: 0.35, y: 0.85, z: 0.4 });
      const centeredFaces = mesh.faces.map((face) => ({
        color: face.color || "#00c8ff",
        smoothPreview: Boolean(face.smoothPreview),
        points: face.points.map((point) => ({
          x: point.x - center.x,
          y: point.y - center.y,
          z: point.z - center.z,
        })),
      }));

      const state = {
        width: 0,
        height: 0,
        rotationX: meshCamera.rotationX,
        rotationY: meshCamera.rotationY,
        zoom: meshCamera.zoom,
        dragging: false,
        lastX: 0,
        lastY: 0,
        animationFrameId: 0,
        renderQueued: false,
      };

      function scheduleRender() {
        if (state.renderQueued) return;
        state.renderQueued = true;
        state.animationFrameId = requestAnimationFrame(() => {
          state.renderQueued = false;
          draw();
        });
      }

      function resizeCanvas() {
        state.width = Math.max(320, Math.floor(host.clientWidth || 620));
        state.height = Math.max(340, Math.floor(host.clientHeight || 380));
        canvas.width = Math.floor(state.width * pixelRatio);
        canvas.height = Math.floor(state.height * pixelRatio);
        canvas.style.width = `${state.width}px`;
        canvas.style.height = `${state.height}px`;
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        scheduleRender();
      }

      function projectRotatedPoint(rotated) {
        const scale = (Math.min(state.width, state.height) / (extent * 2.35)) * state.zoom;
        return {
          x: state.width / 2 + rotated.x * scale,
          y: state.height / 2 - rotated.y * scale,
          z: rotated.z,
        };
      }

      function draw() {
        context.clearRect(0, 0, state.width, state.height);
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, state.width, state.height);
        const projectedFaces = centeredFaces
          .map((face) => {
            const rotatedPoints = face.points.map((point) => rotatePoint(point, state.rotationX, state.rotationY));
            const projectedPoints = rotatedPoints.map(projectRotatedPoint);
            const normal = computeFaceNormal(rotatedPoints);
            const intensity = clamp(dotProduct(normal, lightDirection) * 0.5 + 0.76, 0.34, 1.0);
            const fillIntensity = face.smoothPreview ? 0.82 : intensity;
            const averageDepth = rotatedPoints.reduce((sum, point) => sum + point.z, 0) / rotatedPoints.length;
            return {
              projectedPoints,
              averageDepth,
              fill: shadeColor(face.color, fillIntensity),
              smoothPreview: face.smoothPreview,
            };
          })
          .sort((left, right) => left.averageDepth - right.averageDepth);

        for (const face of projectedFaces) {
          context.beginPath();
          context.moveTo(face.projectedPoints[0].x, face.projectedPoints[0].y);
          for (let index = 1; index < face.projectedPoints.length; index += 1) {
            context.lineTo(face.projectedPoints[index].x, face.projectedPoints[index].y);
          }
          context.closePath();
          context.fillStyle = face.fill;
          context.fill();
          if (renderOptions.wireframe || !face.smoothPreview) {
            context.strokeStyle = renderOptions.edgeColor || (face.smoothPreview ? "rgba(14, 165, 233, 0.82)" : "rgba(8, 145, 178, 0.95)");
            context.lineWidth = face.smoothPreview ? 0.85 : 1.05;
            context.stroke();
          }
        }
      }

      function onPointerDown(event) {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      }

      function onPointerMove(event) {
        if (!state.dragging) return;
        const deltaX = event.clientX - state.lastX;
        const deltaY = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        state.rotationY += deltaX * 0.01;
        state.rotationX = clamp(state.rotationX + deltaY * 0.01, -1.35, 1.35);
        meshCamera.rotationX = state.rotationX;
        meshCamera.rotationY = state.rotationY;
        scheduleRender();
      }

      function onPointerUp(event) {
        state.dragging = false;
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      }

      function onWheel(event) {
        event.preventDefault();
        state.zoom = clamp(state.zoom * (event.deltaY > 0 ? 0.92 : 1.08), 0.55, 2.8);
        meshCamera.zoom = state.zoom;
        scheduleRender();
      }

      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointerleave", onPointerUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });
      const resizeObserver = new ResizeObserver(resizeCanvas);
      resizeObserver.observe(host);
      resizeCanvas();

      trackMeshViewer({
        dispose() {
          cancelAnimationFrame(state.animationFrameId);
          resizeObserver.disconnect();
          canvas.removeEventListener("pointerdown", onPointerDown);
          canvas.removeEventListener("pointermove", onPointerMove);
          canvas.removeEventListener("pointerup", onPointerUp);
          canvas.removeEventListener("pointerleave", onPointerUp);
          canvas.removeEventListener("wheel", onWheel);
        },
      });
    }

    function escapeHtml(text) {
      return String(text == null ? "" : text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }

    function selectedFemMode() {
      // Map the analytical mode keys (b1/b2/b3/a1) to FEM mode numbers.
      // The FEM solver returns modes ordered by frequency, so mode 1 is the
      // fundamental. We expose 1..4 for the four analytical entries.
      const map = { b1: 1, b2: 2, b3: 3, a1: 4 };
      return map[simSelectedMode] || 1;
    }

    function renderStaticStiffness(state) {
      ensureSimOutputPanel();
      const container = document.getElementById("staticStiffnessContainer");
      if (!container) return;
      if (!state) {
        container.innerHTML = "";
        return;
      }
      if (state.status === "loading") {
        container.innerHTML =
          '<div class="sim-contour">' +
          '<p class="sim-fem-msg">Running directional static FEM: fixing the outer core and translating the inner core in X, Y, and Z...</p>' +
          '<div class="sim-fem-progress" aria-hidden="true"><span></span></div>' +
          '</div>';
        return;
      }
      if (state.status === "error") {
        container.innerHTML =
          '<div class="sim-contour"><p class="sim-fem-msg sim-fem-error">Static stiffness failed: ' +
          escapeHtml(state.message) + '</p></div>';
        return;
      }
      const data = state.data;
      if (!data) return;
      const calibration = data.calibration || {};
      const referenceTargets = calibration.reference_targets_n_per_mm || {};
      const directions = Array.isArray(data.directions) ? data.directions : [];
      const directionRows = directions.map((item) => {
        const axis = String(item.engineering_axis || "").toLowerCase();
        const fallbackTargets = {
          x: CLIENT_BUSHING_SPEC.target_kx_n_mm,
          y: CLIENT_BUSHING_SPEC.target_ky_n_mm,
          z: CLIENT_BUSHING_SPEC.target_kz_n_mm,
        };
        const target = Number(referenceTargets["k" + axis] || fallbackTargets[axis] || 0);
        const stiffness = Number(item.stiffness_n_per_mm || 0);
        const errorPercent = target > 0 ? Math.abs(stiffness - target) / target * 100 : Number.NaN;
        const errorClass = errorPercent < 5
          ? "stiffness-error-good"
          : (errorPercent <= 10 ? "stiffness-error-warning" : "stiffness-error-bad");
        return (
          '<tr><td>K' + escapeHtml(axis) + '</td>' +
          '<td class="sim-value">' + formatNumber(target, 2) + ' N/mm</td>' +
          '<td>mesh ' + escapeHtml(item.mesh_axis) + '</td>' +
          '<td class="sim-value">' + formatNumber(item.reaction_force_n, 2) + ' N</td>' +
          '<td class="sim-value">' + formatNumber(stiffness, 2) + ' N/mm</td>' +
          '<td class="stiffness-error-cell ' + errorClass + '">' +
          (Number.isFinite(errorPercent) ? formatNumber(errorPercent, 2) + '%' : '\u2014') +
          '</td></tr>'
        );
      }).join("");
      container.innerHTML =
        '<div class="sim-contour">' +
        '<p class="sim-fem-msg"><strong>Directional static stiffness</strong> \u00b7 CalculiX linear-static solve \u00b7 material ' +
        escapeHtml(data.material || "rubber") + ' \u00b7 ' +
        escapeHtml(calibration.status === "client_calibrated" ? "client calibrated" : "uncalibrated") + '</p>' +
        '<div class="sim-kpi-grid">' +
        '<div class="sim-kpi"><span>Kx axial</span><strong>' + formatNumber(data.kx_n_per_mm, 2) + ' N/mm</strong></div>' +
        '<div class="sim-kpi"><span>Ky radial</span><strong>' + formatNumber(data.ky_n_per_mm, 2) + ' N/mm</strong></div>' +
        '<div class="sim-kpi"><span>Kz radial</span><strong>' + formatNumber(data.kz_n_per_mm, 2) + ' N/mm</strong></div>' +
        '<div class="sim-kpi"><span>Prescribed motion</span><strong>' + formatNumber(directions[0] && directions[0].displacement_mm, 2) + ' mm</strong></div>' +
        '</div>' +
        '<div class="sim-table-scroll"><table class="sim-table sim-table-rows" style="margin-top:10px;min-width:720px"><tr><th>Client axis</th><th style="text-align:right">Client target</th><th>Solver axis</th><th style="text-align:right">Reaction</th><th style="text-align:right">Stiffness</th><th style="text-align:right">Error</th></tr>' +
        directionRows + '</table></div>' +
        '<div class="sim-dashboard-detail" style="margin-top:10px">' +
        '<div class="sim-detail-item"><span>Centerline</span><strong>' + escapeHtml(data.centerline_axis || "X") + '</strong></div>' +
        '<div class="sim-detail-item"><span>Fixed interface</span><strong>' + escapeHtml(data.fixed_interface || "outer core") + '</strong></div>' +
        '<div class="sim-detail-item"><span>Effective modulus</span><strong>' + formatNumber(data.youngs_modulus_mpa, 2) + ' MPa</strong></div>' +
        '<div class="sim-detail-item"><span>Interface nodes</span><strong>' + Number(data.inner_node_count || 0) + ' inner / ' + Number(data.outer_node_count || 0) + ' outer</strong></div>' +
        '</div>' +
        '<p class="muted" style="margin:10px 0 0">' + escapeHtml(data.model_limitations || "") + '</p>' +
        '</div>';
    }

    async function runStaticStiffness() {
      const button = document.getElementById("simStaticBtn");
      const intent = (lastExport && lastExport.intent) || currentEditIntent || {};
      if (!isStructuredBushingIntentClient(intent)) {
        lastStaticStiffness = {
          status: "error",
          message: "Static K currently requires structured Rubber bushing geometry with OD, ID, and core lengths.",
        };
        renderStaticStiffness(lastStaticStiffness);
        return;
      }
      if (button) button.disabled = true;
      lastStaticStiffness = { status: "loading" };
      renderStaticStiffness(lastStaticStiffness);
      try {
        const response = await fetch("/run-static-stiffness", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: (lastExport && lastExport.name) || "bushing",
            intent: intent,
            material: intent.material && intent.material.name,
            displacement_mm: 1,
            global_template: globalMeshTemplate,
          }),
        });
        if (!response.ok) {
          let detail = "Static stiffness failed (HTTP " + response.status + ").";
          try {
            const payload = await response.json();
            detail = payload.detail || detail;
          } catch (error) {}
          lastStaticStiffness = { status: "error", message: detail };
        } else {
          lastStaticStiffness = { status: "ok", data: await response.json() };
        }
      } catch (error) {
        lastStaticStiffness = { status: "error", message: (error && error.message) || "Network error" };
      } finally {
        renderStaticStiffness(lastStaticStiffness);
        if (button) button.disabled = false;
      }
    }

    function renderFemContour(state) {
      ensureSimOutputPanel();
      const container = document.getElementById("simFemContainer");
      if (!container) return;
      cleanupFemViewer();
      if (!state) { container.innerHTML = ""; return; }
      if (state.status === "loading") {
        container.innerHTML =
          '<div class="sim-contour">' +
          '<p class="sim-fem-msg">Running FEM batch (one mesh \u2192 CalculiX modal solve for ' + femBatchCount + ' mode(s) \u2192 selected contour). This can take several minutes...</p>' +
          '<div class="sim-fem-progress" aria-hidden="true"><span></span></div>' +
          '</div>';
        return;
      }
      if (state.status === "error") {
        container.innerHTML =
          '<div class="sim-contour">' +
          '<p class="sim-fem-msg sim-fem-error">FEM contour failed: ' + escapeHtml(state.message) + '</p>' +
          '</div>';
        return;
      }
      if (state.status === "ok" && state.data) {
        const d = state.data;
        const comparison = femComparisonMeshes(d);
        const displayMesh = comparison.surrogate || femSurfaceForDisplay(d);
        const hasMesh = displayMesh && Array.isArray(displayMesh.faces) && displayMesh.faces.length;
        const hasComparison = comparison.uploaded && comparison.surrogate;
        const exactFemNote = d.fem_source === "exact_uploaded_geometry"
          ? '<p class="sim-fem-msg">Exact uploaded-geometry FEM: modal solve and contour are computed on the uploaded STEP/STL volume mesh.</p>'
          : '';
        const uploadFemNote = hasComparison
          ? '<p class="sim-fem-msg">Uploaded STL reference is shown beside the actual structured hex surrogate; FEM stress contour is rendered smooth without dense mesh lines.</p>'
          : (displayMesh && displayMesh.source === "uploaded_stl"
          ? '<p class="sim-fem-msg">Contour preview is mapped onto the uploaded STL surface; modal solve values come from the structured hex surrogate.</p>'
          : '');
        const femView = hasComparison
          ? '<div class="sim-fem-frame">' +
            '<div class="fem-compare" aria-label="Uploaded STL reference and structured hex surrogate FEM comparison">' +
            femPreviewCardHtml("simFemUploadedViewer", "Uploaded STL reference", "mapped stress reference", comparison.uploaded) +
            femPreviewCardHtml("simFemSurrogateViewer", "FEM stress contour", "actual structured hex surrogate", comparison.surrogate) +
            '</div>' +
            femLegendHtml(comparison.surrogate) +
            '</div>'
          : (hasMesh
          ? '<div class="sim-fem-frame">' +
            '<div class="sim-fem-viewer" id="simFemViewer" aria-label="Interactive FEM contour view"></div>' +
            femLegendHtml(displayMesh) +
            '</div>'
          : '<img alt="von Mises stress contour for mode ' + d.mode + '" src="data:image/png;base64,' + d.contour_png_base64 + '"/>');
        const modesRows = (d.modes || []).map((m) => (
          '<tr class="' + (Number(m.mode_number) === Number(d.mode) ? 'sim-row-active' : '') + '">' +
          '<td>Mode ' + m.mode_number + '</td>' +
          '<td class="sim-value">' + formatHz(m.frequency_hz) + '</td></tr>'
        )).join("");
        const pcaHtml = femPcaHtml(d.pca);
        container.innerHTML =
          '<div class="sim-contour">' +
          '<p class="sim-fem-msg"><strong>Validated FEM batch</strong> \u00b7 ' + d.num_modes + ' mode(s) solved \u00b7 showing von Mises contour for mode ' + d.mode +
          ' \u00b7 material ' + escapeHtml(d.material) + '</p>' +
          exactFemNote +
          uploadFemNote +
          femView +
          '<div class="sim-head-actions" style="margin-top:10px;justify-content:flex-end">' +
          '<button type="button" class="sim-btn" id="femCsvBtn">Download CSV</button>' +
          '</div>' +
          pcaHtml +
          (modesRows ? ('<table class="sim-table sim-table-rows" style="margin-top:10px"><tr><th>FEM mode</th><th style="text-align:right">Frequency</th></tr>' + modesRows + '</table>') : '') +
          '</div>';
        if (hasComparison) {
          renderFemMeshCanvas(document.getElementById("simFemUploadedViewer"), comparison.uploaded, { showEdges: false });
          renderFemMeshCanvas(document.getElementById("simFemSurrogateViewer"), comparison.surrogate, { showEdges: false });
        } else if (hasMesh) {
          renderFemMeshCanvas(document.getElementById("simFemViewer"), displayMesh, { showEdges: false });
        }
        bindFemPcaPlot(d.pca);
        const csvBtn = document.getElementById("femCsvBtn");
        if (csvBtn) {
          csvBtn.addEventListener("click", () => {
            downloadBlob(femCsvBlob(d), ((lastExport && lastExport.name) || "model") + "_fem_modes.csv");
          });
        }
      }
    }

    function femComparisonMeshes(data) {
      if (!data || data.fem_source === "exact_uploaded_geometry") {
        return { uploaded: null, surrogate: null };
      }
      const surrogate = data.fem_mesh && Array.isArray(data.fem_mesh.faces) && data.fem_mesh.faces.length
        ? Object.assign({}, data.fem_mesh, { source: data.fem_mesh.source || "structured_hex_surrogate" })
        : null;
      const uploaded = uploadedFemSurfaceForDisplay(surrogate);
      if (uploaded && surrogate) {
        return { uploaded, surrogate };
      }
      return { uploaded: null, surrogate: null };
    }

    function femPreviewCardHtml(viewerId, title, subtitle, mesh) {
      if (!mesh || !Array.isArray(mesh.faces) || !mesh.faces.length) {
        return '<div class="mesh-output err">FEM surface was not available for this preview.</div>';
      }
      return (
        '<div class="mesh-compare-card">' +
        '<div class="mesh-compare-card-head"><strong>' + escapeHtml(title) + '</strong><span>' + escapeHtml(subtitle) + '</span></div>' +
        '<div class="sim-fem-viewer" id="' + escapeHtml(viewerId) + '" aria-label="' + escapeHtml(title) + '"></div>' +
        '</div>'
      );
    }

    function femSurfaceForDisplay(data) {
      if (data && data.fem_source === "exact_uploaded_geometry") {
        return data.fem_mesh || null;
      }
      const uploadedSurface = uploadedFemSurfaceForDisplay(data && data.fem_mesh);
      return uploadedSurface || (data && data.fem_mesh) || null;
    }

    function uploadedFemSurfaceForDisplay(femMesh) {
      if (!rubberBushingWorkflowActive) return null;
      const baseSurface = uploadedMeshSurfaceForMeshResult();
      if (!baseSurface || !Array.isArray(baseSurface.faces) || !baseSurface.faces.length) return null;
      const bounds = computeMeshBounds(baseSurface.faces);
      const span = Math.max(bounds.size.y, bounds.size.x, bounds.size.z, 1e-6);
      const axis = bounds.size.y >= bounds.size.x && bounds.size.y >= bounds.size.z
        ? "y"
        : (bounds.size.x >= bounds.size.z ? "x" : "z");
      const minAxis = bounds.center[axis] - bounds.size[axis] / 2;
      const maxAxis = bounds.center[axis] + bounds.size[axis] / 2;
      const axisSpan = Math.max(maxAxis - minAxis, span, 1e-6);
      return {
        faces: baseSurface.faces.map((face) => {
          const average = face.points.reduce((sum, point) => sum + Number(point[axis] || 0), 0) / Math.max(face.points.length, 1);
          const t = clamp((average - minAxis) / axisSpan, 0, 1);
          return Object.assign({}, face, {
            color: contourPreviewColor(t),
            value: t,
          });
        }),
        field: ((femMesh && femMesh.field) || "S, Mises") + " mapped preview",
        unit: femMesh && femMesh.unit ? femMesh.unit : "",
        scalar_min: femMesh && Number.isFinite(Number(femMesh.scalar_min)) ? Number(femMesh.scalar_min) : 0,
        scalar_max: femMesh && Number.isFinite(Number(femMesh.scalar_max)) ? Number(femMesh.scalar_max) : 1,
        face_count: baseSurface.faces.length,
        source: "uploaded_stl",
      };
    }

    function contourPreviewColor(t) {
      const stops = [
        [0.00, [32, 25, 156]],
        [0.18, [0, 91, 255]],
        [0.36, [0, 200, 255]],
        [0.54, [64, 220, 104]],
        [0.70, [255, 235, 59]],
        [0.84, [255, 135, 0]],
        [1.00, [204, 0, 0]],
      ];
      const value = clamp(Number(t) || 0, 0, 1);
      for (let index = 1; index < stops.length; index += 1) {
        const left = stops[index - 1];
        const right = stops[index];
        if (value <= right[0]) {
          const local = (value - left[0]) / Math.max(right[0] - left[0], 1e-9);
          const rgb = left[1].map((channel, channelIndex) => Math.round(channel + (right[1][channelIndex] - channel) * local));
          return rgbToHex({ r: rgb[0], g: rgb[1], b: rgb[2] });
        }
      }
      return "#cc0000";
    }

    function femPcaHtml(pca) {
      const components = pca && Array.isArray(pca.components) ? pca.components : [];
      if (!components.length) return "";
      const rows = components.map((pc) => {
        const ratio = Number(pc.explained_variance_ratio || 0) * 100;
        const cumulative = Number(pc.cumulative_variance_ratio || 0) * 100;
        const axis = String(pc.dominant_axis || "").toUpperCase();
        const axisEnergy = pc.axis_energy || {};
        const axisLabel = axis ? '<span class="axis-pill">' + escapeHtml(axis) + '</span>' : "";
        return (
          '<tr>' +
          '<td>PC' + pc.component + '</td>' +
          '<td class="sim-value">' + ratio.toFixed(1) + '%</td>' +
          '<td class="sim-value">' + cumulative.toFixed(1) + '%</td>' +
          '<td>' + escapeHtml(pc.characteristic || "Mode-family variation") + ' ' + axisLabel + '</td>' +
          '<td>Mode ' + pc.dominant_mode + ' (' + formatHz(pc.dominant_frequency_hz) + ')</td>' +
          '<td class="sim-value">X ' + percent(axisEnergy.x) + ' / Y ' + percent(axisEnergy.y) + ' / Z ' + percent(axisEnergy.z) + '</td>' +
          '</tr>'
        );
      }).join("");
      const has3d = components.length >= 3 && (pca.mode_scores || []).some((entry) => (entry.scores || []).length >= 3);
      const legend = (pca.mode_scores || []).map((entry) => {
        const modeNo = Number(entry.mode_number) || 0;
        return (
          '<span class="sim-pca-legend-item">' +
          '<span class="sim-pca-legend-dot" style="background:' + pcaPointColor(modeNo) + '"></span>' +
          'Mode ' + modeNo + ' \u00b7 ' + formatHz(entry.frequency_hz) +
          '</span>'
        );
      }).join("");
      return (
        '<details class="sim-pca" id="simPcaDetails" style="margin-top:10px">' +
        '<summary class="sim-pca-summary">' +
        '<span class="sim-pca-summary-copy"><strong>Advanced analysis</strong>' +
        '<small>Mode-shape PCA for comparing deformation patterns across solved modes</small></span>' +
        '<span class="sim-pca-summary-meta">' + pca.mode_count + ' modes \u00b7 ' + pca.node_count + ' nodes</span>' +
        '</summary>' +
        '<div class="sim-pca-content">' +
        '<div class="category-subhead" style="margin:0 0 2px"><strong>Mode-shape PCA</strong><span>Hover over a point for its modal score details</span></div>' +
        '<div class="sim-pca-layout">' +
        '<div class="sim-pca-plot">' +
        '<div class="sim-pca-toolbar"><span class="sim-pca-toolbar-copy"><strong>Mode score map</strong>' +
        '<small>Distance indicates deformation-pattern difference</small></span>' +
        '<span class="sim-pca-toggle"><button type="button" class="active" data-pca-view="2d">2D</button><button type="button" data-pca-view="3d"' + (has3d ? '' : ' disabled') + '>3D</button></span></div>' +
        '<canvas class="sim-pca-canvas" id="simPcaCanvas" aria-label="PCA mode score plot"></canvas>' +
        '<div class="sim-pca-tooltip" id="simPcaTooltip" role="status"></div>' +
        '<div class="sim-pca-legend">' + legend + '</div>' +
        '</div>' +
        '<div class="sim-pca-features">' +
        '<table class="sim-table sim-table-rows"><tr><th>PC</th><th style="text-align:right">Variance</th><th style="text-align:right">Cumulative</th><th>Characteristic feature</th><th>Dominant mode</th><th style="text-align:right">Axis energy</th></tr>' +
        rows +
        '</table>' +
        '</div>' +
        '</div>' +
        '</div>' +
        '</details>'
      );
    }

    function percent(value) {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? (numeric * 100).toFixed(0) + '%' : '0%';
    }

    function bindFemPcaPlot(pca) {
      const canvas = document.getElementById("simPcaCanvas");
      if (!canvas || !pca || !Array.isArray(pca.mode_scores)) return;
      const details = document.getElementById("simPcaDetails");
      const tooltip = document.getElementById("simPcaTooltip");
      const plot = canvas.closest(".sim-pca-plot");
      let view = "2d";
      const buttons = Array.from(document.querySelectorAll("[data-pca-view]"));
      const render = () => {
        if (details && !details.open) return;
        window.requestAnimationFrame(() => drawPcaPlot(canvas, pca, view));
      };
      buttons.forEach((button) => {
        button.addEventListener("click", () => {
          if (button.disabled) return;
          view = button.getAttribute("data-pca-view") || "2d";
          buttons.forEach((item) => item.classList.toggle("active", item === button));
          render();
        });
      });
      if (details) {
        details.addEventListener("toggle", render);
      } else {
        render();
      }
      canvas.addEventListener("pointermove", (event) => {
        const points = canvas._pcaHitPoints || [];
        const canvasRect = canvas.getBoundingClientRect();
        const pointerX = event.clientX - canvasRect.left;
        const pointerY = event.clientY - canvasRect.top;
        let nearest = null;
        points.forEach((point) => {
          const distance = Math.hypot(point.x - pointerX, point.y - pointerY);
          if (distance <= 18 && (!nearest || distance < nearest.distance)) {
            nearest = { point, distance };
          }
        });
        canvas.style.cursor = nearest ? "pointer" : "default";
        if (!tooltip || !plot || !nearest) {
          if (tooltip) tooltip.style.display = "none";
          return;
        }
        const item = nearest.point;
        const scoreLabels = item.scores.map((score, index) => "PC" + (index + 1) + " " + formatPcaScore(score)).join(" \u00b7 ");
        tooltip.innerHTML =
          '<strong>Mode ' + item.modeNumber + ' \u00b7 ' + formatHz(item.frequencyHz) + '</strong>' +
          escapeHtml(scoreLabels);
        const plotRect = plot.getBoundingClientRect();
        tooltip.style.left = clamp(event.clientX - plotRect.left + 12, 8, Math.max(plotRect.width - 174, 8)) + "px";
        tooltip.style.top = clamp(event.clientY - plotRect.top + 12, 48, Math.max(plotRect.height - 66, 48)) + "px";
        tooltip.style.display = "block";
      });
      canvas.addEventListener("pointerleave", () => {
        canvas.style.cursor = "default";
        if (tooltip) tooltip.style.display = "none";
      });
      const resizeObserver = new ResizeObserver(render);
      resizeObserver.observe(canvas.parentElement || canvas);
    }

    function drawPcaPlot(canvas, pca, view) {
      const context = canvas.getContext("2d");
      if (!context) return;
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(360, Math.floor(rect.width || 520));
      const height = Math.max(300, Math.floor(rect.height || 300));
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * pixelRatio);
      canvas.height = Math.floor(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);
      context.fillStyle = "#fbfdff";
      context.fillRect(0, 0, width, height);

      const entries = (pca.mode_scores || []).filter((entry) => Array.isArray(entry.scores) && entry.scores.length >= 2);
      if (!entries.length) {
        canvas._pcaHitPoints = [];
        return;
      }
      const projected = entries.map((entry) => {
        const scores = entry.scores || [];
        const x = Number(scores[0]) || 0;
        const y = Number(scores[1]) || 0;
        const z = view === "3d" ? Number(scores[2]) || 0 : 0;
        return { entry, x: x + z * 0.44, y: y - z * 0.28, z };
      });
      const xs = projected.map((p) => p.x);
      const ys = projected.map((p) => p.y);
      const padding = { left: 64, right: 32, top: 28, bottom: 52 };
      const extentX = Math.max(...xs.map((value) => Math.abs(value)), 1e-6) * 1.18;
      const extentY = Math.max(...ys.map((value) => Math.abs(value)), 1e-6) * 1.18;
      const minX = -extentX;
      const maxX = extentX;
      const minY = -extentY;
      const maxY = extentY;
      const spanX = Math.max(maxX - minX, 1e-9);
      const spanY = Math.max(maxY - minY, 1e-9);
      const plotW = width - padding.left - padding.right;
      const plotH = height - padding.top - padding.bottom;
      const mapX = (value) => padding.left + ((value - minX) / spanX) * plotW;
      const mapY = (value) => padding.top + (1 - (value - minY) / spanY) * plotH;

      const components = Array.isArray(pca.components) ? pca.components : [];
      const varianceX = components[0] ? Number(components[0].explained_variance_ratio || 0) * 100 : 0;
      const varianceY = components[1] ? Number(components[1].explained_variance_ratio || 0) * 100 : 0;
      drawPcaGrid(context, width, height, padding, { minX, maxX, minY, maxY, mapX, mapY, view, varianceX, varianceY });
      const hitPoints = [];
      projected.sort((a, b) => a.z - b.z).forEach((point) => {
        const modeNo = Number(point.entry.mode_number) || 0;
        const radius = view === "3d" ? clamp(6 + Math.abs(point.z) * 3, 6, 11) : 7;
        const x = mapX(point.x);
        const y = mapY(point.y);
        context.save();
        context.shadowColor = "rgba(16, 36, 63, 0.18)";
        context.shadowBlur = 8;
        context.shadowOffsetY = 2;
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fillStyle = pcaPointColor(modeNo);
        context.fill();
        context.restore();
        context.strokeStyle = "#ffffff";
        context.lineWidth = 2;
        context.stroke();
        context.fillStyle = "#10243f";
        context.font = "600 11px Inter, system-ui, sans-serif";
        context.textAlign = point.x > 0 ? "right" : "left";
        context.fillText("M" + modeNo, x + (point.x > 0 ? -radius - 4 : radius + 4), y - radius - 3);
        hitPoints.push({
          x,
          y,
          modeNumber: modeNo,
          frequencyHz: Number(point.entry.frequency_hz) || 0,
          scores: (point.entry.scores || []).slice(0, view === "3d" ? 3 : 2),
        });
      });
      canvas._pcaHitPoints = hitPoints;
    }

    function drawPcaGrid(context, width, height, padding, axes) {
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      context.fillStyle = "#ffffff";
      context.fillRect(padding.left, padding.top, plotWidth, plotHeight);
      context.strokeStyle = "#e4ebf3";
      context.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const x = padding.left + (plotWidth * i) / 4;
        const y = padding.top + (plotHeight * i) / 4;
        context.beginPath(); context.moveTo(x, padding.top); context.lineTo(x, height - padding.bottom); context.stroke();
        context.beginPath(); context.moveTo(padding.left, y); context.lineTo(width - padding.right, y); context.stroke();
        const tickX = axes.minX + ((axes.maxX - axes.minX) * i) / 4;
        const tickY = axes.maxY - ((axes.maxY - axes.minY) * i) / 4;
        context.fillStyle = "#738196";
        context.font = "10px Inter, system-ui, sans-serif";
        context.textAlign = "center";
        context.fillText(formatPcaScore(tickX), x, height - padding.bottom + 17);
        context.textAlign = "right";
        context.fillText(formatPcaScore(tickY), padding.left - 9, y + 3);
      }
      context.strokeStyle = "#9aa8ba";
      context.lineWidth = 1.2;
      context.beginPath(); context.moveTo(axes.mapX(0), padding.top); context.lineTo(axes.mapX(0), height - padding.bottom); context.stroke();
      context.beginPath(); context.moveTo(padding.left, axes.mapY(0)); context.lineTo(width - padding.right, axes.mapY(0)); context.stroke();
      context.fillStyle = "#233b5f";
      context.font = "600 11px Inter, system-ui, sans-serif";
      context.textAlign = "center";
      context.fillText("PC1 (" + axes.varianceX.toFixed(1) + "% variance)", padding.left + plotWidth / 2, height - 8);
      context.save();
      context.translate(15, padding.top + plotHeight / 2);
      context.rotate(-Math.PI / 2);
      context.fillText("PC2 (" + axes.varianceY.toFixed(1) + "% variance)", 0, 0);
      context.restore();
      if (axes.view === "3d") {
        context.textAlign = "right";
        context.fillStyle = "#738196";
        context.fillText("PC3 shown as projected depth and marker size", width - padding.right, padding.top - 10);
      }
    }

    function formatPcaScore(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || Math.abs(numeric) < 1e-10) return "0";
      if (Math.abs(numeric) >= 100) return numeric.toFixed(0);
      if (Math.abs(numeric) >= 10) return numeric.toFixed(1);
      return numeric.toFixed(2);
    }

    function pcaPointColor(modeNo) {
      const colors = ["#2166ac", "#1a9850", "#fdae61", "#d7191c", "#7b3294", "#008c8c", "#b35806"];
      return colors[Math.abs((modeNo || 1) - 1) % colors.length];
    }

    function femLegendHtml(mesh) {
      const field = mesh && mesh.field ? mesh.field : "S, Mises";
      const minLabel = formatEngineering(mesh && mesh.scalar_min, mesh && mesh.unit ? mesh.unit : "");
      const maxLabel = formatEngineering(mesh && mesh.scalar_max, mesh && mesh.unit ? mesh.unit : "");
      return (
        '<div class="mesh-legend" aria-label="' + escapeHtml(field) + ' scale">' +
        '<strong>' + escapeHtml(field) + '</strong>' +
        '<span>' + escapeHtml(maxLabel) + '</span>' +
        '<div class="mesh-legend-bar"></div>' +
        '<span>' + escapeHtml(minLabel) + '</span>' +
        '</div>'
      );
    }

    function cleanupFemViewer() {
      const viewers = Array.isArray(activeFemViewer) ? activeFemViewer : (activeFemViewer ? [activeFemViewer] : []);
      viewers.forEach((viewer) => {
        if (viewer && typeof viewer.dispose === "function") {
          viewer.dispose();
        }
      });
      activeFemViewer = null;
    }

    function trackFemViewer(viewer) {
      if (!viewer) return;
      if (!activeFemViewer) activeFemViewer = [];
      if (!Array.isArray(activeFemViewer)) activeFemViewer = [activeFemViewer];
      activeFemViewer.push(viewer);
    }

    function renderFemMeshCanvas(host, mesh, options) {
      if (!host || !mesh || !Array.isArray(mesh.faces) || !mesh.faces.length) return;
      const renderOptions = options || {};
      const canvas = document.createElement("canvas");
      host.innerHTML = "";
      host.appendChild(canvas);
      const context = canvas.getContext("2d");
      if (!context) return;

      const bounds = computeMeshBounds(mesh.faces);
      const size = bounds.size;
      const extent = Math.max(size.x, size.y, size.z, 1);
      const center = bounds.center;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      const lightDirection = normalizeVector({ x: 0.35, y: 0.85, z: 0.4 });
      const centeredFaces = mesh.faces.map((face) => ({
        color: face.color || "#40c463",
        points: face.points.map((point) => ({
          x: point.x - center.x,
          y: point.y - center.y,
          z: point.z - center.z,
        })),
      }));

      const state = {
        width: 0,
        height: 0,
        rotationX: femCamera.rotationX,
        rotationY: femCamera.rotationY,
        zoom: femCamera.zoom,
        dragging: false,
        lastX: 0,
        lastY: 0,
        animationFrameId: 0,
        renderQueued: false,
      };

      function scheduleRender() {
        if (state.renderQueued) return;
        state.renderQueued = true;
        state.animationFrameId = requestAnimationFrame(() => {
          state.renderQueued = false;
          draw();
        });
      }

      function resizeCanvas() {
        state.width = Math.max(320, Math.floor(host.clientWidth || 520));
        state.height = Math.max(320, Math.floor(host.clientHeight || 360));
        canvas.width = Math.floor(state.width * pixelRatio);
        canvas.height = Math.floor(state.height * pixelRatio);
        canvas.style.width = `${state.width}px`;
        canvas.style.height = `${state.height}px`;
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        scheduleRender();
      }

      function projectRotatedPoint(rotated) {
        const scale = (Math.min(state.width, state.height) / (extent * 2.35)) * state.zoom;
        return {
          x: state.width / 2 + rotated.x * scale,
          y: state.height / 2 - rotated.y * scale,
          z: rotated.z,
        };
      }

      function draw() {
        context.clearRect(0, 0, state.width, state.height);
        context.fillStyle = "#f8fbff";
        context.fillRect(0, 0, state.width, state.height);
        const projectedFaces = centeredFaces
          .map((face) => {
            const rotatedPoints = face.points.map((point) => rotatePoint(point, state.rotationX, state.rotationY));
            const projectedPoints = rotatedPoints.map(projectRotatedPoint);
            const normal = computeFaceNormal(rotatedPoints);
            const intensity = clamp(dotProduct(normal, lightDirection) * 0.45 + 0.78, 0.38, 1.0);
            const averageDepth = rotatedPoints.reduce((sum, point) => sum + point.z, 0) / rotatedPoints.length;
            return {
              projectedPoints,
              averageDepth,
              fill: shadeColor(face.color, intensity),
              stroke: renderOptions.edgeColor || shadeColor(face.color, Math.max(intensity - 0.24, 0.2)),
            };
          })
          .sort((left, right) => left.averageDepth - right.averageDepth);

        for (const face of projectedFaces) {
          context.beginPath();
          context.moveTo(face.projectedPoints[0].x, face.projectedPoints[0].y);
          for (let index = 1; index < face.projectedPoints.length; index += 1) {
            context.lineTo(face.projectedPoints[index].x, face.projectedPoints[index].y);
          }
          context.closePath();
          context.fillStyle = face.fill;
          context.fill();
          if (renderOptions.showEdges) {
            context.strokeStyle = face.stroke;
            context.lineWidth = renderOptions.edgeWidth || 0.35;
            context.stroke();
          }
        }
      }

      function onPointerDown(event) {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      }

      function onPointerMove(event) {
        if (!state.dragging) return;
        const deltaX = event.clientX - state.lastX;
        const deltaY = event.clientY - state.lastY;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        state.rotationY += deltaX * 0.01;
        state.rotationX = clamp(state.rotationX + deltaY * 0.01, -1.35, 1.35);
        femCamera.rotationX = state.rotationX;
        femCamera.rotationY = state.rotationY;
        scheduleRender();
      }

      function onPointerUp(event) {
        state.dragging = false;
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      }

      function onWheel(event) {
        event.preventDefault();
        state.zoom = clamp(state.zoom * (event.deltaY > 0 ? 0.92 : 1.08), 0.55, 2.8);
        femCamera.zoom = state.zoom;
        scheduleRender();
      }

      canvas.addEventListener("pointerdown", onPointerDown);
      canvas.addEventListener("pointermove", onPointerMove);
      canvas.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("pointerleave", onPointerUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });
      const resizeObserver = new ResizeObserver(resizeCanvas);
      resizeObserver.observe(host);
      resizeCanvas();

      trackFemViewer({
        dispose() {
          cancelAnimationFrame(state.animationFrameId);
          resizeObserver.disconnect();
          canvas.removeEventListener("pointerdown", onPointerDown);
          canvas.removeEventListener("pointermove", onPointerMove);
          canvas.removeEventListener("pointerup", onPointerUp);
          canvas.removeEventListener("pointerleave", onPointerUp);
          canvas.removeEventListener("wheel", onWheel);
        },
      });
    }

    async function runFemContour() {
      const femBtn = document.getElementById("simFemBtn");
      const exactUpload = exactUploadedGeometryContext();
      const prompt = (lastExport && lastExport.prompt) || pendingDraftPrompt || (exactUpload ? "Exact uploaded geometry FEM" : "");
      const intent = (lastExport && lastExport.intent) || {};
      if (!exactUpload && !prompt.trim() && !isStructuredBushingIntentClient(intent)) {
        lastFemContour = { status: "error", message: "Send a chat message first, upload STEP/STL geometry, or generate a Rubber bushing before running FEM." };
        renderFemContour(lastFemContour);
        return;
      }
      if (exactUpload && lastMeshResult && lastMeshResult.status === "preview_only") {
        lastFemContour = {
          status: "error",
          message: "FEM needs a real closed volume mesh. This uploaded STL is preview-only because exact meshing failed; upload a watertight STL/STEP solid before running FEM batch.",
        };
        renderFemContour(lastFemContour);
        return;
      }
      const countInput = document.getElementById("simBatchCount");
      const contourInput = document.getElementById("simContourMode");
      femBatchCount = readClampedInput(countInput, 1, 10, femBatchCount || 6);
      femContourMode = readClampedInput(contourInput, 1, femBatchCount, selectedFemMode());
      if (countInput) countInput.value = String(femBatchCount);
      if (contourInput) {
        contourInput.max = String(femBatchCount);
        contourInput.value = String(femContourMode);
      }
      if (femBtn) femBtn.disabled = true;
      lastFemContour = { status: "loading" };
      renderFemContour(lastFemContour);
      try {
        const response = await fetch("/run-fem", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: prompt,
            mode: femContourMode,
            num_modes: femBatchCount,
            name: (lastExport && lastExport.name) || "model",
            intent: intent,
            ...meshRequestOptions(intent),
          }),
        });
        if (!response.ok) {
          let detail = "FEM run failed (HTTP " + response.status + ").";
          try { const payload = await response.json(); detail = payload.detail || detail; } catch (err) {}
          lastFemContour = { status: "error", message: detail };
          renderFemContour(lastFemContour);
          return;
        }
        const data = await response.json();
        lastFemContour = { status: "ok", data: data };
        renderFemContour(lastFemContour);
      } catch (error) {
        lastFemContour = { status: "error", message: (error && error.message) || "Network error" };
        renderFemContour(lastFemContour);
      } finally {
        if (femBtn) femBtn.disabled = false;
      }
    }

    // Called on live slider edits: only refresh results if already shown.
    function updateSimEstimate() {
      if (simShown) {
        renderSimOutput();
      }
    }

    function defaultRubberBushingIntent() {
      return {
        part_type: "bushing",
        material: { name: "rubber", shore_a: 55, sleeve_material: "steel" },
        geometry: {
          outer_diameter_mm: 76,
          inner_diameter_mm: 28,
          height_mm: 40,
          rubber_thickness_mm: 22.5,
          chamfer_mm: 2,
          fillet_mm: 0,
          inner_sleeve: true,
          inner_sleeve_diameter_mm: 31,
          inner_sleeve_length_mm: 40,
          inner_sleeve_thickness_mm: 1.5,
          outer_sleeve: false,
          outer_sleeve_thickness_mm: 0,
          metal_sleeve_thickness_mm: 0,
          flange: "none",
          flange_diameter_mm: 0,
          flange_thickness_mm: 0,
          hole_pattern: "none",
          hole_count: 0,
          hole_diameter_mm: 0,
          bore_shape: "round",
          bore_corner_radius_mm: 4,
          slot_count: 0,
          slot_width_deg: 18,
          slot_depth_mm: 10,
          slot_start_angle_deg: 0,
          slot_radial_mode: "outer",
          slot_axial_mode: "through",
          slot_axial_height_mm: 30,
          inner_core_length_mm: 40,
          outer_core_length_mm: 40,
          arms: [],
          holes: [],
        },
        simulation_hints: { target_output: "cad" },
        missing_information: [],
        ui_workflow: { product_family: "bushing", bushing_type: "rubber-bushing" },
      };
    }

    function cloneJson(value) {
      return JSON.parse(JSON.stringify(value || {}));
    }

    function normalizeRubberBushingIntent(intent) {
      const normalized = Object.assign(defaultRubberBushingIntent(), cloneJson(intent));
      normalized.part_type = "bushing";
      normalized.material = Object.assign({ name: "rubber", shore_a: 55, sleeve_material: "steel" }, normalized.material || {});
      normalized.simulation_hints = Object.assign({ target_output: "cad" }, normalized.simulation_hints || {});
      normalized.missing_information = Array.isArray(normalized.missing_information) ? normalized.missing_information : [];
      normalized.ui_workflow = Object.assign({ product_family: "bushing", bushing_type: "rubber-bushing" }, normalized.ui_workflow || {});
      const geom = Object.assign({}, defaultRubberBushingIntent().geometry, normalized.geometry || {});
      geom.outer_diameter_mm = readPositiveNumber(geom.outer_diameter_mm, 76);
      geom.inner_diameter_mm = Math.min(readPositiveNumber(geom.inner_diameter_mm, 28), geom.outer_diameter_mm - 1);
      geom.height_mm = readPositiveNumber(geom.height_mm, 40);
      geom.chamfer_mm = readNonNegativeNumber(geom.chamfer_mm, 2);
      geom.fillet_mm = readNonNegativeNumber(geom.fillet_mm, 0);
      geom.inner_sleeve = boolValue(geom.inner_sleeve, true);
      geom.outer_sleeve = boolValue(geom.outer_sleeve, false);
      geom.inner_sleeve_diameter_mm = geom.inner_sleeve
        ? Math.max(readPositiveNumber(geom.inner_sleeve_diameter_mm, geom.inner_diameter_mm + 3), geom.inner_diameter_mm)
        : geom.inner_diameter_mm;
      geom.inner_sleeve_length_mm = geom.inner_sleeve ? readPositiveNumber(geom.inner_sleeve_length_mm, geom.height_mm) : 0;
      geom.inner_sleeve_thickness_mm = geom.inner_sleeve ? Math.max(0, (geom.inner_sleeve_diameter_mm - geom.inner_diameter_mm) / 2) : 0;
      geom.outer_sleeve_thickness_mm = geom.outer_sleeve ? readNonNegativeNumber(geom.outer_sleeve_thickness_mm || geom.metal_sleeve_thickness_mm, 0) : 0;
      geom.metal_sleeve_thickness_mm = geom.outer_sleeve ? geom.outer_sleeve_thickness_mm : 0;
      geom.flange = ["none", "top", "bottom", "both"].includes(String(geom.flange || "none")) ? String(geom.flange || "none") : "none";
      geom.flange_diameter_mm = geom.flange === "none" ? 0 : readNonNegativeNumber(geom.flange_diameter_mm, geom.outer_diameter_mm + 16);
      geom.flange_thickness_mm = geom.flange === "none" ? 0 : readNonNegativeNumber(geom.flange_thickness_mm, 4);
      geom.hole_pattern = ["none", "axial", "radial"].includes(String(geom.hole_pattern || "none")) ? String(geom.hole_pattern || "none") : "none";
      geom.hole_count = geom.hole_pattern === "none" ? 0 : clampInt(geom.hole_count, 1, 24, 4);
      geom.hole_diameter_mm = geom.hole_pattern === "none" ? 0 : readPositiveNumber(geom.hole_diameter_mm, 6);
      geom.bore_shape = ["round", "rounded_square"].includes(String(geom.bore_shape || "round")) ? String(geom.bore_shape || "round") : "round";
      geom.bore_corner_radius_mm = geom.bore_shape === "rounded_square" ? readNonNegativeNumber(geom.bore_corner_radius_mm, 4) : 0;
      geom.slot_count = clampInt(geom.slot_count, 0, 24, 0);
      geom.slot_width_deg = geom.slot_count > 0 ? readPositiveNumber(geom.slot_width_deg, 18) : 0;
      geom.slot_depth_mm = geom.slot_count > 0 ? readPositiveNumber(geom.slot_depth_mm, Math.max(1, geom.rubber_thickness_mm * 0.45)) : 0;
      geom.slot_start_angle_deg = Number.isFinite(Number(geom.slot_start_angle_deg)) ? Number(geom.slot_start_angle_deg) : 0;
      geom.slot_radial_mode = ["outer", "through_wall"].includes(String(geom.slot_radial_mode || "outer")) ? String(geom.slot_radial_mode || "outer") : "outer";
      geom.slot_axial_mode = ["through", "centered"].includes(String(geom.slot_axial_mode || "through")) ? String(geom.slot_axial_mode || "through") : "through";
      geom.slot_axial_height_mm = geom.slot_axial_mode === "centered" ? Math.min(readPositiveNumber(geom.slot_axial_height_mm, geom.height_mm * 0.5), geom.height_mm) : geom.height_mm;
      geom.inner_core_length_mm = readPositiveNumber(geom.inner_core_length_mm, geom.height_mm);
      geom.outer_core_length_mm = readPositiveNumber(geom.outer_core_length_mm, geom.height_mm);
      geom.rubber_thickness_mm = readNonNegativeNumber(
        geom.rubber_thickness_mm,
        Math.max(0, (geom.outer_diameter_mm - geom.inner_sleeve_diameter_mm) / 2 - geom.outer_sleeve_thickness_mm)
      );
      geom.arms = Array.isArray(geom.arms) ? geom.arms : [];
      geom.holes = Array.isArray(geom.holes) ? geom.holes : [];
      normalized.geometry = geom;
      return normalized;
    }

    function boolValue(value, fallback) {
      if (value === true || value === "true" || value === "yes" || value === "on") return true;
      if (value === false || value === "false" || value === "no" || value === "off") return false;
      return Boolean(fallback);
    }

    function readPositiveNumber(value, fallback) {
      const parsed = Number(value);
      return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
    }

    function readNonNegativeNumber(value, fallback) {
      const parsed = Number(value);
      return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
    }

    function isRubberBushingWorkflow(intent) {
      const type = String((intent && intent.part_type) || "").toLowerCase();
      const workflow = intent && intent.ui_workflow;
      return rubberBushingWorkflowActive && (type === "bushing" || type === "rubber_mount") && (!workflow || workflow.bushing_type === "rubber-bushing");
    }

    function buildRubberBushingWorkflow(intent) {
      currentEditIntent = normalizeRubberBushingIntent(intent || currentEditIntent || defaultRubberBushingIntent());
      baseGeometry = Object.assign({}, currentEditIntent.geometry || {});
      lastExport.intent = currentEditIntent;
      lastExport.name = "rubber_bushing";
      lastExport.cadEngine = selectedCadEngine;
      jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
      if (!["space", "target"].includes(rubberBushingTab)) {
        rubberBushingTab = "space";
      }
      const tabs = [
        ["space", "Design Space"],
        ["target", "Target Stiffness"],
      ].map(([key, label]) => (
        '<button type="button" class="param-tab' + (rubberBushingTab === key ? ' active' : '') + '" data-rubber-tab="' + key + '">' + label + '</button>'
      )).join("");
      const body = rubberBushingTab === "target" ? targetStiffnessHtml() : designSpaceHtml();
      paramControls.innerHTML = cadEngineSelectorHtml() + '<div class="rubber-workflow"><div class="param-tabs">' + tabs + '</div>' + body + '</div>';
      bindRubberBushingWorkflow();
      if (paramHint) paramHint.textContent = "Explore design variants or find geometry from target stiffness.";
      renderMeshPanel();
      renderSimPanel();
    }

    function designSpaceHtml() {
      return '<div class="param-section" data-rubber-section="space">' +
        '<div class="param-section-title">Design ranges</div>' +
        '<div class="param-form-grid">' +
        rangeField("ds_inner_diameter", "Inner-core diameter (mm)", CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm, CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm) +
        rangeField("ds_inner_core_length", "Inner-core length (mm)", CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm, CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm) +
        rangeField("ds_outer_core_length", "Outer-core length (mm)", CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm, CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm) +
        '<div class="param-form-field full"><label for="ds_sample_count">Samples</label><select id="ds_sample_count"><option value="50">50</option><option value="100">100</option><option value="200">200</option></select></div>' +
        '</div>' +
        bushingConditionsHtml() +
        '<div class="param-actions"><button type="button" class="param-primary" id="generateVariantsBtn">Generate Variants</button></div>' +
        '<div class="variant-output" id="variantOutput">' + designSpaceTableHtml() + '</div>' +
        '</div>';
    }

    function bushingConditionsHtml() {
      return '<div class="best-geometry">' +
        '<strong>Client conditions</strong>' +
        '<span>Diameter inner core: ' + CLIENT_BUSHING_SPEC.inner_diameter_min_mm + ' to ' + CLIENT_BUSHING_SPEC.inner_diameter_max_mm + ' mm</span>' +
        '<span>Length inner core: ' + CLIENT_BUSHING_SPEC.inner_core_length_min_mm + ' to ' + CLIENT_BUSHING_SPEC.inner_core_length_max_mm + ' mm</span>' +
        '<span>Length outer core: ' + CLIENT_BUSHING_SPEC.outer_core_length_min_mm + ' to ' + CLIENT_BUSHING_SPEC.outer_core_length_max_mm + ' mm</span>' +
        '<span>Outer diameter: ' + formatNumber(CLIENT_BUSHING_SPEC.outer_diameter_mm, 1) + ' mm (fixed)</span>' +
        '<span>Swaging value: ' + formatNumber(CLIENT_BUSHING_SPEC.swaging_value_mm, 1) + ' mm</span>' +
        '<span>Decking value: ' + formatNumber(CLIENT_BUSHING_SPEC.decking_value_mm, 1) + ' mm</span>' +
        '<span>Internal teeth: no</span>' +
        '</div>';
    }

    function rangeField(prefix, label, minValue, maxValue, allowedMin = 0, allowedMax = "") {
      return '<div class="param-form-field"><label for="' + prefix + '_min">' + escapeHtml(label) + ' min</label><input id="' + prefix + '_min" type="number" value="' + minValue + '" step="0.5" min="' + allowedMin + '" max="' + allowedMax + '"></div>' +
        '<div class="param-form-field"><label for="' + prefix + '_max">' + escapeHtml(label) + ' max</label><input id="' + prefix + '_max" type="number" value="' + maxValue + '" step="0.5" min="' + allowedMin + '" max="' + allowedMax + '"></div>';
    }

    function designSpaceTableHtml() {
      if (!designSpaceCases.length) {
        return '<p class="muted">Generate variants to populate the case table.</p>';
      }
      const rows = designSpaceCases.map((item, index) => '<tr>' +
        '<td>' + escapeHtml(item.case_id) + '</td>' +
        '<td>' + formatNumber(item.geometry.outer_diameter_mm, 1) + '</td>' +
        '<td>' + formatNumber(item.geometry.inner_diameter_mm, 1) + '</td>' +
        '<td>' + formatNumber(item.geometry.inner_core_length_mm, 1) + '</td>' +
        '<td>' + formatNumber(item.geometry.outer_core_length_mm, 1) + '</td>' +
        '<td>Ready</td>' +
        '<td><button type="button" data-load-variant="' + index + '">Load</button> <button type="button" data-download-variant="' + index + '">JSON</button></td>' +
        '</tr>').join("");
      return '<table class="variant-table"><thead><tr><th>Case ID</th><th>OD</th><th>ID</th><th>L_inner</th><th>L_outer</th><th>CAD Status</th><th>Download</th></tr></thead><tbody>' + rows + '</tbody></table>';
    }

    function targetStiffnessHtml() {
      const resultTitle = targetStiffnessResult && targetStiffnessResult.withinTolerance
        ? "Target match"
        : "Closest screened geometry - outside 10% tolerance";
      const resultHtml = targetStiffnessResult ? '<div class="best-geometry">' +
        '<strong>' + resultTitle + ': ' + escapeHtml(targetStiffnessResult.case_id) + '</strong>' +
        '<span>Source: ' + escapeHtml(targetStiffnessResult.source || "analytical screening fallback") + '</span>' +
        '<span>OD ' + formatNumber(targetStiffnessResult.geometry.outer_diameter_mm, 1) + ' mm, ID ' + formatNumber(targetStiffnessResult.geometry.inner_diameter_mm, 1) + ' mm, inner core ' + formatNumber(targetStiffnessResult.geometry.inner_core_length_mm, 1) + ' mm, outer core ' + formatNumber(targetStiffnessResult.geometry.outer_core_length_mm, 1) + ' mm</span>' +
        '<span>Kx ' + formatTargetStiffness(targetStiffnessResult.kx) + ', Ky ' + formatTargetStiffness(targetStiffnessResult.ky) + ', Kz ' + formatTargetStiffness(targetStiffnessResult.kz) + '</span>' +
        '<span>Maximum target error ' + formatNumber(targetStiffnessResult.maxRelativeError * 100, 1) + '%; RMS target error ' + formatNumber(targetStiffnessResult.rmsRelativeError * 100, 1) + '%</span>' +
        '</div>' : '<p class="muted">Enter target stiffness and search the bounded design space.</p>';
      return '<div class="param-section" data-rubber-section="target">' +
        '<div class="param-section-title">Targets</div>' +
        '<div class="param-form-grid">' +
        '<div class="param-form-field"><label for="target_kx">Target Kx (N/mm)</label><input id="target_kx" data-target-input type="number" value="' + targetSearchInputs.kx + '" min="0" step="0.1"></div>' +
        '<div class="param-form-field"><label for="target_ky">Target Ky (N/mm)</label><input id="target_ky" data-target-input type="number" value="' + targetSearchInputs.ky + '" min="0" step="0.1"></div>' +
        '<div class="param-form-field"><label for="target_kz">Target Kz (N/mm)</label><input id="target_kz" data-target-input type="number" value="' + targetSearchInputs.kz + '" min="0" step="0.1"></div>' +
        '</div>' +
        '<p class="muted">POC test assumption: X is the bushing centerline, the outer core is fixed, and the inner core is displaced. Kx is axial; Ky and Kz are radial. Validate shortlisted designs with client test data or directional static FEM.</p>' +
        '<div class="param-section-title">Design bounds</div>' +
        '<div class="param-form-grid">' +
        rangeField("ts_inner_diameter", "Inner-core diameter (mm)", targetSearchInputs.idMin, targetSearchInputs.idMax, CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm) +
        rangeField("ts_inner_core_length", "Inner-core length (mm)", targetSearchInputs.innerLengthMin, targetSearchInputs.innerLengthMax, CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm) +
        rangeField("ts_outer_core_length", "Outer-core length (mm)", targetSearchInputs.outerLengthMin, targetSearchInputs.outerLengthMax, CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm) +
        '<div class="param-form-field full"><label for="ts_sample_count">Sample count</label><select id="ts_sample_count" data-target-input><option value="50"' + (targetSearchInputs.samples === 50 ? ' selected' : '') + '>50</option><option value="100"' + (targetSearchInputs.samples === 100 ? ' selected' : '') + '>100</option><option value="200"' + (targetSearchInputs.samples === 200 ? ' selected' : '') + '>200</option></select></div>' +
        '</div>' +
        bushingConditionsHtml() +
        '<div class="param-actions"><button type="button" class="param-primary" id="findBestGeometryBtn">Find Best Geometry</button><button type="button" class="param-reset" id="openStiffnessPcaBtn">PCA Dataset</button></div>' +
        resultHtml +
        '</div>';
    }

    function bindRubberBushingWorkflow() {
      bindCadEngineSelector();
      for (const tab of paramControls.querySelectorAll("[data-rubber-tab]")) {
        tab.addEventListener("click", () => {
          rubberBushingTab = tab.dataset.rubberTab || "space";
          buildParamControls(currentEditIntent);
        });
      }
      const variants = document.getElementById("generateVariantsBtn");
      if (variants) variants.addEventListener("click", generateDesignSpaceVariants);
      const best = document.getElementById("findBestGeometryBtn");
      if (best) best.addEventListener("click", findBestGeometry);
      const pcaDataset = document.getElementById("openStiffnessPcaBtn");
      if (pcaDataset) pcaDataset.addEventListener("click", openStiffnessPcaDashboard);
      for (const control of paramControls.querySelectorAll("[data-target-input], #ts_inner_diameter_min, #ts_inner_diameter_max, #ts_inner_core_length_min, #ts_inner_core_length_max, #ts_outer_core_length_min, #ts_outer_core_length_max")) {
        control.addEventListener("change", captureTargetSearchInputs);
      }
      for (const button of paramControls.querySelectorAll("[data-download-variant]")) {
        button.addEventListener("click", () => downloadVariantJson(Number(button.dataset.downloadVariant)));
      }
      for (const button of paramControls.querySelectorAll("[data-load-variant]")) {
        button.addEventListener("click", () => loadVariant(Number(button.dataset.loadVariant)));
      }
    }

    function applyUploadedBushingPocTopology(intent) {
      const normalized = normalizeRubberBushingIntent(intent || currentEditIntent || defaultRubberBushingIntent());
      const geom = normalized.geometry;
      geom.bore_shape = "rounded_square";
      geom.bore_corner_radius_mm = Math.max(2, Number(geom.bore_corner_radius_mm) || 4);
      geom.slot_count = Number(geom.slot_count) > 0 ? geom.slot_count : 4;
      geom.slot_width_deg = Number(geom.slot_width_deg) > 0 ? geom.slot_width_deg : 34;
      geom.slot_depth_mm = Math.max(Number(geom.slot_depth_mm) || 0, (geom.outer_diameter_mm - geom.inner_diameter_mm) * 0.48);
      geom.slot_start_angle_deg = Number.isFinite(Number(geom.slot_start_angle_deg)) ? Number(geom.slot_start_angle_deg) : 0;
      geom.slot_radial_mode = "through_wall";
      geom.slot_axial_mode = "through";
      geom.slot_axial_height_mm = geom.height_mm;
      return normalizeRubberBushingIntent(normalized);
    }

    function readFormNumber(id, fallback) {
      const el = document.getElementById(id);
      const parsed = Number(el && el.value);
      return Number.isFinite(parsed) ? parsed : fallback;
    }

    async function generateRubberParametricCad(intent) {
      const payloadIntent = normalizeRubberBushingIntent(intent || currentEditIntent || defaultRubberBushingIntent());
      uploadNeedsParametricConfirmation = false;
      startActivity("Generating CAD", ["Reading structured JSON", "Calling CAD API", "Rendering preview"]);
      try {
        const response = await fetch("/generate-parametric-cad", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cad_engine: selectedCadEngine, intent: payloadIntent }),
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Parametric CAD generation failed.");
        }
        currentEditIntent = normalizeRubberBushingIntent(payload.cad_intent || payloadIntent);
        lastExport.intent = currentEditIntent;
        lastExport.name = "rubber_bushing";
        lastExport.cadEngine = payload.cad_engine || selectedCadEngine;
        lastExport.prompt = "Rubber bushing structured parametric JSON";
        jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
        downloadBtn.disabled = false;
        preferParametric = true;
        overrideMeshFaces = null;
        meshEditMode = false;
        lastMeshResult = null;
        lastStaticStiffness = null;
        await render3DPreview(currentEditIntent);
        updateSummary(currentEditIntent);
        renderMeshPanel();
        renderSimPanel();
        completeActivity("CAD ready");
      } catch (error) {
        failActivity("CAD failed");
        appendMsg("bot", "Parametric CAD failed: " + (error && error.message ? error.message : error));
      }
    }

    function generateDesignSpaceVariants() {
      const samples = clampInt(readFormNumber("ds_sample_count", 50), 1, 200, 50);
      const idMin = clamp(readFormNumber("ds_inner_diameter_min", CLIENT_BUSHING_SPEC.inner_diameter_min_mm), CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm);
      const idMax = Math.max(idMin, clamp(readFormNumber("ds_inner_diameter_max", CLIENT_BUSHING_SPEC.inner_diameter_max_mm), CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm));
      const liMin = clamp(readFormNumber("ds_inner_core_length_min", CLIENT_BUSHING_SPEC.inner_core_length_min_mm), CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm);
      const liMax = Math.max(liMin, clamp(readFormNumber("ds_inner_core_length_max", CLIENT_BUSHING_SPEC.inner_core_length_max_mm), CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm));
      const loMin = clamp(readFormNumber("ds_outer_core_length_min", CLIENT_BUSHING_SPEC.outer_core_length_min_mm), CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm);
      const loMax = Math.max(loMin, clamp(readFormNumber("ds_outer_core_length_max", CLIENT_BUSHING_SPEC.outer_core_length_max_mm), CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm));
      designSpaceCases = makeRubberCandidates(samples, idMin, idMax, liMin, liMax, loMin, loMax).map((intent, index) => ({
        case_id: "RB-" + String(index + 1).padStart(3, "0"),
        geometry: intent.geometry,
        intent,
      }));
      buildParamControls(currentEditIntent);
    }

    function makeRubberCandidates(samples, idMin, idMax, liMin, liMax, loMin, loMax) {
      const base = normalizeRubberBushingIntent(currentEditIntent || defaultRubberBushingIntent());
      base.material.name = "rubber";
      const count = Math.max(1, samples);
      const sleeveDelta = Math.max(0, (base.geometry.inner_sleeve_diameter_mm || base.geometry.inner_diameter_mm) - base.geometry.inner_diameter_mm);
      return Array.from({ length: count }, (_, index) => {
        const sampleIndex = index + 1;
        const id = lerp(idMin, idMax, halton(sampleIndex, 2));
        const innerLength = lerp(liMin, liMax, halton(sampleIndex, 3));
        const outerLength = lerp(loMin, loMax, halton(sampleIndex, 5));
        const intent = normalizeRubberBushingIntent(base);
        intent.geometry.outer_diameter_mm = CLIENT_BUSHING_SPEC.outer_diameter_mm;
        intent.geometry.inner_diameter_mm = roundTo(id, 0.5);
        intent.geometry.inner_sleeve_diameter_mm = intent.geometry.inner_sleeve ? roundTo(id + sleeveDelta, 0.5) : intent.geometry.inner_diameter_mm;
        intent.geometry.inner_sleeve_thickness_mm = intent.geometry.inner_sleeve ? Math.max(0, (intent.geometry.inner_sleeve_diameter_mm - intent.geometry.inner_diameter_mm) / 2) : 0;
        intent.geometry.inner_core_length_mm = roundTo(innerLength, 0.5);
        intent.geometry.outer_core_length_mm = roundTo(outerLength, 0.5);
        intent.geometry.inner_sleeve_length_mm = intent.geometry.inner_sleeve ? intent.geometry.inner_core_length_mm : 0;
        intent.geometry.height_mm = Math.max(intent.geometry.inner_core_length_mm, intent.geometry.outer_core_length_mm, 1);
        intent.geometry.swaging_value_mm = CLIENT_BUSHING_SPEC.swaging_value_mm;
        intent.geometry.decking_value_mm = CLIENT_BUSHING_SPEC.decking_value_mm;
        intent.geometry.internal_teeth = CLIENT_BUSHING_SPEC.internal_teeth;
        intent.geometry.rubber_thickness_mm = Math.max(0, (intent.geometry.outer_diameter_mm - intent.geometry.inner_sleeve_diameter_mm) / 2 - intent.geometry.outer_sleeve_thickness_mm);
        return intent;
      });
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function halton(index, base) {
      let fraction = 1;
      let result = 0;
      let value = Math.max(1, Math.trunc(index));
      while (value > 0) {
        fraction /= base;
        result += fraction * (value % base);
        value = Math.floor(value / base);
      }
      return result;
    }

    function roundTo(value, step) {
      return Math.round(value / step) * step;
    }

    function downloadVariantJson(index) {
      const item = designSpaceCases[index];
      if (!item) return;
      downloadBlob(new Blob([JSON.stringify(item.intent, null, 2)], { type: "application/json" }), item.case_id.toLowerCase() + ".json");
    }

    async function loadVariant(index) {
      const item = designSpaceCases[index];
      if (!item) return;
      const keptUploadedMesh = applyRubberDesignIntent(item.intent);
      if (!keptUploadedMesh) {
        await generateRubberParametricCad(currentEditIntent);
      }
      rubberBushingTab = "space";
      buildParamControls(currentEditIntent);
    }

    function applyRubberDesignIntent(intent) {
      currentEditIntent = normalizeRubberBushingIntent(intent);
      uploadNeedsParametricConfirmation = false;
      lastExport.intent = currentEditIntent;
      lastExport.name = exportBaseName(currentEditIntent);
      jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
      updateSummary(currentEditIntent);
      lastMeshResult = null;
      lastShapePcaResult = null;
      lastStaticStiffness = null;
      if (meshEditMode && editableMesh) {
        preferParametric = false;
        overrideMeshFaces = warpEditableMeshFaces(currentEditIntent.geometry);
        render3DPreview(currentEditIntent).catch(() => {});
        updateSimEstimate();
        return true;
      }
      return false;
    }

    function captureTargetSearchInputs() {
      const clampBounds = (value, minimum, maximum) => clamp(Number(value), minimum, maximum);
      const idMin = clampBounds(readFormNumber("ts_inner_diameter_min", targetSearchInputs.idMin), CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm);
      const innerLengthMin = clampBounds(readFormNumber("ts_inner_core_length_min", targetSearchInputs.innerLengthMin), CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm);
      const outerLengthMin = clampBounds(readFormNumber("ts_outer_core_length_min", targetSearchInputs.outerLengthMin), CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm);
      targetSearchInputs = {
        kx: readPositiveNumber(readFormNumber("target_kx", targetSearchInputs.kx), CLIENT_BUSHING_SPEC.target_kx_n_mm),
        ky: readPositiveNumber(readFormNumber("target_ky", targetSearchInputs.ky), CLIENT_BUSHING_SPEC.target_ky_n_mm),
        kz: readPositiveNumber(readFormNumber("target_kz", targetSearchInputs.kz), CLIENT_BUSHING_SPEC.target_kz_n_mm),
        idMin,
        idMax: Math.max(idMin, clampBounds(readFormNumber("ts_inner_diameter_max", targetSearchInputs.idMax), CLIENT_BUSHING_SPEC.inner_diameter_min_mm, CLIENT_BUSHING_SPEC.inner_diameter_max_mm)),
        innerLengthMin,
        innerLengthMax: Math.max(innerLengthMin, clampBounds(readFormNumber("ts_inner_core_length_max", targetSearchInputs.innerLengthMax), CLIENT_BUSHING_SPEC.inner_core_length_min_mm, CLIENT_BUSHING_SPEC.inner_core_length_max_mm)),
        outerLengthMin,
        outerLengthMax: Math.max(outerLengthMin, clampBounds(readFormNumber("ts_outer_core_length_max", targetSearchInputs.outerLengthMax), CLIENT_BUSHING_SPEC.outer_core_length_min_mm, CLIENT_BUSHING_SPEC.outer_core_length_max_mm)),
        samples: clampInt(readFormNumber("ts_sample_count", targetSearchInputs.samples), 1, 200, 50),
      };
      return targetSearchInputs;
    }

    function formatTargetStiffness(value) {
      return formatNumber(value, 1) + " N/mm";
    }

    async function findBestGeometry() {
      const search = captureTargetSearchInputs();
      try {
        const response = await fetch("/search-stiffness", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            targets: {
              kx_n_per_mm: search.kx,
              ky_n_per_mm: search.ky,
              kz_n_per_mm: search.kz,
            },
            bounds: {
              inner_diameter_min_mm: search.idMin,
              inner_diameter_max_mm: search.idMax,
              inner_core_length_min_mm: search.innerLengthMin,
              inner_core_length_max_mm: search.innerLengthMax,
              outer_core_length_min_mm: search.outerLengthMin,
              outer_core_length_max_mm: search.outerLengthMax,
            },
            samples: Math.max(2000, search.samples),
          }),
        });
        if (response.ok) {
          const payload = await response.json();
          const predicted = payload.predicted_stiffness || {};
          const best = {
            case_id: payload.case_id || "NN-BEST",
            source: "trained static-FEM neural surrogate",
            intent: payload.intent,
            geometry: payload.design,
            kx: predicted.kx_n_per_mm,
            ky: predicted.ky_n_per_mm,
            kz: predicted.kz_n_per_mm,
            maxRelativeError: payload.max_relative_error,
            rmsRelativeError: payload.rms_relative_error,
            withinTolerance: Boolean(payload.within_tolerance),
          };
          targetStiffnessResult = best;
          if (!best.withinTolerance) {
            appendMsg("bot", "The trained static-FEM surrogate did not find a candidate inside the 10% tolerance. The closest prediction is shown and should be verified with Static K.");
          }
          const keptUploadedMesh = applyRubberDesignIntent(best.intent);
          if (!keptUploadedMesh) {
            await generateRubberParametricCad(currentEditIntent);
          }
          rubberBushingTab = "target";
          buildParamControls(currentEditIntent);
          return;
        }
        if (response.status !== 404) {
          let detail = "Surrogate search was unavailable.";
          try {
            const payload = await response.json();
            detail = payload.detail || detail;
          } catch (error) {}
          appendMsg("bot", detail + " Using the analytical screening fallback.");
        }
      } catch (error) {
        appendMsg("bot", "The trained surrogate could not be reached. Using the analytical screening fallback.");
      }
      let best = null;
      makeRubberCandidates(search.samples, search.idMin, search.idMax, search.innerLengthMin, search.innerLengthMax, search.outerLengthMin, search.outerLengthMax).forEach((intent, index) => {
        const est = estimateBushingModal(intent.geometry, intent.material && intent.material.name);
        if (!est) return;
        const kx = est.kAxial;
        const ky = est.kBending;
        const kz = est.kBending;
        const relativeErrors = [
          relativeErrorMagnitude(kx, search.kx),
          relativeErrorMagnitude(ky, search.ky),
          relativeErrorMagnitude(kz, search.kz),
        ];
        const score = relativeErrors.reduce((total, error) => total + error * error, 0);
        if (!best || score < best.score) {
          best = {
            case_id: "BEST-" + String(index + 1).padStart(3, "0"),
            source: "analytical screening fallback",
            intent,
            geometry: intent.geometry,
            kx,
            ky,
            kz,
            score,
            maxRelativeError: Math.max(...relativeErrors),
            rmsRelativeError: Math.sqrt(score / relativeErrors.length),
          };
        }
      });
      if (!best) {
        appendMsg("bot", "Target stiffness search could not evaluate the current bounds.");
        return;
      }
      best.withinTolerance = best.maxRelativeError <= CLIENT_BUSHING_SPEC.match_tolerance;
      targetStiffnessResult = best;
      if (!best.withinTolerance) {
        appendMsg("bot", "No design in the client bounds reached the 10% stiffness tolerance with the current screening model. The closest candidate is shown for review; validate or calibrate it with directional static FEM or client test data.");
      }
      const keptUploadedMesh = applyRubberDesignIntent(best.intent);
      if (!keptUploadedMesh) {
        await generateRubberParametricCad(currentEditIntent);
      }
      rubberBushingTab = "target";
      buildParamControls(currentEditIntent);
    }

    function relativeErrorMagnitude(value, target) {
      const scale = Math.max(Math.abs(target), 1);
      return Math.abs(value - target) / scale;
    }

    async function openStiffnessPcaDashboard() {
      let payload;
      try {
        const response = await fetch("/stiffness-dashboard-data");
        if (!response.ok) {
          let detail = "No trained PCA stiffness dataset is installed.";
          try {
            const body = await response.json();
            detail = body.detail || detail;
          } catch (error) {}
          appendMsg("bot", detail + " Generate the offline FEM dataset, validate it, and install its artifacts first.");
          return;
        }
        payload = await response.json();
      } catch (error) {
        appendMsg("bot", "The PCA stiffness dataset could not be loaded.");
        return;
      }
      const samples = (payload.samples || []).filter((item) => Array.isArray(item.shape_codes) && item.shape_codes.length >= 3);
      if (!samples.length) {
        appendMsg("bot", "The installed dataset has no three-component shape PCA encoding.");
        return;
      }
      const existing = document.getElementById("stiffnessPcaModal");
      if (existing) existing.remove();
      const overlay = document.createElement("div");
      overlay.id = "stiffnessPcaModal";
      overlay.className = "sim-modal-backdrop";
      overlay.innerHTML =
        '<div class="sim-modal" role="dialog" aria-modal="true" aria-label="Shape PCA stiffness dataset">' +
        '<div class="sim-modal-head"><strong>Shape-code design space</strong><button type="button" class="sim-modal-close" aria-label="Close">×</button></div>' +
        '<div class="sim-modal-body">' +
        '<div class="stiffness-pca-layout">' +
        '<div><div class="stiffness-pca-plot"><canvas id="stiffnessPcaCanvas" aria-label="First three PCA shape codes"></canvas></div>' +
        '<div class="stiffness-pca-legend"><span><i style="background:#15803d"></i>FEM training designs</span><span><i style="background:#e52d2f"></i>Target-near designs</span></div></div>' +
        '<div class="stiffness-pca-conditions"><strong>Client design conditions</strong>' +
        '<span>Inner-core diameter: ' + CLIENT_BUSHING_SPEC.inner_diameter_min_mm + ' to ' + CLIENT_BUSHING_SPEC.inner_diameter_max_mm + ' mm</span>' +
        '<span>Inner-core length: ' + CLIENT_BUSHING_SPEC.inner_core_length_min_mm + ' to ' + CLIENT_BUSHING_SPEC.inner_core_length_max_mm + ' mm</span>' +
        '<span>Outer-core length: ' + CLIENT_BUSHING_SPEC.outer_core_length_min_mm + ' to ' + CLIENT_BUSHING_SPEC.outer_core_length_max_mm + ' mm</span>' +
        '<span>Outer diameter: ' + CLIENT_BUSHING_SPEC.outer_diameter_mm + ' mm</span>' +
        '<span>Kx target: ' + formatTargetStiffness(targetSearchInputs.kx) + '</span>' +
        '<span>Ky target: ' + formatTargetStiffness(targetSearchInputs.ky) + '</span>' +
        '<span>Kz target: ' + formatTargetStiffness(targetSearchInputs.kz) + '</span>' +
        '<span>Samples displayed: ' + samples.length + '</span>' +
        '<span>Axes: PC1, PC2, PC3 shape coefficients</span>' +
        '</div></div></div></div>';
      document.body.appendChild(overlay);
      const close = () => overlay.remove();
      overlay.addEventListener("click", (event) => {
        if (event.target === overlay) close();
      });
      const closeButton = overlay.querySelector(".sim-modal-close");
      if (closeButton) closeButton.addEventListener("click", close);
      drawStiffnessPcaPlot(document.getElementById("stiffnessPcaCanvas"), samples);
    }

    function drawStiffnessPcaPlot(canvas, samples) {
      if (!canvas || !samples.length) return;
      const host = canvas.parentElement;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(520, host.clientWidth || 620);
      const height = Math.max(480, host.clientHeight || 480);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d");
      context.scale(ratio, ratio);
      context.clearRect(0, 0, width, height);
      const codes = samples.map((item) => item.shape_codes.slice(0, 3).map(Number));
      const scales = [0, 1, 2].map((axis) => Math.max(...codes.map((row) => Math.abs(row[axis]) || 0), 1e-6));
      const centerX = width * 0.49;
      const centerY = height * 0.53;
      const radius = Math.min(width * 0.34, height * 0.37);
      const project = (code) => {
        const x = code[0] / scales[0];
        const y = code[1] / scales[1];
        const z = code[2] / scales[2];
        return [
          centerX + radius * (0.72 * x - 0.58 * y),
          centerY + radius * (0.32 * x + 0.30 * y - 0.86 * z),
        ];
      };
      context.strokeStyle = "#aebfd3";
      context.lineWidth = 1;
      const axes = [
        { code: [1, 0, 0], label: "PC1" },
        { code: [0, 1, 0], label: "PC2" },
        { code: [0, 0, 1], label: "PC3" },
      ];
      context.font = "12px Arial, sans-serif";
      context.fillStyle = "#53657d";
      axes.forEach((axis) => {
        const endpoint = project(axis.code);
        context.beginPath();
        context.moveTo(centerX, centerY);
        context.lineTo(endpoint[0], endpoint[1]);
        context.stroke();
        context.fillText(axis.label, endpoint[0] + 5, endpoint[1] - 4);
      });
      const target = [targetSearchInputs.kx, targetSearchInputs.ky, targetSearchInputs.kz];
      const scored = samples.map((item, index) => {
        const stiffness = item.stiffness || {};
        const values = [stiffness.kx_n_per_mm, stiffness.ky_n_per_mm, stiffness.kz_n_per_mm].map(Number);
        const errors = values.map((value, axis) => relativeErrorMagnitude(value, target[axis]));
        return { index, score: Math.max(...errors), near: Math.max(...errors) <= CLIENT_BUSHING_SPEC.match_tolerance };
      });
      if (!scored.some((item) => item.near)) {
        scored.sort((left, right) => left.score - right.score);
        scored.slice(0, Math.min(8, scored.length)).forEach((item) => { item.near = true; });
      }
      const nearIndices = new Set(scored.filter((item) => item.near).map((item) => item.index));
      samples.forEach((item, index) => {
        const point = project(codes[index]);
        context.beginPath();
        context.arc(point[0], point[1], nearIndices.has(index) ? 4 : 2.6, 0, Math.PI * 2);
        context.fillStyle = nearIndices.has(index) ? "#e52d2f" : "rgba(21,128,61,0.72)";
        context.fill();
      });
      context.fillStyle = "#102548";
      context.font = "600 13px Arial, sans-serif";
      context.fillText("Distribution of first three shape codes", 18, 24);
    }

    function cadEngineSelectorHtml() {
      const cadquerySelected = selectedCadEngine === "cadquery" ? " selected" : "";
      const openscadSelected = selectedCadEngine === "openscad" ? " selected" : "";
      return `
        <div class="cad-engine-row">
          <label for="cadEngineSelect">CAD Engine</label>
          <select id="cadEngineSelect">
            <option value="cadquery"${cadquerySelected}>CadQuery</option>
            <option value="openscad"${openscadSelected}>OpenSCAD</option>
          </select>
        </div>`;
    }

    function bindCadEngineSelector() {
      const select = document.getElementById("cadEngineSelect");
      if (!select) {
        syncDownloadItems();
        return;
      }
      select.value = selectedCadEngine;
      select.addEventListener("change", () => {
        selectedCadEngine = select.value === "openscad" ? "openscad" : "cadquery";
        lastExport.cadEngine = selectedCadEngine;
        syncDownloadItems();
      });
      syncDownloadItems();
    }

    function syncDownloadItems() {
      if (!downloadMenu) return;
      for (const item of downloadMenu.querySelectorAll(".download-item")) {
        const scope = item.dataset.engineScope || "both";
        item.hidden = scope !== "both" && scope !== selectedCadEngine;
      }
      for (const group of downloadMenu.querySelectorAll(".download-group")) {
        const hasVisibleItem = Array.from(group.querySelectorAll(".download-item")).some((item) => !item.hidden);
        group.hidden = !hasVisibleItem;
      }
    }

    function buildParamControls(intent) {
      if (!paramControls) {
        return;
      }
      const uploaded = (preferParametric || meshEditMode) ? null : pickUploadedMesh();
      if (uploaded) {
        const dims = measureBushingFromMesh(uploaded);
        if (dims) {
          paramControls.innerHTML =
            '<p class="muted">This is an uploaded mesh, shown exactly as provided. ' +
            'Measured bushing dimensions: OD ' + dims.outer_diameter_mm + ' mm, ID ' +
            dims.inner_diameter_mm + ' mm, height ' + dims.height_mm + ' mm.</p>' +
            '<div class="param-actions"><button type="button" class="param-reset" id="convertBushingBtn">Convert to editable bushing</button></div>';
          const convertBtn = document.getElementById("convertBushingBtn");
          if (convertBtn) convertBtn.addEventListener("click", convertUploadedToBushing);
          if (paramHint) paramHint.textContent = "Convert to edit OD / ID / height live.";
        } else {
          paramControls.innerHTML = '<p class="muted">This is an uploaded mesh, shown exactly as provided. Live parametric editing is available for AI-generated parts.</p>';
          if (paramHint) paramHint.textContent = "Uploaded geometry is read-only.";
        }
        renderMeshPanel();
        renderSimPanel();
        return;
      }
      const type = String((intent && intent.part_type) || "unknown").toLowerCase();
      const spec = PART_FIELD_SETS[type];
      if (!intent || !spec) {
        paramControls.innerHTML = cadEngineSelectorHtml() + '<p class="muted">Adjustable dimensions will appear here once a model is generated. Use the Download menu to export the edited part.</p>';
        bindCadEngineSelector();
        if (paramHint) paramHint.textContent = "Open after a model is generated.";
        renderMeshPanel();
        renderSimPanel();
        return;
      }

      if (isRubberBushingWorkflow(intent)) {
        buildRubberBushingWorkflow(intent);
        return;
      }

      currentEditIntent = intent;
      baseGeometry = Object.assign({}, intent.geometry || {});
      const geom = intent.geometry || {};

      const fields = spec.core.slice();
      for (const key of spec.optional) {
        const value = Number(geom[key]);
        if (Number.isFinite(value) && value > 0) {
          fields.push(key);
        }
      }

      const rows = fields.map((key) => {
        const def = PARAM_FIELDS[key];
        if (!def) return "";
        let value = Number(geom[key]);
        if (!Number.isFinite(value) || value <= 0) value = def.fallback;
        const max = Math.max(def.max, value);
        const display = roundParam(value, def.step);
        const unit = key === "coil_count" ? "" : '<span class="param-unit">mm</span>';
        return `
          <div class="param-row" data-key="${key}">
            <label class="param-label" for="slider_${key}">${def.label}</label>
            <span class="param-value">
              <input type="number" class="param-number" id="num_${key}" value="${display}" min="${def.min}" max="${max}" step="${def.step}">
              ${unit}
            </span>
            <input type="range" class="param-slider" id="slider_${key}" value="${value}" min="${def.min}" max="${max}" step="${def.step}">
          </div>`;
      }).join("");

      paramControls.innerHTML = cadEngineSelectorHtml() + rows + '<div class="param-actions"><button type="button" class="param-reset" id="paramReset">Reset</button></div>';
      bindParamControls();
      if (paramHint) paramHint.textContent = "Drag a slider to resize the model live.";
      renderMeshPanel();
      renderSimPanel();
    }

    function bindParamControls() {
      bindCadEngineSelector();
      const rows = paramControls.querySelectorAll(".param-row");
      for (const row of rows) {
        const key = row.dataset.key;
        const slider = row.querySelector(".param-slider");
        const number = row.querySelector(".param-number");
        if (!slider || !number) continue;
        const onChange = (raw) => {
          const val = Number(raw);
          if (!Number.isFinite(val)) return;
          slider.value = val;
          number.value = roundParam(val, Number(slider.step) || 0.1);
          applyParamEdit(key, val);
        };
        slider.addEventListener("input", () => onChange(slider.value));
        number.addEventListener("input", () => onChange(number.value));
      }
      const reset = document.getElementById("paramReset");
      if (reset) {
        reset.addEventListener("click", () => {
          if (!currentEditIntent || !baseGeometry) return;
          currentEditIntent.geometry = Object.assign({}, baseGeometry);
          lastExport.intent = currentEditIntent;
          jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
          if (meshEditMode) {
            overrideMeshFaces = warpEditableMeshFaces(currentEditIntent.geometry);
          }
          lastMeshResult = null;
          lastStaticStiffness = null;
          buildParamControls(currentEditIntent);
          scheduleParamRender();
        });
      }
    }

    function applyParamEdit(key, value) {
      if (!currentEditIntent) return;
      const geom = currentEditIntent.geometry || (currentEditIntent.geometry = {});
      geom[key] = key === "coil_count" ? Math.max(1, Math.round(value)) : value;
      lastMeshResult = null;
      lastStaticStiffness = null;

      // Keep the inner diameter strictly inside the outer diameter.
      const od = Number(geom.outer_diameter_mm);
      const id = Number(geom.inner_diameter_mm);
      if (Number.isFinite(od) && Number.isFinite(id) && id >= od) {
        geom.inner_diameter_mm = Math.max(1, od - 1);
        const idNum = document.getElementById("num_inner_diameter_mm");
        const idSlider = document.getElementById("slider_inner_diameter_mm");
        if (idNum) idNum.value = roundParam(geom.inner_diameter_mm, 0.5);
        if (idSlider) idSlider.value = geom.inner_diameter_mm;
      }

      lastExport.intent = currentEditIntent;
      jsonOutput.textContent = JSON.stringify(currentEditIntent, null, 2);
      if (meshEditMode) {
        overrideMeshFaces = warpEditableMeshFaces(geom);
      }
      updateSimEstimate();
      scheduleParamRender();
    }

    function scheduleParamRender() {
      if (paramRenderQueued) return;
      paramRenderQueued = true;
      requestAnimationFrame(() => {
        paramRenderQueued = false;
        if (currentEditIntent) {
          render3DPreview(currentEditIntent).catch(() => {});
        }
      });
    }

    function createPreviewMesh(cadIntent) {
      const type = String(cadIntent.part_type || "unknown").toLowerCase();
      const geometry = cadIntent.geometry || {};
      if (type === "bushing" || type === "rubber_mount") {
        return createBushingMesh(geometry);
      }
      if (type === "plate") {
        return createPlateMesh(geometry);
      }
      if (type === "bracket") {
        return createBracketMesh(geometry);
      }
      if (type === "spring") {
        return createSpringMesh(geometry);
      }
      return createUnknownMesh(geometry);
    }

    function createBushingMesh(geometry) {
      const outerDiameter = readPositive(geometry.outer_diameter_mm, 60);
      const innerDiameter = readPositive(geometry.inner_diameter_mm, 20);
      const height = readPositive(geometry.height_mm, 40);
      const outerRadius = Math.max(outerDiameter / 2, 5);
      const innerRadius = Math.min(Math.max(innerDiameter / 2, 2), outerRadius - 1);

      const segments = 36;
      const topY = height / 2;
      const bottomY = -height / 2;

      // Optional Vibracoustic-style features.
      const sleeveOuter = Math.max(0, geometry.metal_sleeve_thickness_mm || 0);
      const sleeveInner = Math.max(0, geometry.inner_sleeve_thickness_mm || 0);
      const flangeDiameter = readPositive(geometry.flange_diameter_mm, 0);
      const flangeThickness = readPositive(geometry.flange_thickness_mm, 0);
      const boreOffset = Number(geometry.bore_offset_mm) || 0;
      const chamferSize = Math.max(0, Number(geometry.chamfer_mm) || 0);
      const filletSize = Math.max(0, Number(geometry.fillet_mm) || 0);
      const previewSlotCount = Math.max(0, Math.round(Number(geometry.slot_count) || 0));
      const previewSlotWidth = Math.max(0, Number(geometry.slot_width_deg) || 0);
      const previewSlotStart = Number(geometry.slot_start_angle_deg) || 0;

      // Clamp shells so they never overlap.
      const safeSleeveOuter = Math.min(sleeveOuter, (outerRadius - innerRadius) * 0.45);
      const safeSleeveInner = Math.min(sleeveInner, (outerRadius - innerRadius) * 0.45);

      // Concentric shell radii (outer -> inner).
      const rOuter = outerRadius;
      const rRubberOuter = outerRadius - safeSleeveOuter;
      const rRubberInner = innerRadius + safeSleeveInner;
      const rInner = innerRadius;
      const offsetX = Math.max(-Math.abs(rRubberInner - rInner) - (rRubberInner - 1), Math.min(boreOffset, rRubberInner - 1));

      // Clamp edge break so it never collapses the rubber wall or eats the height.
      const rubberThickness = Math.max(rRubberOuter - rRubberInner, 0.5);
      const maxEdgeBreak = Math.min(height * 0.45, rubberThickness * 0.45);
      const safeChamfer = Math.min(chamferSize, maxEdgeBreak);
      const safeFillet = Math.min(filletSize, maxEdgeBreak);
      // Fillet wins if both are set; otherwise chamfer.
      const useFillet = safeFillet > 0;
      const useChamfer = !useFillet && safeChamfer > 0;
      const edgeBreak = useFillet ? safeFillet : safeChamfer;
      const filletSteps = 4;

      const colorMetal = "#9aa6b4";
      const colorMetalTop = "#b6c1ce";
      const colorMetalBottom = "#7f8a98";
      const colorRubber = "#3b3a3a";
      const colorRubberTop = "#525151";
      const colorRubberBottom = "#2a2929";

      const faces = [];

      function isVisualSlotSector(index) {
        if (!(previewSlotCount > 0) || !(previewSlotWidth > 0)) return false;
        const angleDeg = ((index + 0.5) / segments) * 360;
        const pitch = 360 / previewSlotCount;
        for (let slotIndex = 0; slotIndex < previewSlotCount; slotIndex += 1) {
          const center = previewSlotStart + slotIndex * pitch;
          const delta = Math.abs(((angleDeg - center + 180) % 360) - 180);
          if (delta <= previewSlotWidth / 2) return true;
        }
        return false;
      }

      // Add a coaxial annulus shell: side walls + top/bottom rings.
      function addAnnulus(rOut, rIn, centerOutX, centerInX, color, topColor, bottomColor) {
        for (let index = 0; index < segments; index += 1) {
          if (isVisualSlotSector(index)) continue;
          const thetaA = (index / segments) * Math.PI * 2;
          const thetaB = ((index + 1) / segments) * Math.PI * 2;
          const outerTopA = pointOnRing(rOut, thetaA, topY, centerOutX);
          const outerTopB = pointOnRing(rOut, thetaB, topY, centerOutX);
          const outerBottomA = pointOnRing(rOut, thetaA, bottomY, centerOutX);
          const outerBottomB = pointOnRing(rOut, thetaB, bottomY, centerOutX);
          const innerTopA = pointOnRing(rIn, thetaA, topY, centerInX);
          const innerTopB = pointOnRing(rIn, thetaB, topY, centerInX);
          const innerBottomA = pointOnRing(rIn, thetaA, bottomY, centerInX);
          const innerBottomB = pointOnRing(rIn, thetaB, bottomY, centerInX);
          // Outer wall
          faces.push(makeFace([outerTopA, outerTopB, outerBottomB, outerBottomA], color));
          // Inner wall (reversed winding so the bore looks correct)
          faces.push(makeFace([innerTopB, innerTopA, innerBottomA, innerBottomB], shadeColor(color, 0.85)));
          // Top annular ring
          faces.push(makeFace([outerTopA, innerTopA, innerTopB, outerTopB], topColor));
          // Bottom annular ring
          faces.push(makeFace([outerBottomB, innerBottomB, innerBottomA, outerBottomA], bottomColor));
        }
      }

      // Build the side wall as a stack of horizontal rings so chamfers / fillets
      // can be applied to the top and bottom edges of both the outer and inner walls.
      function buildProfile() {
        if (edgeBreak <= 0) {
          return {
            outer: [
              { y: topY, dr: 0 },
              { y: bottomY, dr: 0 },
            ],
            inner: [
              { y: topY, dr: 0 },
              { y: bottomY, dr: 0 },
            ],
          };
        }
        if (useChamfer) {
          const c = edgeBreak;
          return {
            outer: [
              { y: topY, dr: -c },
              { y: topY - c, dr: 0 },
              { y: bottomY + c, dr: 0 },
              { y: bottomY, dr: -c },
            ],
            inner: [
              { y: topY, dr: c },
              { y: topY - c, dr: 0 },
              { y: bottomY + c, dr: 0 },
              { y: bottomY, dr: c },
            ],
          };
        }
        // Fillet: quarter-circle sweep with `filletSteps` segments at each corner.
        const r = edgeBreak;
        const outer = [];
        const inner = [];
        for (let step = 0; step <= filletSteps; step += 1) {
          const t = step / filletSteps;
          const angle = (Math.PI / 2) * t;
          const dx = r - r * Math.cos(angle);
          const dy = r * Math.sin(angle);
          outer.push({ y: topY - dy, dr: -dx });
          inner.push({ y: topY - dy, dr: dx });
        }
        for (let step = 0; step <= filletSteps; step += 1) {
          const t = step / filletSteps;
          const angle = (Math.PI / 2) * t;
          const dy = r * Math.sin(angle);
          const dx = r - r * Math.cos(angle);
          outer.push({ y: bottomY + dy - 1e-6, dr: -dx });
          inner.push({ y: bottomY + dy - 1e-6, dr: dx });
        }
        // Add the actual bottom edge so the wall closes.
        outer.push({ y: bottomY, dr: -r });
        inner.push({ y: bottomY, dr: r });
        return { outer, inner };
      }

      // Add a rubber annulus whose outer + inner edges can carry a chamfer or fillet.
      function addProfiledAnnulus(rOutBase, rInBase, centerOutX, centerInX, color, topColor, bottomColor) {
        const profile = buildProfile();
        const outerRings = profile.outer.map((row) => ({
          y: row.y,
          radius: Math.max(rOutBase + row.dr, rInBase + 0.4),
        }));
        const innerRings = profile.inner.map((row) => ({
          y: row.y,
          radius: Math.min(rInBase + row.dr, rOutBase - 0.4),
        }));

        function sweepWall(rings, centerX, isOuter) {
          for (let level = 0; level < rings.length - 1; level += 1) {
            const ringA = rings[level];
            const ringB = rings[level + 1];
            for (let index = 0; index < segments; index += 1) {
              if (isVisualSlotSector(index)) continue;
              const thetaA = (index / segments) * Math.PI * 2;
              const thetaB = ((index + 1) / segments) * Math.PI * 2;
              const a1 = pointOnRing(ringA.radius, thetaA, ringA.y, centerX);
              const a2 = pointOnRing(ringA.radius, thetaB, ringA.y, centerX);
              const b1 = pointOnRing(ringB.radius, thetaA, ringB.y, centerX);
              const b2 = pointOnRing(ringB.radius, thetaB, ringB.y, centerX);
              const shade = level === 0 || level === rings.length - 2 ? topColor : color;
              const face = isOuter
                ? [a1, a2, b2, b1]
                : [a2, a1, b1, b2];
              faces.push(makeFace(face, shade));
            }
          }
        }

        sweepWall(outerRings, centerOutX, true);
        sweepWall(innerRings, centerInX, false);

        // Cap top + bottom annulus rings between the (possibly shrunk) outer/inner edges.
        const topOuter = outerRings[0];
        const topInner = innerRings[0];
        const bottomOuter = outerRings[outerRings.length - 1];
        const bottomInner = innerRings[innerRings.length - 1];
        for (let index = 0; index < segments; index += 1) {
          if (isVisualSlotSector(index)) continue;
          const thetaA = (index / segments) * Math.PI * 2;
          const thetaB = ((index + 1) / segments) * Math.PI * 2;
          const tOA = pointOnRing(topOuter.radius, thetaA, topOuter.y, centerOutX);
          const tOB = pointOnRing(topOuter.radius, thetaB, topOuter.y, centerOutX);
          const tIA = pointOnRing(topInner.radius, thetaA, topInner.y, centerInX);
          const tIB = pointOnRing(topInner.radius, thetaB, topInner.y, centerInX);
          const bOA = pointOnRing(bottomOuter.radius, thetaA, bottomOuter.y, centerOutX);
          const bOB = pointOnRing(bottomOuter.radius, thetaB, bottomOuter.y, centerOutX);
          const bIA = pointOnRing(bottomInner.radius, thetaA, bottomInner.y, centerInX);
          const bIB = pointOnRing(bottomInner.radius, thetaB, bottomInner.y, centerInX);
          faces.push(makeFace([tOA, tIA, tIB, tOB], topColor));
          faces.push(makeFace([bOB, bIB, bIA, bOA], bottomColor));
        }
      }

      // Outer metal sleeve (bonded).
      if (safeSleeveOuter > 0) {
        addAnnulus(rOuter, rRubberOuter, 0, 0, colorMetal, colorMetalTop, colorMetalBottom);
      }

      // Rubber middle layer (the main visible body for Vibracoustic bushings).
      addProfiledAnnulus(rRubberOuter, rRubberInner, 0, offsetX, colorRubber, colorRubberTop, colorRubberBottom);

      // Inner metal sleeve (bonded around the bore).
      if (safeSleeveInner > 0) {
        addAnnulus(rRubberInner, rInner, offsetX, offsetX, colorMetal, colorMetalTop, colorMetalBottom);
      } else if (safeSleeveOuter > 0) {
        // Bore wall when only outer sleeve is set so the inside still looks like a hole.
        addAnnulus(rRubberInner, Math.max(rInner, 0.5), offsetX, offsetX, colorRubber, colorRubberTop, colorRubberBottom);
      }

      // Optional flange.
      if (flangeDiameter > outerDiameter && flangeThickness > 0 && geometry.flange !== "none") {
        const flangeOuterRadius = flangeDiameter / 2;
        const flangeMode = String(geometry.flange || "top").toLowerCase();
        if (flangeMode === "top" || flangeMode === "both") addFlange(topY, 1);
        if (flangeMode === "bottom" || flangeMode === "both") addFlange(bottomY, -1);

        function addFlange(baseY, direction) {
          const outerY = baseY + direction * flangeThickness;
          for (let index = 0; index < segments; index += 1) {
            const thetaA = (index / segments) * Math.PI * 2;
            const thetaB = ((index + 1) / segments) * Math.PI * 2;
            const outerFaceA = pointOnRing(flangeOuterRadius, thetaA, outerY);
            const outerFaceB = pointOnRing(flangeOuterRadius, thetaB, outerY);
            const outerBaseA = pointOnRing(flangeOuterRadius, thetaA, baseY);
            const outerBaseB = pointOnRing(flangeOuterRadius, thetaB, baseY);
            const innerFaceA = pointOnRing(rInner, thetaA, outerY, offsetX);
            const innerFaceB = pointOnRing(rInner, thetaB, outerY, offsetX);
            const innerBaseA = pointOnRing(rOuter, thetaA, baseY);
            const innerBaseB = pointOnRing(rOuter, thetaB, baseY);
            faces.push(makeFace([outerFaceA, outerFaceB, outerBaseB, outerBaseA], colorMetal));
            faces.push(makeFace([outerFaceA, innerFaceA, innerFaceB, outerFaceB], direction > 0 ? colorMetalTop : colorMetalBottom));
            faces.push(makeFace([outerBaseB, innerBaseB, innerBaseA, outerBaseA], direction > 0 ? colorMetalBottom : colorMetalTop));
          }
        }
      }

      // Arms / tabs / lugs attached to the outside of the bushing.
      // Prefer the new list (geometry.arms); fall back to the legacy single-arm fields.
      const armColor = safeSleeveOuter > 0 ? colorMetal : colorRubber;
      const armList = Array.isArray(geometry.arms) ? geometry.arms.slice() : [];
      if (armList.length === 0) {
        const legacyLength = readPositive(geometry.arm_length_mm, 0);
        const legacyWidth = readPositive(geometry.arm_width_mm, 0);
        const legacyThickness = readPositive(geometry.arm_thickness_mm, 0);
        if (legacyLength > 0 && legacyWidth > 0 && legacyThickness > 0) {
          armList.push({
            length_mm: legacyLength,
            width_mm: legacyWidth,
            thickness_mm: legacyThickness,
            angle_deg: 0,
            position: geometry.arm_position || "centered",
          });
        }
      }

      for (const arm of armList) {
        const armLength = readPositive(arm.length_mm, 0);
        const armWidth = readPositive(arm.width_mm, 0);
        const armThickness = readPositive(arm.thickness_mm, 0);
        if (armLength <= 0 || armWidth <= 0 || armThickness <= 0) {
          continue;
        }
        const armPosition = arm.position || "centered";
        const halfThickness = armThickness / 2;
        let centerY;
        if (armPosition === "top") {
          centerY = topY - halfThickness;
        } else if (armPosition === "bottom") {
          centerY = bottomY + halfThickness;
        } else {
          centerY = 0;
        }
        const armCenterX = rOuter + armLength / 2;
        const boxFaces = createBoxFaces(
          armLength,
          armThickness,
          armWidth,
          { x: armCenterX, y: centerY, z: 0 },
          armColor
        );
        const angleRad = ((Number(arm.angle_deg) || 0) * Math.PI) / 180;
        if (Math.abs(angleRad) < 1e-6) {
          faces.push(...boxFaces);
        } else {
          const cos = Math.cos(angleRad);
          const sin = Math.sin(angleRad);
          const rotated = boxFaces.map((face) => ({
            color: face.color,
            points: face.points.map((p) => ({
              x: p.x * cos + p.z * sin,
              y: p.y,
              z: -p.x * sin + p.z * cos,
            })),
          }));
          faces.push(...rotated);
        }
      }

      // Bolt holes through the top face (visualized as dark cylindrical pockets).
      const holeList = Array.isArray(geometry.holes) ? geometry.holes : [];
      const topPlateY = flangeDiameter > outerDiameter && flangeThickness > 0
        ? topY + flangeThickness
        : topY;
      const holeColor = "#1b1b1b";
      const holeSides = 16;
      for (const hole of holeList) {
        const holeDiameter = readPositive(hole.diameter_mm, 0);
        const pcd = readPositive(hole.pitch_circle_diameter_mm, 0);
        const count = Math.max(1, Math.round(Number(hole.count) || 1));
        const startAngle = ((Number(hole.start_angle_deg) || 0) * Math.PI) / 180;
        if (holeDiameter <= 0 || pcd <= 0) {
          continue;
        }
        const holeRadius = holeDiameter / 2;
        const pcdRadius = pcd / 2;
        const holeDepth = Math.max(flangeThickness, 1.5);
        const holeTopY = topPlateY + 0.05;
        const holeBottomY = topPlateY - holeDepth;
        for (let pick = 0; pick < count; pick += 1) {
          const angle = startAngle + (pick / count) * Math.PI * 2;
          const cx = Math.cos(angle) * pcdRadius;
          const cz = Math.sin(angle) * pcdRadius;
          for (let side = 0; side < holeSides; side += 1) {
            const a = (side / holeSides) * Math.PI * 2;
            const b = ((side + 1) / holeSides) * Math.PI * 2;
            const ax = cx + Math.cos(a) * holeRadius;
            const az = cz + Math.sin(a) * holeRadius;
            const bx = cx + Math.cos(b) * holeRadius;
            const bz = cz + Math.sin(b) * holeRadius;
            // Pocket side wall
            faces.push(
              makeFace(
                [
                  { x: ax, y: holeTopY, z: az },
                  { x: bx, y: holeTopY, z: bz },
                  { x: bx, y: holeBottomY, z: bz },
                  { x: ax, y: holeBottomY, z: az },
                ],
                holeColor
              )
            );
            // Pocket floor wedge
            faces.push(
              makeFace(
                [
                  { x: cx, y: holeBottomY, z: cz },
                  { x: ax, y: holeBottomY, z: az },
                  { x: bx, y: holeBottomY, z: bz },
                ],
                "#0e0e0e"
              )
            );
          }
        }
      }

      return { faces };
    }

    function createPlateMesh(geometry) {
      const length = readPositive(geometry.length_mm, 120);
      const width = readPositive(geometry.width_mm, 60);
      const thickness = readPositive(geometry.thickness_mm, 8);
      return {
        faces: createBoxFaces(length, thickness, width, { x: 0, y: 0, z: 0 }, "#95afcc"),
      };
    }

    function createBracketMesh(geometry) {
      const length = readPositive(geometry.length_mm, 120);
      const width = readPositive(geometry.width_mm, 60);
      const thickness = readPositive(geometry.thickness_mm, 8);
      const legHeight = Math.max(length * 0.7, thickness * 6);
      return {
        faces: [
          ...createBoxFaces(length, thickness, width, { x: 0, y: thickness / 2, z: 0 }, "#90abc9"),
          ...createBoxFaces(
            thickness,
            legHeight,
            width,
            { x: length / 2 - thickness / 2, y: legHeight / 2, z: 0 },
            "#7f9cc0"
          ),
        ],
      };
    }

    function createUnknownMesh(geometry) {
      const length = readPositive(geometry.length_mm, 90);
      const width = readPositive(geometry.width_mm, 70);
      const thickness = readPositive(geometry.thickness_mm, 50);
      return {
        faces: createBoxFaces(length, thickness, width, { x: 0, y: 0, z: 0 }, "#a0b6d0"),
      };
    }

    function createSpringMesh(geometry) {
      const coilDiameter = readPositive(geometry.outer_diameter_mm, 40);
      const freeLength = readPositive(geometry.height_mm, 50);
      const wireDiameter = readPositive(geometry.thickness_mm, Math.max(coilDiameter * 0.08, 2));
      const coils = Math.max(2, Math.round(readPositive(geometry.coil_count, 6)));

      const coilRadius = Math.max(coilDiameter / 2, wireDiameter);
      const wireRadius = Math.min(Math.max(wireDiameter / 2, 0.8), coilRadius * 0.45);
      const segmentsPerCoil = 28;
      const totalSteps = Math.max(coils * segmentsPerCoil, 32);
      const tubeSides = 8;
      const pitch = freeLength / coils;
      const startY = -freeLength / 2;
      const color = "#7f9bbd";

      const centerline = [];
      for (let step = 0; step <= totalSteps; step += 1) {
        const t = step / segmentsPerCoil;
        const theta = t * Math.PI * 2;
        centerline.push({
          x: coilRadius * Math.cos(theta),
          y: startY + pitch * t,
          z: coilRadius * Math.sin(theta),
        });
      }

      function ringAt(index) {
        const point = centerline[index];
        const next = centerline[Math.min(index + 1, centerline.length - 1)];
        const prev = centerline[Math.max(index - 1, 0)];
        const tangent = normalizeVector({
          x: next.x - prev.x,
          y: next.y - prev.y,
          z: next.z - prev.z,
        });
        let reference = { x: 0, y: 1, z: 0 };
        if (Math.abs(tangent.y) > 0.92) {
          reference = { x: 1, y: 0, z: 0 };
        }
        const normal = normalizeVector(crossProduct(tangent, reference));
        const binormal = normalizeVector(crossProduct(tangent, normal));
        const ring = [];
        for (let side = 0; side < tubeSides; side += 1) {
          const angle = (side / tubeSides) * Math.PI * 2;
          const cos = Math.cos(angle) * wireRadius;
          const sin = Math.sin(angle) * wireRadius;
          ring.push({
            x: point.x + normal.x * cos + binormal.x * sin,
            y: point.y + normal.y * cos + binormal.y * sin,
            z: point.z + normal.z * cos + binormal.z * sin,
          });
        }
        return ring;
      }

      const faces = [];
      let previousRing = ringAt(0);
      for (let index = 1; index < centerline.length; index += 1) {
        const currentRing = ringAt(index);
        for (let side = 0; side < tubeSides; side += 1) {
          const nextSide = (side + 1) % tubeSides;
          faces.push(
            makeFace(
              [previousRing[side], currentRing[side], currentRing[nextSide], previousRing[nextSide]],
              color
            )
          );
        }
        previousRing = currentRing;
      }

      return { faces };
    }

    function crossProduct(a, b) {
      return {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
      };
    }

    function createBoxFaces(length, height, width, center, color) {
      const halfLength = length / 2;
      const halfHeight = height / 2;
      const halfWidth = width / 2;
      const vertices = [
        { x: center.x - halfLength, y: center.y - halfHeight, z: center.z - halfWidth },
        { x: center.x + halfLength, y: center.y - halfHeight, z: center.z - halfWidth },
        { x: center.x + halfLength, y: center.y + halfHeight, z: center.z - halfWidth },
        { x: center.x - halfLength, y: center.y + halfHeight, z: center.z - halfWidth },
        { x: center.x - halfLength, y: center.y - halfHeight, z: center.z + halfWidth },
        { x: center.x + halfLength, y: center.y - halfHeight, z: center.z + halfWidth },
        { x: center.x + halfLength, y: center.y + halfHeight, z: center.z + halfWidth },
        { x: center.x - halfLength, y: center.y + halfHeight, z: center.z + halfWidth },
      ];
      return [
        makeFace([vertices[0], vertices[1], vertices[2], vertices[3]], color),
        makeFace([vertices[4], vertices[5], vertices[6], vertices[7]], lightenHex(color, 0.05)),
        makeFace([vertices[0], vertices[4], vertices[7], vertices[3]], darkenHex(color, 0.08)),
        makeFace([vertices[1], vertices[5], vertices[6], vertices[2]], darkenHex(color, 0.16)),
        makeFace([vertices[3], vertices[2], vertices[6], vertices[7]], lightenHex(color, 0.12)),
        makeFace([vertices[0], vertices[1], vertices[5], vertices[4]], darkenHex(color, 0.2)),
      ];
    }

    function makeFace(points, color) {
      return { points, color };
    }

    function pointOnRing(radius, angle, y, centerX) {
      const cx = centerX || 0;
      return {
        x: cx + Math.cos(angle) * radius,
        y,
        z: Math.sin(angle) * radius,
      };
    }

    function computeMeshBounds(faces) {
      const points = faces.flatMap((face) => face.points);
      let minX = Infinity;
      let minY = Infinity;
      let minZ = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      let maxZ = -Infinity;

      for (const point of points) {
        minX = Math.min(minX, point.x);
        minY = Math.min(minY, point.y);
        minZ = Math.min(minZ, point.z);
        maxX = Math.max(maxX, point.x);
        maxY = Math.max(maxY, point.y);
        maxZ = Math.max(maxZ, point.z);
      }

      return {
        center: {
          x: (minX + maxX) / 2,
          y: (minY + maxY) / 2,
          z: (minZ + maxZ) / 2,
        },
        size: {
          x: maxX - minX,
          y: maxY - minY,
          z: maxZ - minZ,
        },
      };
    }

    function rotatePoint(point, rotationX, rotationY) {
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);
      const x1 = point.x * cosY - point.z * sinY;
      const z1 = point.x * sinY + point.z * cosY;

      const cosX = Math.cos(rotationX);
      const sinX = Math.sin(rotationX);
      const y2 = point.y * cosX - z1 * sinX;
      const z2 = point.y * sinX + z1 * cosX;

      return { x: x1, y: y2, z: z2 };
    }

    function computeFaceNormal(points) {
      const a = points[0];
      const b = points[1];
      const c = points[2];
      const ab = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
      const ac = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
      return normalizeVector({
        x: ab.y * ac.z - ab.z * ac.y,
        y: ab.z * ac.x - ab.x * ac.z,
        z: ab.x * ac.y - ab.y * ac.x,
      });
    }

    function normalizeVector(vector) {
      const length = Math.hypot(vector.x, vector.y, vector.z) || 1;
      return {
        x: vector.x / length,
        y: vector.y / length,
        z: vector.z / length,
      };
    }

    function dotProduct(left, right) {
      return left.x * right.x + left.y * right.y + left.z * right.z;
    }

    function shadeColor(color, intensity) {
      const rgb = hexToRgb(color);
      const base = intensity >= 0.5
        ? mixColor(rgb, { r: 255, g: 255, b: 255 }, (intensity - 0.5) * 0.5)
        : mixColor(rgb, { r: 30, g: 45, b: 76 }, (0.5 - intensity) * 0.9);
      return `rgb(${base.r}, ${base.g}, ${base.b})`;
    }

    function lightenHex(color, amount) {
      return rgbToHex(mixColor(hexToRgb(color), { r: 255, g: 255, b: 255 }, amount));
    }

    function darkenHex(color, amount) {
      return rgbToHex(mixColor(hexToRgb(color), { r: 28, g: 43, b: 70 }, amount));
    }

    function mixColor(source, target, amount) {
      return {
        r: Math.round(source.r + (target.r - source.r) * clamp(amount, 0, 1)),
        g: Math.round(source.g + (target.g - source.g) * clamp(amount, 0, 1)),
        b: Math.round(source.b + (target.b - source.b) * clamp(amount, 0, 1)),
      };
    }

    function hexToRgb(color) {
      const value = color.replace("#", "");
      const normalized = value.length === 3
        ? value.split("").map((char) => char + char).join("")
        : value;
      return {
        r: parseInt(normalized.slice(0, 2), 16),
        g: parseInt(normalized.slice(2, 4), 16),
        b: parseInt(normalized.slice(4, 6), 16),
      };
    }

    function rgbToHex(color) {
      return `#${[color.r, color.g, color.b]
        .map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0"))
        .join("")}`;
    }

    function niceGridSpacing(extent) {
      const roughSpacing = Math.max(extent / 4.5, 10);
      const magnitude = Math.pow(10, Math.floor(Math.log10(roughSpacing)));
      const normalized = roughSpacing / magnitude;
      if (normalized <= 1.5) return magnitude;
      if (normalized <= 3.5) return magnitude * 2;
      if (normalized <= 7.5) return magnitude * 5;
      return magnitude * 10;
    }

    function readPositive(value, fallback) {
      const numeric = Number(value);
      return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback;
    }

    // ===== Download / export helpers =====

    function exportBaseName(intent) {
      const type = String((intent && intent.part_type) || "model")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
      return type || "model";
    }

    async function handleDownload(format) {
      const name = lastExport.name || "model";
      if (selectedCadEngine === "openscad" && ["scad", "stl", "png", "json"].includes(format)) {
        return downloadOpenScad(format, name);
      }
      switch (format) {
        case "png": return downloadBlob(await canvasPngBlob(), name + ".png");
        case "json": return downloadBlob(jsonBlob(), name + ".json");
        case "stl": return downloadBlob(stlBlob(requireMesh(), name), name + ".stl");
        case "glb": return downloadBlob(glbBlob(requireMesh()), name + ".glb");
        case "dxf": return downloadBlob(dxfBlob(lastExport.intent || {}), name + ".dxf");
        case "pdf": return downloadBlob(await pdfBlob(), name + ".pdf");
        case "step": return downloadStep(name);
        case "scad": return downloadOpenScad(format, name);
        case "zip": return downloadAll(name);
        default: throw new Error("Unknown format: " + format);
      }
    }

    function requireMesh() {
      if (!lastExport.mesh) throw new Error("Generate a 3D model first, then download.");
      return lastExport.mesh;
    }

    function downloadBlob(blob, filename) {
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function femCsvBlob(data) {
      const componentCount = data && data.pca && Array.isArray(data.pca.components) ? data.pca.components.length : 0;
      const pcHeaders = Array.from({ length: componentCount }, (_, index) => "pc" + (index + 1));
      const rows = [["mode_number", "frequency_hz", "eigenvalue", "selected_contour", "material", ...pcHeaders]];
      const scoreByMode = new Map(((data && data.pca && data.pca.mode_scores) || []).map((entry) => [Number(entry.mode_number), entry.scores || []]));
      for (const mode of (data && data.modes) || []) {
        const scores = scoreByMode.get(Number(mode.mode_number)) || [];
        rows.push([
          mode.mode_number,
          mode.frequency_hz,
          mode.eigenvalue,
          Number(mode.mode_number) === Number(data.mode) ? "yes" : "no",
          data.material || "generic",
          ...pcHeaders.map((_, index) => scores[index]),
        ]);
      }
      const csv = rows.map((row) => row.map(csvCell).join(",")).join("\\n") + "\\n";
      return new Blob([csv], { type: "text/csv;charset=utf-8" });
    }

    function csvCell(value) {
      const text = String(value == null ? "" : value);
      return /[",\\n\\r]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
    }

    function exNum(value, fallback) {
      const parsed = Number(value);
      return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
    }

    function concatBytes(arrays) {
      let total = 0;
      for (const part of arrays) total += part.length;
      const out = new Uint8Array(total);
      let offset = 0;
      for (const part of arrays) {
        out.set(part, offset);
        offset += part.length;
      }
      return out;
    }

    function hexToRgb01(hex) {
      const match = /^#?([0-9a-f]{6})$/i.exec(String(hex).trim());
      if (!match) return [0.6, 0.6, 0.6];
      const value = parseInt(match[1], 16);
      return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
    }

    function triNormal(a, b, c) {
      const ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
      const vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
      const nx = uy * vz - uz * vy;
      const ny = uz * vx - ux * vz;
      const nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz) || 1;
      return [nx / len, ny / len, nz / len];
    }

    function triangulateMesh(mesh) {
      const positions = [];
      const colors = [];
      if (!mesh || !Array.isArray(mesh.faces)) return { positions, colors };
      for (const face of mesh.faces) {
        const pts = face.points;
        if (!pts || pts.length < 3) continue;
        const [r, g, b] = hexToRgb01(face.color || "#999999");
        for (let i = 1; i < pts.length - 1; i += 1) {
          for (const point of [pts[0], pts[i], pts[i + 1]]) {
            positions.push(point.x, point.y, point.z);
            colors.push(r, g, b);
          }
        }
      }
      return { positions, colors };
    }

    function jsonBlob() {
      const data = JSON.stringify(lastExport.intent || {}, null, 2);
      return new Blob([data], { type: "application/json" });
    }

    function canvasPngBlob() {
      const canvas = lastExport.canvas;
      if (!canvas) throw new Error("Generate a 3D model first, then download.");
      const flat = flattenCanvas(canvas);
      return new Promise((resolve, reject) => {
        flat.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not render PNG."))), "image/png");
      });
    }

    function flattenCanvas(canvas) {
      const off = document.createElement("canvas");
      off.width = canvas.width;
      off.height = canvas.height;
      const ctx = off.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, off.width, off.height);
      ctx.drawImage(canvas, 0, 0);
      return off;
    }

    function stlBlob(mesh, name) {
      const lines = ["solid " + name];
      for (const face of mesh.faces) {
        const pts = face.points;
        if (!pts || pts.length < 3) continue;
        for (let i = 1; i < pts.length - 1; i += 1) {
          const a = pts[0], b = pts[i], c = pts[i + 1];
          const nrm = triNormal(a, b, c);
          lines.push("  facet normal " + nrm[0] + " " + nrm[1] + " " + nrm[2]);
          lines.push("    outer loop");
          for (const point of [a, b, c]) lines.push("      vertex " + point.x + " " + point.y + " " + point.z);
          lines.push("    endloop");
          lines.push("  endfacet");
        }
      }
      lines.push("endsolid " + name);
      return new Blob([lines.join("\\n")], { type: "model/stl" });
    }

    function align4(value) { return (value + 3) & ~3; }

    function glbBlob(mesh) {
      const { positions, colors } = triangulateMesh(mesh);
      if (!positions.length) throw new Error("No geometry to export.");
      const posArray = new Float32Array(positions);
      const colArray = new Float32Array(colors);
      const count = posArray.length / 3;
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < posArray.length; i += 3) {
        for (let k = 0; k < 3; k += 1) {
          const value = posArray[i + k];
          if (value < min[k]) min[k] = value;
          if (value > max[k]) max[k] = value;
        }
      }
      const posBytes = posArray.byteLength;
      const colBytes = colArray.byteLength;
      const colOffset = align4(posBytes);
      const binLength = align4(colOffset + colBytes);
      const bin = new Uint8Array(binLength);
      bin.set(new Uint8Array(posArray.buffer), 0);
      bin.set(new Uint8Array(colArray.buffer), colOffset);

      const gltf = {
        asset: { version: "2.0", generator: "Resonance AI" },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0, name: lastExport.name || "model" }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0, COLOR_0: 1 }, material: 0, mode: 4 }] }],
        materials: [{ name: "vertexColor", pbrMetallicRoughness: { metallicFactor: 0.1, roughnessFactor: 0.8 } }],
        buffers: [{ byteLength: binLength }],
        bufferViews: [
          { buffer: 0, byteOffset: 0, byteLength: posBytes, target: 34962 },
          { buffer: 0, byteOffset: colOffset, byteLength: colBytes, target: 34962 }
        ],
        accessors: [
          { bufferView: 0, componentType: 5126, count, type: "VEC3", min, max },
          { bufferView: 1, componentType: 5126, count, type: "VEC3" }
        ]
      };

      let jsonBytes = new TextEncoder().encode(JSON.stringify(gltf));
      const jsonPad = align4(jsonBytes.length) - jsonBytes.length;
      if (jsonPad) {
        const padded = new Uint8Array(jsonBytes.length + jsonPad);
        padded.set(jsonBytes);
        padded.fill(0x20, jsonBytes.length);
        jsonBytes = padded;
      }
      const totalLength = 12 + 8 + jsonBytes.length + 8 + bin.length;
      const buffer = new ArrayBuffer(totalLength);
      const dv = new DataView(buffer);
      let off = 0;
      dv.setUint32(off, 0x46546c67, true); off += 4;
      dv.setUint32(off, 2, true); off += 4;
      dv.setUint32(off, totalLength, true); off += 4;
      dv.setUint32(off, jsonBytes.length, true); off += 4;
      dv.setUint32(off, 0x4e4f534a, true); off += 4;
      new Uint8Array(buffer, off, jsonBytes.length).set(jsonBytes); off += jsonBytes.length;
      dv.setUint32(off, bin.length, true); off += 4;
      dv.setUint32(off, 0x004e4942, true); off += 4;
      new Uint8Array(buffer, off, bin.length).set(bin);
      return new Blob([buffer], { type: "model/gltf-binary" });
    }

    function dxfBlob(intent) {
      const geometry = (intent && intent.geometry) || {};
      const type = String((intent && intent.part_type) || "").toLowerCase();
      const entities = [];
      const circle = (cx, cy, r) => {
        if (!(r > 0)) return;
        entities.push("0", "CIRCLE", "8", "PROFILE", "10", String(cx), "20", String(cy), "30", "0", "40", String(r));
      };
      const rect = (w, h) => {
        const x = w / 2, y = h / 2;
        entities.push("0", "LWPOLYLINE", "8", "PROFILE", "90", "4", "70", "1");
        for (const [px, py] of [[-x, -y], [x, -y], [x, y], [-x, y]]) entities.push("10", String(px), "20", String(py));
      };
      if (type === "bushing" || type === "rubber_mount") {
        const od = exNum(geometry.outer_diameter_mm, 60);
        const id = exNum(geometry.inner_diameter_mm, 20);
        const fd = exNum(geometry.flange_diameter_mm, 0);
        if (fd > od) circle(0, 0, fd / 2);
        circle(0, 0, od / 2);
        circle(0, 0, id / 2);
      } else if (type === "plate" || type === "bracket") {
        rect(exNum(geometry.length_mm, 120), exNum(geometry.width_mm, 80));
        const holes = Array.isArray(geometry.holes) ? geometry.holes : [];
        for (const hole of holes) circle(0, 0, exNum(hole.diameter_mm, 0) / 2);
      } else if (type === "spring") {
        circle(0, 0, exNum(geometry.outer_diameter_mm, exNum(geometry.coil_diameter_mm, 40)) / 2);
      } else {
        rect(exNum(geometry.length_mm, 60), exNum(geometry.width_mm, 40));
      }
      const dxf = ["0", "SECTION", "2", "ENTITIES", ...entities, "0", "ENDSEC", "0", "EOF"].join("\\n");
      return new Blob([dxf], { type: "application/dxf" });
    }

    function pdfSummaryLines(intent) {
      const geometry = (intent && intent.geometry) || {};
      const lines = [];
      lines.push("Part type: " + ((intent && intent.part_type) || "unknown"));
      const material = (intent && intent.material) || {};
      if (material.name) {
        lines.push("Material: " + material.name + (material.shore_a ? " (Shore A " + material.shore_a + ")" : ""));
      }
      lines.push("");
      lines.push("Dimensions (mm):");
      const dimKeys = [
        ["outer_diameter_mm", "Outer diameter"], ["inner_diameter_mm", "Inner diameter"],
        ["height_mm", "Height"], ["length_mm", "Length"], ["width_mm", "Width"],
        ["thickness_mm", "Thickness"], ["flange_diameter_mm", "Flange diameter"],
        ["flange_thickness_mm", "Flange thickness"], ["chamfer_mm", "Chamfer"],
        ["fillet_mm", "Fillet"], ["coil_count", "Coil count"]
      ];
      for (const [key, label] of dimKeys) {
        if (geometry[key] !== undefined && geometry[key] !== null && geometry[key] !== "" && Number(geometry[key]) > 0) {
          lines.push("  " + label + ": " + geometry[key]);
        }
      }
      const missing = (intent && intent.missing_information) || [];
      if (Array.isArray(missing) && missing.length) {
        lines.push("");
        lines.push("Assumptions / open items:");
        for (const item of missing) lines.push("  - " + item);
      }
      lines.push("");
      lines.push("Generated by Resonance AI on " + new Date().toISOString().slice(0, 10));
      return lines;
    }

    function pdfEscape(text) {
      return String(text).replace(/\\\\/g, "\\\\\\\\").replace(/\\(/g, "\\\\(").replace(/\\)/g, "\\\\)");
    }

    function dataUrlToUint8(dataUrl) {
      const base64 = dataUrl.split(",")[1] || "";
      const binary = atob(base64);
      const out = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
      return out;
    }

    async function pdfBlob() {
      const intent = lastExport.intent || {};
      const textLines = pdfSummaryLines(intent);
      let jpeg = null, imgW = 0, imgH = 0;
      if (lastExport.canvas) {
        const flat = flattenCanvas(lastExport.canvas);
        jpeg = dataUrlToUint8(flat.toDataURL("image/jpeg", 0.92));
        imgW = flat.width;
        imgH = flat.height;
      }
      return buildPdf(textLines, jpeg, imgW, imgH);
    }

    function buildPdf(textLines, jpeg, imgW, imgH) {
      const enc = (text) => new TextEncoder().encode(text);
      const pageWidth = 595.28, pageHeight = 841.89;
      const hasImage = !!(jpeg && imgW && imgH);

      let drawImage = "";
      let textTop = pageHeight - 90;
      if (hasImage) {
        const maxW = pageWidth - 100;
        const maxH = 300;
        const ratio = Math.min(maxW / imgW, maxH / imgH);
        const w = imgW * ratio, h = imgH * ratio;
        const x = (pageWidth - w) / 2;
        const y = pageHeight - 90 - h;
        drawImage = "q " + w.toFixed(2) + " 0 0 " + h.toFixed(2) + " " + x.toFixed(2) + " " + y.toFixed(2) + " cm /Im0 Do Q\\n";
        textTop = y - 28;
      }

      let content = "BT /F1 18 Tf 50 " + (pageHeight - 60).toFixed(2) + " Td (Resonance AI - CAD Technical Summary) Tj ET\\n";
      content += drawImage;
      content += "BT /F1 11 Tf 50 " + textTop.toFixed(2) + " Td 16 TL\\n";
      for (const line of textLines) content += "(" + pdfEscape(line) + ") Tj T*\\n";
      content += "ET\\n";
      const contentBytes = enc(content);

      const objBodies = [];
      objBodies.push(enc("<< /Type /Catalog /Pages 2 0 R >>"));
      objBodies.push(enc("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"));
      let resources = "<< /Font << /F1 5 0 R >>";
      if (hasImage) resources += " /XObject << /Im0 6 0 R >>";
      resources += " >>";
      objBodies.push(enc("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + pageWidth + " " + pageHeight + "] /Resources " + resources + " /Contents 4 0 R >>"));
      objBodies.push(concatBytes([enc("<< /Length " + contentBytes.length + " >>\\nstream\\n"), contentBytes, enc("\\nendstream")]));
      objBodies.push(enc("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"));
      if (hasImage) {
        const header = enc("<< /Type /XObject /Subtype /Image /Width " + imgW + " /Height " + imgH + " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length " + jpeg.length + " >>\\nstream\\n");
        objBodies.push(concatBytes([header, jpeg, enc("\\nendstream")]));
      }

      const chunks = [];
      let length = 0;
      const offsets = [];
      const push = (bytes) => { chunks.push(bytes); length += bytes.length; };
      push(enc("%PDF-1.4\\n"));
      push(new Uint8Array([0x25, 0xe2, 0xe3, 0xcf, 0xd3, 0x0a]));
      for (let i = 0; i < objBodies.length; i += 1) {
        offsets[i + 1] = length;
        push(enc((i + 1) + " 0 obj\\n"));
        push(objBodies[i]);
        push(enc("\\nendobj\\n"));
      }
      const xrefStart = length;
      const total = objBodies.length + 1;
      let xref = "xref\\n0 " + total + "\\n0000000000 65535 f \\n";
      for (let i = 1; i < total; i += 1) xref += String(offsets[i]).padStart(10, "0") + " 00000 n \\n";
      push(enc(xref));
      push(enc("trailer\\n<< /Size " + total + " /Root 1 0 R >>\\nstartxref\\n" + xrefStart + "\\n%%EOF"));
      return new Blob([concatBytes(chunks)], { type: "application/pdf" });
    }

    async function downloadStep(name) {
      startActivity("Generating STEP", ["Sending request", "Running CadQuery", "Packaging STEP"]);
      try {
        const response = await fetch("/export/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: lastExport.prompt || "", intent: lastExport.intent || {}, name })
        });
        if (!response.ok) {
          let detail = "STEP export is not available on this server.";
          try { const payload = await response.json(); detail = payload.detail || detail; } catch (err) {}
          throw new Error(detail);
        }
        downloadBlob(await response.blob(), name + ".step");
        completeActivity("STEP ready");
      } catch (error) {
        completeActivity("STEP unavailable");
        throw error;
      }
    }

    async function downloadOpenScad(format, name) {
      const labels = { scad: "SCAD", stl: "STL", png: "PNG", json: "JSON" };
      const filenames = { scad: name + ".scad", stl: name + ".stl", png: name + ".png", json: "parameters.json" };
      const phases = format === "scad" || format === "json"
        ? ["Preparing bushing JSON", "Writing OpenSCAD source", "Packaging file"]
        : ["Preparing bushing JSON", "Running OpenSCAD", "Packaging " + labels[format]];
      startActivity("Generating OpenSCAD " + labels[format], phases);
      try {
        const response = await fetch("/export/openscad", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            cad_engine: "openscad",
            format,
            name,
            intent: lastExport.intent || {},
          })
        });
        if (!response.ok) {
          let detail = "OpenSCAD export is not available on this server.";
          try { const payload = await response.json(); detail = payload.detail || detail; } catch (err) {}
          throw new Error(detail);
        }
        downloadBlob(await response.blob(), filenames[format] || (name + "." + format));
        completeActivity("OpenSCAD " + labels[format] + " ready");
      } catch (error) {
        completeActivity("OpenSCAD unavailable");
        throw error;
      }
    }

    function crc32(bytes) {
      let crc = ~0;
      for (let i = 0; i < bytes.length; i += 1) {
        crc ^= bytes[i];
        for (let j = 0; j < 8; j += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
      }
      return (~crc) >>> 0;
    }

    function zipStore(files) {
      const encoder = new TextEncoder();
      const u16 = (value) => new Uint8Array([value & 255, (value >> 8) & 255]);
      const u32 = (value) => new Uint8Array([value & 255, (value >> 8) & 255, (value >> 16) & 255, (value >>> 24) & 255]);
      const locals = [];
      const central = [];
      let offset = 0;
      for (const file of files) {
        const nameBytes = encoder.encode(file.name);
        const crc = crc32(file.data);
        const size = file.data.length;
        const local = concatBytes([
          u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
          u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0),
          nameBytes, file.data
        ]);
        locals.push(local);
        central.push(concatBytes([
          u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
          u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0), u16(0), u16(0), u16(0), u32(0),
          u32(offset), nameBytes
        ]));
        offset += local.length;
      }
      const centralBytes = concatBytes(central);
      const end = concatBytes([
        u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
        u32(centralBytes.length), u32(offset), u16(0)
      ]);
      return new Blob([concatBytes([...locals, centralBytes, end])], { type: "application/zip" });
    }

    async function downloadAll(name) {
      const files = [];
      const add = async (suffix, blob) => files.push({ name: name + suffix, data: new Uint8Array(await blob.arrayBuffer()) });
      await add(".json", jsonBlob());
      await add(".dxf", dxfBlob(lastExport.intent || {}));
      if (lastExport.mesh) {
        await add(".stl", stlBlob(lastExport.mesh, name));
        await add(".glb", glbBlob(lastExport.mesh));
      }
      if (lastExport.canvas) {
        await add(".png", await canvasPngBlob());
        await add(".pdf", await pdfBlob());
      }
      downloadBlob(zipStore(files), name + ".zip");
    }

    function cleanupViewer() {
      lastExport.mesh = null;
      lastExport.canvas = null;
      if (!activeViewer) {
        return;
      }

      activeViewer.dispose();
      preview.innerHTML = "";
      activeViewer = null;
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    window.addEventListener("beforeunload", cleanupViewer);
  </script>
</body>
</html>
"""
