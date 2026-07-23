"""Volume-mesh uploaded STEP/STL geometry for exact FEM.

Gmsh first creates a conformal tetrahedral volume mesh of the uploaded solid.
For the hexahedral modes, Gmsh's all-hexa subdivision then replaces every
tetrahedron with body-fitted hexahedra. This follows the uploaded topology and
does not rebuild a parametric bushing from measured dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UploadedMeshResult:
    """Summary of an uploaded-geometry volume mesh."""

    source_file: Path
    mesh_file: Path
    source_format: str
    node_count: int
    tetra_count: int
    hexa_count: int
    min_edge_mm: float
    max_edge_mm: float
    repaired_surface: bool = False
    mesh_kind: str = "uploaded_geometry_tetra"
    global_compatible: bool = False
    template_id: str = ""
    circumferential_divisions: int = 0
    radial_divisions: int = 0
    axial_divisions: int = 0

    def summary(self) -> str:
        repair = " with STL surface classification" if self.repaired_surface else ""
        return (
            f"Uploaded {self.source_format}{repair}: {self.mesh_file.name} | "
            f"nodes={self.node_count} tetra={self.tetra_count} hex={self.hexa_count} | element size "
            f"{self.min_edge_mm:.3f}-{self.max_edge_mm:.3f} mm"
        )


def uploaded_geometry_to_volume_mesh(
    source_file: Path,
    output_file: Path,
    *,
    target_size_mm: float | None = None,
    min_size_mm: float | None = None,
    second_order: bool = True,
    optimize: bool = True,
    verbose: bool = False,
    mesh_mode: str = "tetra",
    template: dict | None = None,
) -> UploadedMeshResult:
    """Create a body-fitted volume mesh from an uploaded STEP/STL file.

    ``structured`` and ``global`` produce all-hexa meshes through Gmsh's
    subdivision algorithm. ``tetra`` preserves the tetrahedral volume mesh.
    Global mode uses fixed density settings, but arbitrary uploaded models do
    not share connectivity with one another.
    """

    import gmsh

    source_file = Path(source_file).resolve()
    output_file = Path(output_file).resolve()
    if not source_file.exists():
        raise FileNotFoundError(f"Uploaded geometry file not found: {source_file}")

    suffix = source_file.suffix.lower()
    if suffix not in {".step", ".stp", ".stl"}:
        raise ValueError("Exact uploaded-geometry FEM currently supports STEP/STP and watertight STL files.")
    normalized_mode = str(mesh_mode or "tetra").strip().lower()
    all_hexa = normalized_mode in {"structured", "global", "global_template", "dataset"}
    global_mode = normalized_mode in {"global", "global_template", "dataset"}
    template_config = _global_template_config(template or {})

    output_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_source_file = source_file
    repair_notes: list[str] = []
    if suffix == ".stl":
        mesh_source_file, repair_notes = _prepare_stl_for_volume_mesh(source_file, output_file.parent)

    _initialize_gmsh(gmsh)
    repaired_surface = False
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.StlRemoveBadTriangles", 2)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.model.add(source_file.stem + "_uploaded")
        if suffix in {".step", ".stp"}:
            source_format = "STEP"
            gmsh.model.occ.importShapes(str(mesh_source_file))
            gmsh.model.occ.synchronize()
        else:
            source_format = "STL"
            _import_stl_as_volume(gmsh, mesh_source_file)
            repaired_surface = True

        diagonal = _bounding_box_diagonal(gmsh)
        size = _target_mesh_size(
            diagonal,
            target_size_mm=target_size_mm,
            global_mode=global_mode,
            template=template_config,
        )
        lower = min_size_mm if min_size_mm and min_size_mm > 0 else size / 5.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lower)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        if second_order and not all_hexa:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

        gmsh.model.mesh.generate(3)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")
        if all_hexa:
            # Gmsh subdivision algorithm 2 converts every tetrahedron into
            # hexahedra while retaining the uploaded solid as the boundary.
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
            gmsh.model.mesh.refine()
            gmsh.model.mesh.renumberNodes()
            gmsh.model.mesh.renumberElements()
        gmsh.write(str(output_file))

        node_count, element_counts, min_edge, max_edge = _mesh_stats(gmsh)
        tetra_count = sum(
            count for name, count in element_counts.items() if name.startswith("tetra")
        )
        hexa_count = sum(
            count for name, count in element_counts.items() if name.startswith("hexahedron")
        )
    except Exception as exc:
        if suffix == ".stl":
            raise ValueError(_stl_volume_error(str(exc), repair_notes)) from exc
        raise
    finally:
        gmsh.finalize()

    if all_hexa and hexa_count <= 0:
        raise ValueError("Gmsh all-hexa subdivision did not create hexahedral volume elements.")
    if all_hexa and tetra_count > 0:
        raise ValueError("Gmsh all-hexa subdivision left tetrahedral volume elements in the mesh.")
    if not all_hexa and tetra_count <= 0:
        if suffix == ".stl":
            raise ValueError(_stl_volume_error("Gmsh did not create tetrahedral volume elements.", repair_notes))
        raise ValueError("Gmsh did not create tetrahedral volume elements. Check that the uploaded geometry is a closed solid.")

    mesh_kind = "uploaded_geometry_tetra"
    if all_hexa:
        mesh_kind = "uploaded_geometry_global_hex" if global_mode else "uploaded_geometry_subdivided_hex"
    return UploadedMeshResult(
        source_file=source_file,
        mesh_file=output_file,
        source_format=source_format,
        node_count=node_count,
        tetra_count=tetra_count,
        hexa_count=hexa_count,
        min_edge_mm=min_edge,
        max_edge_mm=max_edge,
        repaired_surface=repaired_surface or bool(repair_notes),
        mesh_kind=mesh_kind,
        template_id=_template_id(template_config) if global_mode else "",
        circumferential_divisions=template_config["circumferential"] if global_mode else 0,
        radial_divisions=template_config["radial"] if global_mode else 0,
        axial_divisions=template_config["axial"] if global_mode else 0,
    )


def _prepare_stl_for_volume_mesh(source_file: Path, work_dir: Path) -> tuple[Path, list[str]]:
    """Write a cleaned STL for Gmsh and return repair notes."""

    try:
        import trimesh
    except ImportError:
        return source_file, ["trimesh cleanup unavailable"]

    try:
        loaded = trimesh.load_mesh(str(source_file), process=True)
        if isinstance(loaded, trimesh.Scene):
            geometries = [geom for geom in loaded.geometry.values() if hasattr(geom, "faces") and len(geom.faces)]
            if not geometries:
                return source_file, ["no triangle geometry found"]
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded
    except Exception as exc:  # noqa: BLE001 - keep original STL if cleanup cannot read it
        return source_file, [f"trimesh cleanup failed: {exc}"]

    notes: list[str] = []
    try:
        before_faces = int(len(mesh.faces))
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        mesh.merge_vertices()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        filled = bool(mesh.fill_holes())
        after_faces = int(len(mesh.faces))
        if after_faces != before_faces:
            notes.append(f"cleaned triangles {before_faces}->{after_faces}")
        if filled:
            notes.append("filled small holes")
        if not mesh.is_watertight:
            notes.append("STL is still not watertight after cleanup")
        if getattr(mesh, "is_winding_consistent", True) is False:
            notes.append("STL winding is inconsistent")
    except Exception as exc:  # noqa: BLE001 - export best-effort processed mesh
        notes.append(f"partial cleanup only: {exc}")

    repaired_file = work_dir / f"{source_file.stem}_cleaned_for_volume.stl"
    try:
        mesh.export(repaired_file)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"cleaned STL export failed: {exc}")
        return source_file, notes
    return repaired_file, notes or ["processed STL surface"]


def uploaded_geometry_to_tet_mesh(
    source_file: Path,
    output_file: Path,
    *,
    target_size_mm: float | None = None,
    min_size_mm: float | None = None,
    second_order: bool = True,
    optimize: bool = True,
    verbose: bool = False,
) -> UploadedMeshResult:
    """Backward-compatible tetra-only wrapper."""

    return uploaded_geometry_to_volume_mesh(
        source_file,
        output_file,
        target_size_mm=target_size_mm,
        min_size_mm=min_size_mm,
        second_order=second_order,
        optimize=optimize,
        verbose=verbose,
        mesh_mode="tetra",
    )


def _import_stl_as_volume(gmsh_module, source_file: Path) -> None:
    angle = 40.0 * 3.141592653589793 / 180.0
    attempts = (
        ("parametrized", True, "geometry"),
        ("relaxed", False, "geometry"),
        ("discrete_topology", False, "topology"),
    )
    errors: list[str] = []
    for index, (label, for_reparametrization, topology_method) in enumerate(attempts):
        if index > 0:
            gmsh_module.clear()
            gmsh_module.model.add(source_file.stem + f"_uploaded_{label}")
        try:
            gmsh_module.merge(str(source_file))
            gmsh_module.model.mesh.removeDuplicateNodes()
            gmsh_module.model.mesh.removeDuplicateElements()
            gmsh_module.model.mesh.classifySurfaces(angle, True, for_reparametrization, angle)
            if topology_method == "topology":
                gmsh_module.model.mesh.createTopology(makeSimplyConnected=True, exportDiscrete=True)
            else:
                gmsh_module.model.mesh.createGeometry()
            surfaces = gmsh_module.model.getEntities(2)
            if not surfaces:
                raise ValueError("No closed STL surfaces were found.")
            loop = gmsh_module.model.geo.addSurfaceLoop([tag for _, tag in surfaces])
            gmsh_module.model.geo.addVolume([loop])
            gmsh_module.model.geo.synchronize()
            return
        except Exception as exc:  # noqa: BLE001 - keep trying alternative STL topology modes
            errors.append(f"{label}: {exc}")

    detail = "; ".join(errors[-2:]) if errors else "unknown STL topology error"
    raise ValueError(
        "STL could not be converted into a closed volume for exact FEM. "
        "Repair the STL surface, export STEP if possible, or use the editable bushing surrogate. "
        f"Gmsh detail: {detail}"
    )


def _stl_volume_error(detail: str, repair_notes: list[str]) -> str:
    notes = "; ".join(note for note in repair_notes if note)
    note_text = f" Cleanup notes: {notes}." if notes else ""
    return (
        "Uploaded STL could not be converted into a closed Gmsh volume mesh. "
        "The file is likely open, self-intersecting, has overlapping internal faces, or is a surface-only shell. "
        "Repair the STL or upload the original STEP solid for body-fitted hexa meshing."
        f"{note_text} Gmsh detail: {detail}"
    )


def _initialize_gmsh(gmsh_module) -> None:
    try:
        gmsh_module.initialize(interruptible=False)
    except TypeError:
        gmsh_module.initialize()


def _bounding_box_diagonal(gmsh_module) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh_module.model.getBoundingBox(-1, -1)
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _global_template_config(template: dict) -> dict[str, int]:
    return {
        "circumferential": _bounded_int(template.get("circumferential_divisions"), 96, 24, 192),
        "radial": _bounded_int(template.get("radial_divisions"), 8, 2, 32),
        "axial": _bounded_int(template.get("axial_divisions"), 16, 3, 64),
    }


def _bounded_int(value, fallback: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = fallback
    return max(minimum, min(maximum, parsed))


def _target_mesh_size(
    diagonal: float,
    *,
    target_size_mm: float | None,
    global_mode: bool,
    template: dict[str, int],
) -> float:
    if target_size_mm and target_size_mm > 0:
        return float(target_size_mm)
    if not global_mode:
        return max(diagonal / 24.0, 1e-3)
    density = max(
        template["circumferential"] / 3.0,
        template["radial"] * 3.0,
        template["axial"] * 1.5,
    )
    return max(diagonal / max(density, 8.0), 1e-3)


def _template_id(template: dict[str, int]) -> str:
    return (
        f"uploaded-gmsh-c{template['circumferential']}"
        f"-r{template['radial']}-a{template['axial']}"
    )


def _mesh_stats(gmsh_module) -> tuple[int, dict[str, int], float, float]:
    node_tags, coords, _ = gmsh_module.model.mesh.getNodes()
    node_count = len(node_tags)
    coord_by_tag: dict[int, tuple[float, float, float]] = {}
    for index, tag in enumerate(node_tags):
        base = index * 3
        coord_by_tag[int(tag)] = (coords[base], coords[base + 1], coords[base + 2])

    element_counts: dict[str, int] = {}
    min_edge = float("inf")
    max_edge = 0.0
    for element_type, name, corner_count, edge_pairs in _VOLUME_ELEMENTS:
        try:
            elem_tags, node_tags_flat = gmsh_module.model.mesh.getElementsByType(element_type)
        except Exception:
            continue
        if len(elem_tags) == 0:
            continue
        nodes_per = len(node_tags_flat) // len(elem_tags)
        element_counts[name] = len(elem_tags)
        for elem_index in range(len(elem_tags)):
            corners = node_tags_flat[
                elem_index * nodes_per : elem_index * nodes_per + corner_count
            ]
            points = [coord_by_tag[int(tag)] for tag in corners]
            for i, j in edge_pairs:
                edge = _distance(points[i], points[j])
                min_edge = min(min_edge, edge)
                max_edge = max(max_edge, edge)

    if min_edge == float("inf"):
        min_edge = 0.0
    return node_count, element_counts, min_edge, max_edge


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)


_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_HEX_EDGES = (
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
)
_VOLUME_ELEMENTS = (
    (4, "tetra", 4, _TET_EDGES),
    (11, "tetra10", 4, _TET_EDGES),
    (5, "hexahedron", 8, _HEX_EDGES),
    (12, "hexahedron27", 8, _HEX_EDGES),
    (17, "hexahedron20", 8, _HEX_EDGES),
)
