"""Volume mesh uploaded STEP/STL geometry for exact FEM.

This module is intentionally tetra-first. Arbitrary uploaded geometry, especially
STL triangle soups, cannot be made pure-hex reliably without topology-specific
block decomposition. The bushing-specific hex path remains separate.
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
    min_edge_mm: float
    max_edge_mm: float
    repaired_surface: bool = False

    def summary(self) -> str:
        repair = " with STL surface classification" if self.repaired_surface else ""
        return (
            f"Uploaded {self.source_format}{repair}: {self.mesh_file.name} | "
            f"nodes={self.node_count} tetra={self.tetra_count} | element size "
            f"{self.min_edge_mm:.3f}-{self.max_edge_mm:.3f} mm"
        )


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
    """Create a tetrahedral volume mesh from an uploaded STEP/STL file."""

    import gmsh

    source_file = Path(source_file).resolve()
    output_file = Path(output_file).resolve()
    if not source_file.exists():
        raise FileNotFoundError(f"Uploaded geometry file not found: {source_file}")

    suffix = source_file.suffix.lower()
    if suffix not in {".step", ".stp", ".stl"}:
        raise ValueError("Exact uploaded-geometry FEM currently supports STEP/STP and watertight STL files.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    _initialize_gmsh(gmsh)
    repaired_surface = False
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add(source_file.stem + "_uploaded")
        if suffix in {".step", ".stp"}:
            source_format = "STEP"
            gmsh.model.occ.importShapes(str(source_file))
            gmsh.model.occ.synchronize()
        else:
            source_format = "STL"
            _import_stl_as_volume(gmsh, source_file)
            repaired_surface = True

        diagonal = _bounding_box_diagonal(gmsh)
        size = target_size_mm if target_size_mm and target_size_mm > 0 else max(diagonal / 24.0, 1e-3)
        lower = min_size_mm if min_size_mm and min_size_mm > 0 else size / 5.0

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lower)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        if second_order:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

        gmsh.model.mesh.generate(3)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")
        gmsh.write(str(output_file))

        node_count, tetra_count, min_edge, max_edge = _mesh_stats(gmsh)
    finally:
        gmsh.finalize()

    if tetra_count <= 0:
        raise ValueError("Gmsh did not create tetrahedral volume elements. Check that the uploaded geometry is a closed solid.")

    return UploadedMeshResult(
        source_file=source_file,
        mesh_file=output_file,
        source_format=source_format,
        node_count=node_count,
        tetra_count=tetra_count,
        min_edge_mm=min_edge,
        max_edge_mm=max_edge,
        repaired_surface=repaired_surface,
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


def _initialize_gmsh(gmsh_module) -> None:
    try:
        gmsh_module.initialize(interruptible=False)
    except TypeError:
        gmsh_module.initialize()


def _bounding_box_diagonal(gmsh_module) -> float:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh_module.model.getBoundingBox(-1, -1)
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _mesh_stats(gmsh_module) -> tuple[int, int, float, float]:
    node_tags, coords, _ = gmsh_module.model.mesh.getNodes()
    node_count = len(node_tags)
    coord_by_tag: dict[int, tuple[float, float, float]] = {}
    for index, tag in enumerate(node_tags):
        base = index * 3
        coord_by_tag[int(tag)] = (coords[base], coords[base + 1], coords[base + 2])

    tetra_count = 0
    min_edge = float("inf")
    max_edge = 0.0
    for tet_type, nodes_per in ((4, 4), (11, 10)):
        try:
            elem_tags, node_tags_flat = gmsh_module.model.mesh.getElementsByType(tet_type)
        except Exception:
            continue
        if len(elem_tags) == 0:
            continue
        tetra_count += len(elem_tags)
        for elem_index in range(len(elem_tags)):
            corners = node_tags_flat[elem_index * nodes_per : elem_index * nodes_per + 4]
            points = [coord_by_tag[int(tag)] for tag in corners]
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = _distance(points[i], points[j])
                    min_edge = min(min_edge, edge)
                    max_edge = max(max_edge, edge)

    if min_edge == float("inf"):
        min_edge = 0.0
    return node_count, tetra_count, min_edge, max_edge


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5)
