"""Render CalculiX ``.frd`` results as engineering contour plots (PNG).

This produces the classic "stress / displacement contour" image — a coloured
3D part with a rainbow colour bar — directly from the CalculiX result file, so
no ParaView/cgx GUI is needed.

    .frd  ->  (this module)  ->  contour PNG (von Mises stress or displacement)

A modal analysis stores, per mode, the mode-shape displacement (``DISP``) and
the 6 stress components (``STRESS``). The von Mises stress is computed from
those components. Modal stress is an eigenvector quantity, so the *pattern* is
meaningful while the absolute magnitude is relative (normalised by the solver).

Example
-------
    python -m simulate.visualize outputs/simulation/bracket/bracket.frd \
        --field mises --mode 1 --warp \
        --output outputs/simulation/bracket/bracket_mode1_mises.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np


# CalculiX .frd element type code -> (VTK cell type id, number of corner+all nodes).
# We mainly mesh with tetrahedra; the others are included for completeness.
_VTK_TETRA = 10
_VTK_QUADRATIC_TETRA = 24
_VTK_HEXAHEDRON = 12
_VTK_QUADRATIC_HEXAHEDRON = 25
_VTK_WEDGE = 13
_VTK_QUADRATIC_WEDGE = 26

_FRD_ELEMENT_TYPES: dict[int, tuple[int, int]] = {
    1: (_VTK_HEXAHEDRON, 8),
    2: (_VTK_WEDGE, 6),
    3: (_VTK_TETRA, 4),
    4: (_VTK_QUADRATIC_HEXAHEDRON, 20),
    5: (_VTK_QUADRATIC_WEDGE, 15),
    6: (_VTK_QUADRATIC_TETRA, 10),
}


@dataclass
class FrdMesh:
    """Nodes and elements parsed from a ``.frd`` file."""

    points: np.ndarray  # (N, 3)
    node_index: dict[int, int]  # frd node id -> row in points
    cells: list[tuple[int, list[int]]] = field(default_factory=list)  # (vtk_type, [row indices])


@dataclass
class FrdField:
    """A single nodal result field for one mode/step."""

    mode: int
    frequency_hz: float
    kind: str  # "DISP" or "STRESS"
    data: np.ndarray  # (N, 3) for DISP, (N, 6) for STRESS, aligned to mesh.points


def _floats_fixed(line: str, start: int, width: int = 12, count: int | None = None) -> list[float]:
    """Read fixed-width float columns (frd values can butt together, e.g. -1.0E0-2.0E0)."""

    values: list[float] = []
    pos = start
    while pos < len(line.rstrip()):
        chunk = line[pos : pos + width].strip()
        if not chunk:
            break
        values.append(float(chunk))
        pos += width
        if count is not None and len(values) >= count:
            break
    return values


def parse_frd(frd_file: Path) -> tuple[FrdMesh, list[FrdField]]:
    """Parse nodes, elements and nodal result blocks from a CalculiX ``.frd`` file."""

    frd_file = Path(frd_file).resolve()
    if not frd_file.exists():
        raise FileNotFoundError(f"Result file not found: {frd_file}")

    raw_points: list[tuple[float, float, float]] = []
    node_index: dict[int, int] = {}
    cells: list[tuple[int, list[int]]] = []
    fields: list[FrdField] = []

    mode = "scan"  # scan | nodes | elements | result
    pending_type: int | None = None
    pending_nodes: list[int] = []
    cur_kind: str | None = None
    cur_freq = 0.0
    cur_mode_no = 0
    cur_ncomp = 0
    cur_rows: dict[int, list[float]] = {}

    def flush_result() -> None:
        nonlocal cur_kind, cur_rows
        if cur_kind in ("DISP", "STRESS") and cur_rows:
            width = 3 if cur_kind == "DISP" else 6
            data = np.zeros((len(raw_points), width), dtype=float)
            for node_id, vals in cur_rows.items():
                row = node_index.get(node_id)
                if row is not None:
                    data[row, : len(vals)] = vals[:width]
            fields.append(
                FrdField(mode=cur_mode_no, frequency_hz=cur_freq, kind=cur_kind, data=data)
            )
        cur_kind = None
        cur_rows = {}

    with frd_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()

            # Block headers (the key sits a few columns in, so match on stripped text).
            if stripped.startswith("2C"):
                mode = "nodes"
                continue
            if stripped.startswith("3C"):
                mode = "elements"
                continue
            if stripped.startswith("100CL"):
                # Result header: frequency in cols ~13-24, mode counter before "MODAL".
                mode = "result"
                tokens = stripped.split()
                cur_freq = float(tokens[2]) if len(tokens) > 2 else 0.0
                cur_mode_no = _trailing_mode_number(line)
                continue

            tag = line[:3].strip()

            if mode == "nodes":
                if tag == "-1":
                    node_id = int(line[3:13])
                    coords = _floats_fixed(line, 13, 12, 3)
                    node_index[node_id] = len(raw_points)
                    raw_points.append((coords[0], coords[1], coords[2]))
                elif tag == "-3":
                    mode = "scan"
                continue

            if mode == "elements":
                if tag == "-1":
                    pending_type = int(line[13:18])
                    pending_nodes = []
                elif tag == "-2":
                    pending_nodes.extend(_ints_fixed(line, 3, 10))
                    info = _FRD_ELEMENT_TYPES.get(pending_type or 0)
                    if info and len(pending_nodes) >= info[1]:
                        vtk_type, n_nodes = info
                        rows = [node_index[n] for n in pending_nodes[:n_nodes] if n in node_index]
                        if len(rows) == n_nodes:
                            cells.append((vtk_type, rows))
                        pending_type = None
                        pending_nodes = []
                elif tag == "-3":
                    mode = "scan"
                continue

            if mode == "result":
                if tag == "-4":
                    flush_result()
                    cur_kind = line[5:13].strip()
                    cur_ncomp = _safe_int(line[13:18])
                elif tag == "-5":
                    continue
                elif tag == "-1":
                    node_id = int(line[3:13])
                    cur_rows[node_id] = _floats_fixed(line, 13, 12)
                elif tag == "-2":
                    # Continuation of a long value row.
                    if cur_rows:
                        last = next(reversed(cur_rows))
                        cur_rows[last].extend(_floats_fixed(line, 3, 12))
                elif tag == "-3":
                    flush_result()
                    mode = "scan"
                continue

    flush_result()
    points = np.asarray(raw_points, dtype=float)
    return FrdMesh(points=points, node_index=node_index, cells=cells), fields


def _ints_fixed(line: str, start: int, width: int) -> list[int]:
    values: list[int] = []
    pos = start
    while pos < len(line.rstrip()):
        chunk = line[pos : pos + width].strip()
        if not chunk:
            break
        values.append(int(chunk))
        pos += width
    return values


def _safe_int(text: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return 0


def _trailing_mode_number(line: str) -> int:
    # ...  2    1MODAL  -> the integer immediately before "MODAL".
    idx = line.find("MODAL")
    if idx == -1:
        return 0
    j = idx - 1
    while j >= 0 and line[j].isdigit():
        j -= 1
    digits = line[j + 1 : idx].strip()
    return int(digits) if digits else 0


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises stress from 6 components ordered SXX, SYY, SZZ, SXY, SYZ, SZX."""

    sxx, syy, szz = stress[:, 0], stress[:, 1], stress[:, 2]
    sxy, syz, szx = stress[:, 3], stress[:, 4], stress[:, 5]
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
    )


def _build_grid(mesh: FrdMesh):
    import pyvista as pv

    cell_array: list[int] = []
    cell_types: list[int] = []
    for vtk_type, rows in mesh.cells:
        cell_array.append(len(rows))
        cell_array.extend(rows)
        cell_types.append(vtk_type)

    if not cell_types:
        raise ValueError("No supported volume elements found in the .frd file.")

    grid = pv.UnstructuredGrid(
        np.asarray(cell_array, dtype=np.int64),
        np.asarray(cell_types, dtype=np.uint8),
        np.asarray(mesh.points, dtype=float),
    )
    return grid


def render_contour(
    frd_file: Path,
    output: Path,
    *,
    field_name: str = "mises",
    mode: int = 1,
    warp: bool = False,
    warp_scale: float | None = None,
    cmap: str = "jet",
    n_colors: int = 12,
    window_size: tuple[int, int] = (1100, 760),
) -> Path:
    """Render a contour PNG (von Mises stress or displacement magnitude)."""

    import pyvista as pv

    mesh, fields = parse_frd(frd_file)
    disp = _select_field(fields, "DISP", mode)
    stress = _select_field(fields, "STRESS", mode)

    if field_name == "mises":
        if stress is None:
            raise ValueError(f"No STRESS field found for mode {mode} in {frd_file.name}.")
        scalars = von_mises(stress.data)
        bar_title = "S, Mises"
        freq = stress.frequency_hz
    elif field_name in ("disp", "displacement", "u"):
        if disp is None:
            raise ValueError(f"No DISP field found for mode {mode} in {frd_file.name}.")
        scalars = np.linalg.norm(disp.data, axis=1)
        bar_title = "U, Magnitude"
        freq = disp.frequency_hz
    else:
        raise ValueError("field_name must be 'mises' or 'disp'.")

    grid = _build_grid(mesh)
    grid.point_data[bar_title] = scalars

    if warp and disp is not None:
        grid.point_data["__disp__"] = disp.data
        scale = warp_scale if warp_scale is not None else _auto_warp_scale(grid, disp.data)
        grid = grid.warp_by_vector("__disp__", factor=scale)

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    pv.OFF_SCREEN = True
    plotter = _new_plotter(window_size)
    plotter.set_background("white")
    plotter.add_mesh(
        grid,
        scalars=bar_title,
        cmap=cmap,
        n_colors=n_colors,
        show_edges=False,
        smooth_shading=True,
        scalar_bar_args={
            "title": bar_title,
            "vertical": True,
            "position_x": 0.02,
            "position_y": 0.20,
            "width": 0.08,
            "height": 0.60,
            "n_labels": n_colors + 1,
            "fmt": "%.3e",
            "title_font_size": 18,
            "label_font_size": 13,
            "color": "black",
        },
    )
    title = f"{bar_title}  |  mode {mode}  |  {freq:.1f} Hz"
    plotter.add_text(title, position="upper_edge", font_size=11, color="black")
    plotter.view_isometric()
    plotter.screenshot(str(output))
    plotter.close()
    return output


def export_contour_surface_mesh(
    frd_file: Path,
    *,
    field_name: str = "mises",
    mode: int = 1,
    warp: bool = False,
    warp_scale: float | None = None,
    max_faces: int = 12000,
) -> dict:
    """Return a browser-friendly coloured surface mesh for interactive viewing."""

    mesh, fields = parse_frd(frd_file)
    disp = _select_field(fields, "DISP", mode)
    stress = _select_field(fields, "STRESS", mode)

    if field_name == "mises":
        if stress is None:
            raise ValueError(f"No STRESS field found for mode {mode} in {frd_file.name}.")
        scalars = von_mises(stress.data)
        scalar_name = "S, Mises"
        freq = stress.frequency_hz
    elif field_name in ("disp", "displacement", "u"):
        if disp is None:
            raise ValueError(f"No DISP field found for mode {mode} in {frd_file.name}.")
        scalars = np.linalg.norm(disp.data, axis=1)
        scalar_name = "U, Magnitude"
        freq = disp.frequency_hz
    else:
        raise ValueError("field_name must be 'mises' or 'disp'.")

    grid = _build_grid(mesh)
    grid.point_data[scalar_name] = scalars

    if warp and disp is not None:
        grid.point_data["__disp__"] = disp.data
        scale = warp_scale if warp_scale is not None else _auto_warp_scale(grid, disp.data)
        grid = grid.warp_by_vector("__disp__", factor=scale)

    surface = grid.extract_surface().triangulate()
    if max_faces > 0 and surface.n_cells > max_faces:
        reduction = min(0.95, max(0.0, 1.0 - (max_faces / float(surface.n_cells))))
        try:
            surface = surface.decimate_pro(reduction, preserve_topology=True).triangulate()
        except Exception:
            surface = surface.extract_surface().triangulate()

    points = np.asarray(surface.points, dtype=float)
    values = np.asarray(surface.point_data.get(scalar_name), dtype=float)
    if values.shape[0] != points.shape[0]:
        values = np.zeros(points.shape[0], dtype=float)

    vmin = float(np.nanmin(values)) if values.size else 0.0
    vmax = float(np.nanmax(values)) if values.size else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    faces: list[dict] = []
    raw_faces = np.asarray(surface.faces, dtype=np.int64)
    index = 0
    while index < raw_faces.size:
        count = int(raw_faces[index])
        ids = raw_faces[index + 1 : index + 1 + count]
        index += count + 1
        if count != 3 or len(ids) != 3:
            continue
        avg = float(np.mean(values[ids])) if values.size else vmin
        faces.append(
            {
                "color": _contour_hex(avg, vmin, vmax),
                "points": [
                    {"x": float(points[node_id][0]), "y": float(points[node_id][1]), "z": float(points[node_id][2])}
                    for node_id in ids
                ],
            }
        )

    return {
        "faces": faces,
        "field": scalar_name,
        "mode": mode,
        "frequency_hz": freq,
        "scalar_min": vmin,
        "scalar_max": vmax,
        "face_count": len(faces),
    }


def _contour_hex(value: float, vmin: float, vmax: float) -> str:
    t = (value - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0.0, 1.0))
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
            span = right_t - left_t or 1.0
            local = (t - left_t) / span
            rgb = tuple(round(left_rgb[i] + (right_rgb[i] - left_rgb[i]) * local) for i in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#{:02x}{:02x}{:02x}".format(*stops[-1][1])


def _new_plotter(window_size: tuple[int, int]):
    import pyvista as pv

    try:
        return pv.Plotter(off_screen=True, window_size=list(window_size))
    except Exception:
        pv.start_xvfb()
        return pv.Plotter(off_screen=True, window_size=list(window_size))


def _select_field(fields: list[FrdField], kind: str, mode: int) -> FrdField | None:
    matches = [f for f in fields if f.kind == kind]
    if not matches:
        return None
    for f in matches:
        if f.mode == mode:
            return f
    return matches[0]


def _auto_warp_scale(grid, disp: np.ndarray) -> float:
    """Scale the mode shape so the peak deflection is ~10% of the model size."""

    max_disp = float(np.linalg.norm(disp, axis=1).max())
    if max_disp <= 0.0:
        return 0.0
    spans = np.ptp(np.asarray(grid.bounds).reshape(3, 2), axis=1)
    diagonal = float(np.linalg.norm(spans))
    return 0.1 * diagonal / max_disp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a CalculiX .frd result as a contour PNG.")
    parser.add_argument("frd_file", type=Path, help="CalculiX .frd result file.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument(
        "--field",
        default="mises",
        choices=("mises", "disp"),
        help="Field to colour: von Mises stress or displacement magnitude.",
    )
    parser.add_argument("--mode", type=int, default=1, help="Mode number to visualise.")
    parser.add_argument("--warp", action="store_true", help="Deform the part by the mode shape.")
    parser.add_argument("--warp-scale", type=float, default=None, help="Explicit deformation scale.")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap name.")
    parser.add_argument("--colors", type=int, default=12, help="Number of discrete contour bands.")
    args = parser.parse_args(argv)

    output = args.output or args.frd_file.with_name(
        f"{args.frd_file.stem}_mode{args.mode}_{args.field}.png"
    )

    try:
        path = render_contour(
            args.frd_file,
            output,
            field_name=args.field,
            mode=args.mode,
            warp=args.warp,
            warp_scale=args.warp_scale,
            cmap=args.cmap,
            n_colors=args.colors,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print(f"Wrote contour image: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
