from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from geometry.bushing_hex_mesh import generate_bushing_hex_mesh
from simulate.materials import resolve_material
from simulate.modal_solver import _extract_solid_elements
from simulate.static_stiffness import (
    StaticStiffnessSetup,
    parse_reaction_forces,
    radial_interface_nodes,
    run_static_stiffness,
)


class StaticStiffnessTests(unittest.TestCase):
    def test_parse_reaction_force_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dat_file = Path(directory) / "case.dat"
            dat_file.write_text(
                "\n"
                " forces (fx,fy,fz) for set NINNER and time  1.0000000E+00\n"
                "\n"
                "       1  1.000000E+01 -2.000000E+00  3.000000E+00\n"
                "       2  4.000000E+00  5.000000E+00 -1.000000E+00\n"
                "\n",
                encoding="utf-8",
            )
            reaction = parse_reaction_forces(dat_file)
        np.testing.assert_allclose(reaction, [14.0, 3.0, 2.0])

    def test_structured_interfaces_are_separate_and_length_filtered(self) -> None:
        import meshio

        with tempfile.TemporaryDirectory() as directory:
            mesh_file = Path(directory) / "bushing.vtk"
            generate_bushing_hex_mesh(
                _intent(),
                mesh_file,
                mesh_mode="global",
                template=_small_template(),
            )
            mesh = meshio.read(mesh_file)
            blocks = _extract_solid_elements(mesh)
            inner, outer = radial_interface_nodes(
                np.asarray(mesh.points),
                blocks,
                inner_length_mm=20.0,
                outer_length_mm=40.0,
            )
        self.assertGreater(inner.size, 8)
        self.assertGreater(outer.size, inner.size)
        self.assertEqual(np.intersect1d(inner, outer).size, 0)

    @unittest.skipUnless(shutil.which("ccx"), "CalculiX is not installed")
    def test_three_direction_calculix_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            mesh_file = output_dir / "bushing.vtk"
            generate_bushing_hex_mesh(
                _intent(),
                mesh_file,
                mesh_mode="global",
                template=_small_template(),
            )
            result = run_static_stiffness(
                StaticStiffnessSetup(
                    mesh_file=mesh_file,
                    material=resolve_material("rubber"),
                    inner_interface_length_mm=30.0,
                    outer_interface_length_mm=40.0,
                ),
                output_dir,
                job_name="test",
            )
        self.assertGreater(result.kx_n_per_mm, 0.0)
        self.assertGreater(result.ky_n_per_mm, 0.0)
        self.assertGreater(result.kz_n_per_mm, 0.0)
        self.assertAlmostEqual(result.ky_n_per_mm, result.kz_n_per_mm, delta=result.ky_n_per_mm * 0.03)


def _intent() -> dict:
    return {
        "part_type": "bushing",
        "material": {"name": "rubber"},
        "geometry": {
            "outer_diameter_mm": 76.0,
            "inner_diameter_mm": 28.0,
            "height_mm": 40.0,
            "inner_core_length_mm": 30.0,
            "outer_core_length_mm": 40.0,
            "slot_count": 0,
            "bore_shape": "round",
        },
    }


def _small_template() -> dict[str, int]:
    return {
        "circumferential_divisions": 24,
        "radial_divisions": 2,
        "axial_divisions": 4,
    }
