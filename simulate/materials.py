"""Material property library for modal/structural simulation.

Properties use a consistent unit system that matches CAD millimetres:

    length   : mm
    force    : N
    stress   : MPa  (N/mm^2)
    density  : tonne/mm^3   (t = 1000 kg)

In this "tonne-mm-s" system, natural frequencies come out directly in Hz.
That is the standard convention for CalculiX/Abaqus when geometry is in mm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    """Linear-elastic isotropic material in the tonne-mm-s unit system."""

    name: str
    youngs_modulus_mpa: float          # E [MPa = N/mm^2]
    poisson_ratio: float               # nu [-]
    density_t_per_mm3: float           # rho [tonne/mm^3]
    description: str = ""

    @property
    def density_kg_per_m3(self) -> float:
        """Convenience: density back in SI kg/m^3."""

        return self.density_t_per_mm3 * 1.0e12


# Common Vibracoustic-relevant materials. Rubber is modelled as a soft linear
# elastic solid here, which is adequate for a first-pass modal estimate.
MATERIAL_LIBRARY: dict[str, Material] = {
    "steel": Material(
        name="steel",
        youngs_modulus_mpa=210_000.0,
        poisson_ratio=0.30,
        density_t_per_mm3=7.85e-9,
        description="Structural steel",
    ),
    "stainless_steel": Material(
        name="stainless_steel",
        youngs_modulus_mpa=193_000.0,
        poisson_ratio=0.31,
        density_t_per_mm3=8.00e-9,
        description="Austenitic stainless steel",
    ),
    "aluminum": Material(
        name="aluminum",
        youngs_modulus_mpa=69_000.0,
        poisson_ratio=0.33,
        density_t_per_mm3=2.70e-9,
        description="Aluminium alloy",
    ),
    "cast_iron": Material(
        name="cast_iron",
        youngs_modulus_mpa=110_000.0,
        poisson_ratio=0.28,
        density_t_per_mm3=7.20e-9,
        description="Grey cast iron",
    ),
    "rubber": Material(
        name="rubber",
        youngs_modulus_mpa=10.0,
        poisson_ratio=0.49,
        density_t_per_mm3=1.10e-9,
        description="Natural rubber (linear-elastic approximation)",
    ),
    "epdm": Material(
        name="epdm",
        youngs_modulus_mpa=6.0,
        poisson_ratio=0.49,
        density_t_per_mm3=1.15e-9,
        description="EPDM elastomer (linear-elastic approximation)",
    ),
    "abs": Material(
        name="abs",
        youngs_modulus_mpa=2_300.0,
        poisson_ratio=0.35,
        density_t_per_mm3=1.05e-9,
        description="ABS thermoplastic",
    ),
    "generic": Material(
        name="generic",
        youngs_modulus_mpa=200_000.0,
        poisson_ratio=0.30,
        density_t_per_mm3=7.85e-9,
        description="Default metal fallback",
    ),
}


# Map free-text material hints (from prompts/specs) onto library keys.
_ALIASES: dict[str, str] = {
    "metal": "steel",
    "mild steel": "steel",
    "carbon steel": "steel",
    "ss": "stainless_steel",
    "inox": "stainless_steel",
    "alu": "aluminum",
    "aluminium": "aluminum",
    "al": "aluminum",
    "iron": "cast_iron",
    "natural rubber": "rubber",
    "nr": "rubber",
    "elastomer": "rubber",
    "plastic": "abs",
}


def resolve_material(hint: str | None) -> Material:
    """Resolve a free-text material hint to a library :class:`Material`."""

    if not hint:
        return MATERIAL_LIBRARY["generic"]

    key = hint.strip().lower()
    if key in MATERIAL_LIBRARY:
        return MATERIAL_LIBRARY[key]
    if key in _ALIASES:
        return MATERIAL_LIBRARY[_ALIASES[key]]

    # Substring fallback: "stainless steel bracket" -> stainless_steel.
    for alias, target in _ALIASES.items():
        if alias in key:
            return MATERIAL_LIBRARY[target]
    for name in MATERIAL_LIBRARY:
        if name in key:
            return MATERIAL_LIBRARY[name]

    return MATERIAL_LIBRARY["generic"]


def list_materials() -> list[str]:
    """Return the available material library keys."""

    return sorted(MATERIAL_LIBRARY)
