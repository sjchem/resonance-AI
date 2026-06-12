"""Tiny CAD preview helpers for the OpenAI parser POC."""

from __future__ import annotations

from app.schemas import CADPromptOutput


def build_preview_svg(parsed: CADPromptOutput) -> str:
    """Return a simple SVG preview from validated CAD intent.

    This is intentionally not STEP/STL generation yet. It gives a quick visual
    confirmation that prompt parsing worked before adding a real CAD kernel.
    """

    if parsed.part_type == "bushing":
        return _bushing_svg(parsed)
    if parsed.part_type in {"plate", "bracket", "rubber_mount"}:
        return _block_svg(parsed)
    return _unknown_svg(parsed)


def _bushing_svg(parsed: CADPromptOutput) -> str:
    geometry = parsed.geometry
    outer = geometry.outer_diameter_mm or 60
    inner = geometry.inner_diameter_mm or outer * 0.35
    height = geometry.height_mm or 40
    chamfer = geometry.chamfer_mm or 0
    material = parsed.material.name or "material unknown"
    inner_radius = max(10, 110 * inner / outer)

    return f"""<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg">
  <rect width="760" height="430" fill="#f8fbfd"/>
  <text x="42" y="42" font-size="22" font-weight="700" fill="#182231">Bushing preview</text>
  <text x="42" y="70" font-size="14" fill="#66758a">{material}</text>
  <g transform="translate(190 220)">
    <circle r="110" fill="#13877e" stroke="#0b4f49" stroke-width="4"/>
    <circle r="{inner_radius:.2f}" fill="#f8fbfd" stroke="#0b4f49" stroke-width="4"/>
    <text x="0" y="160" text-anchor="middle" font-size="14" fill="#182231">OD {outer:g} mm</text>
    <text x="0" y="6" text-anchor="middle" font-size="14" fill="#182231">ID {inner:g} mm</text>
  </g>
  <g transform="translate(470 120)">
    <rect x="0" y="0" width="220" height="150" rx="{min(chamfer * 4, 24):.2f}" fill="#13877e" stroke="#0b4f49" stroke-width="4"/>
    <rect x="86" y="0" width="48" height="150" fill="#f8fbfd" stroke="#0b4f49" stroke-width="3"/>
    <text x="110" y="188" text-anchor="middle" font-size="14" fill="#182231">Height {height:g} mm</text>
    <text x="110" y="212" text-anchor="middle" font-size="14" fill="#182231">Chamfer {chamfer:g} mm</text>
  </g>
</svg>"""


def _block_svg(parsed: CADPromptOutput) -> str:
    geometry = parsed.geometry
    length = geometry.length_mm or geometry.outer_diameter_mm or 120
    width = geometry.width_mm or 60
    thickness = geometry.thickness_mm or geometry.height_mm or 5
    label = parsed.part_type.replace("_", " ")

    return f"""<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg">
  <rect width="760" height="430" fill="#f8fbfd"/>
  <text x="42" y="48" font-size="22" font-weight="700" fill="#182231">{label} preview</text>
  <g transform="translate(170 120)">
    <polygon points="0,60 360,20 470,95 105,145" fill="#18a096" stroke="#0b4f49" stroke-width="4"/>
    <polygon points="105,145 470,95 470,150 105,205" fill="#0f766e" stroke="#0b4f49" stroke-width="4"/>
    <polygon points="0,60 105,145 105,205 0,118" fill="#0b5f59" stroke="#0b4f49" stroke-width="4"/>
    <text x="220" y="260" text-anchor="middle" font-size="16" fill="#182231">L {length:g} mm x W {width:g} mm x T {thickness:g} mm</text>
  </g>
</svg>"""


def _unknown_svg(parsed: CADPromptOutput) -> str:
    return f"""<svg viewBox="0 0 760 430" xmlns="http://www.w3.org/2000/svg">
  <rect width="760" height="430" fill="#f8fbfd"/>
  <text x="42" y="48" font-size="22" font-weight="700" fill="#182231">CAD intent parsed</text>
  <text x="42" y="82" font-size="15" fill="#66758a">No preview is available for part_type: {parsed.part_type}</text>
</svg>"""
