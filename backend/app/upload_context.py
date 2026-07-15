"""Helpers for turning uploaded engineering files into CAD prompt context."""

from __future__ import annotations

import io
import json
import os
import re
import struct
import uuid
import zipfile
from pathlib import Path

from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_CHARS = 12000
LEGACY_UPLOAD_STORAGE_DIR = Path(__file__).resolve().parents[2] / "outputs" / "uploads"
UPLOAD_STORAGE_DIR = Path(os.getenv("RESONANCE_UPLOAD_DIR") or os.getenv("UPLOAD_STORAGE_DIR") or "/home/resonance-ai/uploads")
if not UPLOAD_STORAGE_DIR.is_absolute():
    UPLOAD_STORAGE_DIR = Path(__file__).resolve().parents[2] / UPLOAD_STORAGE_DIR
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".dxf", ".scad"}
JSON_EXTENSIONS = {".json"}
CAD_EXTENSIONS = {".step", ".stp", ".iges", ".igs", ".stl", ".obj", ".dxf", ".scad", ".fcstd"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
REAL_FEM_EXTENSIONS = {".step", ".stp", ".stl"}


class UploadContext(BaseModel):
    """Compact context extracted from an uploaded file."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    content_type: str | None
    size_bytes: int
    file_kind: str
    summary: str
    extracted_text: str | None = None
    prompt_context: str
    upload_id: str | None = None
    exact_fem: dict | None = None


async def build_upload_context(file: UploadFile) -> UploadContext:
    """Read an uploaded file and return context suitable for an LLM prompt."""

    raw = file.file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    filename = Path(file.filename or "uploaded-file").name
    suffix = Path(filename).suffix.lower()
    content_type = file.content_type

    if suffix in PDF_EXTENSIONS:
        return _pdf_context(filename, content_type, raw)
    if suffix in IMAGE_EXTENSIONS:
        return _image_context(filename, content_type, raw)
    if suffix in JSON_EXTENSIONS:
        return _json_context(filename, content_type, raw)
    if suffix in CAD_EXTENSIONS:
        return _cad_context(filename, content_type, raw, suffix)
    if suffix in TEXT_EXTENSIONS or _looks_text(raw):
        return _text_context(filename, content_type, raw, "document")

    return _binary_context(filename, content_type, raw)


def _pdf_context(filename: str, content_type: str | None, raw: bytes) -> UploadContext:
    text = _extract_pdf_text(raw)
    if not text:
        text = "No selectable PDF text could be extracted. The PDF may be scanned or image-only."
    summary = f"PDF document uploaded: {filename}. Extracted {len(text)} characters of text."
    return UploadContext(
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        file_kind="pdf",
        summary=summary,
        extracted_text=_truncate(text),
        prompt_context=_format_prompt_context(filename, "pdf", summary, text),
    )


def _image_context(filename: str, content_type: str | None, raw: bytes) -> UploadContext:
    width, height = _read_image_size(raw)
    dimensions = f"{width}x{height}px" if width and height else "unknown pixel size"
    summary = (
        f"Image uploaded: {filename}, {dimensions}. "
        "Visual feature extraction is not enabled yet; use the user's chat message for dimensions."
    )
    return UploadContext(
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        file_kind="image",
        summary=summary,
        extracted_text=None,
        prompt_context=_format_prompt_context(filename, "image", summary, None),
    )


def _cad_context(filename: str, content_type: str | None, raw: bytes, suffix: str) -> UploadContext:
    if suffix == ".fcstd":
        text = _extract_fcstd_text(raw)
    elif suffix == ".stl":
        text = _extract_stl_summary(raw)
    else:
        text = _decode_text(raw)

    numeric_hints = _numeric_hints(text)
    summary = f"CAD file uploaded: {filename}. Detected format {suffix.lstrip('.').upper()}."
    if numeric_hints:
        summary += f" Numeric hints found: {numeric_hints}."

    upload_id = None
    exact_fem = _exact_fem_context(suffix)
    if suffix in REAL_FEM_EXTENSIONS:
        upload_id = _persist_uploaded_geometry(filename, content_type, raw, suffix)
        summary += " Stored for exact uploaded-geometry mesh/FEM."

    return UploadContext(
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        file_kind="cad",
        summary=summary,
        extracted_text=_truncate(text),
        prompt_context=_format_prompt_context(filename, "cad", summary, text),
        upload_id=upload_id,
        exact_fem=exact_fem,
    )


def uploaded_geometry_path(upload_id: str) -> Path | None:
    """Return the persisted uploaded geometry path for a safe upload id."""

    if not re.fullmatch(r"[a-f0-9]{32}", upload_id or ""):
        return None
    for storage_dir in _upload_storage_dirs():
        path = _uploaded_geometry_path_from_dir(upload_id, storage_dir)
        if path is not None:
            return path
    return None


def persist_uploaded_geometry_bytes(filename: str, content_type: str | None, raw: bytes) -> Path:
    """Persist uploaded STEP/STL bytes and return the stored path."""

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is too large. Maximum size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    suffix = Path(filename or "uploaded_geometry").suffix.lower()
    if suffix not in REAL_FEM_EXTENSIONS:
        raise ValueError("Exact uploaded-geometry FEM currently supports STEP/STP and watertight STL files.")
    upload_id = _persist_uploaded_geometry(filename, content_type, raw, suffix)
    path = uploaded_geometry_path(upload_id)
    if path is None:
        raise FileNotFoundError("Uploaded geometry could not be persisted for meshing.")
    return path


def _uploaded_geometry_path_from_dir(upload_id: str, storage_dir: Path) -> Path | None:
    manifest_path = storage_dir / upload_id / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stored_name = Path(str(manifest.get("stored_filename") or "")).name
    if not stored_name:
        return None
    path = storage_dir / upload_id / stored_name
    try:
        path.relative_to(storage_dir)
    except ValueError:
        return None
    return path if path.exists() else None


def _upload_storage_dirs() -> list[Path]:
    dirs = [UPLOAD_STORAGE_DIR]
    if LEGACY_UPLOAD_STORAGE_DIR != UPLOAD_STORAGE_DIR:
        dirs.append(LEGACY_UPLOAD_STORAGE_DIR)
    return dirs


def _persist_uploaded_geometry(filename: str, content_type: str | None, raw: bytes, suffix: str) -> str:
    upload_id = uuid.uuid4().hex
    storage_dir = _writable_upload_storage_dir()
    upload_dir = storage_dir / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_upload_filename(filename, suffix)
    (upload_dir / safe_name).write_bytes(raw)
    manifest = {
        "upload_id": upload_id,
        "filename": filename,
        "stored_filename": safe_name,
        "content_type": content_type,
        "suffix": suffix,
        "size_bytes": len(raw),
        "exact_fem": _exact_fem_context(suffix),
    }
    (upload_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return upload_id


def _writable_upload_storage_dir() -> Path:
    last_error: Exception | None = None
    for storage_dir in _upload_storage_dirs():
        try:
            storage_dir.mkdir(parents=True, exist_ok=True)
            probe = storage_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return storage_dir
        except OSError as exc:
            last_error = exc
    raise OSError(f"No writable upload storage directory is available: {last_error}")


def _safe_upload_filename(filename: str, suffix: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(filename).stem).strip("_")[:60] or "uploaded_geometry"
    return f"{stem}{suffix}"


def _exact_fem_context(suffix: str) -> dict | None:
    if suffix not in REAL_FEM_EXTENSIONS:
        return None
    if suffix in {".step", ".stp"}:
        return {
            "supported": True,
            "source_format": "STEP",
            "mesh_strategy": "uploaded_geometry_tetra",
            "message": "Ready for exact uploaded-geometry tetra FEM using Gmsh/OpenCASCADE and CalculiX.",
        }
    return {
        "supported": True,
        "source_format": "STL",
        "mesh_strategy": "uploaded_geometry_tetra",
        "message": "Ready for exact STL tetra FEM when the STL surface is closed/watertight; otherwise geometry repair is required.",
    }


def _json_context(filename: str, content_type: str | None, raw: bytes) -> UploadContext:
    text = _decode_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _text_context(filename, content_type, raw, "json")

    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    keys = _json_key_summary(data)
    statuses = _json_status_summary(data)
    summary = f"JSON file uploaded: {filename}."
    if keys:
        summary += f" Top-level keys: {keys}."
    if statuses:
        summary += f" Status fields: {statuses}."

    return UploadContext(
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        file_kind="json",
        summary=summary,
        extracted_text=_truncate(pretty),
        prompt_context=_format_prompt_context(filename, "json", summary, pretty),
    )


def _text_context(filename: str, content_type: str | None, raw: bytes, file_kind: str) -> UploadContext:
    text = _decode_text(raw)
    summary = f"Text document uploaded: {filename}. Extracted {len(text)} characters."
    return UploadContext(
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        file_kind=file_kind,
        summary=summary,
        extracted_text=_truncate(text),
        prompt_context=_format_prompt_context(filename, file_kind, summary, text),
    )


def _binary_context(filename: str, content_type: str | None, raw: bytes) -> UploadContext:
    summary = (
        f"Binary file uploaded: {filename}. No text extractor is available for this format yet; "
        "use filename, file type, and user chat instructions as context."
    )
    return UploadContext(
        filename=filename,
        content_type=content_type,
        size_bytes=len(raw),
        file_kind="binary",
        summary=summary,
        extracted_text=None,
        prompt_context=_format_prompt_context(filename, "binary", summary, None),
    )


def _extract_pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "PDF support requires the pypdf package."

    try:
        reader = PdfReader(io.BytesIO(raw))
        parts = [page.extract_text() or "" for page in reader.pages[:12]]
    except Exception:
        return ""
    return "\n".join(part.strip() for part in parts if part.strip())


def _extract_fcstd_text(raw: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            parts = []
            for name in archive.namelist():
                if name.endswith((".xml", ".txt", ".json")):
                    parts.append(archive.read(name).decode("utf-8", errors="ignore"))
            return "\n".join(parts)
    except zipfile.BadZipFile:
        return "FreeCAD file could not be opened as a zip archive."


def _extract_stl_summary(raw: bytes) -> str:
    text = _decode_text(raw[:MAX_CONTEXT_CHARS])
    if text.lstrip().lower().startswith("solid") and "facet normal" in text.lower():
        facets = len(re.findall(r"\bfacet\s+normal\b", text, flags=re.IGNORECASE))
        return f"ASCII STL detected. Facets visible in sample: {facets}.\n{text}"

    if len(raw) >= 84:
        triangle_count = struct.unpack("<I", raw[80:84])[0]
        expected_size = 84 + triangle_count * 50
        if expected_size == len(raw):
            return f"Binary STL detected. Triangle count: {triangle_count}."
    return "STL file detected, but triangle count could not be determined."


def _read_image_size(raw: bytes) -> tuple[int | None, int | None]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return struct.unpack(">II", raw[16:24])
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return _read_webp_size(raw)
    if raw.startswith(b"\xff\xd8"):
        return _read_jpeg_size(raw)
    return None, None


def _read_webp_size(raw: bytes) -> tuple[int | None, int | None]:
    if raw[12:16] == b"VP8X" and len(raw) >= 30:
        width = int.from_bytes(raw[24:27], "little") + 1
        height = int.from_bytes(raw[27:30], "little") + 1
        return width, height
    return None, None


def _read_jpeg_size(raw: bytes) -> tuple[int | None, int | None]:
    index = 2
    while index + 9 < len(raw):
        if raw[index] != 0xFF:
            index += 1
            continue
        marker = raw[index + 1]
        length = int.from_bytes(raw[index + 2 : index + 4], "big")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = int.from_bytes(raw[index + 5 : index + 7], "big")
            width = int.from_bytes(raw[index + 7 : index + 9], "big")
            return width, height
        index += 2 + max(length, 2)
    return None, None


def _decode_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="ignore").strip()


def _looks_text(raw: bytes) -> bool:
    if not raw:
        return True
    sample = raw[:2048]
    control_bytes = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return control_bytes / len(sample) < 0.08


def _numeric_hints(text: str) -> str:
    if not text:
        return ""
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    unique = []
    for number in numbers:
        if number not in unique:
            unique.append(number)
        if len(unique) >= 12:
            break
    return ", ".join(unique)


def _json_key_summary(data: object) -> str:
    if isinstance(data, dict):
        return ", ".join(str(key) for key in list(data.keys())[:16])
    if isinstance(data, list):
        return f"list[{len(data)}]"
    return type(data).__name__


def _json_status_summary(data: object) -> str:
    found: list[str] = []

    def visit(value: object, path: str = "") -> None:
        if len(found) >= 12:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if any(token in str(key).lower() for token in ("status", "state", "result", "valid", "error")):
                    found.append(f"{child_path}={_compact_json_value(child)}")
                    if len(found) >= 12:
                        return
                visit(child, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value[:4]):
                visit(item, f"{path}[{index}]")

    visit(data)
    return "; ".join(found)


def _compact_json_value(value: object) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _truncate(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:MAX_CONTEXT_CHARS]


def _format_prompt_context(filename: str, file_kind: str, summary: str, text: str | None) -> str:
    details = _truncate(text) if text else "No extractable text content."
    return (
        f"Uploaded {file_kind} context from {filename}:\n"
        f"{summary}\n"
        f"Extracted content:\n{details}"
    )
