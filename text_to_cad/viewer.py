"""Standalone browser viewer for generated Phase A STL files."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
from pathlib import Path
import struct

from text_to_cad.cad_templates import CadSpec


@dataclass(frozen=True)
class StlStats:
    triangle_count: int
    file_size_bytes: int
    is_binary: bool


def write_viewer(stl_path: Path, spec: CadSpec, output_dir: Path, output_name: str) -> Path:
    """Write a self-contained interactive HTML viewer for a generated STL."""

    stl_bytes = stl_path.read_bytes()
    stats = _stl_stats(stl_bytes)
    viewer_path = output_dir / "viewer.html"
    model_payload = {
        "name": output_name,
        "stl_file": stl_path.name,
        "step_file": f"{output_name}.step",
        "spec": json.loads(spec.to_json()),
        "stats": {
            "triangles": stats.triangle_count,
            "file_size_kb": round(stats.file_size_bytes / 1024.0, 1),
            "format": "binary STL" if stats.is_binary else "ASCII STL",
        },
    }
    viewer_path.write_text(
        _render_html(
            model_payload=model_payload,
            stl_base64=base64.b64encode(stl_bytes).decode("ascii"),
        ),
        encoding="utf-8",
    )
    return viewer_path


def _stl_stats(stl_bytes: bytes) -> StlStats:
    if len(stl_bytes) >= 84:
        triangle_count = struct.unpack_from("<I", stl_bytes, 80)[0]
        expected_binary_size = 84 + triangle_count * 50
        if expected_binary_size == len(stl_bytes):
            return StlStats(
                triangle_count=triangle_count,
                file_size_bytes=len(stl_bytes),
                is_binary=True,
            )

    text = stl_bytes[:2048].decode("utf-8", errors="ignore").lower()
    return StlStats(
        triangle_count=text.count("facet normal"),
        file_size_bytes=len(stl_bytes),
        is_binary=False,
    )


def _render_html(model_payload: dict, stl_base64: str) -> str:
    payload_json = json.dumps(model_payload, indent=2)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{model_payload["name"]} CAD viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f9fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #617084;
      --line: #d9e1ea;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      min-height: 100vh;
    }}
    .viewport {{
      position: relative;
      min-height: 100vh;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.9), rgba(235,241,247,0.85)),
        radial-gradient(circle at 50% 30%, rgba(15,118,110,0.08), transparent 42%);
      overflow: hidden;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100vh;
      cursor: grab;
    }}
    canvas:active {{ cursor: grabbing; }}
    .toolbar {{
      position: absolute;
      top: 16px;
      right: 16px;
      display: flex;
      gap: 8px;
    }}
    button {{
      height: 36px;
      min-width: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
      font: inherit;
      font-weight: 650;
      cursor: pointer;
      box-shadow: 0 8px 24px rgba(23,32,42,0.08);
    }}
    button:hover {{ border-color: var(--accent); color: var(--accent-strong); }}
    aside {{
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 24px;
      overflow-y: auto;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .subtitle {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 14px;
    }}
    .section {{
      padding: 18px 0;
      border-top: 1px solid var(--line);
    }}
    .section:first-of-type {{ border-top: 0; padding-top: 0; }}
    h2 {{
      margin: 0 0 12px;
      font-size: 13px;
      line-height: 1.2;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0;
    }}
    dl {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 9px 16px;
      margin: 0;
      font-size: 14px;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      text-align: right;
      font-weight: 650;
      color: var(--ink);
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: var(--accent-strong);
      word-break: break-word;
    }}
    .status {{
      position: absolute;
      left: 16px;
      bottom: 16px;
      color: var(--muted);
      font-size: 13px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
    }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; }}
      .viewport {{ min-height: 62vh; }}
      canvas {{ height: 62vh; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="viewport">
      <canvas id="viewer"></canvas>
      <div class="toolbar">
        <button id="zoomIn" title="Zoom in">+</button>
        <button id="zoomOut" title="Zoom out">-</button>
        <button id="reset" title="Reset view">Reset</button>
      </div>
      <div class="status" id="status">Loading model</div>
    </section>
    <aside>
      <h1 id="modelName"></h1>
      <p class="subtitle">Generated CAD model</p>
      <div class="section">
        <h2>Geometry</h2>
        <dl id="geometry"></dl>
      </div>
      <div class="section">
        <h2>Files</h2>
        <dl id="files"></dl>
      </div>
      <div class="section">
        <h2>Mesh</h2>
        <dl id="mesh"></dl>
      </div>
    </aside>
  </main>
  <script>
    const MODEL = {payload_json};
    const STL_BASE64 = "{stl_base64}";

    const canvas = document.getElementById("viewer");
    const statusEl = document.getElementById("status");
    let gl, program, buffers, indexCount = 0;
    let rotationX = -0.72;
    let rotationY = 0.68;
    let distance = 4.2;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;

    function initDetails() {{
      document.getElementById("modelName").textContent = MODEL.name;
      const spec = MODEL.spec;
      fillDefinitionList("geometry", [
        ["Type", spec.part_type],
        ["Material", spec.material_hint],
        ["Length", `${{spec.length_mm}} mm`],
        ["Width", `${{spec.width_mm}} mm`],
        ["Thickness", `${{spec.thickness_mm}} mm`],
        ["Holes", spec.hole_count],
        ["Hole diameter", `${{spec.hole_diameter_mm}} mm`],
      ]);
      fillDefinitionList("files", [
        ["STEP", `<code>${{MODEL.step_file}}</code>`],
        ["STL", `<code>${{MODEL.stl_file}}</code>`],
      ]);
      fillDefinitionList("mesh", [
        ["Format", MODEL.stats.format],
        ["Triangles", MODEL.stats.triangles.toLocaleString()],
        ["File size", `${{MODEL.stats.file_size_kb}} KB`],
      ]);
    }}

    function fillDefinitionList(id, rows) {{
      const dl = document.getElementById(id);
      dl.innerHTML = rows.map(([key, value]) => `<dt>${{key}}</dt><dd>${{value}}</dd>`).join("");
    }}

    function initViewer() {{
      gl = canvas.getContext("webgl", {{ antialias: true }});
      if (!gl) {{
        statusEl.textContent = "WebGL is not available in this browser";
        return;
      }}

      const mesh = parseStlBase64(STL_BASE64);
      program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER);
      buffers = createBuffers(mesh);
      indexCount = mesh.positions.length / 3;

      gl.enable(gl.DEPTH_TEST);
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.BACK);
      gl.clearColor(0.965, 0.976, 0.988, 1.0);
      bindControls();
      statusEl.textContent = `${{MODEL.stats.triangles.toLocaleString()}} triangles`;
      requestAnimationFrame(render);
    }}

    function parseStlBase64(base64) {{
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return parseBinaryStl(bytes.buffer);
    }}

    function parseBinaryStl(buffer) {{
      const view = new DataView(buffer);
      const triangles = view.getUint32(80, true);
      const positions = new Float32Array(triangles * 9);
      const normals = new Float32Array(triangles * 9);
      let min = [Infinity, Infinity, Infinity];
      let max = [-Infinity, -Infinity, -Infinity];
      let offset = 84;

      for (let t = 0; t < triangles; t++) {{
        const normal = [
          view.getFloat32(offset, true),
          view.getFloat32(offset + 4, true),
          view.getFloat32(offset + 8, true),
        ];
        offset += 12;
        for (let v = 0; v < 3; v++) {{
          const base = t * 9 + v * 3;
          const x = view.getFloat32(offset, true);
          const y = view.getFloat32(offset + 4, true);
          const z = view.getFloat32(offset + 8, true);
          positions[base] = x;
          positions[base + 1] = y;
          positions[base + 2] = z;
          normals[base] = normal[0];
          normals[base + 1] = normal[1];
          normals[base + 2] = normal[2];
          min = [Math.min(min[0], x), Math.min(min[1], y), Math.min(min[2], z)];
          max = [Math.max(max[0], x), Math.max(max[1], y), Math.max(max[2], z)];
          offset += 12;
        }}
        offset += 2;
      }}

      const center = [
        (min[0] + max[0]) / 2,
        (min[1] + max[1]) / 2,
        (min[2] + max[2]) / 2,
      ];
      const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
      const scale = 2.2 / size;
      for (let i = 0; i < positions.length; i += 3) {{
        positions[i] = (positions[i] - center[0]) * scale;
        positions[i + 1] = (positions[i + 1] - center[1]) * scale;
        positions[i + 2] = (positions[i + 2] - center[2]) * scale;
      }}
      return {{ positions, normals }};
    }}

    function createBuffers(mesh) {{
      const positionBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);

      const normalBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.normals, gl.STATIC_DRAW);
      return {{ positionBuffer, normalBuffer }};
    }}

    function bindControls() {{
      canvas.addEventListener("pointerdown", (event) => {{
        dragging = true;
        lastX = event.clientX;
        lastY = event.clientY;
        canvas.setPointerCapture(event.pointerId);
      }});
      canvas.addEventListener("pointermove", (event) => {{
        if (!dragging) return;
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        lastX = event.clientX;
        lastY = event.clientY;
        rotationY += dx * 0.01;
        rotationX += dy * 0.01;
      }});
      canvas.addEventListener("pointerup", () => dragging = false);
      canvas.addEventListener("wheel", (event) => {{
        event.preventDefault();
        distance = clamp(distance + Math.sign(event.deltaY) * 0.24, 2.1, 9.5);
      }}, {{ passive: false }});
      document.getElementById("zoomIn").addEventListener("click", () => distance = clamp(distance - 0.35, 2.1, 9.5));
      document.getElementById("zoomOut").addEventListener("click", () => distance = clamp(distance + 0.35, 2.1, 9.5));
      document.getElementById("reset").addEventListener("click", () => {{
        rotationX = -0.72;
        rotationY = 0.68;
        distance = 4.2;
      }});
      window.addEventListener("resize", resizeCanvas);
    }}

    function render() {{
      resizeCanvas();
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(program);

      const aspect = canvas.width / Math.max(1, canvas.height);
      const projection = perspective(Math.PI / 4, aspect, 0.1, 100);
      const view = translate(0, 0, -distance);
      const model = multiply(rotateX(rotationX), rotateY(rotationY));
      const mvp = multiply(projection, multiply(view, model));

      setMatrix("uMvp", mvp);
      setMatrix("uModel", model);
      setVec3("uLight", normalize([0.35, 0.6, 0.72]));

      bindAttribute("aPosition", buffers.positionBuffer, 3);
      bindAttribute("aNormal", buffers.normalBuffer, 3);
      gl.drawArrays(gl.TRIANGLES, 0, indexCount);
      requestAnimationFrame(render);
    }}

    function resizeCanvas() {{
      const width = canvas.clientWidth * window.devicePixelRatio;
      const height = canvas.clientHeight * window.devicePixelRatio;
      if (canvas.width !== width || canvas.height !== height) {{
        canvas.width = width;
        canvas.height = height;
        gl.viewport(0, 0, canvas.width, canvas.height);
      }}
    }}

    function bindAttribute(name, buffer, size) {{
      const location = gl.getAttribLocation(program, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    }}

    function setMatrix(name, matrix) {{
      gl.uniformMatrix4fv(gl.getUniformLocation(program, name), false, new Float32Array(matrix));
    }}

    function setVec3(name, vector) {{
      gl.uniform3fv(gl.getUniformLocation(program, name), new Float32Array(vector));
    }}

    function createProgram(vertexSource, fragmentSource) {{
      const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
      const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
      const linkedProgram = gl.createProgram();
      gl.attachShader(linkedProgram, vertexShader);
      gl.attachShader(linkedProgram, fragmentShader);
      gl.linkProgram(linkedProgram);
      if (!gl.getProgramParameter(linkedProgram, gl.LINK_STATUS)) {{
        throw new Error(gl.getProgramInfoLog(linkedProgram));
      }}
      return linkedProgram;
    }}

    function compileShader(type, source) {{
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
        throw new Error(gl.getShaderInfoLog(shader));
      }}
      return shader;
    }}

    function perspective(fov, aspect, near, far) {{
      const f = 1 / Math.tan(fov / 2);
      const nf = 1 / (near - far);
      return [
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, (2 * far * near) * nf, 0,
      ];
    }}

    function translate(x, y, z) {{
      return [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        x, y, z, 1,
      ];
    }}

    function rotateX(angle) {{
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      return [
        1, 0, 0, 0,
        0, c, s, 0,
        0, -s, c, 0,
        0, 0, 0, 1,
      ];
    }}

    function rotateY(angle) {{
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      return [
        c, 0, -s, 0,
        0, 1, 0, 0,
        s, 0, c, 0,
        0, 0, 0, 1,
      ];
    }}

    function multiply(a, b) {{
      const result = new Array(16).fill(0);
      for (let row = 0; row < 4; row++) {{
        for (let col = 0; col < 4; col++) {{
          for (let k = 0; k < 4; k++) {{
            result[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
          }}
        }}
      }}
      return result;
    }}

    function normalize(vector) {{
      const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
      return [vector[0] / length, vector[1] / length, vector[2] / length];
    }}

    function clamp(value, min, max) {{
      return Math.max(min, Math.min(max, value));
    }}

    const VERTEX_SHADER = `
      attribute vec3 aPosition;
      attribute vec3 aNormal;
      uniform mat4 uMvp;
      uniform mat4 uModel;
      varying vec3 vNormal;
      varying vec3 vPosition;
      void main() {{
        vNormal = mat3(uModel) * aNormal;
        vPosition = aPosition;
        gl_Position = uMvp * vec4(aPosition, 1.0);
      }}
    `;

    const FRAGMENT_SHADER = `
      precision mediump float;
      uniform vec3 uLight;
      varying vec3 vNormal;
      varying vec3 vPosition;
      void main() {{
        vec3 normal = normalize(vNormal);
        float diffuse = max(dot(normal, uLight), 0.0);
        float rim = pow(1.0 - max(dot(normal, normalize(vec3(0.0, 0.0, 1.0))), 0.0), 2.0);
        vec3 base = vec3(0.08, 0.47, 0.43);
        vec3 color = base * (0.42 + diffuse * 0.72) + vec3(0.10, 0.16, 0.20) * rim;
        gl_FragColor = vec4(color, 1.0);
      }}
    `;

    try {{
      initDetails();
      initViewer();
    }} catch (error) {{
      console.error(error);
      statusEl.textContent = `Viewer error: ${{error.message}}`;
    }}
  </script>
</body>
</html>
"""
