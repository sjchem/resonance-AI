# FEM-capable container for the Resonance AI web app.
#
# Bundles the native CalculiX 'ccx' solver plus the CadQuery / gmsh / VTK /
# PyVista stack so the /run-fem endpoint can mesh, solve, and render a real
# von Mises stress contour image in production (Azure Web App for Containers).
#
# Headless rendering: PyVista/VTK need an OpenGL context. The image installs
# Mesa + an X virtual framebuffer (xvfb) and runs gunicorn under xvfb-run.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYVISTA_OFF_SCREEN=true \
    DISPLAY=:99 \
    PORT=8000 \
    PYTHONPATH=/app

# Native packages:
#   calculix-ccx      -> the 'ccx' finite-element solver (Debian bookworm main)
#   libglu1-mesa/libgl1 + Mesa -> OpenGL for gmsh and VTK
#   libx* / xvfb      -> headless X context for off-screen PyVista rendering
# Split into two installs so a missing CalculiX package fails loudly and early.
RUN apt-get update && apt-get install -y --no-install-recommends \
        calculix-ccx \
    && apt-get install -y --no-install-recommends \
        libglu1-mesa \
        libgl1 \
        libgl1-mesa-dri \
        libxrender1 \
        libxext6 \
        libsm6 \
        libxt6 \
        libxcursor1 \
        libxinerama1 \
        libxfixes3 \
        libxft2 \
        libxi6 \
        libxrandr2 \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements-web-fem.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements-web-fem.txt

# Application code (everything the entrypoint + /run-fem pipeline imports).
COPY main.py ./
COPY backend/ ./backend/
COPY text_to_cad/ ./text_to_cad/
COPY simulate/ ./simulate/
COPY geometry/ ./geometry/
COPY skills/ ./skills/

EXPOSE 8000

# xvfb-run provides the GL context; gunicorn serves the FastAPI app (main:app
# adds backend/ to sys.path and exposes app.main:app).
CMD ["sh", "-c", "xvfb-run -a --server-args='-screen 0 1280x1024x24' gunicorn -w ${WEB_CONCURRENCY:-2} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000} --timeout ${GUNICORN_TIMEOUT:-600} main:app"]
