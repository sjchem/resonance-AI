#!/usr/bin/env bash
# Container entrypoint for the FEM-capable web app.
#
# Starts a background X virtual framebuffer (so headless VTK/PyVista rendering
# in /run-fem has a GL context) and then runs gunicorn on a fixed port that
# matches WEBSITES_PORT / EXPOSE. The web server starts even if Xvfb fails, so a
# rendering issue never takes the whole site down (the endpoint degrades).
set -u

# Startup diagnostics (captured in the Azure docker.log). If the container
# exits early these lines pinpoint which step failed.
echo "[entrypoint] starting; PATH=$PATH"
echo "[entrypoint] python: $(command -v python || echo MISSING)"
echo "[entrypoint] gunicorn module: $(python -c 'import gunicorn, sys; print(gunicorn.__file__)' 2>&1 || echo MISSING)"
echo "[entrypoint] Xvfb: $(command -v Xvfb || echo MISSING)"

# Headless display for VTK/PyVista. Backgrounded: if it dies, the API still runs.
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
  export DISPLAY=:99
else
  echo "[entrypoint] WARNING: Xvfb not found; off-screen rendering may fail."
fi

# Run gunicorn as a module so startup never depends on the console-script path
# being resolvable (avoids exec 'command not found' / exit 127).
echo "[entrypoint] launching gunicorn on 0.0.0.0:8000"
exec python -m gunicorn \
  -w "${WEB_CONCURRENCY:-1}" \
  -k uvicorn.workers.UvicornWorker \
  -b "0.0.0.0:8000" \
  --timeout "${GUNICORN_TIMEOUT:-600}" \
  --access-logfile - \
  --error-logfile - \
  main:app

