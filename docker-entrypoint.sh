#!/usr/bin/env bash
# Container entrypoint for the FEM-capable web app.
#
# Starts a background X virtual framebuffer (so headless VTK/PyVista rendering
# in /run-fem has a GL context) and then runs gunicorn on a fixed port that
# matches WEBSITES_PORT / EXPOSE. The web server starts even if Xvfb fails, so a
# rendering issue never takes the whole site down (the endpoint degrades).
set -u

# Headless display for VTK/PyVista. Backgrounded: if it dies, the API still runs.
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
export DISPLAY=:99

exec gunicorn \
  -w "${WEB_CONCURRENCY:-2}" \
  -k uvicorn.workers.UvicornWorker \
  -b "0.0.0.0:8000" \
  --timeout "${GUNICORN_TIMEOUT:-600}" \
  --access-logfile - \
  --error-logfile - \
  main:app
