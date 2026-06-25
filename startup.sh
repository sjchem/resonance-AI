#!/usr/bin/env bash
set -e

export PYTHONPATH="/home/site/wwwroot/.python_packages/lib/site-packages:${PYTHONPATH:-}"

gunicorn -w "${WEB_CONCURRENCY:-2}" \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:"${PORT:-8000}" \
  --timeout "${GUNICORN_TIMEOUT:-600}" \
  main:app
