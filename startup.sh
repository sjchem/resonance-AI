#!/usr/bin/env bash
set -e

gunicorn -w "${WEB_CONCURRENCY:-2}" \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:"${PORT:-8000}" \
  --timeout "${GUNICORN_TIMEOUT:-600}" \
  main:app
