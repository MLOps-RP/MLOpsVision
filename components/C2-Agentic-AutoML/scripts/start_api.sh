#!/usr/bin/env bash
# FastAPI backend for the dashboard -> http://127.0.0.1:8000  (API docs: /docs)
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn c2_automl.api:app --host 127.0.0.1 --port 8000 --reload --reload-dir src
