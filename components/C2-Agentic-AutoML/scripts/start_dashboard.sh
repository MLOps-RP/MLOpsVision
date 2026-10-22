#!/usr/bin/env bash
# Start everything for a demo: API (:8000), JupyterLab (:8888) and the React dashboard (:5173).
# Stop with Ctrl+C - all three are stopped together.
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT
./scripts/start_api.sh &
./scripts/start_jupyter.sh > outputs/jupyter.log 2>&1 &
(cd frontend && [ -d node_modules ] || npm install) && (cd frontend && npm run dev -- --open) &
wait
