#!/usr/bin/env bash
# JupyterLab for the notebook buttons in the dashboard -> http://localhost:8888
# Local demo only: bound to 127.0.0.1 and started without a token so the dashboard links open directly.
cd "$(dirname "$0")/.."
exec .venv/bin/jupyter lab --no-browser --ip 127.0.0.1 --port 8888 \
  --IdentityProvider.token='' --ServerApp.password='' --ServerApp.root_dir="$(pwd)"
