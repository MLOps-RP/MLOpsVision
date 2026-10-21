"""Run the API:  python -m mqvg.api

Settings come from MQVG_* environment variables (see docs/api.md). The server binds
to the loopback interface; to listen elsewhere set MQVG_HOST, which also requires
MQVG_API_KEY.
"""
from __future__ import annotations

import os
import sys

from mqvg.api.settings import Settings, SettingsError

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def main() -> int:
    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2
    host = os.environ.get("MQVG_HOST", "127.0.0.1")
    port = int(os.environ.get("MQVG_PORT", "8000"))
    if host not in LOOPBACK and not settings.api_key:
        print("cannot start: listening on a non-loopback address needs MQVG_API_KEY", file=sys.stderr)
        return 2
    if settings.api_key is None and not settings.allow_no_auth:
        print("note: MQVG_API_KEY is not set, so every endpoint except /health will answer 503. "
              "Set a key, or MQVG_ALLOW_NO_AUTH=true for local development.", file=sys.stderr)
    import uvicorn
    from mqvg.api.app import create_app
    try:
        app = create_app(settings)
    except SettingsError as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
