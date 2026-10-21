"""
ROLLBACK - fully implemented inside C4 (not a stub).

Moves the MLflow alias 'production' from the current version to the 'previous' version.
The runner loads the model by alias every window, so the next window is already
served by the previous model.
"""
from src.common import registry


def rollback(reason: str) -> dict:
    try:
        old, new = registry.rollback(reason)
        return {"ok": True, "from_version": old, "to_version": new}
    except Exception as e:  # e.g. already on the previous version
        return {"ok": False, "error": str(e)}
