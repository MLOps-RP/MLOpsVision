"""Facts about the machine and libraries a validation ran on.

Latency, memory and size results depend on the hardware, so they are only
meaningful next to a machine profile.
"""
from __future__ import annotations

import os
import platform
from importlib import metadata


def current_versions() -> dict[str, str]:
    """Versions keyed like the training metadata's `versions` field."""
    versions = {"python": platform.python_version()}
    for key, package in (("scikit_learn", "scikit-learn"), ("pandas", "pandas"),
                         ("numpy", "numpy"), ("joblib", "joblib")):
        try:
            versions[key] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return versions


def machine_profile() -> dict:
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:  # profile is informational; never fail on it
        ram_gb = None
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
        "versions": current_versions(),
    }
