"""The study reports the API can serve: fixed names, read-only, from one folder.

A "study" is a report produced by a helper script (`scripts/baseline_comparison.py`,
`scripts/sensitivity_analysis.py`, `scripts/evaluate_hidden_groups.py`). The API serves
what those scripts wrote and never runs them. A client cannot name a file: only the ids
below exist, each mapping to constant file names inside `docs_dir`, so there is no path
to traverse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudySpec:
    id: str
    title: str
    description: str
    data_file: str | None      # machine-readable JSON, or None if the script writes only Markdown
    report_file: str
    script: str                # what to run to (re)generate it


STUDIES: dict[str, StudySpec] = {s.id: s for s in (
    StudySpec("baseline_comparison", "Baseline comparison",
              "A plain metrics gate against the full gate on the same sample models.",
              "baseline_comparison.json", "baseline_comparison.md", "scripts/baseline_comparison.py"),
    StudySpec("sensitivity_analysis", "Sensitivity analysis",
              "How stable each recommendation is when weights, thresholds and bands change.",
              "sensitivity_analysis.json", "sensitivity_analysis.md", "scripts/sensitivity_analysis.py"),
    StudySpec("hidden_groups_results", "Hidden failure group evaluation",
              "Detection power and false alarms of hidden failure group detection on synthetic data.",
              None, "hidden_groups_results.md", "scripts/evaluate_hidden_groups.py"),
)}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def generated_at(docs_dir: Path, spec: StudySpec) -> datetime | None:
    """When the study was generated, from its JSON (None if unknown)."""
    if spec.data_file is None:
        return None
    data = _read_json(docs_dir / spec.data_file)
    stamp = None if data is None else data.get("generated_at_utc")
    try:
        return datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
    except ValueError:
        return None


def load_data(docs_dir: Path, spec: StudySpec) -> dict[str, Any] | None:
    return None if spec.data_file is None else _read_json(docs_dir / spec.data_file)


def load_report(docs_dir: Path, spec: StudySpec) -> str | None:
    try:
        return (docs_dir / spec.report_file).read_text(encoding="utf-8")
    except OSError:
        return None
