"""Load a candidate's artifacts and record what could and could not be loaded.

Loading never raises for a bad artifact: a missing file, a corrupt model or
unparseable metadata is a validation finding, so the problem is stored on the
LoadedCandidate and reported by the checks.

SECURITY: joblib/pickle loading executes code embedded in the file. Only
load models from trusted local paths. The prototype performs no sandboxing.
"""
from __future__ import annotations

import gc
import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from mqvg.schemas import CandidateBundle, TrainingMetadata

BYTES_PER_MB = 1024 * 1024
MEMORY_PROBE_ROWS = 200


@dataclass
class LoadedCandidate:
    """A candidate plus the outcome of loading each of its parts."""
    bundle: CandidateBundle
    model: Any | None = None
    model_error: str | None = None
    load_warnings: list[str] = field(default_factory=list)
    model_size_bytes: int | None = None
    model_sha256: str | None = None
    memory_delta_mb: float | None = None
    eval_df: pd.DataFrame | None = None
    eval_error: str | None = None
    eval_sha256: str | None = None
    metadata: TrainingMetadata | None = None
    metadata_provided: bool = False
    metadata_error: str | None = None
    metadata_sha256: str | None = None
    # Results shared between checks (for example the evaluation predictions).
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str] | None:
        """Columns the model expects: from the model, else the metadata."""
        names = getattr(self.model, "feature_names_in_", None)
        if names is not None:
            return [str(n) for n in names]
        if self.metadata is not None and self.metadata.feature_columns:
            return list(self.metadata.feature_columns)
        return None

    @property
    def target_column(self) -> str | None:
        """Bundle override first, then the metadata."""
        if self.bundle.target_column is not None:
            return self.bundle.target_column
        if self.metadata is not None:
            return self.metadata.target_column
        return None

    @property
    def sensitive_columns(self) -> list[str] | None:
        """Bundle override first, then the metadata; None if undeclared."""
        if self.bundle.sensitive_columns is not None:
            return list(self.bundle.sensitive_columns)
        if self.metadata is not None and self.metadata.sensitive_columns:
            return list(self.metadata.sensitive_columns)
        return None

    @property
    def positive_label(self) -> str | int | bool | None:
        """Bundle override first, then the metadata; None if undeclared."""
        if self.bundle.positive_label is not None:
            return self.bundle.positive_label
        if self.metadata is not None:
            return self.metadata.positive_label
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / BYTES_PER_MB
    except Exception:
        return None


def _load_eval(candidate: LoadedCandidate) -> None:
    path = candidate.bundle.eval_data_path
    if not path.is_file():
        candidate.eval_error = f"evaluation data not found: {path}"
        return
    try:
        candidate.eval_df = pd.read_csv(path)
        candidate.eval_sha256 = file_sha256(path)
    except Exception as exc:
        candidate.eval_error = f"cannot read evaluation data: {_describe(exc)}"
        return
    if candidate.eval_df.empty:
        candidate.eval_error = "evaluation data has no rows"
        candidate.eval_df = None


def _load_metadata(candidate: LoadedCandidate) -> None:
    path = candidate.bundle.metadata_path
    if path is None:
        return
    candidate.metadata_provided = True
    if not path.is_file():
        candidate.metadata_error = f"metadata file not found: {path}"
        return
    try:
        text = path.read_text(encoding="utf-8")
        raw = (yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml")
               else json.loads(text))
        candidate.metadata = TrainingMetadata.model_validate(raw)
        candidate.metadata_sha256 = file_sha256(path)
    except Exception as exc:
        candidate.metadata_error = f"cannot parse metadata: {_describe(exc)}"


def _load_model(candidate: LoadedCandidate) -> None:
    path = candidate.bundle.model_path
    if not path.is_file():
        candidate.model_error = f"model artifact not found: {path}"
        return
    candidate.model_size_bytes = path.stat().st_size
    try:
        candidate.model_sha256 = file_sha256(path)
    except OSError as exc:
        candidate.model_error = f"cannot read model artifact: {_describe(exc)}"
        return

    gc.collect()
    before = _rss_mb()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        candidate.load_warnings = [str(w.message) for w in caught]
    except Exception as exc:
        candidate.model_error = f"cannot load model: {_describe(exc)}"
        return
    if not hasattr(model, "predict"):
        candidate.model_error = (
            f"loaded object of type {type(model).__name__} has no predict method")
        return
    candidate.model = model

    # Memory: growth in resident memory from loading and one small prediction.
    # Noisy, and includes lazy library imports on the first load in a process.
    features = candidate.feature_columns
    if candidate.eval_df is not None and features and set(features) <= set(
            candidate.eval_df.columns):
        try:
            model.predict(candidate.eval_df[features].head(MEMORY_PROBE_ROWS))
        except Exception:
            pass  # reported by the input-schema check, not here
    after = _rss_mb()
    if before is not None and after is not None:
        candidate.memory_delta_mb = max(0.0, after - before)


def load_candidate(bundle: CandidateBundle) -> LoadedCandidate:
    """Load every part of a candidate; problems are recorded, never raised."""
    candidate = LoadedCandidate(bundle=bundle)
    _load_eval(candidate)
    _load_metadata(candidate)
    _load_model(candidate)
    return candidate
