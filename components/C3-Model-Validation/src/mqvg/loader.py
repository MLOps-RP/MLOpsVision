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

from mqvg.config import ModelSafetyConfig
from mqvg.model_safety import DANGEROUS, UNRECOGNISED, ScanError, ScanReport, scan_model_file
from mqvg.schemas import CandidateBundle, TrainingMetadata

BYTES_PER_MB = 1024 * 1024
MEMORY_PROBE_ROWS = 200

# Why a model could not be loaded. The first two say something about the
# artifact; the last two say only that THIS gate cannot read it, which is not
# a statement about the model's quality (so they are UNAVAILABLE, not FAIL).
NOT_FOUND = "not_found"
UNREADABLE = "unreadable"
CORRUPT = "corrupt"
NOT_A_MODEL = "not_a_model"
UNSUPPORTED_FORMAT = "unsupported_format"
MISSING_DEPENDENCY = "missing_dependency"
# Set only when the file-safety scan is on. BLOCKED_UNSAFE: the scan found dangerous
# code references, so the file was not loaded. BLOCKED_REVIEW: it found code the
# allowlist does not recognise (or could not scan the file), so a person must decide.
BLOCKED_UNSAFE = "blocked_unsafe"
BLOCKED_REVIEW = "blocked_review"
CANNOT_ASSESS = {UNSUPPORTED_FORMAT, MISSING_DEPENDENCY, BLOCKED_REVIEW}

SUPPORTED_SUFFIXES = (".joblib", ".pkl", ".pickle")
# Formats the gate refuses to load (and never executes).
UNSUPPORTED_SUFFIXES = {
    ".onnx": "ONNX", ".h5": "Keras/TensorFlow", ".hdf5": "Keras/TensorFlow",
    ".keras": "Keras", ".pb": "TensorFlow", ".tflite": "TensorFlow Lite",
    ".pt": "PyTorch", ".pth": "PyTorch", ".ckpt": "PyTorch/TensorFlow checkpoint",
    ".safetensors": "safetensors", ".xgb": "XGBoost", ".ubj": "XGBoost",
    ".cbm": "CatBoost", ".pmml": "PMML", ".mlmodel": "Core ML", ".gguf": "GGUF",
}


@dataclass
class LoadedCandidate:
    """A candidate plus the outcome of loading each of its parts."""
    bundle: CandidateBundle
    model: Any | None = None
    model_error: str | None = None
    model_failure: str | None = None   # one of the kind constants above
    load_warnings: list[str] = field(default_factory=list)
    model_size_bytes: int | None = None
    model_sha256: str | None = None
    memory_delta_mb: float | None = None
    safety_scan: ScanReport | None = None     # set when the file-safety scan ran
    safety_error: str | None = None           # set when it ran but could not read the file
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


def _fail(candidate: LoadedCandidate, kind: str, message: str) -> None:
    candidate.model_failure, candidate.model_error = kind, message


def _classify_load_error(exc: BaseException) -> tuple[str, str]:
    """(kind, message) for an exception raised while loading a model file."""
    scope = ("This says nothing about the model's quality; the gate simply "
             "cannot load it here.")
    if isinstance(exc, ImportError):
        name = getattr(exc, "name", None) or str(exc)
        return MISSING_DEPENDENCY, (
            f"The model needs the Python module '{name}', which is not "
            f"installed in the gate's environment. {scope}")
    if isinstance(exc, AttributeError) and "attribute" in str(exc):
        return MISSING_DEPENDENCY, (
            f"The model refers to code that is not available in the gate's "
            f"environment ({_describe(exc)}). {scope}")
    return CORRUPT, f"cannot load model: {_describe(exc)}"


def _safety_gate(candidate: LoadedCandidate, safety: ModelSafetyConfig, path: Path) -> bool:
    """Scan the file without executing it. True means: do not load it.

    Records the outcome on the candidate. A file the scanner cannot parse is
    never handed to the real loader, because a file one parser rejects and
    another accepts is exactly what an attacker looks for.
    """
    try:
        report = scan_model_file(path, safety)
    except ScanError as exc:
        candidate.safety_error = str(exc)
        if safety.refuse_at == "never":
            return False
        if exc.parse:
            _fail(candidate, CORRUPT, f"cannot load model: it could not be parsed safely "
                                      f"and was not loaded ({exc})")
        else:
            _fail(candidate, BLOCKED_REVIEW, f"the model file was not loaded because the "
                  f"safety scan could not read it ({exc}). This says nothing about the "
                  "model's quality.")
        return True
    candidate.safety_scan = report
    refuse = {"dangerous": DANGEROUS, "unrecognised": UNRECOGNISED}.get(safety.refuse_at)
    if refuse is None or report.severity < refuse:
        return False
    if report.dangerous:
        _fail(candidate, BLOCKED_UNSAFE, (
            "the model file was NOT loaded: the safety scan found dangerous code "
            f"references ({', '.join(report.dangerous[:6])}) that loading would execute."))
    else:
        _fail(candidate, BLOCKED_REVIEW, (
            "the model file was not loaded: it references code the allowlist does not "
            f"recognise ({', '.join(report.unrecognised[:6])}), which loading would "
            "execute. If you trust it, add its module to model_safety.allowed_module_prefixes. "
            "This says nothing about the model's quality."))
    return True


def _load_model(candidate: LoadedCandidate, safety: ModelSafetyConfig | None = None) -> None:
    path = candidate.bundle.model_path
    if not path.is_file():
        _fail(candidate, NOT_FOUND, f"model artifact not found: {path}")
        return
    candidate.model_size_bytes = path.stat().st_size
    try:
        candidate.model_sha256 = file_sha256(path)
    except OSError as exc:
        _fail(candidate, UNREADABLE,
              f"cannot read model artifact: {_describe(exc)}")
        return

    suffix = path.suffix.lower()
    if suffix in UNSUPPORTED_SUFFIXES:
        _fail(candidate, UNSUPPORTED_FORMAT, (
            f"'{suffix}' files ({UNSUPPORTED_SUFFIXES[suffix]}) are not "
            "supported: the gate loads Python models saved with joblib or "
            f"pickle ({', '.join(SUPPORTED_SUFFIXES)}). This says nothing "
            "about the model's quality."))
        return

    if safety is not None and _safety_gate(candidate, safety, path):
        return

    gc.collect()
    before = _rss_mb()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = joblib.load(path)
        candidate.load_warnings = [str(w.message) for w in caught]
    except Exception as exc:
        _fail(candidate, *_classify_load_error(exc))
        return
    if not hasattr(model, "predict"):
        _fail(candidate, NOT_A_MODEL,
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


def load_candidate(bundle: CandidateBundle,
                   safety: ModelSafetyConfig | None = None) -> LoadedCandidate:
    """Load every part of a candidate; problems are recorded, never raised.

    With `safety`, the model file is scanned (not executed) first and is not
    loaded if the scan refuses it. Without it, behaviour is unchanged.
    """
    candidate = LoadedCandidate(bundle=bundle)
    _load_eval(candidate)
    _load_metadata(candidate)
    _load_model(candidate, safety)
    return candidate
