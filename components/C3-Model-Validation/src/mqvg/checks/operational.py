"""Operational readiness checks: can this artifact be used safely?

Everything here runs without a deployment environment, using only the model
file, the evaluation data and (for one check) the declared metadata.
Timing and memory results depend on the machine they ran on.
"""
from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from mqvg.catalogue import CATALOGUE
from mqvg.checks.predictions import prediction_blocker
from mqvg.checks.base import (
    CheckContext, CheckFunction, binary_result, guarded, scored_result,
    unavailable_result,
)
from mqvg.config import GateConfig
from mqvg.environment import current_versions, machine_profile
from mqvg.loader import BYTES_PER_MB, CANNOT_ASSESS, LoadedCandidate
from mqvg.model_safety import MAX_REFS_REPORTED
from mqvg.schemas import CheckResult, Dimension, EvidenceSource

SAMPLE_ROWS = 1000

OPERATIONAL_CHECKS: dict[str, CheckFunction] = {}


def _register(check_id: str) -> Callable[[CheckFunction], CheckFunction]:
    def decorator(func: CheckFunction) -> CheckFunction:
        OPERATIONAL_CHECKS[check_id] = guarded(check_id)(func)
        return OPERATIONAL_CHECKS[check_id]
    return decorator


def _sample(candidate: LoadedCandidate) -> pd.DataFrame:
    return candidate.eval_df[candidate.feature_columns].head(SAMPLE_ROWS)


# --------------------------------------------------------------------------
@_register("operational.model_loadable")
def model_loadable(ctx: CheckContext) -> CheckResult:
    c = ctx.candidate
    check_id = "operational.model_loadable"
    if c.model is not None:
        extra = f" with {len(c.load_warnings)} warning(s)" if c.load_warnings else ""
        return binary_result(
            ctx, check_id, True,
            f"Model loaded as {type(c.model).__name__}{extra}.",
            details={"model_type": type(c.model).__name__,
                     "load_warnings": c.load_warnings,
                     "model_sha256": c.model_sha256})
    details = {"error": c.model_error, "failure_kind": c.model_failure}
    if c.model_failure in CANNOT_ASSESS:
        # The gate cannot read this model; that is not evidence the model is bad.
        return unavailable_result(ctx, check_id, str(c.model_error),
                                  details=details)
    return binary_result(ctx, check_id, False, str(c.model_error),
                         details=details)


@_register("operational.input_schema")
def input_schema(ctx: CheckContext) -> CheckResult:
    c = ctx.candidate
    check_id = "operational.input_schema"
    if c.model is None:
        return unavailable_result(ctx, check_id, "Cannot test: " + prediction_blocker(c))
    if c.eval_df is None:
        return unavailable_result(ctx, check_id, "Cannot test: " + prediction_blocker(c))
    features = c.feature_columns
    if features is None:
        return unavailable_result(ctx, check_id, "Cannot test: " + prediction_blocker(c))
    missing = [col for col in features if col not in c.eval_df.columns]
    if missing:
        return binary_result(
            ctx, check_id, False,
            f"Evaluation data is missing model feature columns {missing}.",
            details={"missing_columns": missing})
    sample = _sample(c)
    try:
        predictions = np.asarray(c.model.predict(sample))
    except Exception as exc:  # the model rejecting the schema is the finding
        return binary_result(
            ctx, check_id, False,
            f"Model failed to predict on the evaluation data: "
            f"{type(exc).__name__}: {exc}",
            details={"exception": type(exc).__name__})
    if len(predictions) != len(sample):
        return binary_result(
            ctx, check_id, False,
            f"Model returned {len(predictions)} predictions for "
            f"{len(sample)} rows.")
    return binary_result(
        ctx, check_id, True,
        f"Model predicted on {len(sample)} evaluation rows using "
        f"{len(features)} feature columns.",
        details={"rows_tested": len(sample), "feature_columns": features})


@_register("operational.prediction_determinism")
def prediction_determinism(ctx: CheckContext) -> CheckResult:
    check_id = "operational.prediction_determinism"
    reason = prediction_blocker(ctx.candidate)
    if reason:
        return unavailable_result(ctx, check_id, "Cannot test: " + reason)
    model, sample = ctx.candidate.model, _sample(ctx.candidate)
    first, second = np.asarray(model.predict(sample)), np.asarray(model.predict(sample))
    same = bool(np.array_equal(first, second))
    probabilities_checked = False
    if same and hasattr(model, "predict_proba"):
        probabilities_checked = True
        same = bool(np.array_equal(np.asarray(model.predict_proba(sample)),
                                   np.asarray(model.predict_proba(sample))))
    message = ("Repeated predictions on the same rows were identical."
               if same else
               "Repeated predictions on the same rows differed.")
    return binary_result(ctx, check_id, same, message,
                         details={"rows_tested": len(sample),
                                  "probabilities_checked": probabilities_checked})


@_register("operational.malformed_input_handling")
def malformed_input_handling(ctx: CheckContext) -> CheckResult:
    """The model must reject malformed input, not quietly predict anyway.

    Limitation: this runs in-process, so a hard crash (for example a
    segmentation fault) would end the whole run and cannot be detected here.
    """
    check_id = "operational.malformed_input_handling"
    reason = prediction_blocker(ctx.candidate)
    if reason:
        return unavailable_result(ctx, check_id, "Cannot test: " + reason)
    model, sample = ctx.candidate.model, _sample(ctx.candidate)
    features = list(sample.columns)

    def outcome(frame: pd.DataFrame) -> tuple[str, int | None]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return "returned", len(np.asarray(model.predict(frame)))
            except Exception:  # noqa: BLE001 - rejection is the good outcome
                return "rejected", None

    cases: dict[str, bool] = {}
    status, _ = outcome(sample.drop(columns=[features[0]]).head(5))
    cases["missing_column_rejected"] = status == "rejected"

    numeric = [c for c in features if pd.api.types.is_numeric_dtype(sample[c])]
    if numeric:
        bad = sample.head(5).copy()
        bad[numeric[0]] = "not-a-number"
        status, _ = outcome(bad)
        cases["non_numeric_value_rejected"] = status == "rejected"

    status, n_rows = outcome(sample.iloc[0:0])
    cases["empty_input_handled"] = status == "rejected" or n_rows == 0

    failed = [name for name, ok in cases.items() if not ok]
    if failed:
        return binary_result(
            ctx, check_id, False,
            f"Malformed input was accepted or mishandled: {failed}.",
            details={"cases": cases})
    return binary_result(ctx, check_id, True,
                         f"All {len(cases)} malformed-input cases were handled.",
                         details={"cases": cases})


@_register("operational.artifact_size")
def artifact_size(ctx: CheckContext) -> CheckResult:
    check_id = "operational.artifact_size"
    size = ctx.candidate.model_size_bytes
    if size is None:
        return unavailable_result(
            ctx, check_id, f"No model file to measure ({ctx.candidate.model_error}).")
    megabytes = size / BYTES_PER_MB
    shown = f"{size / 1024:.1f} KB" if size < BYTES_PER_MB else f"{megabytes:.1f} MB"
    return scored_result(ctx, check_id, megabytes, f"Model file is {shown}.",
                         details={"bytes": size})


@_register("operational.latency")
def latency(ctx: CheckContext) -> CheckResult:
    check_id = "operational.latency"
    reason = prediction_blocker(ctx.candidate)
    if reason:
        return unavailable_result(ctx, check_id, "Cannot test: " + reason)
    evaluation = ctx.config.evaluation
    rows = ctx.candidate.eval_df[ctx.candidate.feature_columns]
    n = min(evaluation.latency_repeats, len(rows))
    model = ctx.candidate.model

    for i in range(evaluation.latency_warmup_runs):
        model.predict(rows.iloc[[i % len(rows)]])
    timings_ms = []
    for i in range(n):  # a different row per call, so nothing is cached
        row = rows.iloc[[i]]
        start = time.perf_counter()
        model.predict(row)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    p50, p95, p99 = (float(np.percentile(timings_ms, q)) for q in (50, 95, 99))
    return scored_result(
        ctx, check_id, p95,
        f"Single-row prediction p95 latency is {p95:.1f} ms "
        f"(p50 {p50:.1f} ms) over {n} calls.",
        details={"p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
                 "mean_ms": float(np.mean(timings_ms)),
                 "max_ms": float(np.max(timings_ms)), "calls": n,
                 "warmup_runs": evaluation.latency_warmup_runs,
                 "machine": machine_profile()})


@_register("operational.memory")
def memory(ctx: CheckContext) -> CheckResult:
    check_id = "operational.memory"
    delta = ctx.candidate.memory_delta_mb
    if ctx.candidate.model is None:
        return unavailable_result(
            ctx, check_id,
            f"Model could not be loaded ({ctx.candidate.model_error}).")
    if delta is None:
        return unavailable_result(ctx, check_id,
                                  "Resident memory could not be measured.")
    return scored_result(
        ctx, check_id, delta,
        f"Loading the model and predicting on a small sample grew memory by "
        f"{delta:.0f} MB.",
        details={"note": "Resident-memory growth; noisy and includes library "
                         "imports on first load.",
                 "machine": machine_profile()})


@_register("operational.dependency_match")
def dependency_match(ctx: CheckContext) -> CheckResult:
    check_id = "operational.dependency_match"
    meta = ctx.candidate.metadata
    declared = None if meta is None else meta.versions
    if not declared:
        return unavailable_result(
            ctx, check_id, "No library versions declared in the training metadata.")
    current = current_versions()

    def same(key: str) -> bool:
        have, want = current.get(key), declared[key]
        if key == "python":  # patch releases are compatible
            return have is not None and have.split(".")[:2] == want.split(".")[:2]
        return have == want

    comparisons = {key: {"declared": declared[key], "current": current.get(key),
                         "match": same(key)} for key in declared}
    matched = sum(item["match"] for item in comparisons.values())
    fraction = matched / len(comparisons)
    mismatches = [k for k, v in comparisons.items() if not v["match"]]
    message = (f"{matched} of {len(comparisons)} declared library versions "
               "match the current environment."
               + (f" Mismatched: {mismatches}." if mismatches else ""))
    return scored_result(ctx, check_id, fraction, message,
                         evidence=EvidenceSource.DECLARED,
                         details={"comparisons": comparisons})


@_register("operational.model_file_safety")
def model_file_safety(ctx: CheckContext) -> CheckResult:
    check_id = "operational.model_file_safety"
    c = ctx.candidate
    report = c.safety_scan
    if report is None:
        if c.safety_error:
            return unavailable_result(
                ctx, check_id,
                f"The model file could not be scanned: {c.safety_error}. It is not counted "
                "as safe.", details={"error": c.safety_error})
        return unavailable_result(
            ctx, check_id,
            "The model file was not scanned (" + (c.model_error or "the scan did not run")
            + "). It is not counted as safe.")
    shown = ", ".join((report.dangerous or report.unrecognised)[:5])
    if report.severity == 0:
        message = (f"The scan found {len(report.references)} code references in the model "
                   "file, all on the allowlist.")
    elif report.dangerous:
        message = ("The model file references dangerous code that loading would run: "
                   f"{shown}.")
    else:
        message = ("The model file references code the allowlist does not recognise: "
                   f"{shown}. A person should review it.")
    message += (" This is a static scan, not a sandbox: it does not prove the file is safe, "
                "and only trusted files should be validated.")
    return scored_result(
        ctx, check_id, float(report.severity), message,
        details={"verdict": report.verdict, "dangerous": report.dangerous,
                 "unrecognised": report.unrecognised,
                 "references": report.references[:MAX_REFS_REPORTED],
                 "reference_count": len(report.references),
                 "loaded": c.model is not None, "refuse_at": ctx.config.model_safety.refuse_at,
                 "file_bytes": report.file_bytes})


@_register("operational.cpu_time")
def cpu_time(ctx: CheckContext) -> CheckResult:
    check_id = "operational.cpu_time"
    reason = prediction_blocker(ctx.candidate)
    if reason:
        return unavailable_result(ctx, check_id, "Cannot test: " + reason)
    evaluation = ctx.config.evaluation
    rows = ctx.candidate.eval_df[ctx.candidate.feature_columns].head(evaluation.cpu_sample_rows)
    model = ctx.candidate.model
    model.predict(rows)                       # warm-up: imports, lazy initialisation

    # The OS reports process CPU time in coarse steps, so predict repeatedly until
    # enough time has passed for the figure to be meaningful.
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    rounds = 0
    while rounds < evaluation.cpu_max_rounds:
        model.predict(rows)
        rounds += 1
        if time.perf_counter() - wall_start >= evaluation.cpu_min_seconds:
            break
    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    predicted = rounds * len(rows)
    ms_per_row = cpu / predicted * 1000.0
    return scored_result(
        ctx, check_id, ms_per_row,
        f"Predicting {predicted} rows used {cpu:.2f} s of process CPU time over "
        f"{wall:.2f} s of wall time: {ms_per_row:.3f} ms of CPU per row "
        f"(CPU/wall ratio {cpu / wall:.2f}; above 1 means several cores were busy).",
        details={"cpu_seconds": cpu, "wall_seconds": wall, "rows_predicted": predicted,
                 "rounds": rounds, "rows_per_round": len(rows),
                 "cpu_to_wall_ratio": cpu / wall if wall else None,
                 "machine": machine_profile(),
                 "note": "Batch prediction, so it differs from the single-row latency "
                         "check; depends on the machine and on other load."})


# --------------------------------------------------------------------------
def run_operational_checks(candidate: LoadedCandidate,
                           config: GateConfig) -> list[CheckResult]:
    """Run every enabled operational check, in catalogue order."""
    ctx = CheckContext(candidate=candidate, config=config)
    return [OPERATIONAL_CHECKS[spec.id](ctx)
            for spec in CATALOGUE
            if spec.dimension == Dimension.OPERATIONAL_READINESS
            and config.checks[spec.id].enabled]
