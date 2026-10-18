"""Helpers shared by every check module.

A check is a function `(CheckContext) -> CheckResult`. These helpers build
results that satisfy the CheckResult invariants, so a check cannot produce a
score without evidence or a pass without a score.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable

from mqvg.catalogue import CHECKS_BY_ID
from mqvg.config import GateConfig
from mqvg.loader import LoadedCandidate
from mqvg.normalize import binary_outcome, normalize
from mqvg.schemas import CheckResult, CheckStatus, EvidenceSource


@dataclass
class CheckContext:
    candidate: LoadedCandidate
    config: GateConfig


CheckFunction = Callable[[CheckContext], CheckResult]


def _common(ctx: CheckContext, check_id: str) -> dict[str, Any]:
    spec = CHECKS_BY_ID[check_id]
    cfg = ctx.config.checks[check_id]
    thresholds = ({} if cfg.thresholds is None
                  else cfg.thresholds.model_dump())
    return dict(id=check_id, dimension=spec.dimension, name=spec.description,
                direction=spec.direction, thresholds=thresholds,
                required=cfg.required)


def scored_result(ctx: CheckContext, check_id: str, value: float, message: str,
                  evidence: EvidenceSource = EvidenceSource.MEASURED,
                  details: dict[str, Any] | None = None) -> CheckResult:
    """Result for a numeric check, scored with the configured thresholds."""
    thresholds = ctx.config.checks[check_id].thresholds
    score, status = normalize(float(value), thresholds,
                              CHECKS_BY_ID[check_id].direction)
    return CheckResult(**_common(ctx, check_id), status=status, score=score,
                       raw_value=float(value), evidence_source=evidence,
                       message=message, details=details or {})


def binary_result(ctx: CheckContext, check_id: str, passed: bool, message: str,
                  evidence: EvidenceSource = EvidenceSource.MEASURED,
                  details: dict[str, Any] | None = None) -> CheckResult:
    score, status = binary_outcome(passed)
    return CheckResult(**_common(ctx, check_id), status=status, score=score,
                       raw_value=passed, evidence_source=evidence,
                       message=message, details=details or {})


def unavailable_result(ctx: CheckContext, check_id: str, message: str,
                       details: dict[str, Any] | None = None,
                       status: CheckStatus = CheckStatus.UNAVAILABLE
                       ) -> CheckResult:
    """A check that could not run: no score, no evidence, never a pass."""
    return CheckResult(**_common(ctx, check_id), status=status, score=None,
                       evidence_source=EvidenceSource.UNAVAILABLE,
                       message=message, details=details or {})


def guarded(check_id: str) -> Callable[[CheckFunction], CheckFunction]:
    """Turn an exception inside a check into an ERROR result.

    One broken check must not abort the whole validation run.
    """
    def decorator(func: CheckFunction) -> CheckFunction:
        @functools.wraps(func)
        def wrapper(ctx: CheckContext) -> CheckResult:
            try:
                return func(ctx)
            except Exception as exc:  # noqa: BLE001 - deliberate catch-all
                return unavailable_result(
                    ctx, check_id,
                    f"check crashed: {type(exc).__name__}: {exc}",
                    details={"exception": type(exc).__name__},
                    status=CheckStatus.ERROR)
        return wrapper
    return decorator
