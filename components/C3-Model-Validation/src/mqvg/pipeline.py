"""The validation pipeline: one call from a candidate bundle to a result.

    load candidate -> training-validity, behaviour and operational checks
        -> scorecard (TVS / MBS / ORS / MQS) -> rules engine -> ValidationResult

Contract:

* A bad candidate never raises. A missing or corrupt model, unreadable
  evaluation data or broken metadata become findings (failed or unavailable
  checks) and a decision. Only an unusable *configuration* raises
  (`ConfigError`), because without it there is no rulebook to judge by.
* A crash inside one check becomes an ERROR result for that check (see
  `checks.base.guarded`). If a whole check group crashes, every enabled check
  of that dimension is recorded as ERROR and the run continues.
* The result is a recommendation. Standalone, nothing is executed.

Reproducibility: with the same inputs and configuration the result is
identical apart from `run_id`, `created_at_utc`, `environment` and the
timing-based checks (latency, memory), which vary with the machine.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from mqvg import __version__
from mqvg.catalogue import CATALOGUE
from mqvg.checks.base import CheckContext, unavailable_result
from mqvg.checks.behaviour import run_behaviour_checks
from mqvg.checks.operational import run_operational_checks
from mqvg.checks.training_validity import run_training_validity_checks
from mqvg.config import ConfigError, GateConfig, config_hash, load_config
from mqvg.environment import machine_profile
from mqvg.explain import explain_failures
from mqvg.loader import LoadedCandidate, load_candidate
from mqvg.rules import evaluate_rules
from mqvg.schemas import (
    CandidateBundle, CandidateSummary, CheckResult, CheckStatus, Dimension,
    ScenarioMode, ValidationResult,
)
from mqvg.scoring import compute_scorecard

GroupRunner = Callable[[LoadedCandidate, GateConfig], list[CheckResult]]

# One entry per dimension, in the order results are reported. The lambdas look
# the runner up when called, so it can be replaced in tests.
_RUNNERS: dict[Dimension, GroupRunner] = {
    Dimension.TRAINING_VALIDITY:
        lambda candidate, config: run_training_validity_checks(candidate, config),
    Dimension.MODEL_BEHAVIOUR:
        lambda candidate, config: run_behaviour_checks(candidate, config),
    Dimension.OPERATIONAL_READINESS:
        lambda candidate, config: run_operational_checks(candidate, config),
}


def _resolve_config(bundle: CandidateBundle,
                    config: GateConfig | None) -> GateConfig:
    if config is not None:
        return config
    if bundle.config_path is not None:
        return load_config(bundle.config_path)
    raise ConfigError("no configuration: pass config=... or set config_path "
                      "in the candidate bundle")


def _run_group(dimension: Dimension, candidate: LoadedCandidate,
               config: GateConfig) -> list[CheckResult]:
    """Run one dimension's checks; a crash of the whole group becomes ERROR
    results for its enabled checks instead of ending the run."""
    runner = _RUNNERS[dimension]
    try:
        return runner(candidate, config)
    except Exception as exc:  # noqa: BLE001 - isolation is the point
        ctx = CheckContext(candidate=candidate, config=config)
        message = f"check group crashed: {type(exc).__name__}: {exc}"
        return [unavailable_result(ctx, spec.id, message,
                                   details={"exception": type(exc).__name__},
                                   status=CheckStatus.ERROR)
                for spec in CATALOGUE
                if spec.dimension == dimension and config.checks[spec.id].enabled]


def run_validation(bundle: CandidateBundle, config: GateConfig | None = None, *,
                   run_id: str | None = None,
                   created_at: datetime | None = None) -> ValidationResult:
    """Validate one candidate model and return the full, auditable result.

    `config` overrides `bundle.config_path`. `run_id` and `created_at` exist so
    tests and callers can fix the otherwise generated values.

    SECURITY: loading the model executes code stored in the file; only
    validate models from trusted sources.
    """
    config = _resolve_config(bundle, config)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    candidate = load_candidate(bundle)
    timings["load"] = time.perf_counter() - started

    checks: list[CheckResult] = []
    for dimension in _RUNNERS:
        started = time.perf_counter()
        checks += _run_group(dimension, candidate, config)
        timings[dimension.value] = time.perf_counter() - started

    started = time.perf_counter()
    scorecard = compute_scorecard(checks, config)
    decision = evaluate_rules(checks, scorecard, config)
    timings["score_and_decide"] = time.perf_counter() - started

    return ValidationResult(
        gate_version=__version__,
        run_id=run_id or uuid.uuid4().hex,
        created_at_utc=created_at or datetime.now(timezone.utc),
        candidate=CandidateSummary(
            candidate_id=bundle.candidate_id, model_path=str(bundle.model_path),
            model_sha256=candidate.model_sha256,
            eval_data_sha256=candidate.eval_sha256,
            metadata_sha256=candidate.metadata_sha256),
        scenario=bundle.scenario,
        config_hash=config_hash(config),
        config_snapshot=config.model_dump(mode="json"),
        environment={"machine": machine_profile(),
                     "timings_seconds": timings},
        checks=checks, scorecard=scorecard, decision=decision,
        explanations=explain_failures(
            checks, scorecard, decision, config,
            simulated_inputs=(bundle.scenario.simulated_inputs
                              if bundle.scenario.mode == ScenarioMode.SIMULATED else ())))
