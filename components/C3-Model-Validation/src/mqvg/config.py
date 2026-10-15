"""Gate configuration: typed models plus a validating YAML loader.

A config holds everything that can change a validation outcome: dimension
weights, per-check weights and thresholds, the missing-evidence policy,
decision thresholds, and critical rules. Bad configs fail at load time with
every problem listed, never in the middle of a run.

Threshold semantics (used by the scoring module) for each non-binary check:

* score is 0 at `floor` and 100 at `target`, linear in between, clipped;
* status is PASS if the value is at least as good as `pass_at`, FAIL if it is
  worse than `fail_at`, and WARN in between.

"Better" follows the metric's direction in the check catalogue, so for a
higher-is-better metric `floor <= fail_at <= pass_at <= target`, and the
order is reversed for a lower-is-better metric.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mqvg.catalogue import CHECKS_BY_ID
from mqvg.schemas import (
    WEIGHT_TOLERANCE, CheckStatus, Dimension, MissingPolicy,
)

MAX_EXTENDS_DEPTH = 5


class ConfigError(Exception):
    """Raised for any problem loading or validating a config."""

    def __init__(self, messages: list[str] | str):
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("\n".join(f"- {m}" for m in self.messages))


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Thresholds(_Strict):
    target: float
    pass_at: float
    fail_at: float
    floor: float


class CheckConfig(_Strict):
    enabled: bool = True
    weight: float = Field(default=1.0, gt=0)
    required: bool = False  # missing evidence for a required check blocks PROMOTE
    thresholds: Thresholds | None = None


class RuleCondition(_Strict):
    """Exactly one of the three tests must be given."""
    check: str
    status_in: list[CheckStatus] | None = None
    value_above: float | None = None
    value_below: float | None = None

    @model_validator(mode="after")
    def _exactly_one_test(self) -> RuleCondition:
        given = [n for n in ("status_in", "value_above", "value_below")
                 if getattr(self, n) not in (None, [])]
        if len(given) != 1:
            raise ValueError(
                "give exactly one of status_in, value_above, value_below "
                f"(got {given or 'none'})")
        return self


class CriticalRule(_Strict):
    """A condition that forces ROLLBACK regardless of the MQS."""
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    description: str | None = None
    condition: RuleCondition


class DimensionWeights(_Strict):
    training_validity: float = Field(ge=0)
    model_behaviour: float = Field(ge=0)
    operational_readiness: float = Field(ge=0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> DimensionWeights:
        total = (self.training_validity + self.model_behaviour
                 + self.operational_readiness)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"must sum to 1 (got {total})")
        return self

    def as_dict(self) -> dict[Dimension, float]:
        return {
            Dimension.TRAINING_VALIDITY: self.training_validity,
            Dimension.MODEL_BEHAVIOUR: self.model_behaviour,
            Dimension.OPERATIONAL_READINESS: self.operational_readiness,
        }


class CoverageConfig(_Strict):
    # A dimension with less evidence than this gets no score at all.
    min_dimension: float = Field(ge=0, le=1)
    # Below this overall coverage PROMOTE is not allowed.
    min_overall: float = Field(ge=0, le=1)


class DecisionConfig(_Strict):
    promote_at: float = Field(ge=0, le=100)
    reject_below: float = Field(ge=0, le=100)
    # A WARN or FAIL on any of these checks blocks PROMOTE.
    must_pass_checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ordered(self) -> DecisionConfig:
        if self.reject_below >= self.promote_at:
            raise ValueError("reject_below must be lower than promote_at")
        return self


class EvaluationConfig(_Strict):
    random_seed: int = 42
    min_group_size: int = Field(default=30, ge=1)
    ece_bins: int = Field(default=10, ge=2)
    latency_warmup_runs: int = Field(default=5, ge=0)
    latency_repeats: int = Field(default=50, ge=1)
    # Robustness: Gaussian noise with standard deviation
    # `robustness_noise_scale` x (each numeric feature's own std), repeated.
    robustness_noise_scale: float = Field(default=0.1, gt=0)
    robustness_repeats: int = Field(default=5, ge=1)
    # Leakage probe: cross-validated single-feature models, on at most this
    # many (randomly sampled, seeded) evaluation rows.
    leakage_max_rows: int = Field(default=20000, ge=100)
    leakage_cv_folds: int = Field(default=5, ge=2)


class MetadataConfig(_Strict):
    # Dotted field names looked up in TrainingMetadata.
    required_fields: list[str]
    reproducibility_fields: list[str]


class GateConfig(_Strict):
    profile: str
    description: str | None = None
    weights: DimensionWeights
    missing_policy: MissingPolicy
    coverage: CoverageConfig
    decision: DecisionConfig
    critical_rules: list[CriticalRule] = Field(default_factory=list)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    metadata: MetadataConfig
    checks: dict[str, CheckConfig]

    @model_validator(mode="after")
    def _cross_validate(self) -> GateConfig:
        problems: list[str] = []
        problems += self._check_entries()
        problems += self._dimensions_have_checks()
        problems += self._references()
        if problems:
            raise ValueError("\n".join(problems))
        return self

    def _check_entries(self) -> list[str]:
        problems = []
        for check_id, cfg in self.checks.items():
            spec = CHECKS_BY_ID.get(check_id)
            if spec is None:
                problems.append(f"checks.{check_id}: unknown check id")
                continue
            t = cfg.thresholds
            if spec.direction == "binary":
                if t is not None:
                    problems.append(f"checks.{check_id}: binary checks take no thresholds")
                continue
            if t is None:
                problems.append(f"checks.{check_id}: thresholds are required")
                continue
            ordered = [t.floor, t.fail_at, t.pass_at, t.target]
            if spec.direction == "lower_is_better":
                ordered.reverse()
            if ordered != sorted(ordered):
                order = ("floor <= fail_at <= pass_at <= target"
                         if spec.direction == "higher_is_better"
                         else "floor >= fail_at >= pass_at >= target")
                problems.append(
                    f"checks.{check_id}: thresholds must satisfy {order} "
                    f"for a {spec.direction.replace('_', ' ')} metric")
            if t.target == t.floor:
                problems.append(f"checks.{check_id}: target and floor must differ")
        missing = sorted(set(CHECKS_BY_ID) - set(self.checks))
        if missing:
            problems.append(f"checks: missing entries for {missing}")
        return problems

    def _enabled(self) -> set[str]:
        return {i for i, c in self.checks.items() if c.enabled and i in CHECKS_BY_ID}

    def _dimensions_have_checks(self) -> list[str]:
        problems = []
        enabled = self._enabled()
        for dimension, weight in self.weights.as_dict().items():
            has_check = any(CHECKS_BY_ID[i].dimension == dimension for i in enabled)
            if weight > 0 and not has_check:
                problems.append(
                    f"weights.{dimension.value} is {weight} but no enabled check "
                    "belongs to that dimension")
        return problems

    def _references(self) -> list[str]:
        problems = []
        enabled = self._enabled()
        for check_id, cfg in self.checks.items():
            if cfg.required and check_id not in enabled:
                problems.append(f"checks.{check_id}: a required check must be enabled")
        for check_id in self.decision.must_pass_checks:
            if check_id not in enabled:
                problems.append(
                    f"decision.must_pass_checks: '{check_id}' is unknown or disabled")
        seen: set[str] = set()
        for rule in self.critical_rules:
            if rule.id in seen:
                problems.append(f"critical_rules: duplicate rule id '{rule.id}'")
            seen.add(rule.id)
            cond = rule.condition
            if cond.check not in enabled:
                problems.append(
                    f"critical_rules.{rule.id}: check '{cond.check}' is unknown or disabled")
            elif (cond.status_in is None
                  and CHECKS_BY_ID[cond.check].direction == "binary"):
                problems.append(
                    f"critical_rules.{rule.id}: value tests need a numeric check; "
                    f"'{cond.check}' is binary, use status_in")
        for field in ("required_fields", "reproducibility_fields"):
            if not getattr(self.metadata, field):
                problems.append(f"metadata.{field}: must not be empty")
        return problems


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if data is None:
        raise ConfigError(f"{path}: config file is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Mappings merge recursively; any other value (including lists) replaces."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        raise ConfigError(
            "circular 'extends': " + " -> ".join(p.name for p in (*chain, path)))
    if len(chain) >= MAX_EXTENDS_DEPTH:
        raise ConfigError(f"'extends' chain deeper than {MAX_EXTENDS_DEPTH}")
    data = _read_yaml(path)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ConfigError(f"{path}: 'extends' must be a file path string")
    base = _resolve(path.parent / parent, (*chain, path))
    return _deep_merge(base, data)


def load_config(path: str | Path) -> GateConfig:
    """Load and validate a config file.

    A file may start from another with `extends: other.yaml` (path relative to
    the file). Mappings are merged recursively; lists are replaced whole.

    Raises ConfigError. Problems come in two stages: basic field errors
    (types, ranges, weights) are listed first; cross-field errors (threshold
    order, unknown or disabled check references) appear once those are fixed.
    """
    data = _resolve(Path(path), ())
    try:
        return GateConfig.model_validate(data)
    except ValidationError as exc:
        messages = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"])
            text = error["msg"].removeprefix("Value error, ")
            # Cross-checks raise at the top level and already name their path.
            for line in text.splitlines():
                messages.append(f"{where}: {line}" if where else line)
        raise ConfigError(messages) from None


def config_hash(config: GateConfig) -> str:
    """Stable SHA-256 of the effective config, for audit records."""
    canonical = json.dumps(config.model_dump(mode="json"),
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
