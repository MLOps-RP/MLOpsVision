"""Build CheckResult lists for scoring and rules tests."""
from mqvg.catalogue import CHECKS_BY_ID
from mqvg.schemas import CheckResult, CheckStatus, EvidenceSource

NA = "NOT_APPLICABLE"
UNAVAILABLE = "UNAVAILABLE"
ERROR = "ERROR"


def enabled_ids(config):
    return [i for i, c in config.checks.items() if c.enabled]


def make_results(config, base=100.0, scores=None, per_dimension=None):
    """One result per enabled check.

    `scores` maps a check id to one of:
      * a number: PASS with that 0-100 score;
      * "UNAVAILABLE" / "ERROR" / "NOT_APPLICABLE": no evidence;
      * a (status, score) tuple, e.g. ("WARN", 60);
      * a dict with optional keys status, score, raw (the measured value).
    `per_dimension` gives a default for a whole dimension; `base` for the rest.
    """
    scores, per_dimension = scores or {}, per_dimension or {}
    results = []
    for check_id in enabled_ids(config):
        spec = CHECKS_BY_ID[check_id]
        value = scores.get(check_id, per_dimension.get(spec.dimension, base))
        common = dict(id=check_id, dimension=spec.dimension, name=check_id,
                      message="m")
        if isinstance(value, str):
            results.append(CheckResult(
                **common, status=CheckStatus(value), score=None,
                evidence_source=EvidenceSource.UNAVAILABLE))
            continue
        if isinstance(value, tuple):
            value = {"status": value[0], "score": value[1]}
        elif isinstance(value, dict):
            value = dict(value)
        else:
            value = {"score": value}
        results.append(CheckResult(
            **common, status=CheckStatus(value.get("status", "PASS")),
            score=float(value.get("score", 100.0)),
            raw_value=value.get("raw"),
            evidence_source=EvidenceSource.MEASURED))
    return results
