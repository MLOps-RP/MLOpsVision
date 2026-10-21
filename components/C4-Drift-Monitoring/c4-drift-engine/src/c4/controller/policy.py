"""
REMEDIATION DECISION (processing step 4)

Reads configs/policy.yaml and returns the first rule matching (diagnosis, severity).
"""


def _matches(rule_value, value) -> bool:
    return rule_value == "any" or value in rule_value


def decide(diagnosis: str, severity_level: str, policy: dict) -> tuple[str, str, int]:
    """Return (action, target, rule_number)."""
    for i, rule in enumerate(policy["rules"], start=1):
        if _matches(rule["diagnosis"], diagnosis) and _matches(rule["severity"], severity_level):
            return rule["action"], rule["target"], i
    return "WATCH", "dashboard", 0


def explain(diagnosis: str, level: str, action: str, evidence: list) -> str:
    """One readable sentence for the Drift Report and the audit log."""
    why = "; ".join(evidence) if evidence else "no drift signals above threshold"
    meaning = {
        "NONE": "no action needed",
        "WATCH": "keep watching until labels confirm the impact",
        "WARN_ONLY": "inputs changed but the model still performs - warn, do not retrain",
        "ROUTE_C1": "send to C1 to adapt preprocessing / features and retrain",
        "ROUTE_C2": "send to C2 for model retraining / recalibration",
        "ROLLBACK": "roll back to the previous model version immediately",
        "BLOCK_AND_FLAG": "data is broken - block retraining and flag the data pipeline",
    }.get(action, action)
    return f"{diagnosis} ({level}): {why}. Decision: {meaning}."
