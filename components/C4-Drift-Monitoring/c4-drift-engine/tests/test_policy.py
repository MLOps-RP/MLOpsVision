import pytest

from src.common.config import load_config
from src.c4.controller import policy

POLICY = load_config("policy")


@pytest.mark.parametrize("diagnosis, level, action", [
    ("DATA_QUALITY", "Critical", "BLOCK_AND_FLAG"),      # broken data is never retrained or rolled back
    ("DATA_QUALITY", "Low", "BLOCK_AND_FLAG"),
    ("CONCEPT_DRIFT", "Critical", "ROLLBACK"),
    ("COVARIATE_DEGRADING", "Critical", "ROLLBACK"),
    ("COVARIATE_BENIGN", "High", "WARN_ONLY"),            # benign drift never triggers retraining
    ("COVARIATE_DEGRADING", "Medium", "ROUTE_C1"),
    ("COVARIATE_DEGRADING", "High", "ROUTE_C1"),
    ("CONCEPT_DRIFT", "Medium", "ROUTE_C2"),
    ("LABEL_SHIFT", "High", "ROUTE_C2"),
    ("CONCEPT_DRIFT", "Low", "WARN_ONLY"),
    ("SUSPECTED", "Low", "WATCH"),
    ("NO_DRIFT", "None", "NONE"),
])
def test_policy_decisions(diagnosis, level, action):
    assert policy.decide(diagnosis, level, POLICY)[0] == action


def test_first_matching_rule_wins():
    custom = {"rules": [
        {"diagnosis": ["A"], "severity": "any", "action": "FIRST", "target": "x"},
        {"diagnosis": "any", "severity": "any", "action": "SECOND", "target": "y"},
    ]}
    assert policy.decide("A", "Low", custom) == ("FIRST", "x", 1)
    assert policy.decide("B", "Low", custom) == ("SECOND", "y", 2)


def test_unmatched_case_falls_back_to_watch():
    assert policy.decide("UNKNOWN", "None", {"rules": []}) == ("WATCH", "dashboard", 0)


def test_explain_mentions_diagnosis_action_and_evidence():
    text = policy.explain("CONCEPT_DRIFT", "Medium", "ROUTE_C2", ["F1 dropped 0.15"])
    assert "CONCEPT_DRIFT" in text and "F1 dropped 0.15" in text and "C2" in text


def test_every_scenario_expectation_is_a_known_policy_action():
    actions = {r["action"] for r in POLICY["rules"]}
    for name, sc in load_config("scenarios")["scenarios"].items():
        assert sc["expected_action"] in actions, name
