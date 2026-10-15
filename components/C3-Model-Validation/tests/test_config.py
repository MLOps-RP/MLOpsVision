from pathlib import Path

import pytest
import yaml

from mqvg.catalogue import CATALOGUE, CHECKS_BY_ID
from mqvg.config import ConfigError, config_hash, load_config
from mqvg.schemas import CheckResult, Dimension, MissingPolicy

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT = CONFIGS / "default.yaml"
STRICT = CONFIGS / "strict.yaml"


def default_dict() -> dict:
    return yaml.safe_load(DEFAULT.read_text(encoding="utf-8"))


def load_from_dict(tmp_path, data: dict, name: str = "cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def error_text(tmp_path, data: dict) -> str:
    with pytest.raises(ConfigError) as info:
        load_from_dict(tmp_path, data)
    return str(info.value)


# ------------------------------------------------------------- catalogue
def test_catalogue_ids_are_unique_and_valid_check_ids():
    assert len(CHECKS_BY_ID) == len(CATALOGUE)
    for spec in CATALOGUE:
        # the same id format the CheckResult schema enforces
        CheckResult(id=spec.id, dimension=spec.dimension, name=spec.id,
                    status="UNAVAILABLE", evidence_source="UNAVAILABLE",
                    message="x")


def test_catalogue_covers_every_dimension():
    assert {s.dimension for s in CATALOGUE} == set(Dimension)


# ------------------------------------------------------- shipped profiles
@pytest.mark.parametrize("path", [DEFAULT, STRICT])
def test_shipped_profiles_load(path):
    config = load_config(path)
    assert set(config.checks) == set(CHECKS_BY_ID)


def test_default_profile_values():
    config = load_config(DEFAULT)
    assert config.profile == "default"
    assert config.weights.as_dict() == {
        Dimension.TRAINING_VALIDITY: 0.30,
        Dimension.MODEL_BEHAVIOUR: 0.50,
        Dimension.OPERATIONAL_READINESS: 0.20,
    }
    assert config.missing_policy == MissingPolicy.EXCLUDE_AND_CAP
    assert not config.checks["robustness.noise_degradation"].enabled


def test_strict_extends_default_and_is_stricter():
    default, strict = load_config(DEFAULT), load_config(STRICT)
    assert strict.profile == "strict"
    assert strict.decision.promote_at > default.decision.promote_at
    assert strict.decision.reject_below > default.decision.reject_below
    assert strict.coverage.min_overall > default.coverage.min_overall
    assert strict.missing_policy == MissingPolicy.PENALIZE
    assert len(strict.critical_rules) > len(default.critical_rules)
    # inherited, not restated
    assert (strict.checks["behaviour.recall"]
            == default.checks["behaviour.recall"])
    # overridden thresholds are stricter
    assert (strict.checks["behaviour.f1"].thresholds.pass_at
            > default.checks["behaviour.f1"].thresholds.pass_at)


def test_required_checks_exist_in_every_dimension():
    config = load_config(DEFAULT)
    required_dims = {CHECKS_BY_ID[i].dimension
                     for i, c in config.checks.items() if c.required}
    assert required_dims == set(Dimension)


# ---------------------------------------------------------------- hashing
def test_config_hash_is_stable_and_profile_specific():
    assert config_hash(load_config(DEFAULT)) == config_hash(load_config(DEFAULT))
    assert config_hash(load_config(DEFAULT)) != config_hash(load_config(STRICT))


def test_config_hash_ignores_file_location(tmp_path):
    copy_path = tmp_path / "copy.yaml"
    copy_path.write_text(DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    assert config_hash(load_config(copy_path)) == config_hash(load_config(DEFAULT))


def test_config_hash_changes_with_a_threshold(tmp_path):
    data = default_dict()
    data["checks"]["behaviour.f1"]["thresholds"]["pass_at"] = 0.61
    assert (config_hash(load_from_dict(tmp_path, data))
            != config_hash(load_config(DEFAULT)))


# ------------------------------------------------------ file-level errors
def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(path)


def test_top_level_must_be_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


# -------------------------------------------------------------- extends
def test_extends_merges_mappings_and_replaces_lists(tmp_path):
    (tmp_path / "base.yaml").write_text(DEFAULT.read_text(encoding="utf-8"),
                                        encoding="utf-8")
    child = {"extends": "base.yaml", "profile": "child",
             "decision": {"must_pass_checks": ["calibration.ece"]}}
    config = load_from_dict(tmp_path, child, "child.yaml")
    assert config.profile == "child"
    assert config.decision.must_pass_checks == ["calibration.ece"]  # replaced
    assert config.decision.promote_at == 75                           # merged


def test_extends_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_extends_missing_parent(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: gone.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "a.yaml")


# --------------------------------------------------------- weights/policy
def test_dimension_weights_must_sum_to_one(tmp_path):
    data = default_dict()
    data["weights"]["training_validity"] = 0.5
    assert "must sum to 1" in error_text(tmp_path, data)


def test_negative_dimension_weight(tmp_path):
    data = default_dict()
    data["weights"] = {"training_validity": -0.1, "model_behaviour": 0.6,
                       "operational_readiness": 0.5}
    assert "training_validity" in error_text(tmp_path, data)


def test_unknown_missing_policy(tmp_path):
    data = default_dict()
    data["missing_policy"] = "assume_pass"
    assert "missing_policy" in error_text(tmp_path, data)


def test_unknown_top_level_key(tmp_path):
    data = default_dict()
    data["wieghts"] = {}
    assert "wieghts" in error_text(tmp_path, data)


def test_coverage_out_of_range(tmp_path):
    data = default_dict()
    data["coverage"]["min_overall"] = 1.5
    assert "coverage.min_overall" in error_text(tmp_path, data)


# ----------------------------------------------------------------- checks
def test_unknown_check_id(tmp_path):
    data = default_dict()
    data["checks"]["behaviour.f2"] = {"weight": 1,
                                      "thresholds": {"target": 1, "pass_at": 1,
                                                     "fail_at": 1, "floor": 0}}
    assert "behaviour.f2: unknown check id" in error_text(tmp_path, data)


def test_missing_check_entry(tmp_path):
    data = default_dict()
    del data["checks"]["calibration.ece"]
    assert "calibration.ece" in error_text(tmp_path, data)


@pytest.mark.parametrize("check_id, thresholds", [
    # higher is better but pass_at above target
    ("behaviour.f1", {"target": 0.5, "pass_at": 0.9, "fail_at": 0.4, "floor": 0}),
    # lower is better written as if higher were better
    ("calibration.ece", {"target": 0.25, "pass_at": 0.1, "fail_at": 0.05,
                         "floor": 0.01}),
])
def test_threshold_order_follows_metric_direction(tmp_path, check_id, thresholds):
    data = default_dict()
    data["checks"][check_id]["thresholds"] = thresholds
    assert f"{check_id}: thresholds must satisfy" in error_text(tmp_path, data)


def test_target_and_floor_must_differ(tmp_path):
    data = default_dict()
    data["checks"]["behaviour.f1"]["thresholds"] = {
        "target": 0.5, "pass_at": 0.5, "fail_at": 0.5, "floor": 0.5}
    assert "target and floor must differ" in error_text(tmp_path, data)


def test_binary_check_rejects_thresholds(tmp_path):
    data = default_dict()
    data["checks"]["operational.model_loadable"]["thresholds"] = {
        "target": 1, "pass_at": 1, "fail_at": 0, "floor": 0}
    assert "binary checks take no thresholds" in error_text(tmp_path, data)


def test_numeric_check_requires_thresholds(tmp_path):
    data = default_dict()
    del data["checks"]["behaviour.f1"]["thresholds"]
    assert "behaviour.f1: thresholds are required" in error_text(tmp_path, data)


@pytest.mark.parametrize("weight", [0, -1])
def test_check_weight_must_be_positive(tmp_path, weight):
    data = default_dict()
    data["checks"]["behaviour.f1"]["weight"] = weight
    assert "weight" in error_text(tmp_path, data)


def test_required_check_cannot_be_disabled(tmp_path):
    data = default_dict()
    data["checks"]["behaviour.f1"]["enabled"] = False
    assert "a required check must be enabled" in error_text(tmp_path, data)


def test_dimension_with_weight_needs_an_enabled_check(tmp_path):
    data = default_dict()
    for check_id, spec in CHECKS_BY_ID.items():
        if spec.dimension == Dimension.OPERATIONAL_READINESS:
            data["checks"][check_id]["enabled"] = False
            data["checks"][check_id]["required"] = False
    data["decision"]["must_pass_checks"] = [
        c for c in data["decision"]["must_pass_checks"]
        if not c.startswith("operational.")]
    assert "no enabled check belongs to that dimension" in error_text(tmp_path, data)


# --------------------------------------------------------------- decision
def test_reject_must_be_below_promote(tmp_path):
    data = default_dict()
    data["decision"]["reject_below"] = 80
    assert "reject_below must be lower than promote_at" in error_text(tmp_path, data)


def test_must_pass_check_must_exist_and_be_enabled(tmp_path):
    data = default_dict()
    data["decision"]["must_pass_checks"].append("robustness.noise_degradation")
    assert "robustness.noise_degradation" in error_text(tmp_path, data)


# --------------------------------------------------------- critical rules
def rule(**condition):
    return {"id": "r1", "name": "Rule", "condition": condition}


def test_rule_needs_exactly_one_test(tmp_path):
    data = default_dict()
    data["critical_rules"] = [rule(check="behaviour.f1")]
    assert "exactly one" in error_text(tmp_path, data)
    data["critical_rules"] = [rule(check="behaviour.f1", status_in=["FAIL"],
                                   value_below=0.1)]
    assert "exactly one" in error_text(tmp_path, data)


def test_rule_with_numeric_test_is_accepted(tmp_path):
    data = default_dict()
    data["critical_rules"] = [rule(check="behaviour.f1", value_below=0.3)]
    assert load_from_dict(tmp_path, data).critical_rules[0].condition.value_below == 0.3


def test_rule_value_test_on_binary_check_is_rejected(tmp_path):
    data = default_dict()
    data["critical_rules"] = [rule(check="operational.model_loadable",
                                   value_below=1)]
    assert "is binary, use status_in" in error_text(tmp_path, data)


def test_rule_must_reference_known_enabled_check(tmp_path):
    data = default_dict()
    data["critical_rules"] = [rule(check="nope.missing", status_in=["FAIL"])]
    assert "unknown or disabled" in error_text(tmp_path, data)


def test_rule_status_must_be_valid(tmp_path):
    data = default_dict()
    data["critical_rules"] = [rule(check="behaviour.f1", status_in=["BROKEN"])]
    assert "status_in" in error_text(tmp_path, data)


def test_duplicate_rule_ids(tmp_path):
    data = default_dict()
    data["critical_rules"] = [rule(check="behaviour.f1", status_in=["FAIL"]),
                              rule(check="behaviour.f1", status_in=["WARN"])]
    assert "duplicate rule id" in error_text(tmp_path, data)


# ---------------------------------------------------------------- metadata
def test_metadata_field_lists_cannot_be_empty(tmp_path):
    data = default_dict()
    data["metadata"]["required_fields"] = []
    assert "metadata.required_fields: must not be empty" in error_text(tmp_path, data)


# --------------------------------------------------------- error reporting
# Problems are reported in two stages: basic field errors (types, ranges,
# weights) first; once those are fixed, cross-field errors (threshold order,
# unknown references). Within a stage every problem is listed together.
def test_basic_field_problems_are_reported_together(tmp_path):
    data = default_dict()
    data["weights"]["training_validity"] = 0.9
    data["decision"]["reject_below"] = 99
    data["coverage"]["min_overall"] = 2
    with pytest.raises(ConfigError) as info:
        load_from_dict(tmp_path, data)
    assert len(info.value.messages) == 3


def test_cross_field_problems_are_reported_together(tmp_path):
    data = default_dict()
    data["checks"]["behaviour.f1"]["thresholds"] = {
        "target": 0.1, "pass_at": 0.9, "fail_at": 0.4, "floor": 0}
    data["decision"]["must_pass_checks"].append("nope.missing")
    data["critical_rules"] = [rule(check="gone.away", status_in=["FAIL"])]
    with pytest.raises(ConfigError) as info:
        load_from_dict(tmp_path, data)
    assert len(info.value.messages) == 3


def test_extends_must_be_a_string(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: [1, 2]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="file path string"):
        load_config(tmp_path / "a.yaml")


# ---------------------------------------------------------- fixture manifest
def test_manifest_findings_use_catalogue_check_ids():
    manifest_path = CONFIGS.parent / "fixtures" / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["fixtures"].items():
        for check_id in entry["expected_findings"]:
            assert check_id in CHECKS_BY_ID, f"{name}: unknown check {check_id}"
