"""The plain-language profile guide is generated from configuration, never from a profile's name."""
import shutil

import pytest
import yaml
from fastapi.testclient import TestClient

from mqvg.api import profile_guide
from mqvg.api.app import create_app
from mqvg.api.settings import Settings
from mqvg.config import load_config

from conftest import CONFIGS

KEY = "k"
AUTH = {"X-API-Key": KEY}
HIDDEN = "behaviour.hidden_failure_groups"


def settings(tmp_path, **kw):
    values = dict(artifact_root=tmp_path, report_dir=tmp_path / "reports", config_dir=CONFIGS, api_key=KEY)
    values.update(kw)
    return Settings(**values)


# --------------------------------------------------------------------- strictness
def test_strictness_follows_the_promotion_threshold_and_missing_evidence_policy(make_config):
    assert profile_guide.strictness_of(make_config()) == "standard"                      # promote_at 75
    assert profile_guide.strictness_of(make_config(lambda d: d["decision"].update(promote_at=85, reject_below=60))) == "strict"
    assert profile_guide.strictness_of(make_config(lambda d: d.update(missing_policy="penalize"))) == "strict"
    assert profile_guide.strictness_of(make_config(lambda d: d["decision"].update(promote_at=60, reject_below=40))) == "lenient"
    assert profile_guide.strictness_of(make_config(lambda d: d["decision"].update(promote_at=69.9, reject_below=40))) == "lenient"
    assert profile_guide.strictness_of(make_config(lambda d: d["decision"].update(promote_at=70, reject_below=40))) == "standard"


def test_the_numbers_in_the_text_are_the_configurations_own(make_config):
    config = make_config(lambda d: d["decision"].update(promote_at=82.5, reject_below=55, must_pass_checks=["calibration.ece"]))
    text = profile_guide.describe(config, scans_files=False).strictness_text
    assert "82.5 or more" in text and "below 55" in text
    assert "must pass" in text and f"{len(config.critical_rules)} critical rules" in text
    assert "1 check must pass" in text                                                    # singular agrees with the count


def test_missing_evidence_is_described_as_the_policy_treats_it(make_config):
    exclude = profile_guide.describe(make_config(), False).strictness_text
    assert "never counted as a pass" in exclude
    penalize = profile_guide.describe(make_config(lambda d: d.update(missing_policy="penalize")), False).strictness_text
    assert "counts as zero" in penalize and "never counted" not in penalize


def test_a_single_critical_rule_is_described_in_the_singular(make_config):
    config = make_config(lambda d: d.update(critical_rules=d["critical_rules"][:1]))
    assert "1 critical rule forces a rollback" in profile_guide.describe(config, False).strictness_text


# ----------------------------------------------------------------------- highlights
def test_the_summary_counts_the_checks_that_actually_run(make_config):
    config = make_config()
    enabled = sum(c.enabled for c in config.checks.values())
    assert f"Runs {enabled} checks" in profile_guide.describe(config, False).summary


def test_highlights_appear_only_for_what_the_profile_really_does(make_config):
    plain = profile_guide.describe(make_config(), scans_files=False)
    assert plain.highlights == []
    scanning = profile_guide.describe(make_config(), scans_files=True)
    assert scanning.highlights == ["Scans the model file before loading it"]


@pytest.mark.parametrize("consequence, phrase", [
    ("score_only", "only lowers the score"), ("hold", "puts the model on hold"), ("critical", "rollback recommendation")])
def test_hidden_group_detection_is_described_with_its_configured_consequence(make_config, consequence, phrase):
    config = make_config(lambda d: (d["checks"][HIDDEN].update(enabled=True), d["hidden_groups"].update(consequence=consequence)))
    highlights = profile_guide.describe(config, False).highlights
    assert any("hidden failing groups" in h and phrase in h for h in highlights)


def test_extra_checks_are_listed_by_what_they_check(make_config):
    def tweak(d):
        for check in ("training_validity.eval_label_conflicts", "training_validity.train_duplicate_rows", "operational.cpu_time"):
            d["checks"][check]["enabled"] = True
    text = " ".join(profile_guide.describe(make_config(tweak), False).highlights)
    assert "conflicting labels in the evaluation data, repeated training rows and how much CPU the model uses" in text
    one = make_config(lambda d: d["checks"]["operational.cpu_time"].update(enabled=True))
    assert "Also checks how much CPU the model uses" in " ".join(profile_guide.describe(one, False).highlights)


# ---------------------------------------------------------------------- the endpoint
def profiles(tmp_path, **kw):
    return TestClient(create_app(settings(tmp_path, **kw))).get("/v1/profiles", headers=AUTH).json()


def test_every_profile_carries_a_plain_language_guide(tmp_path):
    body = profiles(tmp_path)
    assert body["profiles"]
    for p in body["profiles"]:
        assert p["summary"].startswith("Runs ") and p["strictness"] in ("lenient", "standard", "strict")
        assert "Recommends promotion at a score of" in p["strictness_text"] and isinstance(p["highlights"], list)


def test_the_guide_matches_each_profiles_real_configuration(tmp_path):
    for p in profiles(tmp_path)["profiles"]:
        config = load_config(CONFIGS / f"{p['name']}.yaml")
        assert f"{config.decision.promote_at:g} or more" in p["strictness_text"], p["name"]
        assert p["strictness"] == profile_guide.strictness_of(config), p["name"]
        assert (("Scans the model file before loading it" in p["highlights"]) == p["scans_model_files"]), p["name"]
        assert (any("hidden failing groups" in h for h in p["highlights"]) == config.checks[HIDDEN].enabled), p["name"]


def test_the_strict_profile_is_reported_strict_and_the_rest_standard(tmp_path):
    by_name = {p["name"]: p for p in profiles(tmp_path)["profiles"]}
    assert by_name["strict"]["strictness"] == "strict"
    assert by_name["extended"]["strictness"] == "standard"


def test_a_restricted_profile_says_why_and_an_allowed_one_does_not(tmp_path):
    by_name = {p["name"]: p for p in profiles(tmp_path)["profiles"]}
    assert not by_name["default"]["allowed"]
    assert "without scanning it first" in by_name["default"]["unavailable_reason"]
    assert "runs the code inside it" in by_name["default"]["unavailable_reason"]
    assert by_name["extended"]["allowed"] and by_name["extended"]["unavailable_reason"] is None


def test_allowing_unscanned_profiles_lifts_the_restriction_and_the_reason(tmp_path):
    by_name = {p["name"]: p for p in profiles(tmp_path, allow_unscanned_profiles=True)["profiles"]}
    assert by_name["default"]["allowed"] and by_name["default"]["unavailable_reason"] is None


def test_a_profile_added_by_an_operator_is_described_without_any_code_change(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    (configs / "my_team_profile.yaml").write_text(yaml.safe_dump({
        "extends": "extended.yaml", "profile": "my_team_profile", "description": "Our own rules.",
        "decision": {"promote_at": 91, "reject_below": 70}}), encoding="utf-8")
    by_name = {p["name"]: p for p in profiles(tmp_path, config_dir=configs)["profiles"]}
    mine = by_name["my_team_profile"]
    assert mine["strictness"] == "strict" and "91 or more" in mine["strictness_text"] and "below 70" in mine["strictness_text"]
    assert mine["allowed"] and "Scans the model file before loading it" in mine["highlights"]
    assert by_name["extended"]["strictness"] == "standard"                                # unaffected


def test_restricting_a_profile_follows_its_configuration_not_its_name(tmp_path):
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    (configs / "extended_but_report_only.yaml").write_text(yaml.safe_dump({
        "extends": "extended.yaml", "profile": "x", "model_safety": {"refuse_at": "never"}}), encoding="utf-8")
    by_name = {p["name"]: p for p in profiles(tmp_path, config_dir=configs)["profiles"]}
    odd = by_name["extended_but_report_only"]
    assert not odd["allowed"] and not odd["scans_model_files"] and odd["unavailable_reason"]
