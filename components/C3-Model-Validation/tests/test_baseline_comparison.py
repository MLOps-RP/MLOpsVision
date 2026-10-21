"""The baseline-comparison helper: baselines, ground truth and the measured outcome."""
import importlib.util
import sys

import pytest

from builders import make_results
from conftest import ROOT


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "baseline_comparison", ROOT / "scripts" / "baseline_comparison.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["baseline_comparison"] = module
    spec.loader.exec_module(module)
    return module


F1, PRECISION = "behaviour.f1", "behaviour.precision"


def test_the_f1_baseline_promotes_only_when_f1_passes(script, default_config):
    assert script.promotes_f1_only(make_results(default_config))
    assert not script.promotes_f1_only(
        make_results(default_config, scores={F1: {"status": "WARN", "score": 60}}))
    assert not script.promotes_f1_only(make_results(default_config, scores={F1: "UNAVAILABLE"}))


def test_the_metrics_baseline_needs_every_performance_metric_to_pass(script, default_config):
    assert script.promotes_metrics_only(make_results(default_config))
    for metric in script.METRICS:
        weak = make_results(default_config, scores={metric: {"status": "WARN", "score": 60}})
        assert not script.promotes_metrics_only(weak), metric
        assert script.promotes_f1_only(weak) or metric == F1


def test_the_baselines_ignore_everything_else(script, default_config):
    broken = make_results(default_config, scores={
        "leakage.single_feature_predictiveness": {"status": "FAIL", "score": 0},
        "calibration.ece": {"status": "FAIL", "score": 0},
        "operational.model_loadable": {"status": "FAIL", "score": 0}})
    assert script.promotes_f1_only(broken) and script.promotes_metrics_only(broken)


def test_rates_count_defects_stopped_and_false_alarms(script):
    gates = dict.fromkeys(script.GATES, True)
    stop = lambda *names: {g: g not in names for g in script.GATES}   # noqa: E731
    outcomes = {
        "d1": {"defect": True, "promoted": stop("F1 only")},
        "d2": {"defect": True, "promoted": stop()},
        "c1": {"defect": False, "promoted": stop("F1 only")},
        "skip": {"defect": None, "promoted": stop("F1 only", "metrics only")},
    }
    summary = script.rates(outcomes)
    assert summary["F1 only"] == {"caught": 1, "defects": 2, "false_alarms": 1, "clean": 1}
    assert summary["metrics only"] == {"caught": 0, "defects": 2, "false_alarms": 0, "clean": 1}
    assert gates and set(summary) == set(script.GATES)


def test_ground_truth_comes_from_the_injected_defect_not_the_expected_decision(script):
    cases = script.load_cases()
    assert not script.is_defect(cases["good"]) and not script.is_defect(cases["hidden_group_null"])
    # PROMOTE is expected by the default profile for this one, but a defect is injected
    assert script.is_defect(cases["hidden_group"])
    assert all(script.is_defect(cases[n]) for n in
               ("leaky", "overfit", "biased", "miscalibrated", "corrupt",
                "good_incomplete_metadata", "good_no_sensitive"))


def test_the_full_comparison_on_the_sample_models(script, tmp_path):
    out = tmp_path / "baseline.md"
    assert script.main(["--out", str(out)]) == 0
    outcomes = script.collect(script.load_cases())
    summary = script.rates(outcomes)
    defects = summary["F1 only"]["defects"]
    assert defects == 8 and summary["F1 only"]["clean"] == 2        # `slow` is excluded
    # the claims the document makes
    assert summary["full gate (hidden_groups)"]["caught"] == defects
    assert summary["full gate (default)"]["caught"] == defects - 1  # misses hidden_group
    assert not outcomes["hidden_group"]["promoted"]["full gate (hidden_groups)"]
    assert outcomes["hidden_group"]["promoted"]["full gate (default)"]
    for gate in ("full gate (default)", "full gate (hidden_groups)"):
        assert summary[gate]["false_alarms"] == 0
    for name in ("leaky", "good_incomplete_metadata", "good_no_sensitive"):
        assert outcomes[name]["promoted"]["metrics only"]           # invisible to a metrics gate
        assert not outcomes[name]["promoted"]["full gate (default)"]
    assert summary["F1 only"]["caught"] < summary["full gate (default)"]["caught"]
    text = out.read_text(encoding="utf-8")
    assert "overstates what it detects" in text and "built by the project author" in text
    # the machine-readable copy the dashboard reads says the same thing
    import json
    study = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert study["study"] == "baseline_comparison" and study["generated_at_utc"]
    assert study["gates"] == list(script.GATES) and study["summary"] == summary
    assert {m["name"] for m in study["models"]} == set(outcomes)
    assert {m["name"]: m["truth"] for m in study["models"]}["hidden_group"] == "defect"
    assert {m["name"]: m["truth"] for m in study["models"]}["slow"] == "excluded"
    assert any("two clean models" in c for c in study["caveats"])
