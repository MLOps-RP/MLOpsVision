"""Conflicting labels, training duplicates and CPU time, plus the extended profile."""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from mqvg.checks.base import CheckContext
from mqvg.checks.operational import cpu_time
from mqvg.checks.training_validity import eval_label_conflicts, train_duplicate_rows
from mqvg.config import load_config
from mqvg.loader import LoadedCandidate
from mqvg.pipeline import run_validation
from mqvg.schemas import (
    CandidateBundle, CheckStatus, Decision, Dimension, EvidenceSource,
)

from conftest import CONFIGS, FIXTURES, ROOT

CONFLICTS = "training_validity.eval_label_conflicts"
DUPLICATES = "training_validity.train_duplicate_rows"
SAFETY = "operational.model_file_safety"
CPU = "operational.cpu_time"
EXTENDED_CHECKS = {CONFLICTS, DUPLICATES, SAFETY, CPU}


def candidate(eval_df=None, model=None, hashes=None, target="y", features=("a", "b")):
    bundle = CandidateBundle(candidate_id="t", model_path=Path("m.joblib"),
                             eval_data_path=Path("e.csv"), target_column=target,
                             train_row_hashes_path=hashes)
    stub = model or SimpleNamespace(feature_names_in_=list(features), predict=lambda X: np.zeros(len(X)))
    return LoadedCandidate(bundle=bundle, model=stub, eval_df=eval_df)


def run(check, cand, config):
    return check(CheckContext(candidate=cand, config=config))


@pytest.fixture
def config(make_config):
    return make_config(lambda d: [d["checks"][i].update(enabled=True) for i in EXTENDED_CHECKS])


# ------------------------------------------------------------- conflicting labels
def frame(rows):
    return pd.DataFrame(rows, columns=["a", "b", "y"])


def test_no_conflicts_is_a_pass(config):
    df = frame([(i, "x", i % 2) for i in range(50)])
    result = run(eval_label_conflicts, candidate(df), config)
    assert result.status == CheckStatus.PASS and result.raw_value == 0
    assert result.score == 100 and result.evidence_source == EvidenceSource.MEASURED


def test_the_fraction_counts_every_row_of_a_conflicting_group(config):
    # groups: (1,x): labels 0,1,0 -> conflict (3 rows); (2,y): 1,1 -> agree; rest unique
    rows = [(1, "x", 0), (1, "x", 1), (1, "x", 0), (2, "y", 1), (2, "y", 1)]
    rows += [(10 + i, "z", i % 2) for i in range(15)]
    result = run(eval_label_conflicts, candidate(frame(rows)), config)
    assert result.raw_value == pytest.approx(3 / 20)
    d = result.details
    assert (d["conflict_rows"], d["conflicting_groups"], d["repeated_groups"]) == (3, 1, 2)
    assert d["rows_in_repeated_groups"] == 5 and d["eval_rows"] == 20
    assert result.status == CheckStatus.FAIL                    # 15% is past fail_at 3%
    assert result.score == 0                                    # and at the 15% floor


def test_agreeing_duplicates_are_not_conflicts(config):
    df = frame([(1, "x", 1)] * 10 + [(i + 5, "y", 0) for i in range(10)])
    result = run(eval_label_conflicts, candidate(df), config)
    assert result.raw_value == 0 and result.details["repeated_groups"] == 1


def test_missing_values_in_features_match_each_other(config):
    df = frame([(np.nan, "x", 0), (np.nan, "x", 1)] + [(i, "z", 0) for i in range(8)])
    result = run(eval_label_conflicts, candidate(df), config)
    assert result.details["conflict_rows"] == 2


def test_the_score_follows_the_thresholds(config):
    th = config.checks[CONFLICTS].thresholds
    rows = [(1, "x", 0), (1, "x", 1)] + [(10 + i, "z", 0) for i in range(98)]
    result = run(eval_label_conflicts, candidate(frame(rows)), config)
    fraction = 2 / 100
    assert result.raw_value == pytest.approx(fraction)
    assert result.score == pytest.approx((th.floor - fraction) / (th.floor - th.target) * 100)
    assert result.status == CheckStatus.WARN                    # between 0.5% and 3%


@pytest.mark.parametrize("what, cand", [
    ("no evaluation data", candidate(None)),
    ("no target column", candidate(frame([(1, "x", 0)]), target="missing")),
    ("missing feature columns", candidate(frame([(1, "x", 0)]),
                                          model=SimpleNamespace(feature_names_in_=["a", "q"]))),
    ("no feature columns", candidate(frame([(1, "x", 0)]),
                                     model=SimpleNamespace(predict=lambda X: X))),
    ("no labels", candidate(pd.DataFrame({"a": [1], "b": ["x"], "y": [np.nan]}))),
])
def test_conflicts_are_unavailable_never_a_pass(config, what, cand):
    result = run(eval_label_conflicts, cand, config)
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None, what
    assert result.evidence_source == EvidenceSource.UNAVAILABLE


def test_the_conflict_check_works_without_a_loaded_model_via_the_metadata_columns(config):
    cand = candidate(frame([(1, "x", 0), (1, "x", 1)] + [(i + 5, "z", 0) for i in range(8)]))
    cand.model = None
    cand.metadata = SimpleNamespace(feature_columns=["a", "b"])
    result = run(eval_label_conflicts, cand, config)
    assert result.details["conflict_rows"] == 2


# --------------------------------------------------------------- training duplicates
def hash_file(tmp_path, lines):
    path = tmp_path / "hashes.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_duplicates_are_counted_from_the_hashes(config, tmp_path):
    lines = [f"h{i}" for i in range(18)] + ["h0", "h1"]            # 20 rows, 2 repeats
    result = run(train_duplicate_rows, candidate(hashes=hash_file(tmp_path, lines)), config)
    assert result.raw_value == pytest.approx(0.1)
    assert result.details["duplicate_rows"] == 2 and result.details["distinct_rows"] == 18
    assert result.evidence_source == EvidenceSource.DECLARED       # the hashes are producer-supplied
    assert result.status == CheckStatus.WARN                       # exactly fail_at is not past it


def test_no_duplicates_is_a_pass(config, tmp_path):
    result = run(train_duplicate_rows,
                 candidate(hashes=hash_file(tmp_path, [f"h{i}" for i in range(30)])), config)
    assert result.status == CheckStatus.PASS and result.score == 100


def test_many_duplicates_fail(config, tmp_path):
    result = run(train_duplicate_rows,
                 candidate(hashes=hash_file(tmp_path, ["a"] * 4 + ["b"] * 4 + ["c", "d"])), config)
    assert result.status == CheckStatus.FAIL and result.raw_value == pytest.approx(0.6)


def test_blank_lines_are_ignored(config, tmp_path):
    result = run(train_duplicate_rows,
                 candidate(hashes=hash_file(tmp_path, ["a", "", "b", "  ", "a"])), config)
    assert result.details["training_rows"] == 3


@pytest.mark.parametrize("what", ["no file", "missing file", "empty file"])
def test_duplicates_are_unavailable_never_a_pass(config, tmp_path, what):
    hashes = {"no file": None, "missing file": tmp_path / "nope.txt",
              "empty file": hash_file(tmp_path, [""])}[what]
    result = run(train_duplicate_rows, candidate(hashes=hashes), config)
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None


# ------------------------------------------------------------------------- CPU time
class Burner:
    """Spins for about `ms` per predict call, on the precise wall clock.

    (The OS process-CPU clock ticks in steps of about 15 ms on Windows, so
    spinning on it would make the burn itself coarse.)"""

    def __init__(self, ms):
        self.ms, self.feature_names_in_ = ms, ["a", "b"]

    def predict(self, X):
        end = time.perf_counter() + self.ms / 1000
        while time.perf_counter() < end:
            pass
        return np.zeros(len(X))


@pytest.fixture
def fast_cpu_config(make_config):
    def tweak(d):
        d["checks"][CPU]["enabled"] = True
        d["evaluation"].update(cpu_sample_rows=10, cpu_min_seconds=0.1, cpu_max_rounds=100)
    return make_config(tweak)


def test_cpu_time_reports_milliseconds_of_cpu_per_row(fast_cpu_config):
    df = frame([(i, "x", 0) for i in range(10)])
    result = run(cpu_time, candidate(df, model=Burner(10)), fast_cpu_config)
    d = result.details
    assert result.raw_value == pytest.approx(d["cpu_seconds"] / d["rows_predicted"] * 1000)
    assert result.raw_value == pytest.approx(1.0, rel=0.4)         # 10 ms for 10 rows = 1 ms/row
    assert d["rows_predicted"] == d["rounds"] * 10 and d["wall_seconds"] >= 0.1
    assert result.status == CheckStatus.WARN                       # 0.5 < 1 ms <= 3 ms
    assert result.evidence_source == EvidenceSource.MEASURED


def test_a_heavier_model_costs_more_cpu(fast_cpu_config):
    df = frame([(i, "x", 0) for i in range(10)])
    light = run(cpu_time, candidate(df, model=Burner(1)), fast_cpu_config)
    heavy = run(cpu_time, candidate(df, model=Burner(20)), fast_cpu_config)
    assert heavy.raw_value > 8 * light.raw_value
    assert heavy.status == CheckStatus.WARN and heavy.score < light.score


def test_a_very_heavy_model_fails(fast_cpu_config):
    df = frame([(i, "x", 0) for i in range(10)])
    result = run(cpu_time, candidate(df, model=Burner(60)), fast_cpu_config)
    assert result.status == CheckStatus.FAIL                       # 6 ms/row is past 3 ms


def test_cpu_time_uses_only_the_configured_sample(fast_cpu_config):
    df = frame([(i, "x", 0) for i in range(500)])
    result = run(cpu_time, candidate(df, model=Burner(1)), fast_cpu_config)
    assert result.details["rows_per_round"] == 10


@pytest.mark.parametrize("what", ["no model", "no data"])
def test_cpu_time_is_unavailable_without_a_model_or_data(fast_cpu_config, what):
    cand = candidate(frame([(1, "x", 0)]), model=Burner(1))
    if what == "no model":
        cand.model, cand.model_error = None, "gone"
    else:
        cand.eval_df, cand.eval_error = None, "no csv"
    result = run(cpu_time, cand, fast_cpu_config)
    assert result.status == CheckStatus.UNAVAILABLE and result.score is None


# ------------------------------------------------------- the profile and the default
def test_the_default_profile_leaves_all_four_checks_off():
    config = load_config(CONFIGS / "default.yaml")
    assert not any(config.checks[i].enabled for i in EXTENDED_CHECKS)


def test_the_default_run_does_not_contain_or_perform_the_new_checks():
    config = load_config(CONFIGS / "default.yaml")
    path = FIXTURES / "good" / "model.joblib"
    if not path.exists():
        pytest.skip("fixtures not generated")
    from conftest import ROOT as root
    result = run_validation(CandidateBundle(
        candidate_id="good", model_path=path, eval_data_path=root / "data" / "eval.csv",
        metadata_path=FIXTURES / "good" / "metadata.json",
        train_row_hashes_path=root / "data" / "train_row_hashes.txt"), config)
    assert not EXTENDED_CHECKS & {c.id for c in result.checks}
    assert len(result.checks) == 21


def test_the_extended_profile_enables_exactly_the_four_checks():
    default = load_config(CONFIGS / "default.yaml")
    extended = load_config(CONFIGS / "extended.yaml")
    assert {i for i, c in extended.checks.items() if c.enabled} - \
        {i for i, c in default.checks.items() if c.enabled} == EXTENDED_CHECKS
    for check_id, check in default.checks.items():
        if check_id not in EXTENDED_CHECKS:
            assert extended.checks[check_id] == check, check_id
    assert extended.weights == default.weights
    assert extended.missing_policy == default.missing_policy
    assert extended.coverage == default.coverage
    assert extended.decision.promote_at == default.decision.promote_at
    assert extended.decision.reject_below == default.decision.reject_below


def test_the_extended_profile_adds_policy_without_dropping_the_default_rules():
    default = load_config(CONFIGS / "default.yaml")
    extended = load_config(CONFIGS / "extended.yaml")
    assert set(default.decision.must_pass_checks) < set(extended.decision.must_pass_checks)
    assert set(extended.decision.must_pass_checks) - set(default.decision.must_pass_checks) == {
        CONFLICTS, SAFETY}
    assert [r.id for r in default.critical_rules] == [r.id for r in extended.critical_rules][:4]
    assert extended.critical_rules[-1].id == "unsafe_model_file"
    assert extended.checks[SAFETY].required and extended.model_safety.refuse_at == "unrecognised"


def test_the_new_checks_sit_in_the_documented_dimensions():
    from mqvg.catalogue import CHECKS_BY_ID
    assert CHECKS_BY_ID[CONFLICTS].dimension == Dimension.TRAINING_VALIDITY
    assert CHECKS_BY_ID[DUPLICATES].dimension == Dimension.TRAINING_VALIDITY
    assert CHECKS_BY_ID[SAFETY].dimension == Dimension.OPERATIONAL_READINESS
    assert CHECKS_BY_ID[CPU].dimension == Dimension.OPERATIONAL_READINESS


def test_the_extended_scores_use_the_new_checks_in_their_dimension_averages(make_config):
    from builders import make_results
    from mqvg.scoring import compute_scorecard
    config = make_config(lambda d: [d["checks"][i].update(enabled=True) for i in EXTENDED_CHECKS])
    clean = compute_scorecard(make_results(config), config)
    bad = compute_scorecard(make_results(
        config, scores={CONFLICTS: {"status": "FAIL", "score": 0, "raw": 0.15}}), config)
    tv = Dimension.TRAINING_VALIDITY
    weights = {i: config.checks[i].weight for i in (
        "training_validity.metadata_completeness", "training_validity.reproducibility_info",
        "training_validity.overfit_gap", "training_validity.train_eval_overlap",
        "leakage.single_feature_predictiveness", CONFLICTS, DUPLICATES)}
    expected_drop = 100 * weights[CONFLICTS] / sum(weights.values())
    assert clean.dimensions[tv].score - bad.dimensions[tv].score == pytest.approx(expected_drop)


# ------------------------------------------------------------------ the fixtures
def _manifest():
    return yaml.safe_load((FIXTURES / "extended_manifest.yaml").read_text(encoding="utf-8"))["fixtures"]


def _ready(name):
    entry = _manifest()[name]
    needed = [ROOT / entry[k] for k in ("model", "eval_data", "train_row_hashes")]
    if not all(p.exists() for p in needed):
        pytest.skip("extended fixtures not generated (run scripts/make_extended_fixtures.py)")
    return entry


def _run(entry, name, profile):
    return run_validation(CandidateBundle(
        candidate_id=name, model_path=ROOT / entry["model"], eval_data_path=ROOT / entry["eval_data"],
        metadata_path=ROOT / entry["metadata"], train_row_hashes_path=ROOT / entry["train_row_hashes"]),
        load_config(CONFIGS / f"{profile}.yaml"))


@pytest.mark.parametrize("name, profile", [
    ("label_conflicts", "default"), ("label_conflicts", "extended"),
    ("duplicate_training", "default"), ("duplicate_training", "extended"),
    ("unreviewed_model", "default"), ("unreviewed_model", "extended"),
    ("unsafe_model", "extended"),       # the default run would execute the payload
])
def test_the_extended_fixture_decisions_hold(name, profile):
    entry = _ready(name)
    result = _run(entry, name, profile)
    assert result.decision.decision.value == entry["expected_decision"][profile], (
        name, profile, [r.rule_id for r in result.decision.triggered_rules])


def test_each_fixture_produces_its_intended_finding_under_the_extended_profile():
    expected = {"label_conflicts": (CONFLICTS, CheckStatus.FAIL),
                "duplicate_training": (DUPLICATES, CheckStatus.FAIL),
                "unsafe_model": (SAFETY, CheckStatus.FAIL),
                "unreviewed_model": (SAFETY, CheckStatus.WARN)}
    for name, (check_id, status) in expected.items():
        result = _run(_ready(name), name, "extended")
        got = {c.id: c for c in result.checks}[check_id]
        assert got.status == status, name
        assert got.score is not None or status == CheckStatus.UNAVAILABLE


def test_the_unsafe_fixture_is_stopped_by_the_scan_and_its_payload_never_runs(capfd):
    entry = _ready("unsafe_model")
    result = _run(entry, "unsafe_model", "extended")
    out = capfd.readouterr().out
    assert "PAYLOAD EXECUTED" not in out
    ids = [r.rule_id for r in result.decision.triggered_rules]
    assert "unsafe_model_file" in ids and result.decision.decision == Decision.ROLLBACK
    checks = {c.id: c for c in result.checks}
    assert checks[SAFETY].details["loaded"] is False
    assert checks["operational.model_loadable"].status == CheckStatus.FAIL
    assert "NOT loaded" in checks["operational.model_loadable"].message
    assert result.decision.action_mode.value == "RECOMMENDATION"


def test_the_unreviewed_fixture_asks_for_review_instead_of_rolling_back():
    result = _run(_ready("unreviewed_model"), "unreviewed_model", "extended")
    checks = {c.id: c for c in result.checks}
    assert checks[SAFETY].status == CheckStatus.WARN
    assert checks["operational.model_loadable"].status == CheckStatus.UNAVAILABLE
    assert result.decision.decision == Decision.HOLD
    assert "unsafe_model_file" not in [r.rule_id for r in result.decision.triggered_rules]


def test_the_corrupt_fixture_is_still_a_rollback_under_the_extended_profile():
    path = FIXTURES / "corrupt" / "model.joblib"
    if not path.exists():
        pytest.skip("fixtures not generated")
    result = run_validation(CandidateBundle(
        candidate_id="corrupt", model_path=path, eval_data_path=ROOT / "data" / "eval.csv",
        metadata_path=FIXTURES / "corrupt" / "metadata.json",
        train_row_hashes_path=ROOT / "data" / "train_row_hashes.txt"),
        load_config(CONFIGS / "extended.yaml"))
    assert result.decision.decision == Decision.ROLLBACK
    checks = {c.id: c for c in result.checks}
    assert checks[SAFETY].status == CheckStatus.UNAVAILABLE and checks[SAFETY].score is None


def test_the_extended_manifest_findings_use_catalogue_check_ids():
    from mqvg.catalogue import CHECKS_BY_ID
    for name, entry in _manifest().items():
        assert all(c in CHECKS_BY_ID for c in entry["expected_findings"]), name


def test_the_original_manifest_is_untouched():
    text = (FIXTURES / "manifest.yaml").read_text(encoding="utf-8")
    assert "label_conflicts" not in text and "unsafe_model" not in text


def test_the_extended_profile_runs_all_four_checks_and_never_passes_missing_evidence(tiny_bundle):
    result = run_validation(tiny_bundle, load_config(CONFIGS / "extended.yaml"))
    checks = {c.id: c for c in result.checks}
    assert EXTENDED_CHECKS <= set(checks)
    assert checks[SAFETY].status == CheckStatus.PASS and checks[CPU].status == CheckStatus.PASS
    # the tiny bundle has no training row hashes: that evidence is missing, not clean
    assert checks[DUPLICATES].status == CheckStatus.UNAVAILABLE and checks[DUPLICATES].score is None
