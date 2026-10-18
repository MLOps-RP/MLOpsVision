import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mqvg import synthetic as syn
from mqvg.config import HiddenGroupsConfig
from mqvg.hidden_groups import (
    bh_adjust, build_features, confirm, detect, expected_errors, fold_indices,
    format_rule, platt_recalibrator, search, test_group as group_test,
    wilson_interval,
)

# `test_group` is an algorithm function, not a pytest test; imported under another name.


def positive_probability(model, frame):
    return model.predict_proba(frame[syn.FEATURES])[:, list(model.classes_).index(syn.POSITIVE)]


def scenario(pattern, n_eval=3000, seed=7, mode="misspec", strength=syn.DEFAULT_STRENGTH,
             fit=syn.fit_logistic, n_train=5000):
    train, evaluation, region = syn.make_pair(n_train, n_eval, seed, pattern, mode, strength)
    model = fit(train)
    p = positive_probability(model, evaluation)
    return {"X": evaluation[syn.FEATURES], "y": (evaluation[syn.TARGET] == syn.POSITIVE).to_numpy(),
            "pred": p >= 0.5, "p": p, "region": region, "frame": evaluation}


def run(s, cfg=None, seed=42):
    cfg = cfg or HiddenGroupsConfig()
    return detect(s["X"], s["y"], s["pred"], s["p"], cfg, seed=seed)


def recall(detection, region):
    covered = detection.covered_rows()
    return (covered & region).sum() / region.sum()


# ------------------------------------------------------------- statistics
def test_benjamini_hochberg_by_hand():
    # sorted p: .005 .01 .03 .04 -> times m/rank: .02 .02 .04 .04
    q = bh_adjust([0.01, 0.04, 0.03, 0.005])
    assert q.tolist() == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_benjamini_hochberg_properties():
    rng = np.random.default_rng(0)
    p = rng.random(50)
    q = bh_adjust(p)
    assert np.all(q >= p - 1e-12) and np.all(q <= 1)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)              # monotone in p
    assert bh_adjust([]).size == 0
    assert bh_adjust([0.3]).tolist() == [0.3]


def test_wilson_interval_values():
    low, high = wilson_interval(5, 10)
    assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert wilson_interval(0, 20)[0] == 0.0 and wilson_interval(20, 20)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_expected_errors_follow_the_prediction():
    expected = expected_errors(np.array([True, False, True]), np.array([0.8, 0.8, 0.3]))
    assert expected.tolist() == pytest.approx([0.2, 0.8, 0.7])


def test_group_test_by_hand():
    err = np.zeros(200, bool)
    err[:30] = True                                          # 30 errors in 100 rows
    expected = np.full(200, 0.2)
    members = np.arange(100)
    g = group_test("tree", 0, "rule", members, err, expected, 200, 0.1)
    # mean 20, variance 100*.2*.8 = 16, z = (30 - 0.5 - 20) / 4 = 2.375
    assert g.p_value == pytest.approx(norm.sf(2.375))
    assert g.observed_rate == 0.3 and g.expected_rate == pytest.approx(0.2)
    assert g.effect == pytest.approx(0.1) and g.share == 0.5
    low, high = wilson_interval(30, 100)
    assert (g.effect_ci_low, g.effect_ci_high) == pytest.approx((low - 0.2, high - 0.2))


def test_a_group_with_fewer_errors_than_expected_is_not_significant():
    err = np.zeros(100, bool)
    g = group_test("tree", 0, "r", np.arange(100), err, np.full(100, 0.3), 100, 0.1)
    assert g.p_value > 0.99 and g.effect < 0


# ---------------------------------------------------------------- features
def test_build_features_encodes_without_labels():
    X = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0] * 5,
        "cat": ["a", "b", "a", None] * 5,
        "flag": [True, False, True, True] * 5,
        "const": [7] * 20,
        "same_text": ["x"] * 20,
        "sentence": [f"this is free text number {i}" for i in range(20)]})
    f = build_features(X, max_categories=20)
    assert f.used == ["num", "cat", "flag"]
    assert f.excluded == {"const": "constant", "same_text": "constant",
                          "sentence": "free text or identifier-like"}
    assert f.names == ["num", "cat=a", "cat=__missing__", "cat=b", "flag"] or \
        set(f.names) == {"num", "cat=a", "cat=b", "cat=__missing__", "flag"}
    assert not np.isnan(f.matrix).any()
    assert f.matrix[2, 0] == pytest.approx(np.nanmedian([1, 2, 4]))     # median imputation


def test_high_cardinality_categories_are_pooled_and_reported():
    X = pd.DataFrame({"c": [f"v{i % 50}" for i in range(500)], "n": np.arange(500.0)})
    f = build_features(X, max_categories=10)
    assert sum(name.startswith("c=") for name in f.names) == 10
    assert f.pooled == {"c": 40}


def test_feature_encoding_is_deterministic_with_ties():
    X = pd.DataFrame({"c": ["b", "a", "c", "b", "a", "c"] * 5})
    assert build_features(X, 20).names == build_features(X.iloc[::-1], 20).names


def test_format_rule_reads_naturally():
    X = pd.DataFrame({"age": np.arange(40.0), "seg": ["d", "e"] * 20, "r": list("abcd") * 10})
    f = build_features(X, 20)
    index = {name: i for i, name in enumerate(f.names)}
    rule = format_rule([(index["age"], ">", 20.0), (index["age"], "<=", 35.0),
                        (index["seg=d"], ">", 0.5), (index["r=a"], "<=", 0.5),
                        (index["r=b"], "<=", 0.5)], f)
    assert rule == "20 < age <= 35 AND seg = d AND r not in {a, b}"
    assert format_rule([], f) == "all rows"
    assert format_rule([(index["r=a"], "<=", 0.5)], f) == "r != a"
    assert format_rule([(index["age"], ">", 31.5)], f) == "age > 31.5"


# ------------------------------------------------------------ cross-fitting
def test_folds_partition_the_rows_and_are_seeded():
    folds = fold_indices(101, 3, seed=5)
    assert sorted(np.concatenate(folds)) == list(range(101))
    assert sum(len(f) for f in folds) == 101 and max(len(f) for f in folds) - min(len(f) for f in folds) <= 1
    assert all(np.array_equal(a, b) for a, b in zip(folds, fold_indices(101, 3, seed=5)))
    assert not np.array_equal(folds[0], fold_indices(101, 3, seed=6)[0])


def test_every_group_is_tested_only_on_its_held_out_fold():
    s = scenario("two_feature")
    cfg = HiddenGroupsConfig()
    feats = build_features(s["X"], cfg.max_categories)
    groups = search(feats, s["y"], s["pred"], s["p"], cfg, seed=42)
    folds = fold_indices(len(s["y"]), cfg.folds, 42)
    assert groups
    for g in groups:
        assert set(g.members) <= set(folds[g.fold])          # never a discovery row
        assert g.discovery_effect > 0                         # chosen on discovery rows only


@pytest.mark.parametrize("recalibrate, temperature", [(False, 1.0), (True, 1.0), (True, 1.6)])
def test_a_groups_discovery_never_uses_its_own_test_fold_labels(recalibrate, temperature):
    """Changing the labels of fold f must not change what is discovered for fold f.

    With temperature 1.6 the probabilities are over-confident, so recalibration
    is triggered; it is fitted on discovery rows only and must obey the same rule.
    """
    s = scenario("two_feature")
    p = _tempered(s["p"], temperature)
    cfg = HiddenGroupsConfig(recalibrate=recalibrate)
    feats = build_features(s["X"], cfg.max_categories)
    before = search(feats, s["y"], s["pred"], p, cfg, seed=42)

    folds = fold_indices(len(s["y"]), cfg.folds, 42)
    altered = s["y"].copy()
    altered[folds[0]] = ~altered[folds[0]]                    # scramble fold 0's labels
    after = search(feats, altered, s["pred"], p, cfg, seed=42)

    rules = lambda gs, fold: [(g.method, g.rule) for g in gs if g.fold == fold]   # noqa: E731
    assert rules(before, 0) == rules(after, 0)                # fold 0 discovered from fold 1 only
    assert rules(before, 1) != rules(after, 1)                # fold 1 discovery DID see fold 0


# ----------------------------------------------------------- recalibration
def _tempered(p, temperature):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)) * temperature
    return 1 / (1 + np.exp(-z))


def test_platt_map_corrects_overconfidence_and_a_shifted_intercept():
    from mqvg.checks.calibration import reliability_summary
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 20000)
    y = rng.random(20000) < true_p
    for distorted in (_tempered(true_p, 2.5), 1 / (1 + np.exp(-(np.log(true_p / (1 - true_p)) + 1.5)))):
        before = reliability_summary(y, distorted, 10)["ece"]
        after = reliability_summary(y, platt_recalibrator(distorted, y)(distorted), 10)["ece"]
        assert before > 0.1 and after < 0.02


def test_platt_map_leaves_calibrated_probabilities_nearly_unchanged():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.random(20000) < p
    assert np.max(np.abs(platt_recalibrator(p, y)(p) - p)) < 0.03


def test_platt_map_is_the_identity_when_there_is_one_class():
    p = np.array([0.2, 0.7])
    assert platt_recalibrator(p, np.array([True, True]))(p).tolist() == [0.2, 0.7]


def test_recalibration_prevents_false_alarms_from_moderate_miscalibration():
    """Without it a model with ECE 0.05-0.08 looks full of 'hidden groups'."""
    alarms = {True: 0, False: 0}
    for seed in range(15):
        s = scenario(None, n_eval=2000, seed=700 + seed, n_train=3000)
        p = _tempered(s["p"], 1.5)                            # moderately over-confident
        for recalibrate in (True, False):
            d = detect(s["X"], s["y"], p >= 0.5, p, HiddenGroupsConfig(recalibrate=recalibrate), seed=seed)
            alarms[recalibrate] += bool(d.available and d.confirmed)
    assert alarms[True] <= 3 and alarms[False] >= 8, alarms


def test_recalibration_is_applied_only_when_the_discovery_rows_are_miscalibrated(monkeypatch):
    from mqvg import hidden_groups
    calls = []
    real = hidden_groups.platt_recalibrator

    def spy(score, outcome):
        calls.append(len(score))
        return real(score, outcome)

    monkeypatch.setattr(hidden_groups, "platt_recalibrator", spy)
    s = scenario(None)
    run(s)
    assert calls == []                                        # calibrated: left alone
    p = _tempered(s["p"], 1.6)
    detect(s["X"], s["y"], p >= 0.5, p, HiddenGroupsConfig(), seed=42)
    assert len(calls) == 2                                    # one fit per fold
    calls.clear()
    detect(s["X"], s["y"], p >= 0.5, p, HiddenGroupsConfig(recalibrate=False), seed=42)
    assert calls == []                                        # and it can be switched off


def test_recalibration_does_not_hide_a_real_group():
    s = scenario("two_feature", mode="shift", fit=syn.fit_gradient_boosting)
    d = run(s)
    assert d.available and d.confirmed and recall(d, s["region"]) >= 0.5


# ------------------------------------------------------------------ power
@pytest.mark.parametrize("methods", [("tree",), ("tree", "kmeans")])
def test_a_planted_group_is_found_and_described(methods):
    s = scenario("two_feature")
    d = run(s, HiddenGroupsConfig(methods=list(methods)))
    assert d.available and d.confirmed
    assert recall(d, s["region"]) >= 0.5
    assert d.max_effect >= 0.10
    tree_rules_ = [g.rule for g in d.confirmed if g.method == "tree"]
    assert any("segment = d" in r and "tenure_years" in r for r in tree_rules_)


def test_the_tree_finds_a_numeric_threshold_region_and_kmeans_alone_may_not():
    """K-means clusters on overall similarity, so a region defined by one numeric
    feature among many is often not a cluster. This is a known limitation,
    quantified in docs/hidden_groups.md; the tree and the combination find it."""
    s = scenario("single_threshold")
    for methods in (["tree"], ["tree", "kmeans"]):
        d = run(s, HiddenGroupsConfig(methods=methods))
        assert d.available and d.confirmed and recall(d, s["region"]) >= 0.5, methods
    assert run(s, HiddenGroupsConfig(methods=["kmeans"])).available


def test_residual_weighted_kmeans_finds_a_categorical_region():
    found = 0
    for seed in range(100, 108):
        s = scenario("categorical_pair", seed=seed)
        d = run(s, HiddenGroupsConfig(methods=["kmeans"]), seed=seed - 100)
        found += bool(d.confirmed) and recall(d, s["region"]) >= 0.5
    assert found >= 4                                       # development seeds; see docs/hidden_groups.md


def test_the_weighting_option_changes_the_clusters_but_stays_valid():
    s = scenario("categorical_pair")
    weighted = run(s, HiddenGroupsConfig(methods=["kmeans"], kmeans_weighting="residual"))
    plain = run(s, HiddenGroupsConfig(methods=["kmeans"], kmeans_weighting="none"))
    assert weighted.available and plain.available
    assert [g.rule for g in weighted.tests] != [g.rule for g in plain.tests]


def test_reported_statistics_are_self_consistent():
    d = run(scenario("two_feature"))
    for g in d.confirmed:
        assert g.q_value <= 0.05 and g.effect >= 0.10
        assert g.effect_ci_low < g.effect < g.effect_ci_high
        assert g.observed_rate == pytest.approx(g.observed_errors / g.n_rows)
        assert g.effect == pytest.approx(g.observed_rate - g.expected_rate)
        assert g.q_value >= g.p_value - 1e-12                 # adjustment never lowers p
    assert d.max_effect == max(g.effect for g in d.confirmed)


def test_the_mean_expected_error_of_confirmed_groups_is_below_what_was_observed():
    d = run(scenario("two_feature"))
    assert all(g.observed_rate > g.expected_rate for g in d.confirmed)


# ---------------------------------------------------- false alarms (null)
def test_false_alarm_rate_on_data_with_no_hidden_group():
    """30 independent null datasets; the nominal rate is 5% so up to 20% is allowed."""
    alarms = {"tree": 0, "kmeans": 0, "both": 0}
    for seed in range(30):
        s = scenario(None, n_eval=2000, seed=500 + seed, n_train=3000)
        for name, methods in (("tree", ["tree"]), ("kmeans", ["kmeans"]),
                              ("both", ["tree", "kmeans"])):
            d = run(s, HiddenGroupsConfig(methods=methods), seed=seed)
            assert d.available
            alarms[name] += bool(d.confirmed)
    assert max(alarms.values()) <= 6, alarms                  # <= 20% of 30


def test_a_naive_same_data_test_would_have_failed_this():
    """The reason for cross-fitting: fit a tree and test it on the same rows."""
    from sklearn.tree import DecisionTreeRegressor
    false_alarms = 0
    for seed in range(15):
        s = scenario(None, n_eval=2000, seed=900 + seed, n_train=3000)
        feats = build_features(s["X"], 20)
        err = s["y"] != s["pred"]
        expected = expected_errors(s["pred"], s["p"])
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=60, random_state=0).fit(
            feats.matrix, err - expected)
        leaf = tree.apply(feats.matrix)
        groups = [group_test("tree", 0, "", np.where(leaf == l)[0], err, expected, len(err), 1)
                  for l in np.unique(leaf)]
        false_alarms += any(g.effect >= 0.10 and g.q_value is None and g.p_value < 0.05 / len(groups)
                            for g in groups)
    assert false_alarms >= 5     # selection bias makes the same-data test unreliable


# --------------------------------------------------------- significance + effect size
def test_both_significance_and_effect_size_are_required():
    s = scenario("two_feature")
    significant_only = run(s, HiddenGroupsConfig(min_effect=0.95))
    assert any(g.q_value <= 0.05 for g in significant_only.tests)       # significant...
    assert not significant_only.confirmed                                # ...but too small an effect
    assert significant_only.max_effect == 0.0
    assert run(s).confirmed


def test_a_stricter_alpha_confirms_fewer_groups():
    s = scenario("two_feature", strength=syn.SUBTLE_STRENGTH)
    loose = run(s, HiddenGroupsConfig(alpha=0.2, min_effect=0.05))
    strict = run(s, HiddenGroupsConfig(alpha=0.001, min_effect=0.05))
    assert len(strict.confirmed) <= len(loose.confirmed)


def test_methods_subset_recomputes_the_correction():
    s = scenario("two_feature")
    both = run(s, HiddenGroupsConfig(methods=["tree", "kmeans"]))
    cfg_tree = HiddenGroupsConfig(methods=["tree"])
    tree_only = run(s, cfg_tree)
    assert {g.method for g in tree_only.tests} == {"tree"}
    resubset = confirm(both.tests, ["tree"], cfg_tree)
    assert [(g.rule, round(g.q_value, 12), g.confirmed) for g in resubset] == \
           [(g.rule, round(g.q_value, 12), g.confirmed) for g in tree_only.tests]


def test_small_groups_are_never_tested():
    d = run(scenario("two_feature"))
    cfg = HiddenGroupsConfig()
    smallest = max(cfg.min_group_rows, math.ceil(cfg.min_group_fraction * 1500))
    assert d.tests and all(g.n_rows >= smallest for g in d.tests)


# ----------------------------------------------------------- reproducibility
def test_same_inputs_and_seed_give_identical_results():
    s = scenario("two_feature")
    a, b = run(s, seed=3), run(s, seed=3)
    key = lambda d: [(g.method, g.fold, g.rule, g.n_rows, g.p_value, g.q_value, g.confirmed)   # noqa: E731
                     for g in d.tests]
    assert key(a) == key(b) and a.ece == b.ece


def test_a_different_seed_changes_the_folds():
    s = scenario("two_feature")
    assert [g.rule for g in run(s, seed=1).tests] != [g.rule for g in run(s, seed=2).tests] \
        or [g.n_rows for g in run(s, seed=1).tests] != [g.n_rows for g in run(s, seed=2).tests]


# --------------------------------------------------------- unavailable paths
def test_missing_probabilities_make_it_unavailable():
    s = scenario("two_feature")
    d = detect(s["X"], s["y"], s["pred"], None, HiddenGroupsConfig())
    assert not d.available and "no probability scores" in d.reason


def test_a_small_evaluation_sample_makes_it_unavailable():
    s = scenario("two_feature", n_eval=800)
    d = run(s)
    assert not d.available and "only 800 evaluation rows" in d.reason and "1000" in d.reason


def test_too_few_errors_make_it_unavailable():
    s = scenario(None, n_eval=2000)
    perfect = s["y"].copy()                                    # a model that is never wrong
    d = detect(s["X"], s["y"], perfect, np.where(perfect, 0.99, 0.01), HiddenGroupsConfig())
    assert not d.available and "prediction errors" in d.reason


def test_grossly_miscalibrated_probabilities_make_it_unavailable():
    s = scenario(None)
    shifted = 1 / (1 + np.exp(-(np.log(s["p"] / (1 - s["p"])) + 3.0)))   # ECE well above 0.15
    d = detect(s["X"], s["y"], shifted >= 0.5, shifted, HiddenGroupsConfig())
    assert not d.available and "not calibrated" in d.reason and d.ece > 0.15


def test_moderately_miscalibrated_probabilities_are_corrected_not_refused():
    s = scenario(None)
    p = _tempered(s["p"], 1.6)
    d = detect(s["X"], s["y"], p >= 0.5, p, HiddenGroupsConfig())
    assert 0.03 < d.ece < 0.15 and d.available


def test_an_overall_error_rate_far_from_the_expected_rate_is_unavailable():
    s = scenario(None)
    overconfident = np.where(s["p"] >= 0.5, 0.999, 0.001)     # ECE moderate, errors underestimated
    d = detect(s["X"], s["y"], s["pred"], overconfident, HiddenGroupsConfig())
    assert not d.available and "not calibrated" in d.reason


def test_the_calibration_limits_are_configurable():
    s = scenario(None)
    shifted = 1 / (1 + np.exp(-(np.log(s["p"] / (1 - s["p"])) + 3.0)))
    relaxed = HiddenGroupsConfig(max_ece=1.0, max_overall_gap=1.0)
    assert detect(s["X"], s["y"], shifted >= 0.5, shifted, relaxed).available


def test_no_usable_feature_makes_it_unavailable():
    s = scenario(None)
    X = pd.DataFrame({"const": 1, "note": [f"free text row number {i}" for i in range(len(s["y"]))]})
    d = detect(X, s["y"], s["pred"], s["p"], HiddenGroupsConfig())
    assert not d.available and "no usable feature" in d.reason
    assert d.features_excluded == {"const": "constant", "note": "free text or identifier-like"}


def test_excluded_and_pooled_columns_are_reported_when_available():
    s = scenario("two_feature")
    X = s["X"].assign(const=5, notes=[f"free text number {i} here" for i in range(len(s["X"]))])
    d = detect(X, s["y"], s["pred"], s["p"], HiddenGroupsConfig())
    assert d.available and set(d.features_excluded) == {"const", "notes"}
    assert set(d.features_used) == set(syn.FEATURES)


# ----------------------------------------------------------- k-means output
def test_cluster_groups_are_described_by_their_distinctive_features():
    for seed in range(100, 108):
        s = scenario("categorical_pair", seed=seed)
        d = run(s, HiddenGroupsConfig(methods=["kmeans"]), seed=seed - 100)
        if d.confirmed:
            break
    profiles = [g.rule for g in d.confirmed]
    assert profiles and all(p.startswith("cluster of similar rows:") for p in profiles)
    assert any("segment = c" in p or "region = e" in p for p in profiles)
    assert all("(overall " in p for p in profiles)           # compared with the whole data


def test_confirmed_rows_never_include_a_row_twice_in_one_group():
    d = run(scenario("two_feature"))
    for g in d.tests:
        assert len(set(g.members)) == len(g.members) == g.n_rows


# ------------------------------------------- what was actually recalibrated
def test_the_detection_records_which_folds_were_recalibrated():
    s = scenario(None)
    p = _tempered(s["p"], 1.6)                                # clearly miscalibrated
    default = detect(s["X"], s["y"], p >= 0.5, p, HiddenGroupsConfig(), seed=1)
    assert default.recalibrated_folds == [True, True]
    never = HiddenGroupsConfig(recalibration_trigger_ece=0.5)  # trigger never reached
    assert detect(s["X"], s["y"], p >= 0.5, p, never, seed=1).recalibrated_folds == [False, False]
    off = HiddenGroupsConfig(recalibrate=False)
    assert detect(s["X"], s["y"], p >= 0.5, p, off, seed=1).recalibrated_folds == [False, False]


def test_the_recalibration_trigger_is_approximate_for_borderline_data():
    """ECE is biased upward on a half-size discovery fold, so a calibrated model can
    occasionally cross the trigger; this is harmless (a Platt map of calibrated
    probabilities is nearly the identity) and is recorded, not hidden."""
    s = scenario(None)
    d = run(s)
    assert len(d.recalibrated_folds) == 2 and d.ece < 0.03    # whole-sample ECE is within range
