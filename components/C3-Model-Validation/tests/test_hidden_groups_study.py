"""The evaluation script, the synthetic generators and the fixtures."""
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from mqvg import synthetic as syn

from conftest import FIXTURES, ROOT


def load_script():
    path = ROOT / "scripts" / "evaluate_hidden_groups.py"
    spec = importlib.util.spec_from_file_location("evaluate_hidden_groups", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_hidden_groups"] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- synthetic data
def test_the_generator_is_deterministic():
    a, ra = syn.make_dataset(500, 3, "two_feature")
    b, rb = syn.make_dataset(500, 3, "two_feature")
    pd.testing.assert_frame_equal(a, b)
    assert np.array_equal(ra, rb)


def test_a_null_dataset_has_no_region():
    _, region = syn.make_dataset(500, 3, None)
    assert not region.any()


@pytest.mark.parametrize("pattern, low, high", [
    ("two_feature", 0.05, 0.14), ("single_threshold", 0.05, 0.14),
    ("categorical_pair", 0.03, 0.10), ("oblique", 0.05, 0.16),
    ("small_group", 0.01, 0.04), ("three_way_categorical", 0.015, 0.045),
    ("young_low_income", 0.04, 0.10)])
def test_each_pattern_has_the_advertised_size(pattern, low, high):
    _, region = syn.make_dataset(30000, 1, pattern)
    assert low < region.mean() < high


def test_the_plant_makes_the_model_fail_inside_the_region_only():
    train, evaluation, region = syn.make_pair(5000, 4000, 5, "two_feature")
    model = syn.fit_logistic(train)
    wrong = (model.predict(evaluation[syn.FEATURES]) != evaluation[syn.TARGET]).to_numpy()
    assert wrong[region].mean() > wrong[~region].mean() + 0.2


def test_shift_mode_plants_in_the_evaluation_data_only():
    train, evaluation, region = syn.make_pair(5000, 4000, 5, "two_feature", mode="shift")
    clean_train, _ = syn.make_dataset(5000, 5, None)
    pd.testing.assert_frame_equal(train, clean_train)             # training data untouched
    flipped = (evaluation[syn.TARGET] != syn.make_dataset(4000, 5 + 10_000, None)[0][syn.TARGET])
    assert flipped[region].mean() > 0.5 and flipped[~region].mean() < 0.02


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        syn.make_pair(100, 100, 1, "two_feature", mode="bogus")


def test_held_out_patterns_are_a_separate_registered_set():
    assert set(syn.HELD_OUT_PATTERNS) <= set(syn.PATTERNS)
    assert set(syn.HELD_OUT_PATTERNS) == {"three_way_categorical", "young_low_income"}


# --------------------------------------------------------- evaluation script
def test_the_quick_study_runs_and_is_reproducible(tmp_path):
    script = load_script()
    runs = []
    for name in ("a", "b"):
        out, doc = tmp_path / name, tmp_path / f"{name}.md"
        assert script.main(["--quick", "--jobs", "1", "--out-dir", str(out), "--doc", str(doc)]) == 0
        runs.append((pd.read_csv(out / "results.csv"), doc.read_text(encoding="utf-8")))
    (df_a, text_a), (df_b, text_b) = runs
    comparable = ["scenario", "size", "rep", "method", "available", "flagged", "recall",
                  "precision", "n_confirmed", "max_effect"]
    pd.testing.assert_frame_equal(df_a[comparable], df_b[comparable])
    # every table matches exactly; only the run time (header) and the Cost section may differ
    tables = lambda text: text.split("## Detection power")[1].split("## Cost")[0]   # noqa: E731
    assert tables(text_a) == tables(text_b)


def test_the_quick_study_reports_every_method_and_ablation(tmp_path):
    script = load_script()
    script.main(["--quick", "--jobs", "1", "--out-dir", str(tmp_path / "o"),
                 "--doc", str(tmp_path / "d.md")])
    df = pd.read_csv(tmp_path / "o" / "results.csv")
    assert set(df.method) == {"tree", "kmeans", "both", "both, no recalibration",
                              "kmeans, unweighted"}
    text = (tmp_path / "d.md").read_text(encoding="utf-8")
    for heading in ("## Detection power", "## Localization", "## False alarms",
                    "## Ablations", "## Cost", "Synthetic data only"):
        assert heading in text
    assert set(df.kind) == {"planted", "control"}


def test_subsets_share_one_search_but_get_their_own_correction():
    script = load_script()
    from mqvg.config import HiddenGroupsConfig
    from mqvg.hidden_groups import detect
    train, evaluation, _ = syn.make_pair(5000, 3000, 7, "two_feature")
    model = syn.fit_logistic(train)
    p = model.predict_proba(evaluation[syn.FEATURES])[:, list(model.classes_).index(syn.POSITIVE)]
    y = (evaluation[syn.TARGET] == syn.POSITIVE).to_numpy()
    cfg = HiddenGroupsConfig()
    both = detect(evaluation[syn.FEATURES], y, p >= 0.5, p, cfg, seed=1)
    tree = script.detect_from(both, ["tree"], cfg)
    direct = detect(evaluation[syn.FEATURES], y, p >= 0.5, p,
                    HiddenGroupsConfig(methods=["tree"]), seed=1)
    assert [(g.rule, round(g.q_value, 10)) for g in tree.tests] == \
           [(g.rule, round(g.q_value, 10)) for g in direct.tests]


def test_the_study_scenarios_include_held_out_nulls_and_miscalibration():
    script = load_script()
    names = [s.name for s in script.SCENARIOS]
    assert any(s.held_out for s in script.SCENARIOS)
    assert any(s.kind == "control" and s.temperature for s in script.SCENARIOS)
    assert any(s.mode == "shift" and s.model == "gbm" for s in script.SCENARIOS)
    assert len(names) == len(set(names))
    assert script.SEED_BASE >= 20_000          # disjoint from the development seeds


# ------------------------------------------------------------------- fixtures
def _need(name):
    if not (FIXTURES / name / "model.joblib").exists():
        pytest.skip("hidden-group fixtures not generated "
                    "(run scripts/make_hidden_group_fixtures.py)")


def test_the_planted_fixture_really_fails_inside_the_group():
    _need("hidden_group")
    truth = json.loads((FIXTURES / "hidden_group" / "ground_truth.json").read_text())
    assert truth["pattern"] == "two_feature" and truth["rule_keywords"]
    m = truth["measured"]
    assert m["error_rate_inside"] > m["error_rate_outside"] + 0.15
    assert len(truth["planted_row_indices"]) / 3000 == pytest.approx(truth["region_share"])


def test_the_null_fixture_has_no_planted_rows():
    _need("hidden_group_null")
    truth = json.loads((FIXTURES / "hidden_group_null" / "ground_truth.json").read_text())
    assert truth["pattern"] is None and truth["planted_row_indices"] == []
    assert truth["measured"]["error_rate_inside"] is None


def test_the_fixture_metadata_is_complete_and_uses_the_models_features():
    _need("hidden_group")
    meta = json.loads((FIXTURES / "hidden_group" / "metadata.json").read_text())
    assert meta["feature_columns"] == syn.FEATURES and meta["sensitive_columns"] == [syn.SENSITIVE]
    for field in ("target_column", "positive_label", "hyperparameters", "training_metrics",
                  "train_data_sha256", "row_hash_columns", "versions", "code"):
        assert meta.get(field)


def test_existing_fixtures_were_not_modified_by_the_new_script():
    manifest = (FIXTURES / "manifest.yaml").read_text(encoding="utf-8")
    assert "hidden_group" not in manifest        # the scenario manifest is untouched


# ------------------------------------------------- the fixture manifest
def _manifest():
    import yaml
    return yaml.safe_load((FIXTURES / "hidden_groups_manifest.yaml").read_text(
        encoding="utf-8"))["fixtures"]


@pytest.mark.parametrize("profile", ["default", "hidden_groups_score_only", "hidden_groups",
                                     "hidden_groups_critical"])
def test_the_hidden_group_manifest_decisions_hold_for_every_profile(profile):
    from mqvg.config import load_config
    from mqvg.pipeline import run_validation
    from mqvg.schemas import CandidateBundle
    config = load_config(ROOT / "configs" / f"{profile}.yaml")
    for name, entry in _manifest().items():
        _need(name)
        result = run_validation(CandidateBundle(
            candidate_id=name, model_path=ROOT / entry["model"],
            eval_data_path=ROOT / entry["eval_data"], metadata_path=ROOT / entry["metadata"],
            train_row_hashes_path=ROOT / entry["train_row_hashes"]), config)
        assert result.decision.decision.value == entry["expected_decision"][profile], (
            name, profile, [r.rule_id for r in result.decision.triggered_rules])


def test_the_manifest_findings_use_catalogue_check_ids():
    from mqvg.catalogue import CHECKS_BY_ID
    for name, entry in _manifest().items():
        assert all(check in CHECKS_BY_ID for check in entry["expected_findings"]), name
