import pandas as pd

from src.common.config import load_config
from src.c4.monitoring import data_quality

RULES = load_config("monitoring")["data_quality"]

BASELINE = {
    "numeric_features": ["hours"], "categorical_features": ["sex"],
    "numeric": {"hours": {"null_rate": 0.0, "min": 1, "max": 99}},
    "categorical": {"sex": {"null_rate": 0.0, "shares": {"Male": 0.5, "Female": 0.5}}},
}


def window(hours, sex):
    return pd.DataFrame({"hours": hours, "sex": sex})


def test_clean_window_does_not_fire():
    res = data_quality.check(window([40] * 100, ["Male", "Female"] * 50), BASELINE, RULES)
    assert res["fired"] is False and res["issues"] == []


def test_null_spike_fires():
    hours = [None] * 40 + [40] * 60
    res = data_quality.check(window(hours, ["Male"] * 100), BASELINE, RULES)
    assert res["fired"] and any("null rate" in i for i in res["issues"])


def test_wrong_units_fire_out_of_range():
    res = data_quality.check(window([2400] * 100, ["Male"] * 100), BASELINE, RULES)
    assert res["fired"] and any("outside baseline range" in i for i in res["issues"])


def test_unseen_category_fires():
    res = data_quality.check(window([40] * 100, ["Other"] * 100), BASELINE, RULES)
    assert res["fired"] and any("unseen" in i for i in res["issues"])
