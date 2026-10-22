import numpy as np
import pandas as pd
import pytest

from src.c4.monitoring import stats


def test_decile_edges_gives_nine_interior_edges():
    edges = stats.decile_edges(pd.Series(np.arange(1000)))
    assert len(edges) == 9
    assert edges == sorted(edges)


def test_bin_shares_sum_to_one_and_ignore_nan():
    edges = [2.0, 4.0]
    shares = stats.bin_shares(pd.Series([1, 1, 3, 5, np.nan]), edges)
    assert shares.sum() == pytest.approx(1.0)
    assert list(shares) == [0.5, 0.25, 0.25]


def test_bin_shares_empty_window_is_uniform():
    shares = stats.bin_shares(pd.Series([np.nan, np.nan]), [1.0, 2.0])
    assert shares == pytest.approx([1 / 3] * 3)


def test_psi_is_zero_for_identical_distributions():
    p = [0.25, 0.25, 0.25, 0.25]
    assert stats.psi(p, p) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift_and_is_symmetric():
    base = [0.25, 0.25, 0.25, 0.25]
    small = [0.30, 0.25, 0.25, 0.20]
    big = [0.60, 0.20, 0.10, 0.10]
    assert 0 < stats.psi(base, small) < stats.psi(base, big)
    assert stats.psi(base, big) > 0.25          # the usual "major shift" rule of thumb
    assert stats.psi(base, big) == pytest.approx(stats.psi(big, base))


def test_category_shares_ignore_missing_values():
    shares = stats.category_shares(pd.Series(["a", "a", "b", None]))
    assert shares == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_jsd_identical_is_zero_and_disjoint_is_near_one():
    same = {"a": 0.5, "b": 0.5}
    assert stats.jsd_from_dicts(same, same) == pytest.approx(0.0, abs=1e-3)
    assert stats.jsd_from_dicts({"a": 1.0}, {"b": 1.0}) > 0.95


def test_confidence_hist_sums_to_one_and_handles_empty():
    assert stats.confidence_hist([0.1, 0.5, 0.9, 0.95]).sum() == pytest.approx(1.0)
    assert stats.confidence_hist([]) == pytest.approx([0.1] * 10)


def test_classification_summary_values():
    out = stats.classification_summary([1, 1, 0, 0], [1, 0, 0, 0])
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["tpr"] == pytest.approx(0.5)
    assert out["tnr"] == pytest.approx(1.0)
    assert out["positive_share"] == pytest.approx(0.5)
    assert out["n"] == 4
