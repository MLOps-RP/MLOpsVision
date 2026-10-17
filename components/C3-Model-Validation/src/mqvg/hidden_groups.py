"""Hidden failure group detection.

Finds feature-defined groups of rows on which a model makes clearly more
errors than the model itself expects, beyond the overall metrics. It is
the algorithm only: arrays and a config in, findings out. The validation check
that wraps it is `mqvg.checks.hidden_failure`.

Method (see docs/hidden_groups.md for the justification and the evaluation):

1. **Expected error.** For every row the model's own probability gives the
   chance its prediction is wrong: ``1 - p`` if it predicted the positive
   class, ``p`` otherwise. If the probabilities are calibrated, the number of
   errors in any group is then approximately a sum of independent Bernoulli
   variables with known means, which gives an honest null hypothesis. A group
   is interesting only if it has MORE errors than that.
2. **Cross-fitted discovery.** The rows are split into folds. Candidate groups
   are discovered on all folds but one and TESTED on the held-out fold, then
   the roles rotate. A group is never evaluated on the rows used to find it;
   discovering and testing on the same rows gave a 30/30 false-alarm rate in
   the audit experiment.
3. **Two discovery methods.** A regression tree on the residual (error minus
   expected error) gives groups that are readable rules. K-means on the
   features gives groups with no axis-aligned shape (a cluster of similar
   rows, described by its most distinctive features). Discovery uses only
   features and the discovery-fold residuals; the test fold's labels and
   errors are never used.
4. **Test and correct.** Each held-out group gets a one-sided normal-approximation
   test of observed against expected errors (with continuity correction);
   Benjamini-Hochberg controls the false discovery rate over every group tested.
5. **Effect size.** A group is confirmed only if it is also significant AND
   its observed error rate exceeds the expected rate by at least `min_effect`.

Safeguards return "unavailable" instead of a result: no probability scores,
too few rows or errors, or probabilities that are not calibrated (the test is
invalid then).

Limitations (stated in the check output too): the null assumes calibrated
probabilities; groups are axis-aligned rules or clusters, so an oblique region
may be found only partly; the test is approximate for small groups; and a
group found is evidence of a problem to investigate, not proof of its cause.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from mqvg.config import HiddenGroupsConfig

Z_95 = 1.959963984540054
FREE_TEXT_REASON = "free text or identifier-like"
# Groups covering more of the data than this indicate a model that is poor or
# miscalibrated in general, not a hidden group.
MAX_COVERED_SHARE = 0.5


# --------------------------------------------------------------- statistics
def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n <= 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p, kind="stable")
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def expected_errors(y_pred: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Probability, according to the model, that each prediction is wrong."""
    return np.where(y_pred, 1.0 - y_score, y_score)


# ------------------------------------------------------------------ results
@dataclass(frozen=True)
class Group:
    """One tested group, with its statistics (computed on held-out rows)."""
    method: str                 # "tree" or "kmeans"
    fold: int                   # the held-out fold it was tested on
    rule: str                   # readable rule, or a cluster profile
    n_rows: int
    share: float                # of all rows
    observed_errors: int
    observed_rate: float
    expected_rate: float
    effect: float               # observed_rate - expected_rate
    effect_ci_low: float        # Wilson interval of the rate, minus the expected rate
    effect_ci_high: float
    p_value: float
    discovery_effect: float     # mean residual on the discovery rows (a pre-selection)
    q_value: float | None = None
    confirmed: bool = False
    members: np.ndarray = field(default=None, compare=False, repr=False)

    def as_dict(self) -> dict:
        return {
            "method": self.method, "fold": self.fold, "rule": self.rule,
            "n_rows": self.n_rows, "share": self.share,
            "observed_errors": self.observed_errors,
            "observed_error_rate": self.observed_rate,
            "expected_error_rate": self.expected_rate,
            "effect": self.effect,
            "effect_ci": [self.effect_ci_low, self.effect_ci_high],
            "p_value": self.p_value, "adjusted_p_value": self.q_value,
            "confirmed": self.confirmed,
            "discovery_effect": self.discovery_effect,
        }


@dataclass
class Detection:
    available: bool
    reason: str | None = None
    n_rows: int = 0
    n_errors: int = 0
    ece: float | None = None                  # all rows
    overall_gap: float | None = None          # |observed - expected| error rate, all rows
    features_used: list[str] = field(default_factory=list)
    features_excluded: dict[str, str] = field(default_factory=dict)
    pooled_categories: dict[str, int] = field(default_factory=dict)
    tests: list[Group] = field(default_factory=list)   # every group tested
    methods: tuple[str, ...] = ()
    seed: int = 0
    recalibrated_folds: list[bool] = field(default_factory=list)

    @property
    def confirmed(self) -> list[Group]:
        return sorted((g for g in self.tests if g.confirmed),
                      key=lambda g: (-g.effect, g.p_value))

    @property
    def max_effect(self) -> float:
        return max((g.effect for g in self.confirmed), default=0.0)

    def covered_rows(self) -> np.ndarray:
        covered = np.zeros(self.n_rows, bool)
        for group in self.confirmed:
            covered[group.members] = True
        return covered


# ------------------------------------------------------------------ features
@dataclass
class Features:
    matrix: np.ndarray
    names: list[str]                                   # encoded column names
    origin: list[tuple[str, str, str | None]]          # (column, 'num' | 'cat', level)
    used: list[str]
    excluded: dict[str, str]
    pooled: dict[str, int]                             # column -> levels folded away


def _is_free_text_like(values: pd.Series) -> bool:
    from mqvg.checks.training_validity import is_free_text_like
    return is_free_text_like(values)


def build_features(X: pd.DataFrame, max_categories: int) -> Features:
    """Encode features without looking at any label.

    Numeric columns are used as they are (missing values take the median);
    categorical columns become 0/1 indicators for their most frequent levels.
    Constant columns and free-text or identifier-like columns are left out and
    listed, so the search never silently skips a column.
    """
    columns, names, origin, used = [], [], [], []
    excluded: dict[str, str] = {}
    pooled: dict[str, int] = {}
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").astype(float)
            if values.nunique(dropna=True) < 2:
                excluded[column] = "constant"
                continue
            columns.append(values.fillna(values.median()).to_numpy())
            names.append(str(column))
            origin.append((str(column), "num", None))
            used.append(str(column))
            continue
        if _is_free_text_like(series):
            excluded[column] = FREE_TEXT_REASON
            continue
        text = series.astype(object).where(series.notna(), "__missing__").astype(str)
        counts = text.value_counts()
        ordered = sorted(counts.index, key=lambda level: (-counts[level], level))
        if len(ordered) < 2:
            excluded[column] = "constant"
            continue
        kept = ordered[:max_categories]
        if len(ordered) > max_categories:
            pooled[str(column)] = len(ordered) - max_categories
        for level in kept:
            columns.append((text == level).to_numpy(float))
            names.append(f"{column}={level}")
            origin.append((str(column), "cat", level))
        used.append(str(column))
    matrix = np.column_stack(columns) if columns else np.empty((len(X), 0))
    return Features(matrix, names, origin, used, excluded, pooled)


# ------------------------------------------------------------------- rules
def _number(value: float) -> str:
    return f"{value:.3g}"


def format_rule(conditions: list[tuple[int, str, float]], feats: Features) -> str:
    """Readable rule from tree-path conditions (feature index, '<=' or '>', threshold)."""
    if not conditions:
        return "all rows"
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    equals: dict[str, str] = {}
    not_equal: dict[str, list[str]] = {}
    order: list[str] = []
    for index, op, threshold in conditions:
        column, kind, level = feats.origin[index]
        if column not in order:
            order.append(column)
        if kind == "num":
            if op == ">":
                lower[column] = max(lower.get(column, -math.inf), threshold)
            else:
                upper[column] = min(upper.get(column, math.inf), threshold)
        elif op == ">":
            equals[column] = level
        else:
            not_equal.setdefault(column, []).append(level)
    parts = []
    for column in order:
        if column in equals:
            parts.append(f"{column} = {equals[column]}")
        elif column in not_equal:
            levels = sorted(not_equal[column])
            parts.append(f"{column} != {levels[0]}" if len(levels) == 1
                         else f"{column} not in {{{', '.join(levels)}}}")
        elif column in lower and column in upper:
            parts.append(f"{_number(lower[column])} < {column} <= {_number(upper[column])}")
        elif column in lower:
            parts.append(f"{column} > {_number(lower[column])}")
        else:
            parts.append(f"{column} <= {_number(upper[column])}")
    return " AND ".join(parts)


def tree_rules(tree: DecisionTreeRegressor, feats: Features) -> dict[int, str]:
    """Leaf id -> readable rule."""
    t = tree.tree_
    rules: dict[int, str] = {}

    def walk(node: int, conditions: list) -> None:
        if t.children_left[node] == -1:
            rules[node] = format_rule(conditions, feats)
            return
        feature, threshold = int(t.feature[node]), float(t.threshold[node])
        walk(t.children_left[node], conditions + [(feature, "<=", threshold)])
        walk(t.children_right[node], conditions + [(feature, ">", threshold)])

    walk(0, [])
    return rules


def cluster_profile(centroid_scaled: np.ndarray, scaler: StandardScaler,
                    feats: Features, top: int = 3) -> str:
    """Describe a k-means cluster by its most distinctive features.

    Values are rounded before formatting so that floating-point noise (for
    example -1e-17) can never change the text between identical runs.
    """
    raw = centroid_scaled * scaler.scale_ + scaler.mean_
    # Most distinctive first. Magnitudes are rounded so floating-point noise
    # cannot reorder ties (a binary feature's two levels are exact mirrors),
    # and each source column is described once.
    order = sorted(range(len(centroid_scaled)),
                   key=lambda i: (-round(abs(float(centroid_scaled[i])), 6), i))
    parts, described = [], set()
    for index in order:
        column, kind, level = feats.origin[index]
        if column in described:
            continue
        described.add(column)
        overall = float(scaler.mean_[index])
        if kind == "num":
            parts.append(f"{column} averages {_number(round(float(raw[index]), 6) + 0.0)} "
                         f"(overall {_number(round(overall, 6) + 0.0)})")
        else:
            share = float(np.clip(raw[index], 0.0, 1.0))
            parts.append(f"{column} = {level} for {share:.0%} of rows "
                         f"(overall {overall:.0%})")
        if len(parts) == top:
            break
    return "cluster of similar rows: " + "; ".join(parts)


# ----------------------------------------------------------------- the test
def test_group(method: str, fold: int, rule: str, members: np.ndarray,
               err: np.ndarray, expected: np.ndarray, n_total: int,
               discovery_effect: float) -> Group:
    """Observed versus expected errors for one group of held-out rows."""
    n = len(members)
    observed = int(err[members].sum())
    e = expected[members]
    mean, variance = float(e.sum()), float((e * (1.0 - e)).sum())
    z = (observed - 0.5 - mean) / math.sqrt(max(variance, 1e-12))   # continuity correction
    low, high = wilson_interval(observed, n)
    expected_rate = mean / n
    return Group(
        method=method, fold=fold, rule=rule, n_rows=n, share=n / n_total,
        observed_errors=observed, observed_rate=observed / n,
        expected_rate=expected_rate, effect=observed / n - expected_rate,
        effect_ci_low=low - expected_rate, effect_ci_high=high - expected_rate,
        p_value=float(norm.sf(z)), discovery_effect=discovery_effect,
        members=members)


def fold_indices(n: int, folds: int, seed: int) -> list[np.ndarray]:
    """Random, seeded, near-equal folds (sorted row positions)."""
    permutation = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(permutation, folds)]


# ---------------------------------------------------------------- discovery
def _tree_groups(feats, err, expected, residual, disc, test, fold, cfg, seed, min_rows):
    min_leaf = max(cfg.min_group_rows, int(cfg.tree_min_leaf_fraction * len(disc)))
    if len(disc) < 2 * min_leaf:
        return []
    tree = DecisionTreeRegressor(max_depth=cfg.tree_depth, min_samples_leaf=min_leaf,
                                 random_state=seed).fit(feats.matrix[disc], residual[disc])
    leaf_disc, leaf_test = tree.apply(feats.matrix[disc]), tree.apply(feats.matrix[test])
    rules, groups = tree_rules(tree, feats), []
    for leaf in np.unique(leaf_disc):
        discovery_effect = float(residual[disc][leaf_disc == leaf].mean())
        if discovery_effect <= 0:           # chosen on discovery rows only
            continue
        members = test[leaf_test == leaf]
        if len(members) >= min_rows:
            groups.append(test_group("tree", fold, rules[leaf], members, err,
                                     expected, len(err), discovery_effect))
    return groups


def _kmeans_groups(feats, err, expected, residual, disc, test, fold, cfg, seed, min_rows):
    k = min(cfg.kmeans_clusters, len(disc) // max(cfg.min_group_rows, 10))
    if k < 2:
        return []
    scaler = StandardScaler().fit(feats.matrix[disc])
    weights = None
    if cfg.kmeans_weighting == "residual":
        weights = np.maximum(residual[disc], 0.0) + 1e-6      # discovery-fold labels only
    model = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(
        scaler.transform(feats.matrix[disc]), sample_weight=weights)
    label_test = model.predict(scaler.transform(feats.matrix[test]))
    groups = []
    for cluster in range(k):
        discovery_effect = float(residual[disc][model.labels_ == cluster].mean())
        if discovery_effect <= 0:
            continue
        members = test[label_test == cluster]
        if len(members) >= min_rows:
            groups.append(test_group(
                "kmeans", fold, cluster_profile(model.cluster_centers_[cluster],
                                                scaler, feats),
                members, err, expected, len(err), discovery_effect))
    return groups


def platt_recalibrator(score: np.ndarray, outcome: np.ndarray):
    """Fit p -> sigmoid(a * logit(p) + b) on (score, outcome); return the function.

    A two-parameter map that exactly corrects temperature-style over- or
    under-confidence and a shifted intercept. Returns the identity if the
    outcome has a single class.
    """
    if outcome.min() == outcome.max():
        return lambda p: p
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=500).fit(logit(score), outcome.astype(int))
    return lambda p: model.predict_proba(logit(p))[:, 1]


def _ece(y_true: np.ndarray, y_score: np.ndarray, bins: int) -> float:
    from mqvg.checks.calibration import reliability_summary
    return reliability_summary(y_true, y_score, bins)["ece"]


def search(feats: Features, y_true: np.ndarray, y_pred: np.ndarray,
           y_score: np.ndarray, cfg: HiddenGroupsConfig, seed: int,
           methods: Sequence[str] | None = None, ece_bins: int = 10) -> list[Group]:
    """The groups tested (see `search_with_info` for which folds were recalibrated)."""
    return search_with_info(feats, y_true, y_pred, y_score, cfg, seed, methods, ece_bins)[0]


def search_with_info(feats: Features, y_true: np.ndarray, y_pred: np.ndarray,
                     y_score: np.ndarray, cfg: HiddenGroupsConfig, seed: int,
                     methods: Sequence[str] | None = None, ece_bins: int = 10
                     ) -> tuple[list[Group], list[bool]]:
    """Discover on the other folds, test on each held-out fold, for each method.

    For each fold the expected errors of ALL rows are computed from the
    probabilities recalibrated on that fold's discovery rows only, so no
    held-out label influences either what is searched or what is expected.
    """
    methods = tuple(methods or cfg.methods)
    n = len(y_true)
    err = y_true != y_pred
    groups: list[Group] = []
    recalibrated: list[bool] = []
    for fold, test in enumerate(fold_indices(n, cfg.folds, seed)):
        disc = np.setdiff1d(np.arange(n), test)
        score = y_score
        needed = (cfg.recalibrate
                  and _ece(y_true[disc], y_score[disc], ece_bins) > cfg.recalibration_trigger_ece)
        if needed:
            score = platt_recalibrator(y_score[disc], y_true[disc])(y_score)
        recalibrated.append(bool(needed))
        expected = expected_errors(y_pred, score)
        residual = err.astype(float) - expected
        min_rows = max(cfg.min_group_rows, math.ceil(cfg.min_group_fraction * len(test)))
        if "tree" in methods:
            groups += _tree_groups(feats, err, expected, residual, disc, test, fold,
                                   cfg, seed, min_rows)
        if "kmeans" in methods:
            groups += _kmeans_groups(feats, err, expected, residual, disc, test, fold,
                                     cfg, seed, min_rows)
    return groups, recalibrated


def confirm(groups: Sequence[Group], methods: Sequence[str],
            cfg: HiddenGroupsConfig) -> list[Group]:
    """BH-correct over the groups of the chosen methods, then apply the effect floor.

    Returns every group of those methods with `q_value` and `confirmed` set.
    """
    chosen = [g for g in groups if g.method in methods]
    q = bh_adjust([g.p_value for g in chosen])
    return [replace(g, q_value=float(qv),
                    confirmed=bool(qv <= cfg.alpha and g.effect >= cfg.min_effect))
            for g, qv in zip(chosen, q)]


# --------------------------------------------------------------- entry point
def detect(X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
           y_score: np.ndarray | None, cfg: HiddenGroupsConfig, seed: int = 42,
           ece_bins: int = 10) -> Detection:
    """Run the full procedure, or say why it cannot be run.

    `X` holds the feature columns only (no target), one row per evaluated row.
    `y_score` is the model's probability of the positive class.
    """
    n = len(X)
    base = Detection(available=False, n_rows=n, methods=tuple(cfg.methods), seed=seed)
    if y_score is None:
        base.reason = ("the model provides no probability scores, so expected "
                       "errors cannot be derived")
        return base
    if n < cfg.min_rows:
        base.reason = (f"only {n} evaluation rows; at least {cfg.min_rows} are "
                       "needed to test groups reliably")
        return base
    err = (y_true != y_pred)
    base.n_errors = int(err.sum())
    if base.n_errors < cfg.min_errors:
        base.reason = (f"only {base.n_errors} prediction errors; at least "
                       f"{cfg.min_errors} are needed to find groups with excess errors")
        return base

    from mqvg.checks.calibration import reliability_summary
    expected = expected_errors(y_pred, y_score)
    base.ece = reliability_summary(y_true, y_score, ece_bins)["ece"]
    base.overall_gap = abs(float(err.mean()) - float(expected.mean()))
    if base.ece > cfg.max_ece or base.overall_gap > cfg.max_overall_gap:
        base.reason = (
            f"the model's probabilities are not calibrated well enough (ECE "
            f"{base.ece:.3f} against a limit of {cfg.max_ece:g}; observed error rate "
            f"differs from the expected rate by {base.overall_gap:.3f}, limit "
            f"{cfg.max_overall_gap:g}), so a test of errors against the model's own "
            "expectations would be invalid")
        return base

    feats = build_features(X, cfg.max_categories)
    base.features_excluded, base.pooled_categories = feats.excluded, feats.pooled
    if feats.matrix.shape[1] == 0:
        base.reason = "no usable feature columns (all constant or free text)"
        return base
    base.features_used = feats.used

    groups, base.recalibrated_folds = search_with_info(
        feats, y_true, y_pred, y_score, cfg, seed, ece_bins=ece_bins)
    base.tests = confirm(groups, cfg.methods, cfg)

    covered = base.covered_rows()
    if covered.mean() > MAX_COVERED_SHARE:
        base.reason = (f"the groups found cover {covered.mean():.0%} of the rows, "
                       "which points to a model that is wrong or miscalibrated "
                       "in general rather than in a hidden group")
        return base
    base.available = True
    return base
