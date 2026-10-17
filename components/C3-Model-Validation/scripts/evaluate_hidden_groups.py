"""Reproducible evaluation of Hidden Failure Group Detection.

Run from the C3-Model-Validation folder:

    python scripts/evaluate_hidden_groups.py                 # full study (several minutes)
    python scripts/evaluate_hidden_groups.py --quick         # tiny smoke run
    python scripts/evaluate_hidden_groups.py --repeats 40 --jobs 4

For each synthetic scenario and evaluation size it generates fresh data
`repeats` times, fits a model, and runs the detector, comparing:

* ``tree``    - cross-fitted regression tree only
* ``kmeans``  - residual-weighted k-means only
* ``both``    - the union, with one Benjamini-Hochberg correction (the default)

plus two ablations: ``both, no recalibration`` and ``kmeans, unweighted``.

Scenarios cover planted regions of different shapes and sizes (some never
used while the detector was designed), a weaker plant, a concept shift that
any model suffers from, data with NO hidden group, and an over-confident
model with no hidden group (to test the calibration safeguard).

The study uses seeds that were not used during development. It is synthetic:
results show how the method behaves on these patterns, not that it works in
general. Outputs: reports/hidden_groups_eval/results.csv (one row per run and
method set) and docs/hidden_groups_results.md (the summary tables).
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from mqvg import synthetic as syn
from mqvg.config import HiddenGroupsConfig
from mqvg.hidden_groups import confirm, detect, wilson_interval

PROJECT = Path(__file__).resolve().parents[1]
SEED_BASE = 20_000                    # never used during development
N_TRAIN = 5000
DEFAULT_SIZES = (1000, 2000, 5000, 10000)

# method sets compared on the same search; ablations need their own search
SUBSETS = {"tree": ["tree"], "kmeans": ["kmeans"], "both": ["tree", "kmeans"]}


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str                         # "planted" or "control" (no hidden group)
    pattern: str | None = None
    mode: str = "misspec"             # misspec: plant in train and eval; shift: eval only
    model: str = "logistic"           # logistic or gbm
    strength: float = syn.DEFAULT_STRENGTH
    temperature: float | None = None  # >1 makes the probabilities over-confident
    held_out: bool = False            # pattern not used while designing the detector


SCENARIOS = [
    Scenario("two_feature", "planted", "two_feature"),
    Scenario("single_threshold", "planted", "single_threshold"),
    Scenario("categorical_pair", "planted", "categorical_pair"),
    Scenario("oblique (not axis-aligned)", "planted", "oblique"),
    Scenario("small_group (~2%)", "planted", "small_group"),
    Scenario("three_way_categorical (~3%)", "planted", "three_way_categorical", held_out=True),
    Scenario("young_low_income (~7%)", "planted", "young_low_income", held_out=True),
    Scenario("two_feature, weaker (40% flips)", "planted", "two_feature", strength=syn.SUBTLE_STRENGTH),
    Scenario("two_feature, shift + gradient boosting", "planted", "two_feature", mode="shift", model="gbm"),
    Scenario("young_low_income, shift + gradient boosting", "planted", "young_low_income",
             mode="shift", model="gbm", held_out=True),
    Scenario("null, logistic", "control"),
    Scenario("null, gradient boosting", "control", model="gbm"),
    Scenario("null, over-confident (T=1.5)", "control", temperature=1.5),
]


def _probabilities(model, frame, temperature):
    p = model.predict_proba(frame[syn.FEATURES])[:, list(model.classes_).index(syn.POSITIVE)]
    if temperature:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        p = 1 / (1 + np.exp(-np.log(p / (1 - p)) * temperature))
    return p


def _score_run(detection, region):
    """Per-run outcome for one method set."""
    flagged = bool(detection.available and detection.confirmed)
    covered = detection.covered_rows() if detection.available else np.zeros(len(region), bool)
    hit = int((covered & region).sum())
    return {
        "available": int(detection.available), "flagged": int(flagged),
        "recall": hit / region.sum() if region.any() else np.nan,
        "precision": hit / covered.sum() if covered.any() and region.any() else np.nan,
        "n_confirmed": len(detection.confirmed) if detection.available else 0,
        "max_effect": detection.max_effect if detection.available else np.nan,
    }


def run_cell(scenario: Scenario, size: int, rep: int, scenario_index: int, size_index: int):
    seed = SEED_BASE + 1000 * scenario_index + 37 * size_index + rep
    train, evaluation, region = syn.make_pair(
        N_TRAIN, size, seed, scenario.pattern if scenario.kind == "planted" else None,
        scenario.mode, scenario.strength)
    fit = syn.fit_gradient_boosting if scenario.model == "gbm" else syn.fit_logistic
    model = fit(train)
    p = _probabilities(model, evaluation, scenario.temperature)
    y = (evaluation[syn.TARGET] == syn.POSITIVE).to_numpy()
    X = evaluation[syn.FEATURES]
    base = HiddenGroupsConfig()
    rows = []

    def record(label, detection, seconds):
        out = _score_run(detection, region)
        out.update(scenario=scenario.name, kind=scenario.kind, size=size, rep=rep,
                   method=label, seconds=seconds,
                   held_out=int(scenario.held_out), reason=detection.reason or "")
        rows.append(out)

    def timed(config):
        started = time.perf_counter()
        detection = detect(X, y, p >= 0.5, p, config, seed=rep)
        return detection, time.perf_counter() - started

    main, seconds = timed(base)                   # one search serves tree, k-means and both
    for label, methods in SUBSETS.items():
        record(label, detect_from(main, methods, base), seconds)
    record("both, no recalibration", *timed(HiddenGroupsConfig(recalibrate=False)))
    record("kmeans, unweighted",
           *timed(HiddenGroupsConfig(methods=["kmeans"], kmeans_weighting="none")))
    return rows


def detect_from(main, methods, cfg):
    """Re-run the multiple-testing correction for a subset of methods."""
    from copy import copy
    sub = copy(main)
    sub.methods = tuple(methods)
    if main.available:
        sub.tests = confirm([g for g in main.tests if g.method in methods], methods, cfg)
    return sub


# ------------------------------------------------------------- summaries
def _rate(series_of_bools):
    k, n = int(series_of_bools.sum()), len(series_of_bools)
    low, high = wilson_interval(k, n)
    return k, n, low, high


def summarize(df: pd.DataFrame, repeats: int, sizes) -> str:
    methods_main = ["tree", "kmeans", "both"]
    lines = []
    planted = df[df.kind == "planted"]
    null = df[df.kind == "control"]
    present = set(df.scenario)
    scenarios_run = [sc for sc in SCENARIOS if sc.name in present]

    def cell(sub, label):
        sub = sub[sub.method == label]
        detected = ((sub.flagged == 1) & (sub.recall >= 0.5)).sum()
        return f"{int(detected)}"

    lines += ["## Detection power", "",
              f"Number of runs (of {repeats}) in which confirmed groups covered at least 50% of the "
              "planted rows. Each cell lists **tree / k-means / both**. A held-out pattern "
              "(marked †) was added after the detector's design choices were made.", ""]
    header = "| Scenario | " + " | ".join(f"n={s}" for s in sizes) + " |"
    lines += [header, "|---|" + "---|" * len(sizes)]
    for scenario in scenarios_run:
        if scenario.kind != "planted":
            continue
        sub = planted[planted.scenario == scenario.name]
        cells = []
        for size in sizes:
            part = sub[sub["size"] == size]
            cells.append(" / ".join(cell(part, m) for m in methods_main))
        mark = " †" if scenario.held_out else ""
        lines.append(f"| {scenario.name}{mark} | " + " | ".join(cells) + " |")

    lines += ["", "## Localization (runs with a confirmed group)", "",
              "Mean precision and recall of the rows covered by confirmed groups, against the "
              "planted rows, for the combined method.", ""]
    lines += ["| Scenario | " + " | ".join(f"n={s}" for s in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for scenario in scenarios_run:
        if scenario.kind != "planted":
            continue
        sub = planted[(planted.scenario == scenario.name) & (planted.method == "both")]
        cells = []
        for size in sizes:
            part = sub[(sub["size"] == size) & (sub.flagged == 1)]
            cells.append("—" if part.empty else f"P {part.precision.mean():.0%} / R {part.recall.mean():.0%}")
        lines.append(f"| {scenario.name} | " + " | ".join(cells) + " |")

    lines += ["", "## False alarms (no hidden group)", "",
              f"Runs (of {repeats}) in which any group was confirmed, as **tree / k-means / both**; "
              "`unavail` counts runs where the check refused to run. The false discovery rate "
              "was set to 5%.", ""]
    lines += ["| Scenario | " + " | ".join(f"n={s}" for s in sizes) + " |", "|---|" + "---|" * len(sizes)]
    for scenario in scenarios_run:
        if scenario.kind != "control":
            continue
        sub = null[null.scenario == scenario.name]
        cells = []
        for size in sizes:
            part = sub[sub["size"] == size]
            alarms = " / ".join(str(int(((part[part.method == m].flagged == 1)).sum())) for m in methods_main)
            unavailable = int((part[part.method == "both"].available == 0).sum())
            cells.append(alarms + (f" (unavail {unavailable})" if unavailable else ""))
        lines.append(f"| {scenario.name} | " + " | ".join(cells) + " |")

    both_null = null[(null.method == "both") & (null.available == 1)]
    if len(both_null):
        k, n, low, high = _rate(both_null.flagged == 1)
        lines += ["", f"Pooled false-alarm rate of the combined method on available null runs: "
                      f"{k}/{n} = {k / n:.1%} (95% Wilson interval {low:.1%} to {high:.1%}).", ""]

    lines += ["## Ablations", "",
              "Detections (of the planted runs) and false alarms (of the null runs), summed over all sizes.",
              "", "| Variant | planted detected | null false alarms | null unavailable |", "|---|---|---|---|"]
    for label in ["both", "tree", "kmeans", "both, no recalibration", "kmeans, unweighted"]:
        p = planted[planted.method == label]
        q = null[null.method == label]
        lines.append(f"| {label} | {int(((p.flagged == 1) & (p.recall >= 0.5)).sum())}/{len(p)} | "
                     f"{int((q.flagged == 1).sum())}/{len(q)} | {int((q.available == 0).sum())}/{len(q)} |")

    lines += ["", "## Cost", "",
              f"Mean seconds per detection (combined method, evaluation set size n): "
              + ", ".join(f"n={s}: {df[(df.method == 'both') & (df['size'] == s)].seconds.mean():.2f}s"
                          for s in sizes) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="2 repeats, one size, 3 scenarios")
    parser.add_argument("--out-dir", default=str(PROJECT / "reports" / "hidden_groups_eval"))
    parser.add_argument("--doc", default=str(PROJECT / "docs" / "hidden_groups_results.md"))
    args = parser.parse_args(argv)

    scenarios = SCENARIOS
    if args.quick:
        args.repeats, args.sizes = 2, [1500]
        scenarios = [SCENARIOS[0], SCENARIOS[7], SCENARIOS[10]]

    tasks = [(sc, size, rep, SCENARIOS.index(sc), si)
             for sc in scenarios for si, size in enumerate(args.sizes) for rep in range(args.repeats)]
    started = time.perf_counter()
    print(f"{len(tasks)} runs ({len(scenarios)} scenarios x {len(args.sizes)} sizes x "
          f"{args.repeats} repeats), {args.jobs} jobs ...", flush=True)
    results = Parallel(n_jobs=args.jobs, verbose=0)(delayed(run_cell)(*t) for t in tasks)
    df = pd.DataFrame([row for batch in results for row in batch])
    df = df.sort_values(["scenario", "size", "rep", "method"]).reset_index(drop=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False, float_format="%.6g", lineterminator="\n")

    cfg = HiddenGroupsConfig()
    header = [
        "# Hidden failure group detection: evaluation results", "",
        "*Generated by `scripts/evaluate_hidden_groups.py`; do not edit by hand.*", "",
        f"* Seeds: data seeds start at {SEED_BASE} (not used during development); detector seed = repeat index.",
        f"* {args.repeats} repeats per cell; training set {N_TRAIN} rows; evaluation sizes {args.sizes}.",
        f"* Detector settings: `{json.dumps(cfg.model_dump())}`",
        f"* Libraries: python {platform.python_version()}, scikit-learn {sklearn.__version__}, "
        f"numpy {np.__version__}, pandas {pd.__version__}.",
        f"* Run on {datetime.now(timezone.utc).date().isoformat()}; {len(df)} detections in "
        f"{time.perf_counter() - started:.0f} s.", "",
        "**Synthetic data only.** These tables describe behaviour on the patterns above; they do not "
        "show that the method works in general. With 20 repeats a rate of 0/20 has a 95% upper bound "
        "of about 16%, and 20/20 a lower bound of about 84%.", ""]
    Path(args.doc).write_text("\n".join(header) + "\n" + summarize(df, args.repeats, args.sizes),
                              encoding="utf-8", newline="\n")
    print(summarize(df, args.repeats, args.sizes))
    print(f"results: {out / 'results.csv'}\nsummary: {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
