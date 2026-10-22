"""Summarise results/g1v2/*.csv -> results/g1v2_summary.md. Paired Wilcoxon over datasets (per-dataset mean over seeds/folds)."""
import glob

import pandas as pd
from scipy.stats import wilcoxon

from experiments.run_g1v2 import METHODS

PAIRS = [("rules", "B0"), ("rules", "B_thr"), ("rules", "B_prac"), ("rules", "random"), ("rules", "brute"), ("brute", "B0")]


def cmp(t, a, b):
    d = (t[a] - t[b]).dropna()
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return f"- {a} vs {b}: diff {d.mean():+.4f}, W/T/L {(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()}, p={p:.3f}"


if __name__ == "__main__":
    df = pd.concat(pd.read_csv(f) for f in glob.glob("results/g1v2/*.csv"))
    out = [f"{df.dataset.nunique()} datasets, seeds {sorted(df.seed.unique())}, rows {len(df)}\n"]
    for (cond, model), g in df.groupby(["cond", "model"]):
        for label, gg in [("ALL", g), ("HELD-OUT", g[g.split == "test"])]:
            if gg.empty:
                continue
            t = gg.groupby(["dataset", "method"])["f1_macro"].mean().unstack()[METHODS]
            rt = gg.groupby("method").runtime_s.mean().round(2).to_dict()
            out += [f"### {cond} / {model} / {label} ({len(t)} datasets)", "mean f1_macro: " + str(t.mean().round(4).to_dict()),
                    "runtime_s: " + str(rt)] + [cmp(t, a, b) for a, b in PAIRS] + [""]
    # what does the selector actually pick?
    pick = df[df.method == "brute"].assign(k=lambda d: d.opts)
    rows = []
    for _, r in df[df.method == "rules"].iterrows():
        o = eval(r.opts)
        rows += [{"cond": r.cond, "model": r.model, "stage": k, "value": v} for k, v in o.items() if k != "note"]
    pk = pd.DataFrame(rows)
    out += ["### rules selector: share of choices", pk.groupby(["cond", "model", "stage"]).value.value_counts(normalize=True).round(2).to_string()]
    open("results/g1v2_summary.md", "w").write("\n".join(out))
    print("\n".join(out))
