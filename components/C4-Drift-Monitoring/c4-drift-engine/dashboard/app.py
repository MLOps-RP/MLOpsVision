"""
C4 dashboard - shows INPUT -> PROCESSING -> OUTPUT for one scenario.

Start it from the project root:
    streamlit run dashboard/app.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import (DATA_SCENARIOS, OUTPUTS, RESULTS, load_config, load_json,
                               model_dir)  # noqa: E402

st.set_page_config(page_title="C4 Drift Engine", layout="wide")

DIAG_COLORS = {"NO_DRIFT": "#9e9e9e", "SUSPECTED": "#f2c14e", "COVARIATE_BENIGN": "#7fb3d5",
               "COVARIATE_DEGRADING": "#e67e22", "CONCEPT_DRIFT": "#c0392b",
               "LABEL_SHIFT": "#8e44ad", "DATA_QUALITY": "#2c3e50"}
LEVEL_COLORS = {"None": "gray", "Low": "green", "Medium": "orange", "High": "red", "Critical": "red"}


# ----------------------------------------------------------------- helpers
def production_version() -> str:
    try:
        from src.common import registry
        return registry.get_alias_version("production")
    except Exception as e:  # MLflow not set up yet
        return f"unknown ({e.__class__.__name__})"


def load_run(scenario: str):
    folder = OUTPUTS / "runs" / scenario
    if not (folder / "window_log.csv").exists():
        return None, None, None
    log = pd.read_csv(folder / "window_log.csv")
    scores = pd.read_csv(folder / "feature_scores.csv")
    info = load_json(folder / "run_info.json")
    return log, scores, info


def load_decisions(run_id: str) -> pd.DataFrame:
    db = OUTPUTS / "audit.db"
    if not db.exists():
        return pd.DataFrame()
    with sqlite3.connect(db) as con:
        return pd.read_sql_query(
            "SELECT window, diagnosis, confidence, severity_level, severity_score, action, target, "
            "model_version_before AS v_before, model_version_after AS v_after, reason, report_path "
            "FROM decisions WHERE run_id = ? ORDER BY window DESC", con, params=(run_id,))


def drift_line(ax, start, upto):
    if start is not None and start <= upto:
        ax.axvline(start, color="#c0392b", linestyle="--", linewidth=1)
        ax.text(start + 0.2, ax.get_ylim()[1] * 0.95, "drift injected", color="#c0392b", fontsize=8, va="top")


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def small_fig(w=6.0, h=2.6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


# ----------------------------------------------------------------- sidebar
scen_cfg = load_config("scenarios")
mon_cfg = load_config("monitoring")
scenarios = list(scen_cfg["scenarios"].keys())

st.sidebar.title("C4 controls")
scenario = st.sidebar.selectbox("Scenario", scenarios, index=scenarios.index("S3_concept")
                                if "S3_concept" in scenarios else 0)
s = scen_cfg["scenarios"][scenario]
st.sidebar.caption(s.get("description", ""))
st.sidebar.markdown(f"**Expected:** {s['expected_diagnosis']} → {s['expected_action']}")

if st.sidebar.button("Run C4 on this scenario", type="primary"):
    from src.c4.run_c4 import run_scenario
    with st.spinner("Running C4 window by window..."):
        run_scenario(scenario, reset=True, verbose=False)
    st.sidebar.success("Done")

if st.sidebar.button("Reset production model to v2"):
    from src.common import registry
    registry.reset_production()
    st.sidebar.success("production -> v2")

# ----------------------------------------------------------------- header
st.title("C4 – Drift Diagnosis & Remediation Engine")
log, scores, info = load_run(scenario)
if log is None:
    st.info("No run yet for this scenario. Click **Run C4 on this scenario** in the sidebar.")
    st.stop()

start = info["ground_truth"]["drift_start_window"]
last = int(log["window"].max())
upto = st.slider("Replay: show production up to window", 0, last, last)
view = log[log["window"] <= upto]
cur = view.iloc[-1]

h1, h2, h3, h4 = st.columns(4)
h1.metric("Production model now (MLflow alias)", f"v{production_version()}")
h2.metric("Model serving this window", f"v{cur['model_version']}")
h3.metric("Current window", f"{upto} / {last}")
h4.metric("Drift injected at window", "none" if start is None else str(start))

# ================================================================= INPUT
st.header("1 · Input")
i1, i2, i3 = st.columns(3)
base = load_json(model_dir(cur["model_version"]) / "baseline.json")
with i1:
    st.subheader("Deployed model + baseline")
    st.write(f"**income-classifier v{cur['model_version']}** ({base['model_type']})")
    st.write(f"Test F1 {base['test_metrics']['test_f1']:.3f} · "
             f"ROC-AUC {base['test_metrics']['test_roc_auc']:.3f}")
    st.write(f"Baseline: {base['baseline_data']}, {base['n_rows']:,} rows, "
             f"{base['preprocessing_version']}")
with i2:
    st.subheader("Production stream")
    st.write(f"{last + 1} windows × {mon_cfg['window_size']} rows")
    st.write(f"Scenario: **{scenario}**")
    st.write("Drift active in this window" if cur["injected"] else "Clean window")
with i3:
    st.subheader("Delayed labels")
    st.write(f"Labels arrive **{mon_cfg['label_delay']} windows** late")
    lw = cur["latest_labelled_window"]
    st.write("No labels yet" if pd.isna(lw) else f"Labelled up to window {int(lw)}")

with st.expander("Sample of incoming rows (this window)"):
    stream = pd.read_csv(DATA_SCENARIOS / scenario / "windows.csv")
    st.dataframe(stream[stream["window"] == upto].drop(columns=["injected"]).head(10))

# ================================================================= PROCESSING
st.header("2 · Processing")

# ---- Step 1 monitoring
st.subheader("Step 1 – Multi-signal monitoring")
features = [c for c in scores.columns if c != "window" and not c.startswith("thr__")]
sc = scores[scores["window"] <= upto]
ratios = pd.DataFrame({f: sc[f] / sc[f"thr__{f}"] for f in features})
top_feature = ratios.max().idxmax()

m1, m2 = st.columns(2)
with m1:
    fig, ax = small_fig()
    ax.plot(sc["window"], sc[top_feature], marker="o", markersize=3, label=f"{top_feature} score")
    ax.plot(sc["window"], sc[f"thr__{top_feature}"], color="black", linestyle=":", label="calibrated threshold")
    ax.set_xlabel("window")
    ax.set_title(f"Most drifted feature: {top_feature}", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    drift_line(ax, start, upto)
    show(fig)
with m2:
    fig, ax = small_fig(6, 3.2)
    ax.imshow((ratios.T.to_numpy() > 1).astype(float), aspect="auto", cmap="Reds", vmin=0, vmax=1.3,
              extent=[-0.5, upto + 0.5, len(features) - 0.5, -0.5])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_xlabel("window")
    ax.set_title("Features above their threshold (red = fired)", fontsize=10)
    fig.tight_layout()
    show(fig)

p1, p2 = st.columns(2)
with p1:
    fig, ax = small_fig()
    ax.plot(view["window"], view["f1"], marker="o", markersize=3, label="F1 (last 2 labelled windows)")
    ax.plot(view["window"], view["accuracy"], marker="o", markersize=3, label="accuracy")
    ax.axhline(base["performance"]["f1"], color="gray", linestyle=":", label="baseline F1")
    adw = view[view["adwin_alarm"] == True]  # noqa: E712
    ax.scatter(adw["window"], adw["f1"], color="#c0392b", zorder=5, s=18, label="ADWIN alarm")
    ax.set_ylim(0, 1)
    ax.set_xlabel("window")
    ax.set_title("Performance (from delayed labels)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    drift_line(ax, start, upto)
    show(fig)
with p2:
    fig, ax = small_fig()
    ax.plot(view["window"], view["prediction_share"], marker="o", markersize=3, color="#16a085")
    ax.axhline(base["prediction"]["positive_share"], color="gray", linestyle=":", label="baseline")
    ax.set_xlabel("window")
    ax.set_title("Share of rows predicted >50K", fontsize=10)
    drift_line(ax, start, upto)
    show(fig)
    if cur["dq_fired"]:
        st.error("Data-quality alarm: " + str(cur["dq_issues"]))

# ---- Step 2 diagnosis
st.subheader("Step 2 – Diagnosis")
d1, d2 = st.columns([1, 2])
with d1:
    st.markdown(f"### {cur['diagnosis'].replace('_', ' ')}")
    st.caption(f"confidence: {cur['confidence']} · held for {int(cur['streak'])} window(s)")
with d2:
    fig, ax = small_fig(7, 2.2)
    ax.scatter(view["window"], np.zeros(len(view)), c=[DIAG_COLORS.get(d, "gray") for d in view["diagnosis"]],
               s=80, marker="s")
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("window")
    ax.set_title("Diagnosis per window", fontsize=10)
    for d in view["diagnosis"].unique():
        ax.scatter([], [], c=DIAG_COLORS.get(d, "gray"), marker="s", label=d)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.6))
    drift_line(ax, start, upto)
    fig.subplots_adjust(bottom=0.5, top=0.85)
    show(fig)

# ---- Step 3 severity
st.subheader("Step 3 – Severity")
v1, v2 = st.columns([1, 2])
with v1:
    st.markdown(f"### :{LEVEL_COLORS.get(cur['severity_level'], 'gray')}[{cur['severity_level']}]")
    st.progress(min(1.0, float(cur["severity_score"])), text=f"score {cur['severity_score']:.2f}")
    comp = {k.replace("sev_", ""): cur[k] for k in ["sev_magnitude", "sev_persistence", "sev_scope", "sev_performance"]}
    st.bar_chart(pd.Series(comp, name="component (0-1)"))
with v2:
    fig, ax = small_fig()
    ax.plot(view["window"], view["severity_score"], marker="o", markersize=3, color="#e67e22")
    sev_cfg = load_config("severity")["levels"]
    for name, y in sev_cfg.items():
        ax.axhline(y, color="gray", linestyle=":", linewidth=0.8)
        ax.text(0, y + 0.01, name, fontsize=7, color="gray")
    ax.set_ylim(0, 1)
    ax.set_xlabel("window")
    ax.set_title("Severity score over time", fontsize=10)
    drift_line(ax, start, upto)
    show(fig)

# ---- Step 4 decision
st.subheader("Step 4 – Remediation decision")
st.markdown(f"### Action at window {upto}: **{cur['action']}**")
if isinstance(cur.get("note"), str) and cur["note"]:
    st.caption(cur["note"])

# ================================================================= OUTPUT
st.header("3 · Output")
decisions = load_decisions(info["run_id"])
decisions = decisions[decisions["window"] <= upto] if len(decisions) else decisions
o1, o2 = st.columns([1, 1])
with o1:
    st.subheader("Latest Drift Report (sent to C1 / C2 / C3)")
    if len(decisions):
        path = Path(decisions.iloc[0]["report_path"])
        if path.exists():
            st.json(json.loads(path.read_text()), expanded=True)
    else:
        st.write("No report yet.")
with o2:
    st.subheader("Audit log – decisions")
    if len(decisions):
        st.dataframe(decisions.drop(columns=["report_path"]), hide_index=True)
    else:
        st.write("No decisions yet.")
    if (RESULTS / "unnecessary_retrains.png").exists():
        st.subheader("All scenarios: C4 vs naive retraining")
        st.image(str(RESULTS / "unnecessary_retrains.png"))
    if (RESULTS / "results_table.csv").exists():
        st.dataframe(pd.read_csv(RESULTS / "results_table.csv")[
            ["scenario", "c4_diagnosis", "diagnosis_correct", "c4_first_action", "action_correct",
             "detection_delay", "c4_unnecessary_retrains", "naive_unnecessary_retrains"]],
            hide_index=True)
