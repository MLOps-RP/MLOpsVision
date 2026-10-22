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

st.set_page_config(page_title="C4 Drift Engine", page_icon="🛰️", layout="wide")

DIAG_COLORS = {"NO_DRIFT": "#9e9e9e", "SUSPECTED": "#f2c14e", "COVARIATE_BENIGN": "#7fb3d5",
               "COVARIATE_DEGRADING": "#e67e22", "CONCEPT_DRIFT": "#c0392b",
               "LABEL_SHIFT": "#8e44ad", "DATA_QUALITY": "#2c3e50"}
LEVEL_COLORS = {"None": "gray", "Low": "green", "Medium": "orange", "High": "red", "Critical": "red"}
LEVEL_BG = {"None": "#eceff1", "Low": "#e8f5e9", "Medium": "#fff3e0", "High": "#ffebee", "Critical": "#ffcdd2"}
ACTION_ICON = {"NONE": "✅", "WATCH": "👀", "WARN_ONLY": "⚠️", "ROUTE_C1": "🧩", "ROUTE_C2": "🤖",
               "BLOCK_AND_FLAG": "⛔", "ROLLBACK": "⏪"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
div[data-testid="stMetric"] {background: var(--secondary-background-color); border-radius: 10px;
                             padding: 0.7rem 1rem;}
button[data-baseweb="tab"] {font-size: 1.02rem; padding: 0.6rem 1rem;}
.c4-banner {border-radius: 12px; padding: 1rem 1.4rem; margin: 0.4rem 0 1rem 0; color: #1f2933;}
.c4-banner h3 {margin: 0 0 0.2rem 0; color: #1f2933;}
.c4-banner p {margin: 0; color: #3e4c59;}
</style>
""", unsafe_allow_html=True)


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

st.sidebar.title(" C4 controls")
scenario = st.sidebar.selectbox("Scenario", scenarios, index=scenarios.index("S3_concept")
                                if "S3_concept" in scenarios else 0)
s = scen_cfg["scenarios"][scenario]
st.sidebar.caption(s.get("description", ""))
st.sidebar.markdown(f"**Expected:** {s['expected_diagnosis']} → {s['expected_action']}")

if st.sidebar.button("▶ Run C4 on this scenario", type="primary", use_container_width=True):
    from src.c4.run_c4 import run_scenario
    with st.spinner("Running C4 window by window..."):
        run_scenario(scenario, reset=True, verbose=False)
    st.sidebar.success("Done")

if st.sidebar.button("↺ Reset production model to v2", use_container_width=True):
    from src.common import registry
    registry.reset_production()
    st.sidebar.success("production -> v2")

log, scores, info = load_run(scenario)

# ----------------------------------------------------------------- header
st.title("C4 – Drift Diagnosis & Remediation Engine")
st.caption("Input → Monitoring → Diagnosis → Severity → Decision → Drift Report")

if log is None:
    st.info("No run yet for this scenario. Click **Run C4 on this scenario** in the sidebar.")
    st.stop()

start = info["ground_truth"]["drift_start_window"]
last = int(log["window"].max())

st.sidebar.divider()
st.sidebar.markdown("**Replay**")
upto = st.sidebar.slider("Show production up to window", 0, last, last)
st.sidebar.caption("Drag to watch C4 detect drift over time. All tabs follow this slider.")

view = log[log["window"] <= upto]
cur = view.iloc[-1]
level = cur["severity_level"]
icon = ACTION_ICON.get(cur["action"], "•")

# status banner + KPI row (always visible above the tabs)
st.markdown(
    f"""<div class="c4-banner" style="background:{LEVEL_BG.get(level, '#eceff1')}">
    <h3>{icon} Window {upto}: {cur['diagnosis'].replace('_', ' ')} · severity {level} · action {cur['action']}</h3>
    <p>{'Drift is active in this window.' if cur['injected'] else 'Clean window (no drift injected).'}
    Confidence {cur['confidence']}, held for {int(cur['streak'])} window(s).</p></div>""",
    unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Production model (MLflow alias)", f"v{production_version()}")
h2.metric("Model serving this window", f"v{cur['model_version']}")
h3.metric("Current window", f"{upto} / {last}")
h4.metric("Drift injected at window", "none" if start is None else str(start))

tab_in, tab_mon, tab_diag, tab_out, tab_res = st.tabs(
    [" 1 · Input", " 2 · Monitoring", " 3 · Diagnosis & Severity",
     " 4 · Decision & Output", " 5 · Results (all scenarios)"])

# ================================================================= TAB 1: INPUT
with tab_in:
    base = load_json(model_dir(cur["model_version"]) / "baseline.json")
    i1, i2, i3 = st.columns(3)
    with i1, st.container(border=True):
        st.subheader("Deployed model + baseline")
        st.markdown(f"**income-classifier v{cur['model_version']}** ({base['model_type']})")
        st.write(f"Test F1 **{base['test_metrics']['test_f1']:.3f}** · "
                 f"ROC-AUC **{base['test_metrics']['test_roc_auc']:.3f}**")
        st.write(f"Baseline: {base['baseline_data']}, {base['n_rows']:,} rows, "
                 f"{base['preprocessing_version']}")
    with i2, st.container(border=True):
        st.subheader("Production stream")
        st.write(f"**{last + 1}** windows × **{mon_cfg['window_size']}** rows")
        st.write(f"Scenario: **{scenario}**")
        st.write("🔴 Drift active in this window" if cur["injected"] else "🟢 Clean window")
    with i3, st.container(border=True):
        st.subheader("Delayed labels")
        st.write(f"Labels arrive **{mon_cfg['label_delay']} windows** late")
        lw = cur["latest_labelled_window"]
        st.write("No labels yet" if pd.isna(lw) else f"Labelled up to window **{int(lw)}**")

    with st.expander("Sample of incoming rows (this window)"):
        stream = pd.read_csv(DATA_SCENARIOS / scenario / "windows.csv")
        st.dataframe(stream[stream["window"] == upto].drop(columns=["injected"]).head(10),
                     use_container_width=True)

# ================================================================= TAB 2: MONITORING
with tab_mon:
    features = [c for c in scores.columns if c != "window" and not c.startswith("thr__")]
    sc = scores[scores["window"] <= upto]
    ratios = pd.DataFrame({f: sc[f] / sc[f"thr__{f}"] for f in features})
    top_feature = ratios.max().idxmax()
    fired_now = int((ratios.iloc[-1] > 1).sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Features above threshold now", f"{fired_now} / {len(features)}")
    k2.metric("F1 (delayed labels)", "n/a" if pd.isna(cur["f1"]) else f"{cur['f1']:.3f}")
    k3.metric("Predicted >50K share", f"{cur['prediction_share']:.2f}")

    if cur["dq_fired"]:
        st.error("Data-quality alarm: " + str(cur["dq_issues"]))

    st.markdown("#### Input drift")
    m1, m2 = st.columns(2)
    with m1, st.container(border=True):
        fig, ax = small_fig()
        ax.plot(sc["window"], sc[top_feature], marker="o", markersize=3, label=f"{top_feature} score")
        ax.plot(sc["window"], sc[f"thr__{top_feature}"], color="black", linestyle=":",
                label="calibrated threshold")
        ax.set_xlabel("window")
        ax.set_title(f"Most drifted feature: {top_feature}", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        drift_line(ax, start, upto)
        show(fig)
    with m2, st.container(border=True):
        fig, ax = small_fig(6, 3.2)
        ax.imshow((ratios.T.to_numpy() > 1).astype(float), aspect="auto", cmap="Reds", vmin=0, vmax=1.3,
                  extent=[-0.5, upto + 0.5, len(features) - 0.5, -0.5])
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel("window")
        ax.set_title("Features above their threshold (red = fired)", fontsize=10)
        fig.tight_layout()
        show(fig)

    st.markdown("#### Model behaviour")
    p1, p2 = st.columns(2)
    with p1, st.container(border=True):
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
    with p2, st.container(border=True):
        fig, ax = small_fig()
        ax.plot(view["window"], view["prediction_share"], marker="o", markersize=3, color="#16a085")
        ax.axhline(base["prediction"]["positive_share"], color="gray", linestyle=":", label="baseline")
        ax.set_xlabel("window")
        ax.set_title("Share of rows predicted >50K", fontsize=10)
        drift_line(ax, start, upto)
        show(fig)

# ================================================================= TAB 3: DIAGNOSIS & SEVERITY
with tab_diag:
    d_col, s_col = st.columns(2)

    with d_col, st.container(border=True):
        st.subheader("Diagnosis")
        st.markdown(f"### {cur['diagnosis'].replace('_', ' ')}")
        st.caption(f"confidence: {cur['confidence']} · held for {int(cur['streak'])} window(s)")
        fig, ax = small_fig(6, 2.4)
        ax.scatter(view["window"], np.zeros(len(view)),
                   c=[DIAG_COLORS.get(d, "gray") for d in view["diagnosis"]], s=80, marker="s")
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_xlabel("window")
        ax.set_title("Diagnosis per window", fontsize=10)
        for d in view["diagnosis"].unique():
            ax.scatter([], [], c=DIAG_COLORS.get(d, "gray"), marker="s", label=d)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.6))
        drift_line(ax, start, upto)
        fig.subplots_adjust(bottom=0.5, top=0.85)
        show(fig)

    with s_col, st.container(border=True):
        st.subheader("Severity")
        st.markdown(f"### :{LEVEL_COLORS.get(level, 'gray')}[{level}]")
        st.progress(min(1.0, float(cur["severity_score"])), text=f"score {cur['severity_score']:.2f}")
        fig, ax = small_fig(6, 2.4)
        ax.plot(view["window"], view["severity_score"], marker="o", markersize=3, color="#e67e22")
        for name, y in load_config("severity")["levels"].items():
            ax.axhline(y, color="gray", linestyle=":", linewidth=0.8)
            ax.text(0, y + 0.01, name, fontsize=7, color="gray")
        ax.set_ylim(0, 1)
        ax.set_xlabel("window")
        ax.set_title("Severity score over time", fontsize=10)
        drift_line(ax, start, upto)
        show(fig)

    with st.container(border=True):
        st.subheader("What makes up the severity score")
        comp = {k.replace("sev_", ""): float(cur[k])
                for k in ["sev_magnitude", "sev_persistence", "sev_scope", "sev_performance"]}
        c_cols = st.columns(4)
        for col, (name, val) in zip(c_cols, comp.items()):
            col.metric(name.capitalize(), f"{val:.2f}")
            col.progress(min(1.0, val))

# ================================================================= TAB 4: DECISION & OUTPUT
with tab_out:
    with st.container(border=True):
        st.subheader("Remediation decision")
        st.markdown(f"## {icon} {cur['action']}")
        st.caption(f"at window {upto}")
        if isinstance(cur.get("note"), str) and cur["note"]:
            st.info(cur["note"])

    decisions = load_decisions(info["run_id"])
    decisions = decisions[decisions["window"] <= upto] if len(decisions) else decisions
    o1, o2 = st.columns(2)
    with o1, st.container(border=True):
        st.subheader("Latest Drift Report")
        st.caption("Sent to C1 / C2 / C3")
        if len(decisions):
            path = Path(decisions.iloc[0]["report_path"])
            if path.exists():
                st.json(json.loads(path.read_text()), expanded=True)
        else:
            st.write("No report yet.")
    with o2, st.container(border=True):
        st.subheader("Audit log – decisions")
        if len(decisions):
            st.dataframe(decisions.drop(columns=["report_path"]), hide_index=True,
                         use_container_width=True)
        else:
            st.write("No decisions yet.")

# ================================================================= TAB 5: RESULTS
with tab_res:
    table_path = RESULTS / "results_table.csv"
    if table_path.exists():
        res = pd.read_csv(table_path)
        n = len(res)
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Diagnosis correct", f"{int(res['diagnosis_correct'].sum())}/{n}")
        r2.metric("First action correct", f"{int(res['action_correct'].sum())}/{n}")
        r3.metric("Mean detection delay", f"{res['detection_delay'].mean():.1f} windows")
        r4.metric("Unnecessary retrains (naive → C4)",
                  f"{int(res['naive_unnecessary_retrains'].sum())} → {int(res['c4_unnecessary_retrains'].sum())}")
        t1, t2 = st.columns([3, 2])
        with t1, st.container(border=True):
            st.subheader("Per-scenario results")
            st.dataframe(res[["scenario", "c4_diagnosis", "diagnosis_correct", "c4_first_action",
                              "action_correct", "detection_delay", "c4_unnecessary_retrains",
                              "naive_unnecessary_retrains"]], hide_index=True, use_container_width=True)
        with t2, st.container(border=True):
            st.subheader("C4 vs naive retraining")
            if (RESULTS / "unnecessary_retrains.png").exists():
                st.image(str(RESULTS / "unnecessary_retrains.png"))
    else:
        st.info("Run `python -m src.experiments.run_all` to produce the results table.")
