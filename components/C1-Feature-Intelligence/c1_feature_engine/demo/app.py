"""C1 - PP1 demo: upload a CSV, let the existing C1 engine choose preprocessing and features, preview and download the result.
Run:  streamlit run demo/app.py   (from c1_feature_engine/)
Layout/theme code lives in ui.py; all data and engine calls go through c1_demo_core.py."""
import importlib
import time

import pandas as pd
import streamlit as st

import c1_demo_core as core
import ui

importlib.reload(core)      # always use the current helper code (Streamlit can keep a stale copy after file edits)
importlib.reload(ui)

st.set_page_config(page_title="C1 – Autonomous Feature Intelligence Engine", page_icon="🧪", layout="wide",
                   initial_sidebar_state="auto")

# ------------------------------------------------------------------------------------------------ theme + header
st.session_state.setdefault("dark", ui.dark_default())
_, tcol = st.columns([4, 1])
tcol.toggle("🌙 Dark mode", key="dark", help="Switch between light and dark appearance")
ui.inject_theme(st.session_state["dark"])

with st.sidebar:
    st.markdown("### How it works")
    st.markdown("1. **Upload** a CSV and pick the target column\n2. **Review** the dataset overview\n"
                "3. **Run** the optimization – C1 chooses the preprocessing and the best features\n"
                "4. **Compare** before / after and **download** the optimized CSV")
    st.markdown("### Good to know")
    st.caption("• Results come from the existing C1 engine – nothing is simulated.\n\n"
               "• Your file is processed locally in this session.\n\n"
               "• The exported CSV contains data transformations and the selected features only; training-time and "
               "prediction-time operations are listed but not applied to it.")
    st.divider()
    st.caption("C1 · AI-Driven Integrated MLOps Platform · PP1 demo")

ui.hero("C1 – Autonomous Feature Intelligence Engine", "Intelligent Data Preparation & Feature Optimization",
        "Upload a dataset and C1 chooses the right preprocessing and the most useful features for you.")
stepper = st.empty()


def paint(stage: int) -> None:
    stepper.markdown(ui.stepper_html(stage), unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ 1. upload
paint(0)
with st.container(border=True):
    ui.section(1, "Dataset Upload", "Choose a CSV file and tell C1 which column you want to predict.")
    up = st.file_uploader("Upload a CSV file", type="csv")
    if up is None:
        st.info("Upload a CSV file to begin.")
        st.stop()

    df = pd.read_csv(up)
    default = df.columns.get_loc("target") if "target" in df.columns else len(df.columns) - 1
    target = st.selectbox("Select the target column (the value you want to predict)", df.columns, index=default)
    sig = (up.name, up.size, target)

    X, y, classes, excluded = core.prepare_xy(df, target)
    if excluded:
        st.warning(f"Column `{excluded}` is excluded from the features to prevent target leakage "
                   f"(it duplicates the target `{target}` as text).")

# ------------------------------------------------------------------------------------------------ 2. overview
a = core.analyze(df, target)
problem = core.check_usable(y)
if st.session_state.get("profile_sig") != sig and not problem:   # the engine's own profiler, computed once per file/target
    st.session_state["profile_sig"], st.session_state["profile"] = sig, core.profile(X, y)

with st.container(border=True):
    ui.section(2, "Dataset Overview", "A quick health check of your data before optimization.")
    if a["n_missing_target"]:
        st.warning(f"{a['n_missing_target']} rows have a missing target and are dropped before running C1.")
    if problem:
        st.error(problem)
        st.stop()
    prof = st.session_state["profile"]
    ui.cards([("Total Records", f"{a['shape'][0]:,}"), ("Total Features", prof.n_features),
              ("Numerical Features", prof.n_numeric), ("Categorical Features", prof.n_categorical),
              ("Missing Values", f"{a['missing_total']:,}", "", "warn" if a["missing_total"] else ""),
              ("Duplicate Records", f"{a['duplicates']:,}", "", "warn" if a["duplicates"] else ""),
              ("Minority Class", f"{a['minority_pct']:.2f}%", "smallest class share", "accent")])
    left, right = st.columns([3, 2])
    with left:
        ui.table(df.head(10), "First 10 rows")
    with right:
        ui.table(a["class_dist"], "Class distribution")
    with st.expander("Dataset characteristics measured by C1"):
        d = prof.model_dump()
        ui.table(pd.DataFrame({"characteristic": list(d), "value": [f"{v:.4g}" if isinstance(v, float) else str(v) for v in d.values()]}))
paint(2)

# ------------------------------------------------------------------------------------------------ run
with st.container(border=True):
    st.markdown("**Ready?** C1 first chooses the preprocessing, applies it, and then selects the most useful features "
                "from the prepared data.")
    if len(df) > 20000:
        st.warning("Large file: C1 evaluates many candidates, so this may take several minutes.")
    run = st.button("🚀 Run Intelligent Data Optimization", type="primary")

if run:
    t0 = time.time()
    with st.status("Running the C1 engine…", expanded=True) as status:
        def on_event(kind, stage, value, extra=None):          # reports what the engine is actually doing
            name = core._LABEL[stage][0] if stage in core._LABEL else (stage or "")
            if kind == "profile":
                status.write(f"✔ Dataset analysed: {value['n_rows']} records, {value['n_features']} features")
            elif kind == "stage":
                status.write(f"▶ Preprocessing – **{name}**: testing {value}")
            elif kind == "score":
                status.write(f"   · cross-validated score for {name} = {value}: {extra:.3f}")
            elif kind == "threshold":
                status.write("   · prediction threshold scores: " + ", ".join(f"{k}={v:.3f}" for k, v in value.items()))
        res = core.run_c1(X, y, seed=core.CFG["seed"], on_event=on_event)
        status.write("▶ Applying the selected data transformations…")
        Z, origin = core.gap1_export(X, y, res["opts"], res["model"], core.CFG["seed"])
        status.write(f"▶ Feature selection: generating, scoring and searching feature subsets among {Z.shape[1]} prepared columns "
                     "(this can take a minute or more)…")
        g2 = core.run_gap2(Z, y, seed=core.CFG["seed"])
        eng = g2["eng"]
        F = eng.transform(Z)                                    # the engine's own transform -> selected columns only
        total = time.time() - t0
        status.update(label=f"Optimization finished in {total:.1f}s", state="complete", expanded=False)
    st.session_state["c1"] = {"sig": sig, "res": res, "Z": Z, "origin": origin, "eng": eng, "g2_s": g2["runtime_s"],
                              "total_s": total, "F": core.with_target(F, y, classes, target)}
    st.toast("Optimization complete – scroll down for the results.", icon="✅")

out = st.session_state.get("c1")
if not (out and out["sig"] == sig):
    st.stop()
res, Z, origin, eng, Fin = out["res"], out["Z"], out["origin"], out["eng"], out["F"]
F = Fin.drop(columns=[target])
paint(3)

# ------------------------------------------------------------------------------------------------ 3. preprocessing
with st.container(border=True):
    ui.section(3, "Intelligent Data Preprocessing", "The preparation steps C1 selected for your data.")
    ui.table(core.preprocessing_table(res), pill_col="Application Type")
    ui.callout("<b>Data transformations</b> change the values in your dataset and are included in the exported file. "
               "<b>Training-time operations</b> (class weights, resampling) only affect how a model learns, and "
               "<b>prediction-time operations</b> (the decision threshold) only affect how a model's output is read – "
               "neither can be stored in a CSV, so they are reported here but not applied to the exported file.")
    with st.expander("How C1 chose these methods"):
        ui.table(core.stage_table(res))
        st.caption(f"Scores are cross-validated macro-F1 on the uploaded data using the engine's default model (`{res['model']}`).")

# ------------------------------------------------------------------------------------------------ 4. automated feature engineering
fe = core.feature_engineering_report(eng, Z)          # read from the same engine run as everything below
with st.container(border=True):
    ui.section(4, "Automated Feature Engineering", "New features C1 built from your prepared columns – before any selection.")
    if not fe["enabled"]:
        ui.cards([("Original Features", fe["n_input"], "entering this step"), ("Generated Features", 0, "generation disabled"),
                  ("Total Candidate Features", fe["n_pool"], "")])
        ui.callout("<b>Feature generation was disabled</b> for this run, so no new features were created. "
                   "The candidate pool is just the original columns.")
    else:
        n_gen = len(fe["kept"])
        ui.cards([("Original Features", fe["n_input"], "columns entering this step"),
                  ("Generated Features", n_gen, "new features kept", "accent"),
                  ("Removed Generated Features", len(fe["removed"]), f"of {fe['n_built']} created", "warn" if fe["removed"] else ""),
                  ("Total Candidate Features", fe["n_pool"], "original + generated", "good")])
        ui.callout("<b>Feature engineering is not feature selection.</b> This step only <i>creates</i> candidate features; nothing "
                   "is chosen or discarded for usefulness here. C1 removes only generated features that are constant or almost "
                   f"identical (correlation above {fe['settings']['corr_thr']}) to an existing one"
                   + (f", and keeps at most {fe['settings']['cap']} of them" if fe["n_built"] - len(fe["removed"]) >= fe["settings"]["cap"] else "")
                   + ". Original columns are always kept.")
        if n_gen == 0:
            ui.callout(f"<b>No new features were generated for this dataset.</b> C1 created {fe['n_built']} candidate(s) from the "
                       f"{len(fe['top_cols'])} most informative numeric column(s), but every one was constant or nearly identical to an "
                       "existing column – or the data had no numeric columns to build from. The candidate pool therefore equals the "
                       "original columns.")
        else:
            st.markdown("**Techniques used**")
            ui.table(fe["techniques"], max_height=300)
            ui.table(fe["generated"], f"Generated features and their lineage ({n_gen})")
        left, right = st.columns(2)
        with left:
            ui.table(df.head(10), f"Original Dataset – {df.shape[0]:,} rows × {df.shape[1]} columns")
        with right:
            ui.table(eng.G_.head(10), f"Feature-Engineered Dataset – {eng.G_.shape[0]:,} rows × {eng.G_.shape[1]} columns")
    ui.callout("<b>How this feeds the next step:</b> the original columns plus the generated features form the <b>Candidate Feature Pool</b> "
               f"({fe['n_pool']} features). Every feature in the pool is then ranked and scored, and the final feature set is chosen "
               "from it in the next section.")
    with st.expander("Where this step sits in the pipeline"):
        st.write(f"The feature generator receives exactly the table that was handed to the C1 engine – here, the **prepared data** produced "
                 f"by the preprocessing selected in section 3 (cleaned, imputed, encoded and scaled; {fe['n_input']} columns). "
                 "Class balancing is not applied beforehand (it is a training-time operation), and the generator never sees the "
                 "unprepared upload in this demo. Because categorical columns are already encoded as numbers at this point"
                 + (f" ({fe['n_cat_seen']} categorical column(s) reach the generator)" if fe["enabled"] else "")
                 + ", category-based features (frequency / target encoding, group statistics) are only built if such columns remain.")
        if fe["enabled"]:
            st.write("Numeric columns used as building blocks (most informative first): " + ", ".join(f"`{c}`" for c in fe["top_cols"]))

# ------------------------------------------------------------------------------------------------ 5. feature selection
orig_cols = list(X.columns)
contrib = {origin.get(p, p) for f in eng.feature_scores_ if f.selected for p in (f.parents or [f.name])}
n_orig, n_pool, n_out = len(orig_cols), eng.G_.shape[1], F.shape[1]
red = (1 - n_out / n_orig) * 100
sel = core.selected_table(eng, origin)

with st.container(border=True):
    ui.section(5, "Intelligent Feature Selection", "The most useful features C1 kept from your data.")
    ui.cards([("Original Feature Count", n_orig, "columns you uploaded"),
              ("Candidate Features Evaluated", n_pool, "prepared + generated"),
              ("Selected Feature Count", n_out, "kept by C1", "accent"),
              ("Removed Feature Count", n_pool - n_out, "candidates not selected"),
              ("Feature Reduction", f"{red:.1f}%", "vs original count" + ("" if red >= 0 else " (negative: encoding / generated features)"),
               "good" if red >= 0 else "warn")])
    ui.callout("<b>Feature Selection Strategy:</b> every candidate feature is scored for predictive power, stability, missing data "
               "and redundancy (the <i>Feature Quality Score</i>); then a genetic algorithm searches for the best small subset.")
    with st.expander("Technical details of the strategy"):
        st.write(core.gap2_strategy(eng))
        st.write("Genetic-algorithm best fitness per generation: " + ", ".join(f"{h:.3f}" for h in eng.ga_history_))
    ui.table(sel[["feature", "type", "from uploaded column(s)"]].rename(
        columns={"feature": "Selected feature", "type": "Kind", "from uploaded column(s)": "Built from uploaded column(s)"}),
        "Selected feature names", pill_col="Kind")
    st.caption("An **original column** is an uploaded column kept as is (after preprocessing). An **encoded column** is one piece of an "
               "uploaded categorical column after encoding (for example `odor_n` = “odor is n”). A **generated feature** is a new "
               "feature C1 built from other columns (a ratio, difference, sum, …). Selected features therefore refer to prepared "
               "columns, not always to the uploaded ones; the last column links each back.")
    unused = [c for c in orig_cols if c not in contrib]
    with st.expander(f"Uploaded columns not used by any selected feature ({len(unused)})"):
        st.write(", ".join(f"`{c}`" for c in unused) or "None")
    with st.expander("Feature scoring details"):
        ui.table(sel.drop(columns=["type", "from uploaded column(s)"]).round(3))
        st.caption("P predictive power · S stability · M missing ratio · R redundancy · FQS combined score (all computed by the engine).")

# ------------------------------------------------------------------------------------------------ 5. summary
with st.container(border=True):
    ui.section(6, "Optimization Summary", "What C1 did, at a glance.")
    ui.cards([("Dataset Records", f"{out['res']['profile']['n_rows']:,}"), ("Original → Final Features", f"{n_orig} → {n_out}", "", "accent"),
              ("Feature Reduction", f"{red:.1f}%", "", "good" if red >= 0 else "warn"),
              ("Total Execution Time", f"{out['total_s']:.1f} s", f"preprocessing {res['runtime_s']:.1f} s · selection {out['g2_s']:.1f} s")])
    methods = "; ".join(f"{core._LABEL[s][0]}: {core._METHOD.get(res['opts'][s], res['opts'][s])}" for s in core.STAGES)
    ui.table(pd.DataFrame([
        ("Dataset Records", f"{res['profile']['n_rows']:,}"),
        ("Original Features", str(n_orig)),
        ("Final Features", str(n_out)),
        ("Feature Reduction", f"{red:.1f}%" + (" (more final features than original columns, due to encoding / generated features)"
                                               if red < 0 else "")),
        ("Selected Preprocessing Methods", methods),
        ("Feature Selection Strategy", "Feature Quality Score ranking + genetic-algorithm subset search"),
        ("Total Execution Time", f"{out['total_s']:.1f} s (preprocessing {res['runtime_s']:.1f} s, feature selection {out['g2_s']:.1f} s)"),
    ], columns=["Item", "Result"]))

# ------------------------------------------------------------------------------------------------ 6. before / after
with st.container(border=True):
    ui.section(7, "Data Transformation Results", "Your original data next to the optimized data (first 10 rows).")
    o, p = st.columns(2)
    with o:
        ui.table(df.head(10), f"Original Dataset – {df.shape[0]:,} rows × {df.shape[1]} columns")
    with p:
        ui.table(Fin.head(10), f"Optimized Dataset – {Fin.shape[0]:,} rows × {Fin.shape[1]} columns (incl. target)")

# ------------------------------------------------------------------------------------------------ 7. export
with st.container(border=True):
    ui.section(8, "Export Optimized Dataset", "Download the result as a CSV file.")
    st.caption("The file contains the data transformations, the selected features and the target column. It is prepared using the whole "
               "uploaded file, so use it for inspection and downstream work rather than as a held-out evaluation set. Class weights, "
               "resampling and threshold tuning are not applied to it.")
    st.download_button("⬇ Download Optimized CSV", Fin.to_csv(index=False).encode("utf-8"),
                       file_name=f"{up.name.rsplit('.', 1)[0]}_optimized.csv", mime="text/csv")

# ------------------------------------------------------------------------------------------------ feedback
with st.container(border=True):
    ui.section("★", "Your Feedback", "Help us improve C1 – it takes 20 seconds.")
    with st.form("feedback", clear_on_submit=True):
        stars = st.feedback("stars")
        clear = st.radio("Were the results easy to understand?", ["Yes", "Somewhat", "No"], horizontal=True)
        comment = st.text_area("Comments or suggestions (optional)", max_chars=1000)
        if st.form_submit_button("Submit feedback"):
            ui.save_feedback({"dataset": up.name, "records": len(df), "rating_1to5": (stars + 1) if stars is not None else "",
                              "easy_to_understand": clear, "comment": comment.strip(), "runtime_s": round(out["total_s"], 1)})
            st.toast("Thank you for your feedback!", icon="💬")
            st.success("Feedback saved. Thank you!")
