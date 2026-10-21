"""Presentation layer for the C1 demo: light/dark theme, cards, stepper, HTML tables, feedback storage.
Pure UI - no engine or data logic lives here."""
import html
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PALETTES = {
    "light": dict(bg="#F4F6FB", surface="#FFFFFF", surface2="#EEF2FA", text="#0F172A", muted="#5B6B82", border="#DDE3EE",
                  primary="#4F46E5", primary2="#0891B2", success="#15803D", warn="#B45309", info="#1D4ED8",
                  shadow="0 1px 2px rgba(15,23,42,.06), 0 6px 20px rgba(15,23,42,.06)", hero="linear-gradient(120deg,#4F46E5 0%,#0891B2 100%)"),
    "dark": dict(bg="#0B1020", surface="#131A2E", surface2="#1B2440", text="#E8ECF8", muted="#9AA7C2", border="#26325A",
                 primary="#818CF8", primary2="#22D3EE", success="#4ADE80", warn="#FBBF24", info="#60A5FA",
                 shadow="0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35)", hero="linear-gradient(120deg,#3730A3 0%,#0E7490 100%)"),
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]):not(code):not(pre) { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
[data-testid="stIconMaterial"], [class*="material-symbols"] { font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }
.stApp { background: var(--bg); color: var(--text); transition: background .25s ease, color .25s ease; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
h1,h2,h3,h4,h5,h6, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
label, .stToggle p, [data-testid="stCaptionContainer"] { color: var(--text); }
[data-testid="stCaptionContainer"], .muted { color: var(--muted) !important; }
a { color: var(--primary); }
hr { border-color: var(--border); }

/* sidebar */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text); }

/* hero */
.hero { background: var(--hero); color: #fff; border-radius: 18px; padding: 26px 30px; box-shadow: var(--shadow); }
.hero h1 { color: #fff !important; font-size: clamp(1.35rem, 2.6vw, 2rem); margin: 0 0 4px 0; font-weight: 700; letter-spacing: -.01em; padding: 0; }
.hero .sub { color: rgba(255,255,255,.92); font-size: clamp(.9rem, 1.4vw, 1.08rem); font-weight: 500; }
.hero .desc { color: rgba(255,255,255,.8); font-size: .9rem; margin-top: 8px; max-width: 760px; }

/* stepper */
.stepper { display: flex; gap: 10px; flex-wrap: wrap; margin: 16px 0 6px 0; }
.step { flex: 1 1 150px; display: flex; align-items: center; gap: 10px; background: var(--surface); border: 1px solid var(--border);
        border-radius: 12px; padding: 10px 14px; transition: all .25s ease; }
.step .dot { width: 28px; height: 28px; border-radius: 50%; display: grid; place-items: center; font-weight: 700; font-size: .85rem;
             background: var(--surface2); color: var(--muted); flex: none; }
.step .t { font-weight: 600; font-size: .88rem; color: var(--text); line-height: 1.15; }
.step .s { font-size: .75rem; color: var(--muted); }
.step.done .dot { background: var(--success); color: #fff; }
.step.active { border-color: var(--primary); box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 22%, transparent); }
.step.active .dot { background: var(--primary); color: #fff; }

/* section headers + cards */
.sec { display: flex; align-items: center; gap: 12px; margin: 4px 0 2px 0; }
.sec .num { width: 32px; height: 32px; border-radius: 10px; background: var(--hero); color: #fff; display: grid; place-items: center;
            font-weight: 700; flex: none; }
.sec .ttl { font-size: 1.2rem; font-weight: 700; color: var(--text); letter-spacing: -.01em; }
.sec-desc { color: var(--muted); font-size: .9rem; margin: 0 0 10px 44px; }
[data-testid="stVerticalBlockBorderWrapper"]:not(:has([data-testid="stVerticalBlockBorderWrapper"])) {
  background: var(--surface); border: 1px solid var(--border) !important; border-radius: 16px; box-shadow: var(--shadow); padding: 10px 12px; }

/* metric cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(112px, 1fr)); gap: 12px; margin: 8px 0 14px 0; }
.card { background: var(--surface2); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow); border-color: var(--primary); }
.card .l { font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); font-weight: 600; }
.card .v { font-size: 1.55rem; font-weight: 700; color: var(--text); margin-top: 2px; line-height: 1.2; }
.card .h { font-size: .74rem; color: var(--muted); margin-top: 2px; }
.card.good .v { color: var(--success); } .card.warn .v { color: var(--warn); } .card.accent .v { color: var(--primary); }

/* tables */
.tbl-title { font-weight: 600; margin: 6px 0 6px 2px; color: var(--text); }
.tbl-wrap { overflow: auto; max-height: 380px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
table.c1 { border-collapse: collapse; width: 100%; font-size: .83rem; }
table.c1 th { position: sticky; top: 0; background: var(--surface2); color: var(--muted); text-align: left; font-weight: 600;
              padding: 9px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; z-index: 1; }
table.c1 td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); vertical-align: top; }
table.c1 tr:last-child td { border-bottom: none; }
table.c1 tbody tr:hover td { background: color-mix(in srgb, var(--primary) 7%, transparent); }
table.c1 td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600; white-space: nowrap; }
.pill.tr { background: color-mix(in srgb, var(--success) 16%, transparent); color: var(--success); }
.pill.tt { background: color-mix(in srgb, var(--warn) 18%, transparent); color: var(--warn); }
.pill.pt { background: color-mix(in srgb, var(--info) 16%, transparent); color: var(--info); }
.pill.na { background: var(--surface2); color: var(--muted); }
.pill.gen { background: color-mix(in srgb, var(--primary) 16%, transparent); color: var(--primary); }

/* callout */
.callout { border-left: 4px solid var(--primary); background: var(--surface2); border-radius: 10px; padding: 12px 16px; margin: 8px 0;
           color: var(--text); font-size: .9rem; }
.callout b { color: var(--text); }

/* inputs */
[data-testid="stFileUploaderDropzone"] { background: var(--surface2); border: 2px dashed color-mix(in srgb, var(--primary) 55%, var(--border));
  border-radius: 14px; transition: all .2s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary); background: color-mix(in srgb, var(--primary) 8%, var(--surface2)); }
[data-testid="stFileUploaderDropzone"] * { color: var(--muted); }
[data-testid="stSelectbox"] [data-baseweb="select"], [data-baseweb="select"] > div, [data-baseweb="textarea"], [data-baseweb="base-input"], textarea { background: var(--surface2) !important;
  border-color: var(--border) !important; color: var(--text) !important; border-radius: 10px !important; }
[data-baseweb="select"] div, [data-baseweb="select"] input { color: var(--text) !important; -webkit-text-fill-color: var(--text); }
[data-testid="stSelectbox"] [role="group"], [data-testid="stTextInput"] [role="group"] { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px; }
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] [role="group"] * { color: var(--text) !important; -webkit-text-fill-color: var(--text); }
[role="listbox"], [data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] * { background: var(--surface) !important; color: var(--text) !important; }
[role="option"]:hover, [role="option"][data-focused="true"], [role="option"][aria-selected="true"] { background: var(--surface2) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: var(--surface) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover { background: var(--surface2) !important; }
[data-testid="stExpander"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text); }
[data-testid="stStatus"], [data-testid="stStatusWidget"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 12px; background: color-mix(in srgb, var(--info) 11%, var(--surface)); color: var(--text); border: 1px solid var(--border); }
[data-testid="stAlert"] * { color: var(--text); }
[data-testid="stForm"] { background: var(--surface2); border: 1px solid var(--border); border-radius: 14px; }

[data-testid="stToast"], [data-testid="stToast"] > div { background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border); border-radius: 12px; }
[data-testid="stToast"] * { color: var(--text) !important; }

/* buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] {
  border-radius: 12px; font-weight: 600; padding: .62rem 1.3rem; transition: transform .15s ease, box-shadow .15s ease, filter .15s ease; }
[data-testid="stBaseButton-primary"] { background: var(--hero); border: none; color: #fff; box-shadow: 0 4px 14px color-mix(in srgb, var(--primary) 40%, transparent); }
[data-testid="stBaseButton-primary"] * { color: #fff; }
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-secondaryFormSubmit"] { background: var(--surface2); border: 1px solid var(--primary); color: var(--primary); }
[data-testid="stBaseButton-secondary"] * , [data-testid="stBaseButton-secondaryFormSubmit"] * { color: var(--primary); }
[data-testid="stDownloadButton"] button { background: var(--hero); border: none; box-shadow: 0 4px 14px color-mix(in srgb, var(--primary) 40%, transparent); }
[data-testid="stDownloadButton"] button * { color: #fff; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stDownloadButton"] button:hover { transform: translateY(-2px); filter: brightness(1.08); }
[data-testid="stBaseButton-secondary"]:hover { transform: translateY(-1px); background: color-mix(in srgb, var(--primary) 10%, var(--surface2)); }

@media (max-width: 720px) {
  .hero { padding: 18px; border-radius: 14px; } .sec-desc { margin-left: 0; } .card .v { font-size: 1.3rem; }
  .block-container { padding-left: 1rem; padding-right: 1rem; }
}
"""


def dark_default() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def inject_theme(dark: bool) -> None:
    p = PALETTES["dark" if dark else "light"]
    css = ":root{" + "".join(f"--{k}:{v};" for k, v in p.items()) + "}" + _CSS
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, desc: str) -> None:
    st.markdown(f'<div class="hero"><h1>{html.escape(title)}</h1><div class="sub">{html.escape(subtitle)}</div>'
                f'<div class="desc">{html.escape(desc)}</div></div>', unsafe_allow_html=True)


STEPS = [("Upload", "Choose your CSV"), ("Overview", "Understand the data"), ("Optimize", "Run C1"), ("Export", "Download result")]


def stepper_html(stage: int) -> str:
    """stage = index of the current step (0..3); stage 4 = everything done."""
    out = []
    for i, (t, s) in enumerate(STEPS):
        cls = "done" if i < stage else ("active" if i == stage else "")
        out.append(f'<div class="step {cls}"><div class="dot">{"✓" if i < stage else i + 1}</div>'
                   f'<div><div class="t">{t}</div><div class="s">{s}</div></div></div>')
    return '<div class="stepper">' + "".join(out) + "</div>"


def section(n: int, title: str, desc: str = "") -> None:
    st.markdown(f'<div class="sec"><div class="num">{n}</div><div class="ttl">{html.escape(title)}</div></div>'
                + (f'<div class="sec-desc">{html.escape(desc)}</div>' if desc else ""), unsafe_allow_html=True)


def cards(items: list[tuple]) -> None:
    """items: (label, value, hint='', tone='')"""
    out = []
    for it in items:
        label, value, hint, tone = (list(it) + ["", ""])[:4]
        out.append(f'<div class="card {tone}"><div class="l">{html.escape(str(label))}</div><div class="v">{html.escape(str(value))}</div>'
                   + (f'<div class="h">{html.escape(hint)}</div>' if hint else "") + "</div>")
    st.markdown('<div class="cards">' + "".join(out) + "</div>", unsafe_allow_html=True)


def callout(text_html: str) -> None:
    st.markdown(f'<div class="callout">{text_html}</div>', unsafe_allow_html=True)


def _fmt(v) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _pill(text: str) -> str:
    t = text.lower()
    cls = ("na" if t.startswith("not applied") else "tr" if t.startswith("data transformation")
           else "tt" if t.startswith("training") else "pt" if t.startswith("prediction")
           else "gen" if t.startswith("generated") else "na")
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def table(df: pd.DataFrame, title: str = "", pill_col: str | None = None, max_height: int = 380) -> None:
    """Theme-aware HTML table (st.dataframe is canvas-rendered and cannot follow a custom light/dark toggle)."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for rec in df.itertuples(index=False):
        tds = []
        for c, v in zip(df.columns, rec):
            num = isinstance(v, (int, float)) and not isinstance(v, bool) and not pd.isna(v)
            cell = _pill(str(v)) if c == pill_col else html.escape(_fmt(v))
            tds.append(f'<td class="{"num" if num else ""}">{cell}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    t = f'<div class="tbl-title">{html.escape(title)}</div>' if title else ""
    st.markdown(f'{t}<div class="tbl-wrap" style="max-height:{max_height}px"><table class="c1"><thead><tr>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


FEEDBACK_FILE = Path(__file__).resolve().parent / "feedback" / "feedback.csv"


def save_feedback(row: dict) -> None:
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    pd.DataFrame([{"timestamp": datetime.now().isoformat(timespec="seconds"), **row}]).to_csv(
        FEEDBACK_FILE, mode="a", header=not FEEDBACK_FILE.exists(), index=False)
