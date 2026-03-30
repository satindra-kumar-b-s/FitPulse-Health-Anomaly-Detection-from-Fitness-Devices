import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from prophet import Prophet
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

import base64, tempfile, os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse ML Pipeline",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (M1-3 styles + M4 additions)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:      #070c14;
  --surface: #0d1526;
  --glass:   rgba(255,255,255,0.04);
  --border:  rgba(99,215,196,0.18);
  --accent:  #63d7c4;
  --accent2: #f97316;
  --accent3: #818cf8;
  --text:    #e2eaf4;
  --muted:   #6b7a96;
  --success: #34d399;
  --danger:  #f87171;
  --warn:    #fbbf24;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Syne', sans-serif !important; }

h1 { font-size:2rem !important; font-weight:800 !important; color:var(--text) !important; }
h2 { font-size:1.4rem !important; font-weight:700 !important; color:var(--accent) !important; }
h3 { font-size:1.1rem !important; font-weight:600 !important; color:var(--text) !important; }
p, li, label, span { color: var(--text) !important; }

.stButton > button {
  background: linear-gradient(135deg,#1a2f4a,#0f1f36) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  font-family: 'Syne',sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  padding: 0.5rem 1.4rem !important;
  transition: all 0.25s ease !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  background: linear-gradient(135deg,#1f3d5c,#162c46) !important;
  transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
  background: var(--glass) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color:var(--muted) !important; font-size:0.75rem !important; text-transform:uppercase; letter-spacing:1px; }
[data-testid="stMetricValue"] { color:var(--accent) !important; font-size:1.8rem !important; font-weight:800 !important; }

.stAlert { border-radius:12px !important; border-left-width:4px !important; }
.stDataFrame { border-radius:12px !important; overflow:hidden; }

[data-testid="stFileUploader"] {
  background: var(--glass) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem !important;
}

hr { border-color:var(--border) !important; }

.fp-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 1.6rem;
  margin-bottom: 1.2rem;
  backdrop-filter: blur(10px);
}
.fp-badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; letter-spacing:0.6px; text-transform:uppercase; }
.fp-badge-ok   { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3); }
.fp-badge-miss { background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); }

.step-header {
  display:inline-flex; align-items:center; gap:10px;
  background:rgba(99,215,196,0.07); border:1px solid rgba(99,215,196,0.2);
  border-radius:10px; padding:7px 16px; margin:1.2rem 0 0.8rem 0;
  font-family:'JetBrains Mono',monospace; font-size:0.82rem;
  color:var(--accent); font-weight:600; letter-spacing:0.5px;
}

.null-card {
  background:rgba(255,255,255,0.03); border:1px solid rgba(99,215,196,0.12);
  border-radius:14px; padding:16px 14px 12px 14px;
}
.null-card-name { font-size:0.78rem; color:#6b7a96; margin-bottom:8px; font-family:'JetBrains Mono',monospace; }
.null-val-ok  { font-size:1.6rem; color:#34d399; }
.null-val-bad { font-size:1.3rem; color:#f87171; font-weight:800; font-family:'JetBrains Mono',monospace; }
.null-rows    { font-size:0.7rem; color:#6b7a96; margin-top:5px; }

.log-box {
  background:rgba(255,255,255,0.02); border:1px solid rgba(99,215,196,0.12);
  border-radius:12px; padding:1.1rem 1.4rem;
  font-family:'JetBrains Mono',monospace; font-size:0.8rem; line-height:2;
}

.prog-bar-bg   { background:rgba(255,255,255,0.06); border-radius:10px; height:6px; width:100%; margin:6px 0 14px 0; }
.prog-bar-fill { background:linear-gradient(90deg,#63d7c4,#818cf8); border-radius:10px; height:6px; }

.nav-item { display:flex; align-items:center; gap:9px; padding:5px 0; font-size:0.83rem; }
.nav-dot-active   { width:8px;height:8px;border-radius:50%;background:#63d7c4;flex-shrink:0; }
.nav-dot-inactive { width:8px;height:8px;border-radius:50%;border:1.5px solid #6b7a96;flex-shrink:0; }
.nav-label-active   { color:#e2eaf4; font-weight:600; }
.nav-label-inactive { color:#6b7a96; }

.shared-upload-box {
  background: rgba(99,215,196,0.05);
  border: 1.5px dashed rgba(99,215,196,0.35);
  border-radius: 12px;
  padding: 0.9rem 1rem;
  margin: 0.5rem 0 1rem 0;
}
.shared-upload-title {
  font-size: 0.7rem; color: #63d7c4; font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase; letter-spacing: 1px; font-weight: 700;
  margin-bottom: 0.5rem;
}
.shared-upload-sub {
  font-size: 0.65rem; color: #6b7a96; font-family: 'JetBrains Mono', monospace;
  margin-top: 0.4rem; line-height: 1.6;
}
.file-pill {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3);
  border-radius: 20px; padding: 2px 8px; font-size: 0.62rem;
  font-family: 'JetBrains Mono', monospace; color: #34d399;
  margin: 2px; white-space: nowrap;
}
.file-pill-miss {
  background: rgba(248,113,113,0.1); border-color: rgba(248,113,113,0.3); color: #f87171;
}

/* M3 styles */
.m3-hero {
  background: linear-gradient(135deg,rgba(252,129,129,0.08),rgba(246,135,179,0.06),rgba(10,14,26,0.9));
  border: 1px solid rgba(252,129,129,0.4); border-radius: 20px;
  padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero-badge {
  display:inline-block; background:rgba(252,129,129,0.1); border:1px solid rgba(252,129,129,0.4);
  border-radius:100px; padding:0.3rem 1rem; font-size:0.75rem;
  font-family:'JetBrains Mono',monospace; color:#fc8181; margin-bottom:1rem;
}
.m3-card {
  background:rgba(15,23,42,0.85); border:1px solid rgba(99,179,237,0.2);
  border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem; backdrop-filter:blur(10px);
}
.m3-card-title {
  font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
  color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;
}
.m3-step-pill {
  display:inline-flex; align-items:center; gap:0.5rem;
  background:rgba(99,179,237,0.07); border:1px solid rgba(99,179,237,0.2);
  border-radius:100px; padding:0.3rem 0.9rem; font-size:0.75rem;
  font-family:'JetBrains Mono',monospace; color:#63b3ed; margin-bottom:0.8rem;
}
.m3-metric-grid { display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.8rem 0; }
.m3-metric-card {
  flex:1; min-width:120px; background:rgba(99,179,237,0.07);
  border:1px solid rgba(99,179,237,0.2); border-radius:12px;
  padding:1rem 1.2rem; text-align:center;
}
.m3-metric-val { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#63b3ed; line-height:1; margin-bottom:0.25rem; }
.m3-metric-val-red { color:#fc8181; }
.m3-metric-label { font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; }
.m3-anom-tag {
  display:inline-flex; align-items:center; gap:0.4rem;
  background:rgba(252,129,129,0.1); border:1px solid rgba(252,129,129,0.4);
  border-radius:100px; padding:0.3rem 0.9rem; font-size:0.72rem;
  font-family:'JetBrains Mono',monospace; color:#fc8181; margin-bottom:0.8rem;
}
.m3-screenshot-badge {
  display:inline-flex; align-items:center; gap:0.4rem;
  background:rgba(246,135,179,0.15); border:1px solid rgba(246,135,179,0.4);
  border-radius:100px; padding:0.3rem 0.9rem; font-size:0.72rem;
  font-family:'JetBrains Mono',monospace; color:#f687b3; margin-bottom:0.8rem;
}
.m3-alert-warn    { background:rgba(246,173,85,0.12);  border-left:3px solid #f6ad55; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#fbd38d; }
.m3-alert-success { background:rgba(104,211,145,0.1);  border-left:3px solid #68d391; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#9ae6b4; }
.m3-alert-info    { background:rgba(99,179,237,0.15);  border-left:3px solid #63b3ed; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#bee3f8; }
.m3-alert-danger  { background:rgba(252,129,129,0.1);  border-left:3px solid #fc8181; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#feb2b2; }
.m3-sec-header { display:flex; align-items:center; gap:0.8rem; margin:2rem 0 1rem 0; padding-bottom:0.6rem; border-bottom:1px solid rgba(99,179,237,0.2); }
.m3-sec-icon { font-size:1.4rem; width:2.2rem; height:2.2rem; display:flex; align-items:center; justify-content:center; background:rgba(99,179,237,0.15); border-radius:8px; border:1px solid rgba(99,179,237,0.2); }
.m3-sec-title { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; color:#e2e8f0; margin:0; }
.m3-sec-badge { margin-left:auto; background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.2); border-radius:100px; padding:0.2rem 0.7rem; font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:#63b3ed; }
.m3-divider { border:none; border-top:1px solid rgba(99,179,237,0.2); margin:2rem 0; }

/* M4 Report styles */
.m4-hero {
  background: linear-gradient(135deg,rgba(99,179,237,0.08),rgba(104,211,145,0.05),rgba(10,14,26,0.9));
  border:1px solid rgba(99,179,237,0.2);border-radius:20px;padding:2rem 2.5rem;
  margin-bottom:1.5rem;position:relative;overflow:hidden;
}
.m4-hero-badge {
  display:inline-block;background:rgba(99,179,237,0.15);border:1px solid rgba(99,179,237,0.2);
  border-radius:100px;padding:0.25rem 0.9rem;font-size:0.72rem;
  font-family:'JetBrains Mono',monospace;color:#63b3ed;margin-bottom:0.8rem;
}
.m4-hero-title {
  font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
  color:#e2e8f0;margin:0 0 0.3rem 0;letter-spacing:-0.02em;
}
.m4-hero-sub { font-size:1rem;color:#94a3b8;font-weight:300;margin:0; }
.kpi-grid { display:grid;grid-template-columns:repeat(6,1fr);gap:0.7rem;margin:1rem 0; }
.kpi-card {
  background:rgba(15,23,42,0.85);border:1px solid rgba(99,179,237,0.2);border-radius:14px;
  padding:1rem 1.1rem;text-align:center;backdrop-filter:blur(10px);
}
.kpi-val { font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;line-height:1;margin-bottom:0.2rem; }
.kpi-label { font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em; }
.kpi-sub { font-size:0.65rem;color:#94a3b8;margin-top:0.15rem; }
.m4-card {
  background:rgba(15,23,42,0.85);border:1px solid rgba(99,179,237,0.2);border-radius:14px;
  padding:1.2rem 1.4rem;margin-bottom:0.8rem;backdrop-filter:blur(10px);
}
.m4-card-title {
  font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:700;
  color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;
}
.m4-sec-header { display:flex;align-items:center;gap:0.8rem;margin:1.5rem 0 0.8rem;padding-bottom:0.5rem;border-bottom:1px solid rgba(99,179,237,0.2); }
.m4-sec-icon { font-size:1.3rem;width:2rem;height:2rem;display:flex;align-items:center;justify-content:center;background:rgba(99,179,237,0.15);border-radius:8px;border:1px solid rgba(99,179,237,0.2); }
.m4-sec-title { font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;color:#e2e8f0;margin:0; }
.m4-sec-badge { margin-left:auto;background:rgba(99,179,237,0.15);border:1px solid rgba(99,179,237,0.2);border-radius:100px;padding:0.2rem 0.7rem;font-size:0.7rem;font-family:'JetBrains Mono',monospace;color:#63b3ed; }
.anom-row { display:flex;align-items:center;gap:0.6rem;padding:0.45rem 0;border-bottom:1px solid rgba(99,179,237,0.15);font-size:0.82rem; }
.m4-alert-info    { background:rgba(99,179,237,0.15);border-left:3px solid #63b3ed;border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#bee3f8; }
.m4-alert-success { background:rgba(104,211,145,0.1);border-left:3px solid #68d391;border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#9ae6b4; }
.m4-alert-danger  { background:rgba(252,129,129,0.1);border-left:3px solid #fc8181;border-radius:0 10px 10px 0;padding:0.7rem 1rem;margin:0.5rem 0;font-size:0.83rem;color:#feb2b2; }
.m4-divider { border:none;border-top:1px solid rgba(99,179,237,0.2);margin:1.5rem 0; }
.stTabs [data-baseweb="tab-list"] { background:rgba(99,179,237,0.07);border-radius:10px;padding:0.2rem; }
.stTabs [data-baseweb="tab"] { color:#94a3b8;font-family:'JetBrains Mono',monospace;font-size:0.8rem; }
.stTabs [aria-selected="true"] { background:rgba(15,23,42,0.85);color:#63b3ed;border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1526", "axes.facecolor": "#0d1526",
    "axes.edgecolor":   "#1e3050", "axes.labelcolor": "#e2eaf4",
    "axes.titlecolor":  "#63d7c4", "axes.grid": True,
    "grid.color":       "#1a2d44", "grid.linestyle": "--", "grid.linewidth": 0.5,
    "xtick.color":      "#6b7a96", "ytick.color": "#6b7a96",
    "text.color":       "#e2eaf4", "legend.facecolor": "#0d1526",
    "legend.edgecolor": "#1e3050", "font.family": "monospace",
})
PALETTE = ["#63d7c4","#f97316","#818cf8","#fbbf24","#34d399","#f87171","#60a5fa","#e879f9"]

# M4 Plotly theme constants
M4_BG      = "#0a0e1a"; M4_CARD    = "rgba(15,23,42,0.85)"; M4_BOR     = "rgba(99,179,237,0.2)"
M4_TEXT    = "#e2e8f0"; M4_MUTED   = "#94a3b8"; M4_ACCENT  = "#63b3ed"
M4_ACC2    = "#f687b3"; M4_ACC3    = "#68d391"; M4_RED     = "#fc8181"
M4_ORG     = "#f6ad55"; M4_PURPLE  = "#b794f4"
M4_GRID    = "rgba(255,255,255,0.05)"
M4_PLOT_BG = "#0f172a"
M4_SECTION_BG = "rgba(99,179,237,0.07)"
M4_BADGE_BG   = "rgba(99,179,237,0.15)"
M4_SUCCESS_BG = "rgba(104,211,145,0.1)"; M4_SUCCESS_BOR = "rgba(104,211,145,0.4)"
M4_DANGER_BG  = "rgba(252,129,129,0.1)"; M4_DANGER_BOR  = "rgba(252,129,129,0.4)"
M4_WARN_BG    = "rgba(246,173,85,0.12)"; M4_WARN_BOR    = "rgba(246,173,85,0.4)"

# Old M3 Plotly constants (used by anomaly section)
M3_BG      = "#0a0e1a"; M3_CARD    = "rgba(15,23,42,0.85)"; M3_BOR     = "rgba(99,179,237,0.2)"
M3_TEXT    = "#e2e8f0"; M3_MUTED   = "#94a3b8"; M3_ACCENT  = "#63b3ed"
M3_ACC2    = "#f687b3"; M3_ACC3    = "#68d391"; M3_RED     = "#fc8181"
M3_GRID    = "rgba(255,255,255,0.06)"
M3_DANGER_BG  = "rgba(252,129,129,0.1)"; M3_DANGER_BOR = "rgba(252,129,129,0.4)"
M3_SECTION_BG = "rgba(99,179,237,0.07)"; M3_BADGE_BG   = "rgba(99,179,237,0.15)"
M3_SUCCESS_BG = "rgba(104,211,145,0.1)"; M3_SUCCESS_BOR= "rgba(104,211,145,0.4)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=M3_BG, plot_bgcolor="#0f172a",
    font_color=M3_TEXT, font_family="Inter, sans-serif",
    xaxis=dict(gridcolor=M3_GRID, showgrid=True, zeroline=False, linecolor=M3_BOR, tickfont_color=M3_MUTED),
    yaxis=dict(gridcolor=M3_GRID, showgrid=True, zeroline=False, linecolor=M3_BOR, tickfont_color=M3_MUTED),
    legend=dict(bgcolor=M3_CARD, bordercolor=M3_BOR, borderwidth=1, font_color=M3_TEXT),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor=M3_CARD, bordercolor=M3_BOR, font_color=M3_TEXT),
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    # preprocessing
    "df": None, "clean_df": None, "milestone1_loaded": False,
    # pattern extraction
    "files_loaded": False, "tsfresh_done": False, "prophet_done": False, "cluster_done": False,
    "daily": None, "steps": None, "intensity": None, "sleep": None, "hr": None,
    "features": None, "master_df": None, "cluster_summary": None,
    "tsfresh_fig": None, "prophet_fig": None, "cluster_fig": None,
    "cluster_bar_fig": None, "elbow_fig": None, "steps_sleep_fig": None,
    "X_scaled": None, "km_labels": None, "db_labels": None,
    "k_val_used": None, "available_cols": None, "feat_df": None,
    "raw_files": {},
    # anomaly detector
    "m3_files_loaded": False, "m3_anomaly_done": False, "m3_simulation_done": False,
    "m3_daily": None, "m3_hourly_s": None, "m3_hourly_i": None,
    "m3_sleep": None, "m3_hr": None, "m3_hr_minute": None,
    "m3_master": None,
    "m3_anom_hr": None, "m3_anom_steps": None, "m3_anom_sleep": None,
    "m3_sim_results": None,
    # shared files
    "shared_detected": {},
    "shared_master": None,
    "shared_loaded": False,
    # navigation
    "active_section": "preprocessing",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED FILES REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
SHARED_REQUIRED = {
    "dailyActivity_merged.csv":     {"key_cols": ["ActivityDate","TotalSteps","Calories"],    "label": "Daily Activity",    "icon": "🏃"},
    "hourlySteps_merged.csv":       {"key_cols": ["ActivityHour","StepTotal"],                "label": "Hourly Steps",      "icon": "👣"},
    "hourlyIntensities_merged.csv": {"key_cols": ["ActivityHour","TotalIntensity"],           "label": "Hourly Intensities","icon": "⚡"},
    "minuteSleep_merged.csv":       {"key_cols": ["date","value","logId"],                    "label": "Minute Sleep",      "icon": "💤"},
    "heartrate_seconds_merged.csv": {"key_cols": ["Time","Value"],                            "label": "Heart Rate",        "icon": "❤️"},
}

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

def detect_shared_files(uploaded_list):
    detected = {}
    if not uploaded_list:
        return detected
    for req_name, finfo in SHARED_REQUIRED.items():
        best_s, best_df = 0, None
        for uname, udf in uploaded_list:
            s = score_match(udf, finfo)
            if s > best_s:
                best_s, best_df = s, udf
        if best_s >= 2:
            detected[req_name] = best_df
    return detected

def build_shared_master(detected):
    daily    = detected["dailyActivity_merged.csv"].copy()
    hourly_s = detected["hourlySteps_merged.csv"].copy()
    hourly_i = detected["hourlyIntensities_merged.csv"].copy()
    sleep    = detected["minuteSleep_merged.csv"].copy()
    hr       = detected["heartrate_seconds_merged.csv"].copy()

    daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"],    format="%m/%d/%Y")
    hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
    sleep["date"]            = pd.to_datetime(sleep["date"],            format="%m/%d/%Y %I:%M:%S %p")
    hr["Time"]               = pd.to_datetime(hr["Time"],               format="%m/%d/%Y %I:%M:%S %p")

    hr_minute = (hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index())
    hr_minute.columns = ["Id","Time","HeartRate"]
    hr_minute = hr_minute.dropna()
    hr_minute["Date"] = hr_minute["Time"].dt.date

    hr_daily = (hr_minute.groupby(["Id","Date"])["HeartRate"]
                .agg(["mean","max","min","std"]).reset_index()
                .rename(columns={"mean":"AvgHR","max":"MaxHR","min":"MinHR","std":"StdHR"}))

    sleep["Date"] = sleep["date"].dt.date
    sleep_daily = (sleep.groupby(["Id","Date"])
                   .agg(TotalSleepMinutes=("value","count"),
                        DominantSleepStage=("value", lambda x: x.mode()[0]))
                   .reset_index())

    master = daily.copy().rename(columns={"ActivityDate":"Date"})
    master["Date"] = master["Date"].dt.date
    master = master.merge(hr_daily,    on=["Id","Date"], how="left")
    master = master.merge(sleep_daily, on=["Id","Date"], how="left")
    master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
    master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
    for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
        master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

    return master, daily, hourly_s, hourly_i, sleep, hr, hr_minute

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0); plt.close(fig)
    return buf.getvalue()

def step_hdr(num, label):
    st.markdown(f"""
    <div class='step-header'>
      <span style='color:#f97316'>◆</span> Step {num}
      <span style='color:#f97316;font-size:0.6rem'>·</span> {label}
    </div>""", unsafe_allow_html=True)

def m3_sec(icon, title, badge=None):
    badge_html = f'<span class="m3-sec-badge">{badge}</span>' if badge else ''
    st.markdown(f'<div class="m3-sec-header"><div class="m3-sec-icon">{icon}</div><p class="m3-sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

def m3_success(msg): st.markdown(f'<div class="m3-alert-success">✅ {msg}</div>', unsafe_allow_html=True)
def m3_warn(msg):    st.markdown(f'<div class="m3-alert-warn">⚠️ {msg}</div>', unsafe_allow_html=True)
def m3_info(msg):    st.markdown(f'<div class="m3-alert-info">ℹ️ {msg}</div>', unsafe_allow_html=True)
def m3_danger(msg):  st.markdown(f'<div class="m3-alert-danger">🚨 {msg}</div>', unsafe_allow_html=True)

def m3_metrics(*items, red_indices=None):
    red_indices = red_indices or []
    html = '<div class="m3-metric-grid">'
    for i, (val, label) in enumerate(items):
        vc = "m3-metric-val m3-metric-val-red" if i in red_indices else "m3-metric-val"
        html += f'<div class="m3-metric-card"><div class="{vc}">{val}</div><div class="m3-metric-label">{label}</div></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def apply_plotly_theme(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font_color=M3_TEXT, font_size=14, font_family="Syne, sans-serif"))
    return fig

# M4 section header helper
def m4_sec(icon, title, badge=None):
    badge_html = f'<span class="m4-sec-badge">{badge}</span>' if badge else ''
    st.markdown(f'<div class="m4-sec-header"><div class="m4-sec-icon">{icon}</div><p class="m4-sec-title">{title}</p>{badge_html}</div>', unsafe_allow_html=True)

def m4_info(m):    st.markdown(f'<div class="m4-alert-info">ℹ️ {m}</div>',    unsafe_allow_html=True)
def m4_success(m): st.markdown(f'<div class="m4-alert-success">✅ {m}</div>', unsafe_allow_html=True)
def m4_danger(m):  st.markdown(f'<div class="m4-alert-danger">🚨 {m}</div>',  unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY DETECTION FUNCTIONS  (shared by M3 + M4)
# ─────────────────────────────────────────────────────────────────────────────
def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["AvgHR"].mean().reset_index()
    d.columns = ["Date","AvgHR"]
    d = d.sort_values("Date")
    d["thresh_high"] = d["AvgHR"] > hr_high
    d["thresh_low"]  = d["AvgHR"] < hr_low
    d["rolling_med"]  = d["AvgHR"].rolling(3, center=True, min_periods=1).median()
    d["residual"]     = d["AvgHR"] - d["rolling_med"]
    resid_std = d["residual"].std()
    d["resid_anom"] = d["residual"].abs() > (residual_sigma * resid_std)
    d["is_anomaly"] = d["thresh_high"] | d["thresh_low"] | d["resid_anom"]
    def reason(row):
        r = []
        if row["thresh_high"]:   r.append(f"HR>{hr_high}")
        if row["thresh_low"]:    r.append(f"HR<{hr_low}")
        if row["resid_anom"]:    r.append(f"+/-{residual_sigma:.0f}sigma residual")
        return ", ".join(r) if r else ""
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSteps"].mean().reset_index().sort_values("Date")
    d["rolling_med"]   = d["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    d["residual"]      = d["TotalSteps"] - d["rolling_med"]
    resid_std = d["residual"].std()
    d["thresh_low"]  = d["TotalSteps"] < steps_low
    d["thresh_high"] = d["TotalSteps"] > steps_high
    d["resid_anom"]  = d["residual"].abs() > (residual_sigma * resid_std)
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Steps<{int(steps_low):,}")
        if row["thresh_high"]:   r.append(f"Steps>{int(steps_high):,}")
        if row["resid_anom"]:    r.append(f"+/-{residual_sigma:.0f}sigma residual")
        return ", ".join(r) if r else ""
    d["reason"] = d.apply(reason, axis=1)
    return d

def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    d = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index().sort_values("Date")
    d["rolling_med"]   = d["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    d["residual"]      = d["TotalSleepMinutes"] - d["rolling_med"]
    resid_std = d["residual"].std()
    d["thresh_low"]  = (d["TotalSleepMinutes"] > 0) & (d["TotalSleepMinutes"] < sleep_low)
    d["thresh_high"] = d["TotalSleepMinutes"] > sleep_high
    d["resid_anom"]  = d["residual"].abs() > (residual_sigma * resid_std)
    d["is_anomaly"]  = d["thresh_low"] | d["thresh_high"] | d["resid_anom"]
    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Sleep<{int(sleep_low)}min")
        if row["thresh_high"]:   r.append(f"Sleep>{int(sleep_high)}min")
        if row["resid_anom"]:    r.append(f"+/-{residual_sigma:.0f}sigma residual")
        return ", ".join(r) if r else ""
    d["reason"] = d.apply(reason, axis=1)
    return d

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    hr_sim = df_daily[["Date","AvgHR"]].copy()
    inject_idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[inject_idx, "AvgHR"] = np.random.choice([115,120,125,35,40,45,118,130,38,42], n_inject, replace=True)
    hr_sim["rolling_med"] = hr_sim["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_sim["residual"]    = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    resid_std = hr_sim["residual"].std()
    hr_sim["detected"] = (hr_sim["AvgHR"] > 100) | (hr_sim["AvgHR"] < 50) | (hr_sim["residual"].abs() > 2 * resid_std)
    tp = hr_sim.iloc[inject_idx]["detected"].sum()
    results["Heart Rate"] = {"injected": n_inject, "detected": int(tp), "accuracy": round(tp / n_inject * 100, 1)}
    st_sim = df_daily[["Date","TotalSteps"]].copy()
    inject_idx2 = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[inject_idx2, "TotalSteps"] = np.random.choice([50,100,150,30000,35000,28000,80,200,31000,29000], n_inject, replace=True)
    st_sim["rolling_med"] = st_sim["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    st_sim["residual"]    = st_sim["TotalSteps"] - st_sim["rolling_med"]
    resid_std2 = st_sim["residual"].std()
    st_sim["detected"] = (st_sim["TotalSteps"] < 500) | (st_sim["TotalSteps"] > 25000) | (st_sim["residual"].abs() > 2 * resid_std2)
    tp2 = st_sim.iloc[inject_idx2]["detected"].sum()
    results["Steps"] = {"injected": n_inject, "detected": int(tp2), "accuracy": round(tp2 / n_inject * 100, 1)}
    sl_sim = df_daily[["Date","TotalSleepMinutes"]].copy()
    inject_idx3 = np.random.choice(len(sl_sim), n_inject, replace=False)
    sl_sim.loc[inject_idx3, "TotalSleepMinutes"] = np.random.choice([10,20,30,700,750,800,15,25,710,720], n_inject, replace=True)
    sl_sim["rolling_med"] = sl_sim["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sl_sim["residual"]    = sl_sim["TotalSleepMinutes"] - sl_sim["rolling_med"]
    resid_std3 = sl_sim["residual"].std()
    sl_sim["detected"] = ((sl_sim["TotalSleepMinutes"] > 0) & (sl_sim["TotalSleepMinutes"] < 60)) | \
                          (sl_sim["TotalSleepMinutes"] > 600) | (sl_sim["residual"].abs() > 2 * resid_std3)
    tp3 = sl_sim.iloc[inject_idx3]["detected"].sum()
    results["Sleep"] = {"injected": n_inject, "detected": int(tp3), "accuracy": round(tp3 / n_inject * 100, 1)}
    results["Overall"] = round(np.mean([results[k]["accuracy"] for k in ["Heart Rate","Steps","Sleep"]]), 1)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# M4 CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def m4_ptheme(fig, title="", h=400):
    fig.update_layout(
        paper_bgcolor=M4_BG, plot_bgcolor=M4_PLOT_BG, font_color=M4_TEXT,
        font_family="Inter, sans-serif", height=h,
        legend=dict(bgcolor=M4_CARD, bordercolor=M4_BOR, borderwidth=1, font_color=M4_TEXT),
        margin=dict(l=50, r=30, t=55, b=45),
        hoverlabel=dict(bgcolor=M4_CARD, bordercolor=M4_BOR, font_color=M4_TEXT),
    )
    fig.update_xaxes(gridcolor=M4_GRID, zeroline=False, linecolor=M4_BOR, tickfont_color=M4_MUTED)
    fig.update_yaxes(gridcolor=M4_GRID, zeroline=False, linecolor=M4_BOR, tickfont_color=M4_MUTED)
    if title:
        fig.update_layout(title=dict(text=title, font_color=M4_TEXT, font_size=13, font_family="Syne, sans-serif"))
    return fig

def chart_hr(anom_hr, hr_high, hr_low, sigma, h=380):
    fig = go.Figure()
    upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
    lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=upper, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=lower, mode="lines",
                             fill="tonexty", fillcolor="rgba(99,179,237,0.1)",
                             line=dict(width=0), name=f"+/-{sigma:.0f}sigma Band"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"],
                             mode="lines+markers", name="Avg HR",
                             line=dict(color=M4_ACCENT, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
    fig.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=M4_ACC3, width=1.5, dash="dot")))
    a = anom_hr[anom_hr["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["AvgHR"], mode="markers",
                                 name="🚨 Anomaly",
                                 marker=dict(color=M4_RED, size=13, symbol="circle",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>HR: %{y:.1f}<br><b>ANOMALY</b><extra>⚠️</extra>"))
        for _, row in a.iterrows():
            fig.add_annotation(x=row["Date"], y=row["AvgHR"],
                               text="⚠️", showarrow=True, arrowhead=2,
                               arrowcolor=M4_RED, ax=0, ay=-35,
                               font=dict(color=M4_RED, size=11))
    fig.add_hline(y=hr_high, line_dash="dash", line_color=M4_RED, line_width=1.5, opacity=0.6,
                  annotation_text=f"High ({int(hr_high)} bpm)", annotation_font_color=M4_RED,
                  annotation_position="top right")
    fig.add_hline(y=hr_low, line_dash="dash", line_color=M4_ACC2, line_width=1.5, opacity=0.6,
                  annotation_text=f"Low ({int(hr_low)} bpm)", annotation_font_color=M4_ACC2,
                  annotation_position="bottom right")
    m4_ptheme(fig, "❤️ Heart Rate - Anomaly Detection", h)
    fig.update_layout(xaxis_title="Date", yaxis_title="HR (bpm)")
    return fig

def chart_steps(anom_steps, st_low, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Daily Steps (avg users)","Residual Deviation"])
    a = anom_steps[anom_steps["is_anomaly"]]
    for _, row in a.iterrows():
        fig.add_vrect(x0=str(row["Date"]), x1=str(row["Date"]),
                      fillcolor="rgba(252,129,129,0.12)",
                      line_color="rgba(252,129,129,0.4)", line_width=1.5, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"],
                             mode="lines+markers", name="Avg Steps",
                             line=dict(color=M4_ACC3, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=M4_ACCENT, width=2, dash="dash")),
                  row=1, col=1)
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSteps"],
                                 mode="markers", name="🚨 Alert",
                                 marker=dict(color=M4_RED, size=13, symbol="triangle-up",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(st_low), line_dash="dash", line_color=M4_RED, line_width=1.5,
                  opacity=0.7, row=1, col=1,
                  annotation_text=f"Low ({int(st_low):,})", annotation_font_color=M4_RED)
    res_colors = [M4_RED if v else M4_ACC3 for v in anom_steps["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:,.0f}<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=M4_MUTED, line_width=1, row=2, col=1)
    m4_ptheme(fig, "🚶 Step Count - Trend & Alerts", h)
    return fig

def chart_sleep(anom_sleep, sl_low, sl_high, h=380):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.07,
                        subplot_titles=["Sleep Duration (min/night)","Residual Deviation"])
    fig.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(104,211,145,0.07)", line_width=0,
                  annotation_text="✅ Healthy Zone", annotation_position="top right",
                  annotation_font_color=M4_ACC3, row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"],
                             mode="lines+markers", name="Sleep (min)",
                             line=dict(color=M4_PURPLE, width=2.5), marker=dict(size=5),
                             hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f} min<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"],
                             mode="lines", name="Trend",
                             line=dict(color=M4_ACC3, width=1.5, dash="dot")),
                  row=1, col=1)
    a = anom_sleep[anom_sleep["is_anomaly"]]
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Date"], y=a["TotalSleepMinutes"],
                                 mode="markers", name="🚨 Anomaly",
                                 marker=dict(color=M4_RED, size=13, symbol="diamond",
                                             line=dict(color="white", width=2)),
                                 hovertemplate="<b>%{x|%d %b}</b><br>Sleep: %{y:.0f}<br><b>ANOMALY</b><extra>⚠️</extra>"),
                      row=1, col=1)
    fig.add_hline(y=int(sl_low), line_dash="dash", line_color=M4_RED, line_width=1.5,
                  opacity=0.7, row=1, col=1,
                  annotation_text=f"Min ({int(sl_low)} min)", annotation_font_color=M4_RED)
    fig.add_hline(y=int(sl_high), line_dash="dash", line_color=M4_ACCENT, line_width=1.5,
                  opacity=0.6, row=1, col=1,
                  annotation_text=f"Max ({int(sl_high)} min)", annotation_font_color=M4_ACCENT)
    res_colors = [M4_RED if v else M4_PURPLE for v in anom_sleep["resid_anom"]]
    fig.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"],
                         name="Residual", marker_color=res_colors,
                         hovertemplate="<b>%{x|%d %b}</b><br>Δ: %{y:.0f} min<extra></extra>"),
                  row=2, col=1)
    fig.add_hline(y=0, line_color=M4_MUTED, line_width=1, row=2, col=1)
    m4_ptheme(fig, "💤 Sleep Pattern - Anomaly Visualization", h)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# M4 PDF + CSV GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def try_export_chart_png(fig, width=1100, height=480):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=1.5, engine="kaleido")
    except Exception:
        return None

def generate_pdf(master, anom_hr, anom_steps, anom_sleep,
                 hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                 fig_hr, fig_steps, fig_sleep):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_RIGHT, TA_CENTER
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, Image as RLImage, PageBreak)
    from reportlab.lib.colors import HexColor, white

    C_NAVY  = HexColor("#0f172a"); C_BLUE  = HexColor("#63b3ed"); C_GREEN = HexColor("#68d391")
    C_PINK  = HexColor("#f687b3"); C_RED   = HexColor("#fc8181"); C_PURPLE= HexColor("#b794f4")
    C_MUTED = HexColor("#94a3b8"); C_LIGHT = HexColor("#e2e8f0"); C_CARD  = HexColor("#1e2d45")
    C_HDR   = HexColor("#0a0e1a"); C_ORG   = HexColor("#f6ad55")

    buf = io.BytesIO()
    PAGE_W, PAGE_H = A4

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_HDR)
        canvas.rect(0, PAGE_H - 22*mm, PAGE_W, 22*mm, fill=1, stroke=0)
        canvas.setFont("Helvetica-Bold", 11); canvas.setFillColor(C_BLUE)
        canvas.drawCentredString(PAGE_W/2, PAGE_H-12*mm, "FitPulse Anomaly Detection Report — Milestone 4")
        canvas.setFont("Helvetica", 7); canvas.setFillColor(C_MUTED)
        canvas.drawCentredString(PAGE_W/2, PAGE_H-18*mm, f"Generated: {datetime.now().strftime('%d %B %Y  %H:%M')}")
        canvas.setFont("Helvetica", 7); canvas.setFillColor(C_MUTED)
        canvas.drawCentredString(PAGE_W/2, 10*mm, f"FitPulse ML Pipeline  ·  Page {doc.page}")
        canvas.restoreState()

    def section_hdr(text, bg=C_NAVY):
        t = Table([[Paragraph(f"<b>{text}</b>", ParagraphStyle("sh", fontName="Helvetica-Bold", fontSize=10, textColor=white))]],
                  colWidths=[165*mm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("TOPPADDING",(0,0),(-1,-1),5),
                                ("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),8)]))
        return t

    s_body = ParagraphStyle("body",fontName="Helvetica",fontSize=8.5,textColor=HexColor("#1e293b"),leading=13,spaceAfter=4)

    def kv(key, val):
        return Table([[Paragraph(f"<b>{key}</b>", ParagraphStyle("k",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_MUTED)),
                       Paragraph(str(val), ParagraphStyle("v",fontName="Helvetica-Bold",fontSize=8.5,textColor=HexColor("#1e293b")))]],
                     colWidths=[55*mm,110*mm])

    def data_tbl(df, max_rows=20):
        if df.empty:
            return Paragraph("No anomalies detected.", s_body)
        df2 = df.head(max_rows).copy()
        header = [Paragraph(f"<b>{c}</b>", ParagraphStyle("th",fontName="Helvetica-Bold",fontSize=7.5,textColor=white)) for c in df2.columns]
        rows = [header]
        for _, row in df2.iterrows():
            cells = []
            for val in row:
                text = f"{val:.2f}" if isinstance(val, float) else (val.strftime("%d %b %Y") if hasattr(val,"strftime") else str(val)[:28])
                cells.append(Paragraph(text, ParagraphStyle("td",fontName="Helvetica",fontSize=7,textColor=C_LIGHT)))
            rows.append(cells)
        cw = 165/len(df2.columns)
        t = Table(rows, colWidths=[cw*mm]*len(df2.columns), repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),C_NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[C_CARD,HexColor("#162032")]),
            ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
            ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
            ("GRID",(0,0),(-1,-1),0.3,HexColor("#334155")),
        ]))
        elems = [t]
        if len(df) > max_rows:
            elems.append(Spacer(1,3))
            elems.append(Paragraph(f"… and {len(df)-max_rows} more records",
                                   ParagraphStyle("note",fontName="Helvetica-Oblique",fontSize=7,textColor=C_MUTED)))
        return elems

    SP = lambda h=6: Spacer(1, h)

    n_hr    = int(anom_hr["is_anomaly"].sum())
    n_steps = int(anom_steps["is_anomaly"].sum())
    n_sleep = int(anom_sleep["is_anomaly"].sum())
    n_users = master["Id"].nunique()
    n_days  = master["Date"].nunique()
    date_range_str = (f"{pd.to_datetime(master['Date']).min().strftime('%d %b %Y')}"
                      f" – {pd.to_datetime(master['Date']).max().strftime('%d %b %Y')}")

    story = []
    story.append(SP(4))

    hero_t = Table([[Paragraph("📊 FitPulse Insights Dashboard",
                               ParagraphStyle("h",fontName="Helvetica-Bold",fontSize=18,textColor=C_BLUE)),
                     Paragraph(f"<b>{n_users}</b> users · <b>{n_days}</b> days · <b>{n_hr+n_steps+n_sleep}</b> anomalies",
                               ParagraphStyle("hr",fontName="Helvetica-Bold",fontSize=9,textColor=C_MUTED,alignment=TA_RIGHT))]],
                   colWidths=[115*mm,50*mm])
    hero_t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),("BACKGROUND",(0,0),(-1,-1),HexColor("#0d1830")),
                                 ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
                                 ("LEFTPADDING",(0,0),(-1,-1),12),("RIGHTPADDING",(0,0),(-1,-1),12)]))
    story.append(hero_t); story.append(SP(12))

    story.append(section_hdr("1.  EXECUTIVE SUMMARY")); story.append(SP(6))
    for k, v in [("Dataset","Real Fitbit Device Data — Kaggle (arashnic/fitbit)"),
                 ("Users",f"{n_users} participants"),("Date Range",date_range_str),
                 ("Total Days",f"{n_days} days"),("Pipeline","Milestone 4 — Anomaly Detection Dashboard")]:
        story.append(kv(k, v)); story.append(SP(2))
    story.append(SP(8))

    story.append(section_hdr("2.  ANOMALY SUMMARY", HexColor("#7f1d1d"))); story.append(SP(6))
    kpi_data = [
        [Paragraph("<b>Signal</b>",ParagraphStyle("kh",fontName="Helvetica-Bold",fontSize=8.5,textColor=white)),
         Paragraph("<b>Flagged</b>",ParagraphStyle("kh",fontName="Helvetica-Bold",fontSize=8.5,textColor=white)),
         Paragraph("<b>Colour</b>",ParagraphStyle("kh",fontName="Helvetica-Bold",fontSize=8.5,textColor=white))],
        [Paragraph("Heart Rate",s_body),Paragraph(str(n_hr),ParagraphStyle("v",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_PINK)),Paragraph("●",ParagraphStyle("c",textColor=C_PINK,fontSize=14))],
        [Paragraph("Steps",s_body),    Paragraph(str(n_steps),ParagraphStyle("v",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_GREEN)),Paragraph("●",ParagraphStyle("c",textColor=C_GREEN,fontSize=14))],
        [Paragraph("Sleep",s_body),    Paragraph(str(n_sleep),ParagraphStyle("v",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_PURPLE)),Paragraph("●",ParagraphStyle("c",textColor=C_PURPLE,fontSize=14))],
        [Paragraph("<b>TOTAL</b>",s_body),Paragraph(f"<b>{n_hr+n_steps+n_sleep}</b>",ParagraphStyle("v",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_RED)),Paragraph("")],
    ]
    kpi_tbl = Table(kpi_data, colWidths=[70*mm,40*mm,55*mm])
    kpi_tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_NAVY),
                                  ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_CARD,HexColor("#162032")]),
                                  ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                                  ("LEFTPADDING",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),0.3,HexColor("#334155")),
                                  ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(kpi_tbl); story.append(SP(8))

    story.append(section_hdr("3.  DETECTION THRESHOLDS", HexColor("#14532d"))); story.append(SP(6))
    for k, v in [("Heart Rate High",f"> {int(hr_high)} bpm"),("Heart Rate Low",f"< {int(hr_low)} bpm"),
                 ("Steps Low Alert",f"< {int(st_low):,} steps/day"),("Sleep Low",f"< {int(sl_low)} minutes/night"),
                 ("Sleep High",f"> {int(sl_high)} minutes/night"),("Residual Sigma",f"+/- {float(sigma):.1f}σ from rolling median")]:
        story.append(kv(k, v)); story.append(SP(2))
    story.append(SP(8))

    story.append(section_hdr("4.  METHODOLOGY", HexColor("#1e3a5f"))); story.append(SP(6))
    story.append(Paragraph(
        "<b>Three complementary anomaly detection methods:</b><br/><br/>"
        "<b>1. Threshold Violations</b> — Hard upper/lower bounds on each metric.<br/><br/>"
        "<b>2. Residual-Based Detection</b> — 3-day rolling median baseline. "
        f"Days deviating by >±{float(sigma):.1f}σ are flagged.<br/><br/>"
        "<b>3. DBSCAN Outlier Clustering</b> — Users profiled on 7 activity features. Label −1 = structural outliers.",
        s_body))
    story.append(PageBreak())

    # Charts page
    story.append(section_hdr("5.  ANOMALY CHARTS", HexColor("#0f172a"))); story.append(SP(8))
    _tmp_paths = []
    for label, fig in [
        ("Figure 1 — Heart Rate with Anomaly Highlights", fig_hr),
        ("Figure 2 — Step Count Trend with Alert Bands",   fig_steps),
        ("Figure 3 — Sleep Pattern Visualization",         fig_sleep),
    ]:
        story.append(Paragraph(f"<b>{label}</b>",
                               ParagraphStyle("fl",fontName="Helvetica-Bold",fontSize=8.5,textColor=C_MUTED,spaceAfter=3)))
        img_bytes = try_export_chart_png(fig)
        if img_bytes:
            story.append(RLImage(io.BytesIO(img_bytes), width=165*mm, height=70*mm))
        else:
            ph = Table([[Paragraph(
                f"📊 <b>{label}</b><br/>"
                "<font color='#94a3b8' size='7'>Install kaleido to embed chart images: pip install kaleido</font>",
                ParagraphStyle("ph",fontName="Helvetica",fontSize=8,textColor=C_MUTED,leading=12))]],
                colWidths=[165*mm])
            ph.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),HexColor("#0d1830")),
                                     ("TOPPADDING",(0,0),(-1,-1),18),("BOTTOMPADDING",(0,0),(-1,-1),18),
                                     ("LEFTPADDING",(0,0),(-1,-1),14)]))
            story.append(ph)
        story.append(SP(8))
    story.append(PageBreak())

    # Anomaly tables
    for sec_title, bg_col, df_in, rename_map in [
        ("6.  ANOMALY RECORDS — HEART RATE", HexColor("#7f1d1d"),
         anom_hr[anom_hr["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]],
         {"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation"}),
        ("7.  ANOMALY RECORDS — STEPS", HexColor("#14532d"),
         anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]],
         {"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation"}),
        ("8.  ANOMALY RECORDS — SLEEP", HexColor("#4a1d96"),
         anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]],
         {"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation"}),
    ]:
        story.append(section_hdr(sec_title, bg_col)); story.append(SP(6))
        result = data_tbl(df_in.rename(columns=rename_map).round(2))
        for e in (result if isinstance(result, list) else [result]): story.append(e)
        story.append(SP(10))
    story.append(PageBreak())

    # User profiles + conclusion
    story.append(section_hdr("9.  USER ACTIVITY PROFILES", HexColor("#0f172a"))); story.append(SP(6))
    profile_cols = ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
    available_cols = [c for c in profile_cols if c in master.columns]
    user_profile = master.groupby("Id")[available_cols].mean().round(1).reset_index()
    short_cols = ["User ID"] + [c[:14] for c in available_cols]
    prof_header = [Paragraph(f"<b>{c}</b>",ParagraphStyle("ph",fontName="Helvetica-Bold",fontSize=7.5,textColor=white)) for c in short_cols]
    prof_rows = [prof_header]
    for _, row in user_profile.iterrows():
        cells = [Paragraph(f"...{str(row['Id'])[-6:]}",ParagraphStyle("uid",fontName="Courier",fontSize=7,textColor=C_MUTED))]
        for col in available_cols:
            cells.append(Paragraph(f"{row[col]:,.0f}",ParagraphStyle("pv",fontName="Helvetica",fontSize=7,textColor=C_LIGHT)))
        prof_rows.append(cells)
    cw2 = 165/len(short_cols)
    prof_t = Table(prof_rows, colWidths=[cw2*mm]*len(short_cols), repeatRows=1)
    prof_t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_NAVY),
                                 ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_CARD,HexColor("#162032")]),
                                 ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
                                 ("LEFTPADDING",(0,0),(-1,-1),4),("GRID",(0,0),(-1,-1),0.3,HexColor("#334155")),
                                 ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(prof_t); story.append(SP(12))

    story.append(section_hdr("10.  CONCLUSION", HexColor("#14532d"))); story.append(SP(6))
    story.append(Paragraph(
        f"The FitPulse pipeline processed <b>{n_users}</b> users over <b>{n_days}</b> days. "
        f"<b>{n_hr+n_steps+n_sleep}</b> total anomalies: HR {n_hr}, Steps {n_steps}, Sleep {n_sleep}. "
        "Three combined methods — threshold, residual, and DBSCAN — provided robust detection.",
        s_body))

    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=27*mm, bottomMargin=18*mm, leftMargin=22*mm, rightMargin=22*mm)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return buf

def generate_csv(anom_hr, anom_steps, anom_sleep):
    parts = []
    if anom_hr is not None:
        d = anom_hr[anom_hr["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].copy()
        d["signal"] = "Heart Rate"; d = d.rename(columns={"AvgHR":"value","rolling_med":"expected"})
        parts.append(d)
    if anom_steps is not None:
        d = anom_steps[anom_steps["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].copy()
        d["signal"] = "Steps"; d = d.rename(columns={"TotalSteps":"value","rolling_med":"expected"})
        parts.append(d)
    if anom_sleep is not None:
        d = anom_sleep[anom_sleep["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].copy()
        d["signal"] = "Sleep"; d = d.rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})
        parts.append(d)
    if not parts:
        return b"signal,Date,value,expected,residual,reason\n"
    combined = pd.concat(parts, ignore_index=True)[["signal","Date","value","expected","residual","reason"]].sort_values(["signal","Date"]).round(2)
    buf = io.StringIO(); combined.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:4px 0 6px 0'>
      <span style='font-size:1.4rem'>🩺</span>
      <div>
        <div style='font-size:1.05rem;font-weight:800;color:#e2eaf4'>FitPulse</div>
        <div style='font-size:0.7rem;color:#6b7a96'>Fitness ML Analytics Pipeline</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:10px">📌 Pipeline Sections</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔧 Preprocessing", use_container_width=True,
                     type="primary" if st.session_state.active_section=="preprocessing" else "secondary"):
            st.session_state.active_section = "preprocessing"; st.rerun()
    with col_b:
        if st.button("🤖 Patterns extraction", use_container_width=True,
                     type="primary" if st.session_state.active_section=="pattern_extraction" else "secondary"):
            st.session_state.active_section = "pattern_extraction"; st.rerun()

    col_c, col_d = st.columns(2)
    with col_c:
        if st.button("🚨 Anomaly Detection", use_container_width=True,
                     type="primary" if st.session_state.active_section=="anomaly_detector" else "secondary"):
            st.session_state.active_section = "anomaly_detector"; st.rerun()
    with col_d:
        if st.button("📊 Report Download", use_container_width=True,
                     type="primary" if st.session_state.active_section=="report" else "secondary"):
            st.session_state.active_section = "report"; st.rerun()

    st.divider()
    active = st.session_state.active_section

    # ── SHARED FILE UPLOAD ────────────────────────────────────────────────────
    if active in ("pattern_extraction", "anomaly_detector", "report"):
        st.markdown("""
        <div class="shared-upload-title">📁 Dataset Files</div>
        <div class="shared-upload-sub">Upload all 5 Fitbit CSV files — shared across Pattern Extraction, Anomaly Detector & Dashboard.</div>
        """, unsafe_allow_html=True)

        shared_up = st.file_uploader(
            "Upload 5 Fitbit CSVs",
            type="csv", accept_multiple_files=True,
            key="sidebar_shared_uploader",
            label_visibility="collapsed"
        )

        if shared_up:
            raw_list = []
            for uf in shared_up:
                try:
                    raw_list.append((uf.name, pd.read_csv(uf)))
                except Exception:
                    pass
            new_detected = detect_shared_files(raw_list)
            if new_detected:
                st.session_state.shared_detected = new_detected

        detected = st.session_state.shared_detected
        n_det    = len(detected)

        pills_html = '<div style="display:flex;flex-wrap:wrap;gap:3px;margin:6px 0">'
        for req_name, finfo in SHARED_REQUIRED.items():
            found = req_name in detected
            cls   = "file-pill" if found else "file-pill file-pill-miss"
            ico   = "✓" if found else "✗"
            pills_html += f'<span class="{cls}">{ico} {finfo["icon"]} {finfo["label"][:5]}</span>'
        pills_html += "</div>"
        st.markdown(pills_html, unsafe_allow_html=True)

        if n_det < 5:
            st.markdown(f'<div style="font-size:0.68rem;color:#f6ad55;font-family:JetBrains Mono,monospace">{n_det}/5 files detected</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:0.68rem;color:#34d399;font-family:JetBrains Mono,monospace">✓ All 5 files ready</div>', unsafe_allow_html=True)

        if n_det == 5:
            if st.button("⚡ Load & Build Master", use_container_width=True, key="sidebar_load_btn"):
                with st.spinner("Building master dataframe..."):
                    try:
                        master, daily, hourly_s, hourly_i, sleep, hr, hr_minute = build_shared_master(detected)
                        st.session_state.shared_master   = master
                        st.session_state.shared_loaded   = True
                        st.session_state.daily     = daily
                        st.session_state.steps     = hourly_s
                        st.session_state.intensity = hourly_i
                        st.session_state.sleep     = sleep
                        st.session_state.hr        = hr
                        st.session_state.master_df = master
                        st.session_state.files_loaded = True
                        st.session_state.m3_daily     = daily
                        st.session_state.m3_hourly_s  = hourly_s
                        st.session_state.m3_hourly_i  = hourly_i
                        st.session_state.m3_sleep     = sleep
                        st.session_state.m3_hr        = hr
                        st.session_state.m3_hr_minute = hr_minute
                        st.session_state.m3_master    = master
                        st.session_state.m3_files_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.shared_loaded:
            master_tmp = st.session_state.shared_master
            st.markdown(f"""
            <div style="background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.3);border-radius:8px;padding:0.6rem 0.8rem;margin-top:0.5rem">
              <div style="font-size:0.68rem;color:#34d399;font-family:'JetBrains Mono',monospace;font-weight:700">✓ MASTER LOADED</div>
              <div style="font-size:0.63rem;color:#6b7a96;font-family:'JetBrains Mono',monospace;margin-top:2px">
                {master_tmp['Id'].nunique()} users · {master_tmp['Date'].nunique()} days · {len(master_tmp):,} rows
              </div>
            </div>""", unsafe_allow_html=True)

        st.divider()

    # ── Section-specific sidebar controls ────────────────────────────────────
    if active == "preprocessing":
        st.markdown('<div style="font-size:0.72rem;color:#6b7a96;line-height:2"><div style="color:#63d7c4;font-weight:700;margin-bottom:6px">🔧 Preprocessing Steps</div>① Upload CSV Dataset<br>② View Missing Values<br>③ Run Data Cleaning<br>④ Explore Distributions<br>⑤ Download Clean CSV</div>', unsafe_allow_html=True)
        st.divider()
        def sidebar_status(done, label):
            color = "#34d399" if done else "#6b7a96"; icon = "✓" if done else "○"
            st.markdown(f"<div style='font-size:0.8rem;color:{color};padding:3px 0'>{icon} {label}</div>", unsafe_allow_html=True)
        sidebar_status(st.session_state.get("milestone1_loaded", False), "Dataset Loaded")
        sidebar_status(st.session_state.get("clean_df") is not None, "Data Cleaned")

    elif active == "pattern_extraction":
        fl=st.session_state.files_loaded; tf=st.session_state.tsfresh_done
        pf=st.session_state.prophet_done; cl=st.session_state.cluster_done
        ss=st.session_state.get("steps_sleep_fig") is not None
        pct=int(sum([fl,tf,pf,cl,ss])/5*100)
        st.markdown(f'<div style="font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700">Pipeline Progress <span style="color:#63d7c4;margin-left:6px">{pct}%</span></div><div class="prog-bar-bg"><div class="prog-bar-fill" style="width:{pct}%"></div></div>', unsafe_allow_html=True)
        def nav(a,i,l):
            d="nav-dot-active" if a else "nav-dot-inactive"; lc="nav-label-active" if a else "nav-label-inactive"
            st.markdown(f"<div class='nav-item'><div class='{d}'></div><span>{i}</span><span class='{lc}'>{l}</span></div>", unsafe_allow_html=True)
        nav(fl,"📁","Data Loading"); nav(tf,"⚗️","TSFresh Features")
        nav(pf,"📅","Prophet Forecast"); nav(cl,"🔵","Clustering"); nav(ss,"📈","Steps & Sleep")
        st.divider()
        st.markdown('<div style="font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700">⚙️ ML Controls</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-top:10px">KMeans Clusters</div>', unsafe_allow_html=True)
        k_val = st.slider("k", 2, 9, 3, label_visibility="collapsed")
        st.markdown('<div style="font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-top:10px">DBSCAN EPS</div>', unsafe_allow_html=True)
        eps_val = st.slider("eps", 0.5, 5.0, 2.20, step=0.1, label_visibility="collapsed")

    elif active == "anomaly_detector":
        steps_done = sum([st.session_state.m3_files_loaded, st.session_state.m3_anomaly_done, st.session_state.m3_simulation_done])
        pct_m3 = int(steps_done/3*100)
        st.markdown(f'<div style="margin-bottom:1rem"><div style="font-size:0.7rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">PIPELINE · {pct_m3}%</div><div style="background:rgba(99,179,237,0.2);border-radius:4px;height:6px;overflow:hidden"><div style="width:{pct_m3}%;height:100%;background:linear-gradient(90deg,#fc8181,#f687b3);border-radius:4px"></div></div></div>', unsafe_allow_html=True)
        for done, icon, label in [(st.session_state.m3_files_loaded,"📂","Data Loaded"),(st.session_state.m3_anomaly_done,"🚨","Anomalies Detected"),(st.session_state.m3_simulation_done,"🎯","Accuracy Simulated")]:
            dot='<span style="color:#68d391">●</span>' if done else '<span style="color:#6b7a96">○</span>'
            col="#e2e8f0" if done else "#6b7a96"
            st.markdown(f'<div style="font-size:0.82rem;padding:0.3rem 0;color:{col}">{dot} {icon} {label}</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<div style="font-size:0.72rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin-bottom:0.5rem">THRESHOLDS</div>', unsafe_allow_html=True)
        hr_high = st.number_input("HR High (bpm)",   value=100, min_value=80,  max_value=180, key="m3_hr_high")
        hr_low  = st.number_input("HR Low (bpm)",    value=50,  min_value=30,  max_value=70,  key="m3_hr_low")
        st_low  = st.number_input("Steps Low",       value=500, min_value=0,   max_value=2000, key="m3_st_low")
        sl_low  = st.number_input("Sleep Low (min)", value=60,  min_value=0,   max_value=120, key="m3_sl_low")
        sl_high = st.number_input("Sleep High (min)",value=600, min_value=300, max_value=900, key="m3_sl_high")
        sigma   = st.slider("Residual σ threshold", 1.0, 4.0, 2.0, 0.5, key="m3_sigma")

    elif active == "report":
        # M4 Dashboard sidebar controls
        st.markdown(f'<div style="font-size:0.7rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin-bottom:0.5rem">DETECTION THRESHOLDS</div>', unsafe_allow_html=True)
        hr_high = int(st.number_input("HR High (bpm)",    value=100, min_value=80,  max_value=180, key="rep_hr_high"))
        hr_low  = int(st.number_input("HR Low (bpm)",     value=50,  min_value=30,  max_value=70,  key="rep_hr_low"))
        st_low  = int(st.number_input("Steps Low/day",    value=500, min_value=0,   max_value=2000, key="rep_st_low"))
        sl_low  = int(st.number_input("Sleep Low (min)",  value=60,  min_value=0,   max_value=120, key="rep_sl_low"))
        sl_high = int(st.number_input("Sleep High (min)", value=600, min_value=300, max_value=900, key="rep_sl_high"))
        sigma   = float(st.slider("Residual sigma", 1.0, 4.0, 2.0, 0.5, key="rep_sigma"))
        st.divider()

        # Date + user filters (only when data is loaded)
        date_range    = None
        selected_user = None
        if st.session_state.shared_loaded:
            master_tmp = st.session_state.shared_master
            all_dates  = pd.to_datetime(master_tmp["Date"])
            d_min = all_dates.min().date(); d_max = all_dates.max().date()
            st.markdown(f'<div style="font-size:0.7rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin-bottom:0.4rem">DATE FILTER</div>', unsafe_allow_html=True)
            date_range = st.date_input("Date range", value=(d_min, d_max),
                                       min_value=d_min, max_value=d_max,
                                       key="rep_daterange", label_visibility="collapsed")
            st.markdown(f'<div style="font-size:0.7rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin:0.6rem 0 0.4rem">USER FILTER</div>', unsafe_allow_html=True)
            all_users = sorted(master_tmp["Id"].unique())
            user_options = ["All Users"] + [f"...{str(u)[-6:]}" for u in all_users]
            sel_label = st.selectbox("User", user_options, key="rep_user", label_visibility="collapsed")
            selected_user = None if sel_label == "All Users" else all_users[user_options.index(sel_label) - 1]
            st.divider()

        # Run button
        run_report = st.button("⚡ Run Detection & Dashboard",
                               disabled=(not st.session_state.shared_loaded),
                               key="rep_run_btn")
        if not st.session_state.shared_loaded:
            st.markdown(f'<div style="font-size:0.7rem;color:#f6ad55;text-align:center">Load files first ↑</div>', unsafe_allow_html=True)

        # Pipeline progress bar
        pct_rep = int(st.session_state.m3_anomaly_done) * 100
        st.markdown(f"""
        <div style="font-size:0.68rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin:0.8rem 0 0.3rem">PIPELINE · {pct_rep}%</div>
        <div style="background:rgba(99,179,237,0.2);border-radius:4px;height:5px;overflow:hidden">
          <div style="width:{pct_rep}%;height:100%;background:linear-gradient(90deg,#63b3ed,#68d391);border-radius:4px"></div>
        </div>""", unsafe_allow_html=True)

# Set fallback defaults for vars not set in sidebar branches
if active not in ("pattern_extraction",):
    k_val = 3; eps_val = 2.2
if active not in ("anomaly_detector",):
    if active != "report":
        hr_high=100; hr_low=50; st_low=500; sl_low=60; sl_high=600; sigma=2.0

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
section_map = {
    "preprocessing":      ("🔧","Preprocessing"),
    "pattern_extraction": ("🤖","Pattern Extraction"),
    "anomaly_detector":   ("🚨","Anomaly Detector"),
    "report":             ("📊","Insights Dashboard"),
}
section_icon, section_label = section_map[active]
st.markdown(f"""
<div style='padding:1rem 0 0.3rem 0'>
  <h1 style='margin:0'>🩺 FitPulse <span style='color:#63d7c4'>Analytics</span></h1>
  <p style='color:#6b7a96;font-size:0.85rem;margin-top:4px'>
    {section_icon} Currently viewing: <span style='color:#63d7c4;font-weight:700'>{section_label}</span>
    &nbsp;·&nbsp; Fitness ML Pipeline
  </p>
</div>""", unsafe_allow_html=True)
st.divider()

# =============================================================================
# PREPROCESSING SECTION
# =============================================================================
if active == "preprocessing":
    st.markdown("## 🔧 Data Preprocessing")
    st.caption("Upload a fitness CSV, inspect missing values, clean the data, and explore distributions.")
    st.markdown("### 📂 Upload Dataset")
    file = st.file_uploader("Upload your fitness CSV", type=["csv"])
    if file:
        file.seek(0); df = pd.read_csv(file)
        st.session_state["df"] = df; st.session_state["milestone1_loaded"] = True
        st.success(f"✅ Dataset loaded — **{len(df):,} rows × {len(df.columns)} columns**")
    if st.session_state.get("milestone1_loaded") and st.session_state.get("df") is not None:
        df = st.session_state["df"]
        with st.expander("🔎 Raw Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        st.markdown("### 🔍 Missing Values Analysis")
        missing = df.isnull().sum(); total_missing = missing.sum()
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Rows",f"{len(df):,}"); c2.metric("Columns",f"{len(df.columns)}"); c3.metric("Missing Cells",f"{total_missing:,}")
        if total_missing > 0:
            fig, ax = plt.subplots(figsize=(10,3.5))
            missing_nz = missing[missing>0]
            bars = ax.bar(missing_nz.index, missing_nz.values, color=PALETTE[0], edgecolor="none")
            ax.set_title("Missing Values per Column", fontsize=13, pad=12)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(int(bar.get_height())), ha="center", va="bottom", fontsize=8, color="#e2eaf4")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.success("🎉 No missing values found!")
        st.markdown("### 🛠 Data Cleaning")
        if st.button("▶  Run Preprocessing"):
            with st.spinner("Cleaning data…"):
                df_clean = df.copy()
                for col in df_clean.select_dtypes(include=np.number).columns:
                    df_clean[col] = df_clean[col].interpolate()
                df_clean = df_clean.fillna("Unknown")
                st.session_state["clean_df"] = df_clean
            st.success("✅ Cleaning complete.")
    if st.session_state.get("clean_df") is not None:
        st.markdown("### 📑 Cleaned Dataset")
        st.dataframe(st.session_state["clean_df"].head(10), use_container_width=True)
        csv = st.session_state["clean_df"].to_csv(index=False).encode()
        st.download_button("⬇  Download Clean CSV", csv, "clean_dataset.csv", mime="text/csv")
        st.markdown("### 📊 Exploratory Data Analysis")
        df_eda = st.session_state["clean_df"]; num_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            pairs = [num_cols[i:i+2] for i in range(0, len(num_cols), 2)]
            for pair in pairs:
                grid = st.columns(len(pair))
                for ax_col, col in zip(grid, pair):
                    with ax_col:
                        fig, ax = plt.subplots(figsize=(5,3))
                        sns.histplot(df_eda[col], kde=True, ax=ax, color=PALETTE[0], edgecolor="none", line_kws={"color":PALETTE[1],"linewidth":2})
                        ax.set_title(col, fontsize=11); ax.set_xlabel("")
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("No numeric columns found.")

# =============================================================================
# PATTERN EXTRACTION SECTION
# =============================================================================
elif active == "pattern_extraction":
    st.markdown("## 🤖 Pattern Extraction — ML Pipeline")
    st.caption("Files are uploaded in the sidebar and shared. Run TSFresh, Prophet, and Clustering below.")

    if not st.session_state.shared_loaded:
        st.markdown(f"""
        <div class="fp-card" style="text-align:center;padding:2.5rem">
          <div style="font-size:2.5rem;margin-bottom:0.8rem">📁</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#e2eaf4;margin-bottom:0.4rem">Upload & Load Files First</div>
          <div style="color:#6b7a96;font-size:0.85rem">Use the <b style="color:#63d7c4">sidebar file uploader</b> to upload all 5 Fitbit CSV files,<br>then click <b>⚡ Load & Build Master</b>.</div>
        </div>""", unsafe_allow_html=True)
    else:
        master_df = st.session_state.shared_master
        m3_success(f"Master DataFrame ready — {master_df.shape[0]:,} rows · {master_df['Id'].nunique()} users · {master_df['Date'].nunique()} days")

        step_hdr(4, "Null Value Check")
        dsets = {"dailyActivity":st.session_state.daily,"hourlySteps":st.session_state.steps,
                 "hourlyIntensities":st.session_state.intensity,"minuteSleep":st.session_state.sleep,"heartrate":st.session_state.hr}
        nc = st.columns(5)
        for col, (name, df_) in zip(nc, dsets.items()):
            nulls = int(df_.isnull().sum().sum()); rows = len(df_)
            val = f"<div class='null-val-bad'>{nulls:,}</div>" if nulls>0 else "<div class='null-val-ok'>⊙</div>"
            with col:
                st.markdown(f"<div class='null-card'><div class='null-card-name'>{name}</div>{val}<div class='null-rows'>nulls · {rows:,} rows</div></div>", unsafe_allow_html=True)

        step_hdr(5, "Dataset Overview")
        hr2 = st.session_state.hr; sl2 = st.session_state.sleep
        hr_id_c = next((c for c in hr2.columns if c.lower()=="id"), None)
        sl_id_c = next((c for c in sl2.columns if c.lower()=="id"), None)
        daily_ = st.session_state.daily; d_id = next((c for c in daily_.columns if c.lower()=="id"), daily_.columns[0])
        o1,o2,o3,o4,o5 = st.columns(5)
        o1.metric(str(daily_[d_id].nunique()),"DAILY USERS")
        o2.metric(str(hr2[hr_id_c].nunique() if hr_id_c else 0),"HR USERS")
        o3.metric(str(sl2[sl_id_c].nunique() if sl_id_c else 0),"SLEEP USERS")
        o4.metric(f"{len(master_df):,}","MASTER ROWS")
        o5.metric(f"{master_df['Date'].nunique()}","DAYS")

        step_hdr(9, "Master Dataset Preview")
        st.dataframe(master_df.head(30), use_container_width=True)
        csv_bytes = master_df.to_csv(index=False).encode()
        st.download_button("⬇  Download Master CSV", csv_bytes, "fitpulse_master.csv", mime="text/csv")

        # TSFresh
        st.divider()
        step_hdr("ML‑1", "TSFresh Feature Extraction")
        st.caption("Extracts statistical features from Heart Rate time-series per user.")
        if st.button("▶  Run TSFresh"):
            with st.spinner("Extracting features…"):
                try:
                    hr3 = st.session_state.hr.copy(); hr3.columns = [c.strip() for c in hr3.columns]
                    id_col = next(c for c in hr3.columns if c.lower()=="id")
                    t_col  = next(c for c in hr3.columns if c.lower() in ("time","datetime","timestamp","date","activityminute"))
                    v_col  = next(c for c in hr3.columns if c.lower()=="value")
                    ts_hr = hr3[[id_col,t_col,v_col]].rename(columns={id_col:"id",t_col:"time",v_col:"value"})
                    ts_hr["time"]  = pd.to_datetime(ts_hr["time"], errors="coerce")
                    ts_hr["value"] = pd.to_numeric(ts_hr["value"], errors="coerce"); ts_hr.dropna(inplace=True)
                    features = extract_features(ts_hr, column_id="id", column_sort="time", column_value="value", default_fc_parameters=MinimalFCParameters())
                    features.dropna(axis=1, how="all", inplace=True)
                    scaler = MinMaxScaler(); norm = scaler.fit_transform(features)
                    fig, ax = plt.subplots(figsize=(11,4))
                    sns.heatmap(norm, cmap="YlOrRd", ax=ax, linewidths=0, cbar_kws={"shrink":0.7})
                    ax.set_title("TSFresh Feature Heatmap (normalised)", fontsize=12)
                    ax.set_xlabel("Feature Index"); ax.set_ylabel("User ID"); fig.tight_layout()
                    st.session_state["tsfresh_fig"] = fig_to_bytes(fig)
                    st.session_state.features = features; st.session_state.tsfresh_done = True
                    st.success(f"✅ Extracted **{features.shape[1]}** features for **{features.shape[0]}** users.")
                except Exception as e:
                    st.error(f"❌ TSFresh error: {e}")
        if st.session_state["tsfresh_fig"] is not None:
            st.image(st.session_state["tsfresh_fig"], use_container_width=True)

        if st.session_state.tsfresh_done:
            st.divider()
            step_hdr("ML‑2", "Prophet Forecast — Heart Rate")
            st.caption("30-day heart rate forecast using Meta's Prophet model.")
            if st.button("▶  Run Prophet"):
                with st.spinner("Fitting Prophet model…"):
                    try:
                        hr4 = st.session_state.hr.copy(); hr4.columns = [c.strip() for c in hr4.columns]
                        t_col = next((c for c in hr4.columns if c.lower() in ("time","datetime","timestamp","date","activityminute")), None)
                        v_col = next((c for c in hr4.columns if c.lower()=="value"), None)
                        hr4["_dt"] = pd.to_datetime(hr4[t_col], errors="coerce"); hr4["_date"] = hr4["_dt"].dt.date
                        agg = hr4.groupby("_date")[v_col].mean().reset_index(); agg.columns = ["ds","y"]
                        agg["ds"] = pd.to_datetime(agg["ds"]); agg["y"] = pd.to_numeric(agg["y"], errors="coerce"); agg.dropna(inplace=True)
                        agg.sort_values("ds", inplace=True); agg = agg.tail(60).reset_index(drop=True)
                        model = Prophet(interval_width=0.90, daily_seasonality=False, weekly_seasonality=True, mcmc_samples=0)
                        model.fit(agg); future = model.make_future_dataframe(periods=30); forecast = model.predict(future)
                        fig, ax = plt.subplots(figsize=(11,4))
                        ax.scatter(agg["ds"], agg["y"], color=PALETTE[0], s=18, alpha=0.8, zorder=3, label="Actual HR")
                        ax.plot(forecast["ds"], forecast["yhat"], color=PALETTE[1], linewidth=2, label="Forecast")
                        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.18, color=PALETTE[1], label="90% CI")
                        ax.axvline(agg["ds"].max(), linestyle="--", color=PALETTE[3], alpha=0.6, linewidth=1)
                        ax.set_title("Heart Rate — 30-Day Prophet Forecast", fontsize=12); ax.legend(fontsize=9); fig.tight_layout()
                        st.session_state["prophet_fig"] = fig_to_bytes(fig); st.session_state.prophet_done = True
                        st.success("✅ Prophet forecast complete.")
                    except Exception as e:
                        st.error(f"❌ Prophet error: {e}")
            if st.session_state["prophet_fig"] is not None:
                st.image(st.session_state["prophet_fig"], use_container_width=True)

        if st.session_state.prophet_done:
            st.divider()
            step_hdr("ML‑3", "Clustering & Dimensionality Reduction")
            st.caption(f"KMeans (k={k_val}), DBSCAN (eps={eps_val}), PCA & t-SNE on Daily Activity features.")
            if st.button("▶  Run Clustering Pipeline"):
                with st.spinner("Running clustering…"):
                    try:
                        daily2 = st.session_state.daily.copy(); daily2.columns = [c.strip() for c in daily2.columns]
                        cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"]
                        available = [c for c in cluster_cols if c in daily2.columns]
                        id_col2 = next((c for c in daily2.columns if c.lower()=="id"), daily2.columns[0])
                        feat_df = daily2.groupby(id_col2)[available].mean().dropna()
                        scaler2 = StandardScaler(); X = scaler2.fit_transform(feat_df)
                        pca = PCA(n_components=2); X_pca = pca.fit_transform(X)
                        kmeans = KMeans(n_clusters=k_val, n_init=10, random_state=42); km_labels = kmeans.fit_predict(X)
                        fig, axes = plt.subplots(1,3,figsize=(15,4.5))
                        for c in np.unique(km_labels):
                            m=km_labels==c; axes[0].scatter(X_pca[m,0],X_pca[m,1],color=PALETTE[c%len(PALETTE)],s=60,alpha=0.85,edgecolors="none",label=f"Cluster {c}")
                        axes[0].set_title(f"KMeans (k={k_val}) — PCA",fontsize=11); axes[0].legend(fontsize=8)
                        db = DBSCAN(eps=eps_val, min_samples=2); db_labels = db.fit_predict(X)
                        for i, c in enumerate(np.unique(db_labels)):
                            m=db_labels==c; label="Noise" if c==-1 else f"Cluster {c}"; color="#6b7a96" if c==-1 else PALETTE[i%len(PALETTE)]
                            axes[1].scatter(X_pca[m,0],X_pca[m,1],color=color,s=60,alpha=0.85,edgecolors="none",label=label)
                        n_noise=(db_labels==-1).sum()
                        axes[1].set_title(f"DBSCAN (eps={eps_val}) — {len(np.unique(db_labels))-1} clusters, {n_noise} noise",fontsize=11); axes[1].legend(fontsize=8)
                        perp=min(30,max(5,len(X)-1)); tsne=TSNE(n_components=2,perplexity=perp,random_state=42,max_iter=1000); X_tsne=tsne.fit_transform(X)
                        for c in np.unique(km_labels):
                            m=km_labels==c; axes[2].scatter(X_tsne[m,0],X_tsne[m,1],color=PALETTE[c%len(PALETTE)],s=60,alpha=0.85,edgecolors="none",label=f"Cluster {c}")
                        axes[2].set_title("t-SNE (KMeans labels)",fontsize=11); axes[2].legend(fontsize=8)
                        for ax in axes: ax.spines[["top","right","left","bottom"]].set_visible(False)
                        fig.suptitle("Clustering Pipeline",fontsize=13,y=1.02,color="#63d7c4",fontweight="bold"); fig.tight_layout()
                        st.session_state["cluster_fig"] = fig_to_bytes(fig)
                        feat_df["Cluster"] = km_labels
                        st.session_state["cluster_summary"] = feat_df.groupby("Cluster")[available].mean().round(1)
                        st.session_state["X_scaled"]=X; st.session_state["km_labels"]=km_labels
                        st.session_state["db_labels"]=db_labels; st.session_state["k_val_used"]=k_val
                        st.session_state["available_cols"]=available; st.session_state["feat_df"]=feat_df.copy()
                        st.session_state.cluster_done=True; st.success("🎉 Clustering complete!")
                    except Exception as e:
                        st.error(f"❌ Clustering error: {e}")
            if st.session_state["cluster_fig"] is not None:
                st.image(st.session_state["cluster_fig"], use_container_width=True)
            if st.session_state.get("cluster_summary") is not None:
                step_hdr("ML‑4","Cluster Profiles")
                st.dataframe(st.session_state["cluster_summary"], use_container_width=True)

        if st.session_state.cluster_done:
            st.divider()
            step_hdr("ML‑7","KMeans Elbow Curve")
            _X = st.session_state.get("X_scaled")
            if _X is not None:
                if st.button("▶  Run Elbow Curve"):
                    with st.spinner("Computing inertia…"):
                        try:
                            inertias=[]; K_range=range(2,10)
                            for k in K_range:
                                km=KMeans(n_clusters=k,random_state=42,n_init=10); km.fit(_X); inertias.append(km.inertia_)
                            fig,ax=plt.subplots(figsize=(9,4))
                            ax.plot(list(K_range),inertias,"o-",color="#63d7c4",linewidth=2.5,markersize=9,markerfacecolor="#f97316")
                            ax.set_title("KMeans Elbow Curve",fontsize=13); ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.set_xticks(list(K_range))
                            ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
                            st.session_state["elbow_fig"]=fig_to_bytes(fig); st.success("✅ Elbow curve complete!")
                        except Exception as e:
                            st.error(f"❌ {e}")
                if st.session_state.get("elbow_fig") is not None:
                    st.image(st.session_state["elbow_fig"], use_container_width=True)

            st.divider()
            step_hdr("ML‑8","Steps & Sleep Prophet Forecasts")
            if st.button("▶  Run Steps & Sleep Forecast"):
                with st.spinner("Fitting Prophet for Steps and Sleep…"):
                    try:
                        daily3=st.session_state.daily.copy(); sleep3=st.session_state.sleep.copy()
                        daily3.columns=[c.strip() for c in daily3.columns]; sleep3.columns=[c.strip() for c in sleep3.columns]
                        d_date=next((c for c in daily3.columns if "date" in c.lower()),None)
                        sl_d=next((c for c in sleep3.columns if "date" in c.lower()),None); sl_v=next((c for c in sleep3.columns if c.lower()=="value"),None)
                        configs=[]
                        if d_date and "TotalSteps" in daily3.columns:
                            configs.append(("TotalSteps",d_date,daily3,"#63d7c4","Steps"))
                        if sl_d and sl_v:
                            sleep3["_dt"]=pd.to_datetime(sleep3[sl_d],errors="coerce"); sleep3["_date"]=sleep3["_dt"].dt.date.astype(str)
                            sleep3[sl_v]=pd.to_numeric(sleep3[sl_v],errors="coerce")
                            sl_daily=sleep3.groupby("_date").size().reset_index(name="SleepMinutes"); sl_daily.columns=["_date","SleepMinutes"]
                            configs.append(("SleepMinutes","_date",sl_daily,"#818cf8","Sleep (minutes)"))
                        if configs:
                            fig,axes=plt.subplots(len(configs),1,figsize=(14,5*len(configs)))
                            if len(configs)==1: axes=[axes]
                            for ax,(metric,date_col,df_src,color,label) in zip(axes,configs):
                                agg=df_src.groupby(date_col)[metric].mean().reset_index(); agg.columns=["ds","y"]
                                agg["ds"]=pd.to_datetime(agg["ds"],errors="coerce"); agg["y"]=pd.to_numeric(agg["y"],errors="coerce")
                                agg=agg.dropna().sort_values("ds").tail(60)
                                if len(agg)<2: ax.text(0.5,0.5,f"Not enough data",ha="center",va="center",transform=ax.transAxes,color="#f87171",fontsize=11); continue
                                m=Prophet(weekly_seasonality=True,yearly_seasonality=False,daily_seasonality=False,interval_width=0.80,changepoint_prior_scale=0.1,mcmc_samples=0)
                                m.fit(agg); future=m.make_future_dataframe(periods=30); forecast=m.predict(future)
                                ax.scatter(agg["ds"],agg["y"],color=color,s=20,alpha=0.75,label=f"Actual {label}",zorder=3)
                                ax.plot(forecast["ds"],forecast["yhat"],color="#e2eaf4",linewidth=2,label="Trend")
                                ax.fill_between(forecast["ds"],forecast["yhat_lower"],forecast["yhat_upper"],alpha=0.22,color=color,label="80% CI")
                                ax.axvline(agg["ds"].max(),color="#fbbf24",linestyle="--",linewidth=1.5,label="Forecast Start")
                                ax.set_title(f"{label} — Prophet Trend Forecast",fontsize=13); ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
                            fig.tight_layout(); st.session_state["steps_sleep_fig"]=fig_to_bytes(fig); st.success("✅ Steps & Sleep forecasts complete!")
                    except Exception as e:
                        st.error(f"❌ {e}")
            if st.session_state.get("steps_sleep_fig") is not None:
                st.image(st.session_state["steps_sleep_fig"], use_container_width=True)

        if st.session_state.cluster_done:
            st.balloons()
            st.markdown('<div class="fp-card" style="border-color:rgba(99,215,196,0.4);text-align:center;padding:2rem;margin-top:1rem"><div style="font-size:2rem">🎉</div><h2 style="color:#63d7c4">Pipeline Complete</h2><p style="color:#6b7a96">TSFresh → Prophet → KMeans → DBSCAN → t-SNE → Elbow → Steps/Sleep</p></div>', unsafe_allow_html=True)

# =============================================================================
# ANOMALY DETECTOR SECTION  (M3 - unchanged)
# =============================================================================
elif active == "anomaly_detector":

    st.markdown(f"""
    <div class="m3-hero">
      <div class="hero-badge">MILESTONE 3 · ANOMALY DETECTION & VISUALIZATION</div>
      <h1 style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:#e2e8f0;margin:0 0 0.4rem 0'>🚨 FitPulse Anomaly Detector</h1>
      <p style='font-size:1.05rem;color:#94a3b8;font-weight:300;margin:0'>Threshold Violations · Residual Analysis · Outlier Clusters · Interactive Charts</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.shared_loaded:
        st.markdown(f"""
        <div class="m3-card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">📁</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem">Upload & Load Files First</div>
          <div style="color:#94a3b8;font-size:0.88rem">Use the <b style="color:#63b3ed">sidebar file uploader</b> on the left — upload all 5 Fitbit CSV files and click <b>⚡ Load & Build Master</b>.</div>
        </div>""", unsafe_allow_html=True)
    else:
        master = st.session_state.shared_master
        m3_success(f"Master DataFrame ready — {master.shape[0]} rows · {master['Id'].nunique()} users")

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
        m3_sec("🚨", "Anomaly Detection — Three Methods", "Steps 2–4")

        st.markdown(f"""
        <div class="m3-card">
          <div class="m3-card-title">Detection Methods Applied</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.83rem">
            <div style="background:{M3_SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{M3_RED};font-weight:600;margin-bottom:0.4rem">① Threshold Violations</div>
              <div style="color:{M3_MUTED}">Hard upper/lower limits on HR, Steps, Sleep.</div>
            </div>
            <div style="background:{M3_SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{M3_ACC2};font-weight:600;margin-bottom:0.4rem">② Residual-Based</div>
              <div style="color:{M3_MUTED}">Rolling median baseline. Flag ±{hr_high if active=='anomaly_detector' else 2:.0f}σ deviations.</div>
            </div>
            <div style="background:{M3_SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{M3_ACC3};font-weight:600;margin-bottom:0.4rem">③ DBSCAN Outliers</div>
              <div style="color:{M3_MUTED}">Users labelled −1 are structural outliers.</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("🔍 Run Anomaly Detection (All 3 Methods)", key="m3_detect"):
            with st.spinner("Detecting anomalies..."):
                try:
                    anom_hr_m3    = detect_hr_anomalies(master,    hr_high, hr_low,   sigma)
                    anom_steps_m3 = detect_steps_anomalies(master,  st_low,  25000,   sigma)
                    anom_sleep_m3 = detect_sleep_anomalies(master,  sl_low,  sl_high, sigma)
                    st.session_state.m3_anom_hr    = anom_hr_m3
                    st.session_state.m3_anom_steps = anom_steps_m3
                    st.session_state.m3_anom_sleep = anom_sleep_m3
                    st.session_state.m3_anomaly_done = True; st.rerun()
                except Exception as e:
                    st.error(f"Detection error: {e}")

        if st.session_state.m3_anomaly_done:
            anom_hr    = st.session_state.m3_anom_hr
            anom_steps = st.session_state.m3_anom_steps
            anom_sleep = st.session_state.m3_anom_sleep

            n_hr=int(anom_hr["is_anomaly"].sum()); n_steps=int(anom_steps["is_anomaly"].sum())
            n_sleep=int(anom_sleep["is_anomaly"].sum()); n_total=n_hr+n_steps+n_sleep

            m3_danger(f"Total anomalies flagged: {n_total}  (HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})")
            m3_metrics((n_hr,"HR Anomalies"),(n_steps,"Steps Anomalies"),(n_sleep,"Sleep Anomalies"),(n_total,"Total Flags"),red_indices=[0,1,2,3])

            # HR Chart
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("❤️","Heart Rate — Anomaly Chart","Step 2")
            hr_anom = anom_hr[anom_hr["is_anomaly"]]
            fig_hr_m3 = go.Figure()
            rolling_upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
            rolling_lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
            fig_hr_m3.add_trace(go.Scatter(x=anom_hr["Date"],y=rolling_upper,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig_hr_m3.add_trace(go.Scatter(x=anom_hr["Date"],y=rolling_lower,mode="lines",fill="tonexty",fillcolor="rgba(99,179,237,0.1)",line=dict(width=0),name=f"±{sigma:.0f}σ Band"))
            fig_hr_m3.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["AvgHR"],mode="lines+markers",name="Avg Heart Rate",line=dict(color=M3_ACCENT,width=2.5),marker=dict(size=5,color=M3_ACCENT),hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
            fig_hr_m3.add_trace(go.Scatter(x=anom_hr["Date"],y=anom_hr["rolling_med"],mode="lines",name="Rolling Median",line=dict(color=M3_ACC3,width=1.5,dash="dot")))
            if not hr_anom.empty:
                fig_hr_m3.add_trace(go.Scatter(x=hr_anom["Date"],y=hr_anom["AvgHR"],mode="markers",name="🚨 Anomaly",
                    marker=dict(color=M3_RED,size=14,symbol="circle",line=dict(color="white",width=2)),
                    hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<br><b>ANOMALY</b><extra>⚠️</extra>"))
            fig_hr_m3.add_hline(y=hr_high,line_dash="dash",line_color=M3_RED,line_width=1.5,opacity=0.7,
                                annotation_text=f"High ({hr_high} bpm)",annotation_position="top right",annotation_font_color=M3_RED)
            fig_hr_m3.add_hline(y=hr_low, line_dash="dash",line_color=M3_ACC2,line_width=1.5,opacity=0.7,
                                annotation_text=f"Low ({hr_low} bpm)",annotation_position="bottom right",annotation_font_color=M3_ACC2)
            apply_plotly_theme(fig_hr_m3,"❤️ Heart Rate — Anomaly Detection (Real Fitbit Data)")
            fig_hr_m3.update_layout(height=480,xaxis_title="Date",yaxis_title="Heart Rate (bpm)")
            st.plotly_chart(fig_hr_m3, use_container_width=True)
            if not hr_anom.empty:
                with st.expander(f"📋 {len(hr_anom)} HR Anomaly Records"):
                    st.dataframe(hr_anom[["Date","AvgHR","rolling_med","residual","reason"]].rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}).round(2), use_container_width=True)

            # Sleep Chart
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("💤","Sleep Pattern — Anomaly Visualization","Step 3")
            sleep_anom = anom_sleep[anom_sleep["is_anomaly"]]
            fig_sleep_m3 = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.7,0.3],
                                         subplot_titles=["Sleep Duration (minutes/night)","Deviation from Expected"],vertical_spacing=0.08)
            fig_sleep_m3.add_hrect(y0=sl_low,y1=sl_high,fillcolor="rgba(104,211,145,0.08)",line_width=0,
                                   annotation_text="✅ Healthy Sleep Zone",annotation_position="top right",annotation_font_color=M3_ACC3,row=1,col=1)
            fig_sleep_m3.add_trace(go.Scatter(x=anom_sleep["Date"],y=anom_sleep["TotalSleepMinutes"],mode="lines+markers",name="Sleep Minutes",line=dict(color="#b794f4",width=2.5),marker=dict(size=5,color="#b794f4")),row=1,col=1)
            fig_sleep_m3.add_trace(go.Scatter(x=anom_sleep["Date"],y=anom_sleep["rolling_med"],mode="lines",name="Rolling Median",line=dict(color=M3_ACC3,width=1.5,dash="dot")),row=1,col=1)
            if not sleep_anom.empty:
                fig_sleep_m3.add_trace(go.Scatter(x=sleep_anom["Date"],y=sleep_anom["TotalSleepMinutes"],mode="markers",name="🚨 Sleep Anomaly",marker=dict(color=M3_RED,size=14,symbol="diamond",line=dict(color="white",width=2))),row=1,col=1)
            fig_sleep_m3.add_hline(y=sl_low,line_dash="dash",line_color=M3_RED,line_width=1.5,opacity=0.7,row=1,col=1,annotation_text=f"Min ({sl_low} min)",annotation_font_color=M3_RED)
            fig_sleep_m3.add_hline(y=sl_high,line_dash="dash",line_color=M3_ACCENT,line_width=1.5,opacity=0.7,row=1,col=1,annotation_text=f"Max ({sl_high} min)",annotation_font_color=M3_ACCENT)
            colors_resid=[M3_RED if v else M3_ACCENT for v in anom_sleep["resid_anom"]]
            fig_sleep_m3.add_trace(go.Bar(x=anom_sleep["Date"],y=anom_sleep["residual"],name="Residual",marker_color=colors_resid),row=2,col=1)
            fig_sleep_m3.add_hline(y=0,line_dash="solid",line_color=M3_MUTED,line_width=1,row=2,col=1)
            apply_plotly_theme(fig_sleep_m3)
            fig_sleep_m3.update_layout(height=560,paper_bgcolor=M3_BG,plot_bgcolor="#0f172a",font_color=M3_TEXT)
            fig_sleep_m3.update_xaxes(gridcolor=M3_GRID,tickfont_color=M3_MUTED)
            fig_sleep_m3.update_yaxes(gridcolor=M3_GRID,tickfont_color=M3_MUTED)
            st.plotly_chart(fig_sleep_m3, use_container_width=True)
            if not sleep_anom.empty:
                with st.expander(f"📋 {len(sleep_anom)} Sleep Anomaly Records"):
                    st.dataframe(sleep_anom[["Date","TotalSleepMinutes","rolling_med","residual","reason"]].rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}).round(2), use_container_width=True)

            # Steps Chart
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("🚶","Step Count Trend — Alerts & Anomalies","Step 4")
            steps_anom = anom_steps[anom_steps["is_anomaly"]]
            fig_steps_m3 = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],
                                         subplot_titles=["Daily Steps (avg across users)","Residual Deviation from Trend"],vertical_spacing=0.08)
            for _,row in steps_anom.iterrows():
                d=str(row["Date"]); d_next=str(pd.Timestamp(d)+pd.Timedelta(days=1))[:10]
                fig_steps_m3.add_vrect(x0=d,x1=d_next,fillcolor="rgba(252,129,129,0.15)",line_color="rgba(252,129,129,0.5)",line_width=1.5,row=1,col=1)
            fig_steps_m3.add_trace(go.Scatter(x=anom_steps["Date"],y=anom_steps["TotalSteps"],mode="lines+markers",name="Avg Daily Steps",line=dict(color=M3_ACC3,width=2.5),marker=dict(size=5,color=M3_ACC3)),row=1,col=1)
            fig_steps_m3.add_trace(go.Scatter(x=anom_steps["Date"],y=anom_steps["rolling_med"],mode="lines",name="Trend (Rolling Median)",line=dict(color=M3_ACCENT,width=2,dash="dash")),row=1,col=1)
            if not steps_anom.empty:
                fig_steps_m3.add_trace(go.Scatter(x=steps_anom["Date"],y=steps_anom["TotalSteps"],mode="markers",name="🚨 Steps Anomaly",marker=dict(color=M3_RED,size=14,symbol="triangle-up",line=dict(color="white",width=2))),row=1,col=1)
            fig_steps_m3.add_hline(y=st_low,line_dash="dash",line_color=M3_RED,line_width=1.5,opacity=0.8,row=1,col=1,annotation_text=f"Low ({st_low:,} steps)",annotation_font_color=M3_RED)
            fig_steps_m3.add_hline(y=25000,line_dash="dash",line_color=M3_ACC2,line_width=1.5,opacity=0.7,row=1,col=1,annotation_text="High Alert (25,000)",annotation_font_color=M3_ACC2)
            res_colors=[M3_RED if v else M3_ACC3 for v in anom_steps["resid_anom"]]
            fig_steps_m3.add_trace(go.Bar(x=anom_steps["Date"],y=anom_steps["residual"],name="Residual",marker_color=res_colors),row=2,col=1)
            fig_steps_m3.add_hline(y=0,line_dash="solid",line_color=M3_MUTED,line_width=1,row=2,col=1)
            apply_plotly_theme(fig_steps_m3)
            fig_steps_m3.update_layout(height=560,paper_bgcolor=M3_BG,plot_bgcolor="#0f172a",font_color=M3_TEXT)
            fig_steps_m3.update_xaxes(gridcolor=M3_GRID,tickfont_color=M3_MUTED)
            fig_steps_m3.update_yaxes(gridcolor=M3_GRID,tickfont_color=M3_MUTED)
            st.plotly_chart(fig_steps_m3, use_container_width=True)
            if not steps_anom.empty:
                with st.expander(f"📋 {len(steps_anom)} Steps Anomaly Records"):
                    st.dataframe(steps_anom[["Date","TotalSteps","rolling_med","residual","reason"]].rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}).round(2), use_container_width=True)

            # DBSCAN
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("🔍","DBSCAN Outlier Users — Cluster-Based Anomalies","Step 5")
            cluster_cols=["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
            try:
                cf=master.groupby("Id")[cluster_cols].mean().round(3).dropna()
                scaler=StandardScaler(); X_scaled=scaler.fit_transform(cf)
                db=DBSCAN(eps=2.2,min_samples=2); db_labels=db.fit_predict(X_scaled)
                pca=PCA(n_components=2,random_state=42); X_pca=pca.fit_transform(X_scaled)
                var=pca.explained_variance_ratio_*100
                cf["DBSCAN"]=db_labels
                outlier_users=cf[cf["DBSCAN"]==-1].index.tolist()
                n_outliers=len(outlier_users); n_clusters=len(set(db_labels))-(1 if -1 in db_labels else 0)
                m3_metrics((n_clusters,"DBSCAN Clusters"),(n_outliers,"Outlier Users"),(len(cf)-n_outliers,"Normal Users"),red_indices=[1])
                CLUSTER_COLORS=[M3_ACCENT,M3_ACC3,"#f6ad55","#b794f4",M3_ACC2]
                fig_db=go.Figure()
                for lbl in sorted(set(db_labels)):
                    if lbl==-1: continue
                    mask=db_labels==lbl
                    fig_db.add_trace(go.Scatter(x=X_pca[mask,0],y=X_pca[mask,1],mode="markers+text",name=f"Cluster {lbl}",
                        marker=dict(size=14,color=CLUSTER_COLORS[lbl%len(CLUSTER_COLORS)],opacity=0.85,line=dict(color="white",width=1.5)),
                        text=[str(uid)[-4:] for uid in cf.index[mask]],textposition="top center",textfont=dict(size=8,color=M3_TEXT)))
                if n_outliers>0:
                    mask_out=db_labels==-1
                    fig_db.add_trace(go.Scatter(x=X_pca[mask_out,0],y=X_pca[mask_out,1],mode="markers+text",name="🚨 Outlier",
                        marker=dict(size=20,color=M3_RED,symbol="x",line=dict(color="white",width=2.5)),
                        text=[str(uid)[-4:] for uid in cf.index[mask_out]],textposition="top center",textfont=dict(size=9,color=M3_RED)))
                apply_plotly_theme(fig_db,"🔍 DBSCAN Outlier Detection — PCA Projection (eps=2.2)")
                fig_db.update_layout(height=500,xaxis_title=f"PC1 ({var[0]:.1f}% variance)",yaxis_title=f"PC2 ({var[1]:.1f}% variance)")
                st.plotly_chart(fig_db, use_container_width=True)
                if outlier_users:
                    st.markdown(f'<div class="m3-card" style="border-color:{M3_DANGER_BOR}"><div class="m3-card-title" style="color:{M3_RED}">🚨 Outlier User Profiles</div></div>', unsafe_allow_html=True)
                    st.dataframe(cf[cf["DBSCAN"]==-1][cluster_cols].round(2), use_container_width=True)
            except Exception as e:
                m3_warn(f"DBSCAN clustering skipped: {e}")

            # Accuracy simulation
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("🎯","Simulated Detection Accuracy — 90%+ Target","Step 6")
            m3_info("10 known anomalies are injected into each signal. We measure how many the detector catches.")
            if st.button("🎯 Run Accuracy Simulation", key="m3_sim"):
                with st.spinner("Simulating..."):
                    try:
                        sim=simulate_accuracy(master,n_inject=10)
                        st.session_state.m3_sim_results=sim; st.session_state.m3_simulation_done=True; st.rerun()
                    except Exception as e:
                        st.error(f"Simulation error: {e}")
            if st.session_state.m3_simulation_done and st.session_state.m3_sim_results:
                sim=st.session_state.m3_sim_results; overall=sim["Overall"]; passed=overall>=90.0
                if passed: m3_success(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
                else:      m3_warn(f"Overall accuracy: {overall}% — below 90% target")
                html='<div class="m3-metric-grid">'
                for signal in ["Heart Rate","Steps","Sleep"]:
                    r=sim[signal]; acc=r["accuracy"]; col=M3_ACC3 if acc>=90 else M3_RED
                    html+=f'<div class="m3-metric-card" style="border-color:{col}44"><div style="font-size:1.8rem;font-weight:800;color:{col};font-family:Syne,sans-serif">{acc}%</div><div style="font-size:0.8rem;color:{M3_TEXT};font-weight:600;margin:0.3rem 0">{signal}</div><div style="font-size:0.72rem;color:{M3_MUTED}">{r["detected"]}/{r["injected"]} detected</div></div>'
                html+=f'<div class="m3-metric-card"><div style="font-size:1.8rem;font-weight:800;color:{"#68d391" if passed else M3_RED};font-family:Syne,sans-serif">{overall}%</div><div style="font-size:0.8rem;color:{M3_TEXT};font-weight:600;margin:0.3rem 0">Overall</div></div>'
                html+='</div>'
                st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# REPORT SECTION  →  REPLACED WITH M4 DASHBOARD
# =============================================================================
elif active == "report":

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="m4-hero">
      <div class="m4-hero-badge">MILESTONE 4 · INSIGHTS DASHBOARD</div>
      <h1 class="m4-hero-title">📊 FitPulse Insights Dashboard</h1>
      <p class="m4-hero-sub">Upload · Detect · Filter · Export PDF & CSV — Real Fitbit Device Data</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.shared_loaded:
        st.markdown(f"""
        <div class="m4-card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">📂</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e2e8f0;margin-bottom:0.5rem">
            Upload Files & Load Pipeline to Begin
          </div>
          <div style="color:#94a3b8;font-size:0.88rem;margin-bottom:1.5rem">
            1 · Upload all 5 CSV files in the sidebar<br>
            2 · Click <b>⚡ Load & Build Master</b><br>
            3 · Come back here and click <b>⚡ Run Detection & Dashboard</b>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;max-width:600px;margin:0 auto;text-align:left">
            <div style="background:{M4_SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{M4_ACCENT};font-weight:600;font-size:0.85rem">📤 Upload</div>
              <div style="color:{M4_MUTED};font-size:0.75rem;margin-top:0.2rem">All 5 Fitbit CSV files auto-detected</div>
            </div>
            <div style="background:{M4_SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{M4_RED};font-weight:600;font-size:0.85rem">🚨 Detect</div>
              <div style="color:{M4_MUTED};font-size:0.75rem;margin-top:0.2rem">3 detection methods run automatically</div>
            </div>
            <div style="background:{M4_SECTION_BG};border-radius:10px;padding:0.8rem">
              <div style="color:{M4_ACC3};font-weight:600;font-size:0.85rem">📥 Export</div>
              <div style="color:{M4_MUTED};font-size:0.75rem;margin-top:0.2rem">Download PDF report + CSV data</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        master = st.session_state.shared_master

        # Run detection when button clicked
        if run_report:
            with st.spinner("⏳ Running anomaly detection..."):
                try:
                    anom_hr_r    = detect_hr_anomalies(master,    hr_high, hr_low,   sigma)
                    anom_steps_r = detect_steps_anomalies(master,  st_low,  25000,   sigma)
                    anom_sleep_r = detect_sleep_anomalies(master,  sl_low,  sl_high, sigma)
                    st.session_state.m3_anom_hr    = anom_hr_r
                    st.session_state.m3_anom_steps = anom_steps_r
                    st.session_state.m3_anom_sleep = anom_sleep_r
                    st.session_state.m3_anomaly_done = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection error: {e}")

        if not st.session_state.m3_anomaly_done:
            m4_info("Click <b>⚡ Run Detection & Dashboard</b> in the sidebar to start.")
        else:
            anom_hr    = st.session_state.m3_anom_hr
            anom_steps = st.session_state.m3_anom_steps
            anom_sleep = st.session_state.m3_anom_sleep

            # ── Date + User filter ────────────────────────────────────────────
            try:
                if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
                    d_from = pd.Timestamp(date_range[0])
                    d_to   = pd.Timestamp(date_range[1])
                else:
                    all_dates = pd.to_datetime(master["Date"])
                    d_from, d_to = all_dates.min(), all_dates.max()
            except Exception:
                all_dates = pd.to_datetime(master["Date"])
                d_from, d_to = all_dates.min(), all_dates.max()

            def filt(df, date_col="Date"):
                df2 = df.copy()
                df2[date_col] = pd.to_datetime(df2[date_col])
                return df2[(df2[date_col] >= d_from) & (df2[date_col] <= d_to)]

            anom_hr_f    = filt(anom_hr)
            anom_steps_f = filt(anom_steps)
            anom_sleep_f = filt(anom_sleep)
            master_f     = filt(master)
            if selected_user:
                master_f = master_f[master_f["Id"] == selected_user]

            # ── KPI strip ─────────────────────────────────────────────────────
            n_hr_f    = int(anom_hr_f["is_anomaly"].sum())
            n_steps_f = int(anom_steps_f["is_anomaly"].sum())
            n_sleep_f = int(anom_sleep_f["is_anomaly"].sum())
            n_total_f = n_hr_f + n_steps_f + n_sleep_f
            n_users_f = master_f["Id"].nunique()
            n_days_f  = master_f["Date"].nunique()
            worst_hr_row = anom_hr_f[anom_hr_f["is_anomaly"]].copy()
            worst_hr_day = worst_hr_row.iloc[worst_hr_row["residual"].abs().argmax()]["Date"].strftime("%d %b") if not worst_hr_row.empty else "-"

            st.markdown(f"""
            <div class="kpi-grid">
              <div class="kpi-card" style="border-color:{M4_DANGER_BOR}">
                <div class="kpi-val" style="color:{M4_RED}">{n_total_f}</div>
                <div class="kpi-label">Total Anomalies</div>
                <div class="kpi-sub">across all signals</div>
              </div>
              <div class="kpi-card" style="border-color:rgba(246,135,179,0.3)">
                <div class="kpi-val" style="color:{M4_ACC2}">{n_hr_f}</div>
                <div class="kpi-label">HR Flags</div>
                <div class="kpi-sub">heart rate anomalies</div>
              </div>
              <div class="kpi-card" style="border-color:rgba(104,211,145,0.3)">
                <div class="kpi-val" style="color:{M4_ACC3}">{n_steps_f}</div>
                <div class="kpi-label">Steps Alerts</div>
                <div class="kpi-sub">step count anomalies</div>
              </div>
              <div class="kpi-card" style="border-color:rgba(183,148,244,0.3)">
                <div class="kpi-val" style="color:{M4_PURPLE}">{n_sleep_f}</div>
                <div class="kpi-label">Sleep Flags</div>
                <div class="kpi-sub">sleep anomalies</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-val" style="color:{M4_ACCENT}">{n_users_f}</div>
                <div class="kpi-label">Users</div>
                <div class="kpi-sub">in selected range</div>
              </div>
              <div class="kpi-card">
                <div class="kpi-val" style="color:{M4_ORG}">{worst_hr_day}</div>
                <div class="kpi-label">Peak HR Anomaly</div>
                <div class="kpi-sub">highest deviation day</div>
              </div>
            </div>""", unsafe_allow_html=True)

            m4_success(f"Pipeline complete · {n_users_f} users · {n_days_f} days · {n_total_f} anomalies flagged")

            # ── Tabs ──────────────────────────────────────────────────────────
            tab_overview, tab_hr, tab_steps, tab_sleep, tab_export = st.tabs([
                "📊 Overview", "❤️ Heart Rate", "🚶 Steps", "💤 Sleep", "📥 Export"
            ])

            # TAB 1: OVERVIEW
            with tab_overview:
                st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
                m4_sec("📅", "Combined Anomaly Timeline")

                all_anoms = []
                for df_, sig, col in [
                    (anom_hr_f,    "Heart Rate", M4_ACC2),
                    (anom_steps_f, "Steps",      M4_ACC3),
                    (anom_sleep_f, "Sleep",      M4_PURPLE),
                ]:
                    a = df_[df_["is_anomaly"]].copy()
                    a["signal"] = sig; a["color"] = col
                    val_col = "AvgHR" if sig == "Heart Rate" else ("TotalSteps" if sig == "Steps" else "TotalSleepMinutes")
                    a_cols = ["Date","signal","color","reason"]
                    all_anoms.append(a[a_cols])

                if all_anoms:
                    combined = pd.concat(all_anoms, ignore_index=True)
                    combined["Date"] = pd.to_datetime(combined["Date"])
                    combined["y"]    = combined["signal"]
                    fig_timeline = go.Figure()
                    for sig, col in [("Heart Rate", M4_ACC2), ("Steps", M4_ACC3), ("Sleep", M4_PURPLE)]:
                        sub = combined[combined["signal"] == sig]
                        if not sub.empty:
                            fig_timeline.add_trace(go.Scatter(
                                x=sub["Date"], y=sub["y"], mode="markers", name=sig,
                                marker=dict(color=col, size=14, symbol="diamond",
                                            line=dict(color="white", width=2)),
                                hovertemplate=f"<b>{sig}</b><br>%{{x|%d %b %Y}}<br>%{{customdata}}<extra>⚠️ ANOMALY</extra>",
                                customdata=sub["reason"].values
                            ))
                    m4_ptheme(fig_timeline, "📅 Anomaly Event Timeline - All Signals", h=280)
                    fig_timeline.update_layout(
                        xaxis_title="Date", yaxis_title="Signal", showlegend=True,
                        yaxis=dict(categoryorder="array",
                                   categoryarray=["Sleep","Steps","Heart Rate"],
                                   gridcolor=M4_GRID, tickfont_color=M4_MUTED)
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

                st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
                m4_sec("🗂️", "Recent Anomaly Log")
                if all_anoms:
                    log = combined.sort_values("Date", ascending=False).head(10)
                    for _, row in log.iterrows():
                        st.markdown(f"""
                        <div class="anom-row">
                          <span style="font-size:0.9rem">🚨</span>
                          <span style="color:{row['color']};font-family:'JetBrains Mono',monospace;font-size:0.75rem;min-width:90px">{row['signal']}</span>
                          <span style="color:{M4_MUTED};font-size:0.78rem;min-width:90px">{row['Date'].strftime('%d %b %Y')}</span>
                          <span style="color:{M4_TEXT};font-size:0.78rem">{row['reason']}</span>
                        </div>""", unsafe_allow_html=True)

            # TAB 2: HEART RATE
            with tab_hr:
                m4_sec("❤️", "Heart Rate - Deep Dive", f"{n_hr_f} anomalies")
                fig_hr_chart = chart_hr(anom_hr_f, hr_high, hr_low, sigma)
                st.plotly_chart(fig_hr_chart, use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="m4-card">
                      <div class="m4-card-title">HR Statistics</div>
                      <div style="font-size:0.83rem;line-height:2">
                        <div>Mean HR: <b style="color:{M4_ACCENT}">{anom_hr_f['AvgHR'].mean():.1f} bpm</b></div>
                        <div>Max HR: <b style="color:{M4_RED}">{anom_hr_f['AvgHR'].max():.1f} bpm</b></div>
                        <div>Min HR: <b style="color:{M4_ACC2}">{anom_hr_f['AvgHR'].min():.1f} bpm</b></div>
                        <div>Anomaly days: <b style="color:{M4_RED}">{n_hr_f}</b> of {len(anom_hr_f)} total</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    hr_display = anom_hr_f[anom_hr_f["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].round(2)
                    if not hr_display.empty:
                        st.dataframe(hr_display.rename(columns={"AvgHR":"Avg HR","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                     use_container_width=True, height=220)
                    else:
                        m4_success("No HR anomalies in selected range")

            # TAB 3: STEPS
            with tab_steps:
                m4_sec("🚶", "Step Count - Deep Dive", f"{n_steps_f} alerts")
                fig_steps_chart = chart_steps(anom_steps_f, st_low)
                st.plotly_chart(fig_steps_chart, use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="m4-card">
                      <div class="m4-card-title">Steps Statistics</div>
                      <div style="font-size:0.83rem;line-height:2">
                        <div>Mean steps/day: <b style="color:{M4_ACC3}">{anom_steps_f['TotalSteps'].mean():,.0f}</b></div>
                        <div>Max steps/day: <b style="color:{M4_ACCENT}">{anom_steps_f['TotalSteps'].max():,.0f}</b></div>
                        <div>Min steps/day: <b style="color:{M4_RED}">{anom_steps_f['TotalSteps'].min():,.0f}</b></div>
                        <div>Alert days: <b style="color:{M4_RED}">{n_steps_f}</b> of {len(anom_steps_f)} total</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    st_display = anom_steps_f[anom_steps_f["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].round(2)
                    if not st_display.empty:
                        st.dataframe(st_display.rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                     use_container_width=True, height=220)
                    else:
                        m4_success("No step anomalies in selected range")

            # TAB 4: SLEEP
            with tab_sleep:
                m4_sec("💤", "Sleep Pattern - Deep Dive", f"{n_sleep_f} anomalies")
                fig_sleep_chart = chart_sleep(anom_sleep_f, sl_low, sl_high)
                st.plotly_chart(fig_sleep_chart, use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    non_zero = anom_sleep_f[anom_sleep_f["TotalSleepMinutes"] > 0]["TotalSleepMinutes"]
                    st.markdown(f"""
                    <div class="m4-card">
                      <div class="m4-card-title">Sleep Statistics</div>
                      <div style="font-size:0.83rem;line-height:2">
                        <div>Mean sleep/night: <b style="color:{M4_PURPLE}">{anom_sleep_f['TotalSleepMinutes'].mean():.0f} min</b></div>
                        <div>Max sleep/night: <b style="color:{M4_ACCENT}">{anom_sleep_f['TotalSleepMinutes'].max():.0f} min</b></div>
                        <div>Min (non-zero): <b style="color:{M4_RED}">{non_zero.min():.0f} min</b></div>
                        <div>Anomaly days: <b style="color:{M4_RED}">{n_sleep_f}</b> of {len(anom_sleep_f)} total</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    sl_display = anom_sleep_f[anom_sleep_f["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].round(2)
                    if not sl_display.empty:
                        st.dataframe(sl_display.rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Reason"}),
                                     use_container_width=True, height=220)
                    else:
                        m4_success("No sleep anomalies in selected range")

            # TAB 5: EXPORT
            with tab_export:
                m4_sec("📥", "Export — PDF Report & CSV Data", "Downloadable")

                st.markdown(f"""
                <div class="m4-card">
                  <div class="m4-card-title">What's Included in the Exports</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;font-size:0.83rem">
                    <div style="background:{M4_SECTION_BG};border-radius:10px;padding:0.9rem">
                      <div style="color:{M4_ACCENT};font-weight:600;margin-bottom:0.5rem">📄 PDF Report (4 pages)</div>
                      <div style="color:{M4_MUTED};line-height:1.8">
                        ✅ Executive summary<br>
                        ✅ Anomaly counts per signal<br>
                        ✅ Thresholds used<br>
                        ✅ Methodology explanation<br>
                        ✅ Charts embedded (if kaleido)<br>
                        ✅ Full anomaly records tables<br>
                        ✅ User activity profiles
                      </div>
                    </div>
                    <div style="background:{M4_SECTION_BG};border-radius:10px;padding:0.9rem">
                      <div style="color:{M4_ACC3};font-weight:600;margin-bottom:0.5rem">📊 CSV Export</div>
                      <div style="color:{M4_MUTED};line-height:1.8">
                        ✅ All anomaly records<br>
                        ✅ Signal type column<br>
                        ✅ Date of anomaly<br>
                        ✅ Actual vs expected value<br>
                        ✅ Residual deviation<br>
                        ✅ Anomaly reason text<br>
                        ✅ All signals combined
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                try:
                    import kaleido
                    m4_success("kaleido is installed — charts will be embedded as images in the PDF.")
                except ImportError:
                    m4_info("kaleido not installed — charts will show as placeholders. Run <b>pip install kaleido</b> to embed chart images.")

                st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
                col_pdf, col_csv = st.columns(2)

                with col_pdf:
                    m4_sec("📄", "PDF Report")
                    st.markdown(f'<div style="color:{M4_MUTED};font-size:0.82rem;margin-bottom:0.8rem">Full 4-page PDF with anomaly tables, user profiles, and charts.</div>', unsafe_allow_html=True)
                    if st.button("📄 Generate PDF Report", key="gen_pdf_m4"):
                        with st.spinner("⏳ Generating PDF..."):
                            try:
                                fig_hr_exp    = chart_hr(anom_hr_f,    hr_high, hr_low, sigma, h=420)
                                fig_steps_exp = chart_steps(anom_steps_f, st_low, h=420)
                                fig_sleep_exp = chart_sleep(anom_sleep_f, sl_low, sl_high, h=420)
                                pdf_buf = generate_pdf(
                                    master_f, anom_hr_f, anom_steps_f, anom_sleep_f,
                                    hr_high, hr_low, st_low, sl_low, sl_high, sigma,
                                    fig_hr_exp, fig_steps_exp, fig_sleep_exp
                                )
                                fname = f"FitPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                                st.download_button(
                                    label="⬇️ Download PDF Report",
                                    data=pdf_buf,
                                    file_name=fname,
                                    mime="application/pdf",
                                    key="dl_pdf_m4"
                                )
                                m4_success(f"PDF ready — {fname}")
                            except Exception as e:
                                st.error(f"PDF error: {e}")
                                import traceback; st.code(traceback.format_exc())

                with col_csv:
                    m4_sec("📊", "CSV Export")
                    st.markdown(f'<div style="color:{M4_MUTED};font-size:0.82rem;margin-bottom:0.8rem">All anomaly records from all three signals in a single CSV file.</div>', unsafe_allow_html=True)
                    csv_data  = generate_csv(anom_hr_f, anom_steps_f, anom_sleep_f)
                    fname_csv = f"FitPulse_Anomalies_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                    st.download_button(
                        label="⬇️ Download Anomaly CSV",
                        data=csv_data,
                        file_name=fname_csv,
                        mime="text/csv",
                        key="dl_csv_m4"
                    )
                    with st.expander("👁️ Preview CSV data"):
                        preview_df = pd.concat([
                            anom_hr_f[anom_hr_f["is_anomaly"]].assign(signal="Heart Rate").rename(columns={"AvgHR":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                            anom_steps_f[anom_steps_f["is_anomaly"]].assign(signal="Steps").rename(columns={"TotalSteps":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                            anom_sleep_f[anom_sleep_f["is_anomaly"]].assign(signal="Sleep").rename(columns={"TotalSleepMinutes":"value","rolling_med":"expected"})[["signal","Date","value","expected","residual","reason"]],
                        ], ignore_index=True).sort_values(["signal","Date"]).round(2)
                        st.dataframe(preview_df, use_container_width=True, height=280)

                st.markdown('<hr class="m4-divider">', unsafe_allow_html=True)
                m4_sec("📸", "Screenshots Required for Submission")
                st.markdown(f"""
                <div class="m4-card">
                  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
                    <div style="background:{M4_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                      <span style="color:{M4_ACC2}">📸</span> <b>Screenshot 1</b> - Full dashboard UI (Overview tab)
                    </div>
                    <div style="background:{M4_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                      <span style="color:{M4_ACC2}">📸</span> <b>Screenshot 2</b> - Downloadable report buttons (this tab)
                    </div>
                    <div style="background:{M4_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                      <span style="color:{M4_ACC2}">📸</span> <b>Screenshot 3</b> - KPI strip with anomaly counts
                    </div>
                    <div style="background:{M4_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem">
                      <span style="color:{M4_ACC2}">📸</span> <b>Screenshot 4</b> - HR / Steps / Sleep deep dive tabs
                    </div>
                    <div style="background:{M4_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;grid-column:1/-1">
                      <span style="color:{M4_ACC2}">📸</span> <b>Screenshot 5</b> - Sidebar with filters + date range visible
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)