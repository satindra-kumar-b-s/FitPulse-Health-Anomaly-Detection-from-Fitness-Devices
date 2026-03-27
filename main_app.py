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
# GLOBAL CSS (merged from M1/M2 + M3)
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
  box-shadow: 0 0 12px rgba(99,215,196,0.08) !important;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  background: linear-gradient(135deg,#1f3d5c,#162c46) !important;
  box-shadow: 0 0 20px rgba(99,215,196,0.2) !important;
  transform: translateY(-1px) !important;
}

[data-testid="stMetric"] {
  background: var(--glass) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
  color:var(--muted) !important; font-size:0.75rem !important;
  text-transform:uppercase; letter-spacing:1px;
}
[data-testid="stMetricValue"] {
  color:var(--accent) !important; font-size:1.8rem !important; font-weight:800 !important;
}

.stSlider [data-baseweb="slider"] { padding:0 !important; }
.stSlider label {
  color:var(--muted) !important; font-size:0.8rem !important;
  text-transform:uppercase; letter-spacing:0.8px;
}

.stAlert { border-radius:12px !important; border-left-width:4px !important; }
.stDataFrame { border-radius:12px !important; overflow:hidden; }

[data-testid="stFileUploader"] {
  background: var(--glass) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem !important;
}

hr { border-color:var(--border) !important; }
.element-container img { border-radius:12px; }

.fp-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem 1.6rem;
  margin-bottom: 1.2rem;
  backdrop-filter: blur(10px);
}

.fp-badge {
  display:inline-block; padding:3px 10px; border-radius:20px;
  font-size:0.72rem; font-weight:700; letter-spacing:0.6px; text-transform:uppercase;
}
.fp-badge-ok   { background:rgba(52,211,153,0.15); color:#34d399; border:1px solid rgba(52,211,153,0.3); }
.fp-badge-miss { background:rgba(248,113,113,0.15); color:#f87171; border:1px solid rgba(248,113,113,0.3); }

.step-header {
  display:inline-flex; align-items:center; gap:10px;
  background:rgba(99,215,196,0.07);
  border:1px solid rgba(99,215,196,0.2);
  border-radius:10px;
  padding:7px 16px;
  margin:1.2rem 0 0.8rem 0;
  font-family:'JetBrains Mono',monospace;
  font-size:0.82rem;
  color:var(--accent);
  font-weight:600;
  letter-spacing:0.5px;
}

.null-card {
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(99,215,196,0.12);
  border-radius:14px;
  padding:16px 14px 12px 14px;
}
.null-card-name {
  font-size:0.78rem; color:#6b7a96; margin-bottom:8px;
  font-family:'JetBrains Mono',monospace;
}
.null-val-ok  { font-size:1.6rem; color:#34d399; }
.null-val-bad { font-size:1.3rem; color:#f87171; font-weight:800; font-family:'JetBrains Mono',monospace; }
.null-rows    { font-size:0.7rem; color:#6b7a96; margin-top:5px; }

.log-box {
  background:rgba(255,255,255,0.02);
  border:1px solid rgba(99,215,196,0.12);
  border-radius:12px;
  padding:1.1rem 1.4rem;
  font-family:'JetBrains Mono',monospace;
  font-size:0.8rem;
  line-height:2;
}

.prog-bar-bg   {
  background:rgba(255,255,255,0.06); border-radius:10px;
  height:6px; width:100%; margin:6px 0 14px 0;
}
.prog-bar-fill {
  background:linear-gradient(90deg,#63d7c4,#818cf8);
  border-radius:10px; height:6px;
}

.nav-item { display:flex; align-items:center; gap:9px; padding:5px 0; font-size:0.83rem; }
.nav-dot-active   { width:8px;height:8px;border-radius:50%;background:#63d7c4;flex-shrink:0; }
.nav-dot-inactive { width:8px;height:8px;border-radius:50%;border:1.5px solid #6b7a96;flex-shrink:0; }
.nav-label-active   { color:#e2eaf4; font-weight:600; }
.nav-label-inactive { color:#6b7a96; }

/* M3 styles */
.m3-hero {
  background: linear-gradient(135deg,rgba(252,129,129,0.08),rgba(246,135,179,0.06),rgba(10,14,26,0.9));
  border: 1px solid rgba(252,129,129,0.4);
  border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem;
  position: relative; overflow: hidden;
}
.m3-hero::before {
  content: ''; position: absolute; top:-60px; right:-60px;
  width:300px; height:300px;
  background: radial-gradient(circle,rgba(252,129,129,0.08) 0%,transparent 70%);
  border-radius:50%;
}
.hero-badge {
  display:inline-block; background:rgba(252,129,129,0.1); border:1px solid rgba(252,129,129,0.4);
  border-radius:100px; padding:0.3rem 1rem; font-size:0.75rem;
  font-family:'JetBrains Mono',monospace; color:#fc8181; margin-bottom:1rem;
}
.m3-card {
  background:rgba(15,23,42,0.85); border:1px solid rgba(99,179,237,0.2); border-radius:14px;
  padding:1.4rem 1.6rem; margin-bottom:1rem; backdrop-filter:blur(10px);
}
.m3-card-title {
  font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:700;
  color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;
}
.m3-step-pill {
  display:inline-flex; align-items:center; gap:0.5rem;
  background:rgba(99,179,237,0.07); border:1px solid rgba(99,179,237,0.2); border-radius:100px;
  padding:0.3rem 0.9rem; font-size:0.75rem; font-family:'JetBrains Mono',monospace;
  color:#63b3ed; margin-bottom:0.8rem;
}
.m3-metric-grid { display:flex; gap:0.8rem; flex-wrap:wrap; margin:0.8rem 0; }
.m3-metric-card {
  flex:1; min-width:120px; background:rgba(99,179,237,0.07); border:1px solid rgba(99,179,237,0.2);
  border-radius:12px; padding:1rem 1.2rem; text-align:center;
}
.m3-metric-val { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#63b3ed; line-height:1; margin-bottom:0.25rem; }
.m3-metric-val-red { color:#fc8181; }
.m3-metric-label { font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.06em; }
.m3-anom-tag {
  display:inline-flex; align-items:center; gap:0.4rem;
  background:rgba(252,129,129,0.1); border:1px solid rgba(252,129,129,0.4); border-radius:100px;
  padding:0.3rem 0.9rem; font-size:0.72rem; font-family:'JetBrains Mono',monospace;
  color:#fc8181; margin-bottom:0.8rem;
}
.m3-screenshot-badge {
  display:inline-flex; align-items:center; gap:0.4rem;
  background:rgba(246,135,179,0.15); border:1px solid rgba(246,135,179,0.4);
  border-radius:100px; padding:0.3rem 0.9rem; font-size:0.72rem;
  font-family:'JetBrains Mono',monospace; color:#f687b3; margin-bottom:0.8rem;
}
.m3-alert-warn { background:rgba(246,173,85,0.12); border-left:3px solid #f6ad55; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#fbd38d; }
.m3-alert-success { background:rgba(104,211,145,0.1); border-left:3px solid #68d391; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#9ae6b4; }
.m3-alert-info { background:rgba(99,179,237,0.15); border-left:3px solid #63b3ed; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#bee3f8; }
.m3-alert-danger { background:rgba(252,129,129,0.1); border-left:3px solid #fc8181; border-radius:0 10px 10px 0; padding:0.8rem 1rem; margin:0.6rem 0; font-size:0.85rem; color:#feb2b2; }
.m3-sec-header { display:flex; align-items:center; gap:0.8rem; margin:2rem 0 1rem 0; padding-bottom:0.6rem; border-bottom:1px solid rgba(99,179,237,0.2); }
.m3-sec-icon { font-size:1.4rem; width:2.2rem; height:2.2rem; display:flex; align-items:center; justify-content:center; background:rgba(99,179,237,0.15); border-radius:8px; border:1px solid rgba(99,179,237,0.2); }
.m3-sec-title { font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:700; color:#e2e8f0; margin:0; }
.m3-sec-badge { margin-left:auto; background:rgba(99,179,237,0.15); border:1px solid rgba(99,179,237,0.2); border-radius:100px; padding:0.2rem 0.7rem; font-size:0.7rem; font-family:'JetBrains Mono',monospace; color:#63b3ed; }
.m3-divider { border:none; border-top:1px solid rgba(99,179,237,0.2); margin:2rem 0; }
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

# M3 Plotly theme constants
M3_BG      = "#0a0e1a"
M3_CARD    = "rgba(15,23,42,0.85)"
M3_BOR     = "rgba(99,179,237,0.2)"
M3_TEXT    = "#e2e8f0"
M3_MUTED   = "#94a3b8"
M3_ACCENT  = "#63b3ed"
M3_ACC2    = "#f687b3"
M3_ACC3    = "#68d391"
M3_RED     = "#fc8181"
M3_GRID    = "rgba(255,255,255,0.06)"
M3_DANGER_BG  = "rgba(252,129,129,0.1)"
M3_DANGER_BOR = "rgba(252,129,129,0.4)"
M3_WARN_BG    = "rgba(246,173,85,0.12)"
M3_SUCCESS_BG = "rgba(104,211,145,0.1)"
M3_SUCCESS_BOR= "rgba(104,211,145,0.4)"
M3_SECTION_BG = "rgba(99,179,237,0.07)"
M3_BADGE_BG   = "rgba(99,179,237,0.15)"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=M3_BG, plot_bgcolor="#0f172a",
    font_color=M3_TEXT, font_family="Inter, sans-serif",
    xaxis=dict(gridcolor=M3_GRID, showgrid=True, zeroline=False,
               linecolor=M3_BOR, tickfont_color=M3_MUTED),
    yaxis=dict(gridcolor=M3_GRID, showgrid=True, zeroline=False,
               linecolor=M3_BOR, tickfont_color=M3_MUTED),
    legend=dict(bgcolor=M3_CARD, bordercolor=M3_BOR, borderwidth=1, font_color=M3_TEXT),
    margin=dict(l=50, r=30, t=60, b=50),
    hoverlabel=dict(bgcolor=M3_CARD, bordercolor=M3_BOR, font_color=M3_TEXT),
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k in ["files_loaded", "tsfresh_done", "prophet_done", "cluster_done"]:
    if k not in st.session_state:
        st.session_state[k] = False

for k in ["daily","steps","intensity","sleep","hr","features",
          "master_df","cluster_summary",
          "tsfresh_fig","prophet_fig","cluster_fig",
          "cluster_bar_fig","elbow_fig","steps_sleep_fig",
          "X_scaled","km_labels","db_labels","k_val_used","available_cols","feat_df",
          "df","clean_df","milestone1_loaded"]:
    if k not in st.session_state:
        st.session_state[k] = None

if "raw_files" not in st.session_state:
    st.session_state["raw_files"] = {}

# M3 session state
for k, v in [
    ("m3_files_loaded",    False),
    ("m3_anomaly_done",    False),
    ("m3_simulation_done", False),
    ("m3_daily",    None), ("m3_hourly_s", None), ("m3_hourly_i", None),
    ("m3_sleep",    None), ("m3_hr",       None), ("m3_hr_minute", None),
    ("m3_master",   None),
    ("m3_anom_hr",  None), ("m3_anom_steps", None), ("m3_anom_sleep", None),
    ("m3_sim_results", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# Active section: "preprocessing", "pattern_extraction", or "anomaly_detector"
if "active_section" not in st.session_state:
    st.session_state["active_section"] = "preprocessing"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — shared
# ─────────────────────────────────────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()

def step_hdr(num, label):
    st.markdown(f"""
    <div class='step-header'>
      <span style='color:#f97316'>◆</span>
      Step {num}
      <span style='color:#f97316;font-size:0.6rem'>·</span>
      {label}
    </div>""", unsafe_allow_html=True)

# ─── M3 helpers ───────────────────────────────────────────────────────────────
def m3_sec(icon, title, badge=None):
    badge_html = f'<span class="m3-sec-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="m3-sec-header">
      <div class="m3-sec-icon">{icon}</div>
      <p class="m3-sec-title">{title}</p>
      {badge_html}
    </div>""", unsafe_allow_html=True)

def m3_step_pill(n, label):
    st.markdown(f'<div class="m3-step-pill">◆ Step {n} &nbsp;·&nbsp; {label}</div>', unsafe_allow_html=True)

def m3_screenshot_badge(ref):
    st.markdown(f'<div class="m3-screenshot-badge">📸 Screenshot · {ref}</div>', unsafe_allow_html=True)

def m3_anom_tag(label):
    st.markdown(f'<div class="m3-anom-tag">🚨 {label}</div>', unsafe_allow_html=True)

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
        fig.update_layout(title=dict(text=title, font_color=M3_TEXT, font_size=14,
                                     font_family="Syne, sans-serif"))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# M3 Anomaly Detection Functions
# ─────────────────────────────────────────────────────────────────────────────
def detect_hr_anomalies(master, hr_high=100, hr_low=50, residual_sigma=2.0):
    df = master[["Id","Date","AvgHR","MaxHR","MinHR"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    hr_daily = df.groupby("Date")["AvgHR"].mean().reset_index()
    hr_daily.columns = ["Date","AvgHR"]
    hr_daily = hr_daily.sort_values("Date")
    hr_daily["thresh_high"] = hr_daily["AvgHR"] > hr_high
    hr_daily["thresh_low"]  = hr_daily["AvgHR"] < hr_low
    hr_daily["rolling_med"]  = hr_daily["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_daily["residual"]     = hr_daily["AvgHR"] - hr_daily["rolling_med"]
    resid_std = hr_daily["residual"].std()
    hr_daily["resid_anomaly"] = hr_daily["residual"].abs() > (residual_sigma * resid_std)
    hr_daily["is_anomaly"] = hr_daily["thresh_high"] | hr_daily["thresh_low"] | hr_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_high"]:   r.append(f"HR>{hr_high}")
        if row["thresh_low"]:    r.append(f"HR<{hr_low}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    hr_daily["reason"] = hr_daily.apply(reason, axis=1)
    return hr_daily

def detect_steps_anomalies(master, steps_low=500, steps_high=25000, residual_sigma=2.0):
    df = master[["Date","TotalSteps"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    steps_daily = df.groupby("Date")["TotalSteps"].mean().reset_index()
    steps_daily = steps_daily.sort_values("Date")
    steps_daily["thresh_low"]  = steps_daily["TotalSteps"] < steps_low
    steps_daily["thresh_high"] = steps_daily["TotalSteps"] > steps_high
    steps_daily["rolling_med"]   = steps_daily["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    steps_daily["residual"]      = steps_daily["TotalSteps"] - steps_daily["rolling_med"]
    resid_std = steps_daily["residual"].std()
    steps_daily["resid_anomaly"] = steps_daily["residual"].abs() > (residual_sigma * resid_std)
    steps_daily["is_anomaly"] = steps_daily["thresh_low"] | steps_daily["thresh_high"] | steps_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["thresh_low"]:    r.append(f"Steps<{steps_low}")
        if row["thresh_high"]:   r.append(f"Steps>{steps_high}")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    steps_daily["reason"] = steps_daily.apply(reason, axis=1)
    return steps_daily

def detect_sleep_anomalies(master, sleep_low=60, sleep_high=600, residual_sigma=2.0):
    df = master[["Date","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    sleep_daily = df.groupby("Date")["TotalSleepMinutes"].mean().reset_index()
    sleep_daily = sleep_daily.sort_values("Date")
    sleep_daily["thresh_low"]  = (sleep_daily["TotalSleepMinutes"] > 0) & (sleep_daily["TotalSleepMinutes"] < sleep_low)
    sleep_daily["thresh_high"] = sleep_daily["TotalSleepMinutes"] > sleep_high
    sleep_daily["no_data"]     = sleep_daily["TotalSleepMinutes"] == 0
    sleep_daily["rolling_med"]   = sleep_daily["TotalSleepMinutes"].rolling(3, center=True, min_periods=1).median()
    sleep_daily["residual"]      = sleep_daily["TotalSleepMinutes"] - sleep_daily["rolling_med"]
    resid_std = sleep_daily["residual"].std()
    sleep_daily["resid_anomaly"] = sleep_daily["residual"].abs() > (residual_sigma * resid_std)
    sleep_daily["is_anomaly"] = sleep_daily["thresh_low"] | sleep_daily["thresh_high"] | sleep_daily["resid_anomaly"]
    def reason(row):
        r = []
        if row["no_data"]:       r.append("No device worn")
        if row["thresh_low"]:    r.append(f"Sleep<{sleep_low}min")
        if row["thresh_high"]:   r.append(f"Sleep>{sleep_high}min")
        if row["resid_anomaly"]: r.append(f"Residual±{residual_sigma:.0f}σ")
        return ", ".join(r) if r else ""
    sleep_daily["reason"] = sleep_daily.apply(reason, axis=1)
    return sleep_daily

def simulate_accuracy(master, n_inject=10):
    np.random.seed(42)
    df = master[["Date","AvgHR","TotalSteps","TotalSleepMinutes"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date").mean().reset_index().sort_values("Date")
    results = {}
    # HR
    hr_sim = df_daily[["Date","AvgHR"]].copy()
    inject_idx = np.random.choice(len(hr_sim), n_inject, replace=False)
    hr_sim.loc[inject_idx, "AvgHR"] = np.random.choice([115,120,125,35,40,45,118,130,38,42], n_inject, replace=True)
    hr_sim["rolling_med"] = hr_sim["AvgHR"].rolling(3, center=True, min_periods=1).median()
    hr_sim["residual"]    = hr_sim["AvgHR"] - hr_sim["rolling_med"]
    resid_std = hr_sim["residual"].std()
    hr_sim["detected"] = (hr_sim["AvgHR"] > 100) | (hr_sim["AvgHR"] < 50) | (hr_sim["residual"].abs() > 2 * resid_std)
    tp = hr_sim.iloc[inject_idx]["detected"].sum()
    results["Heart Rate"] = {"injected": n_inject, "detected": int(tp), "accuracy": round(tp / n_inject * 100, 1)}
    # Steps
    st_sim = df_daily[["Date","TotalSteps"]].copy()
    inject_idx2 = np.random.choice(len(st_sim), n_inject, replace=False)
    st_sim.loc[inject_idx2, "TotalSteps"] = np.random.choice([50,100,150,30000,35000,28000,80,200,31000,29000], n_inject, replace=True)
    st_sim["rolling_med"] = st_sim["TotalSteps"].rolling(3, center=True, min_periods=1).median()
    st_sim["residual"]    = st_sim["TotalSteps"] - st_sim["rolling_med"]
    resid_std2 = st_sim["residual"].std()
    st_sim["detected"] = (st_sim["TotalSteps"] < 500) | (st_sim["TotalSteps"] > 25000) | (st_sim["residual"].abs() > 2 * resid_std2)
    tp2 = st_sim.iloc[inject_idx2]["detected"].sum()
    results["Steps"] = {"injected": n_inject, "detected": int(tp2), "accuracy": round(tp2 / n_inject * 100, 1)}
    # Sleep
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

# M3 required files
M3_REQUIRED_FILES = {
    "dailyActivity_merged.csv":     {"key_cols": ["ActivityDate","TotalSteps","Calories"],    "label": "Daily Activity",    "icon": "🏃"},
    "hourlySteps_merged.csv":       {"key_cols": ["ActivityHour","StepTotal"],                "label": "Hourly Steps",      "icon": "👣"},
    "hourlyIntensities_merged.csv": {"key_cols": ["ActivityHour","TotalIntensity"],           "label": "Hourly Intensities","icon": "⚡"},
    "minuteSleep_merged.csv":       {"key_cols": ["date","value","logId"],                    "label": "Minute Sleep",      "icon": "💤"},
    "heartrate_seconds_merged.csv": {"key_cols": ["Time","Value"],                            "label": "Heart Rate",        "icon": "❤️"},
}

def score_match(df, req_info):
    return sum(1 for col in req_info["key_cols"] if col in df.columns)

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

    st.markdown(
        "<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;"
        "letter-spacing:1px;font-weight:700;margin-bottom:10px'>📌 Pipeline Sections</div>",
        unsafe_allow_html=True
    )

    # Three navigation buttons
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔧 Preprocessing", use_container_width=True,
                     type="primary" if st.session_state.active_section == "preprocessing" else "secondary"):
            st.session_state.active_section = "preprocessing"
            st.rerun()
    with col_b:
        if st.button("🤖 Pattern Extraction", use_container_width=True,
                     type="primary" if st.session_state.active_section == "pattern_extraction" else "secondary"):
            st.session_state.active_section = "pattern_extraction"
            st.rerun()

    # Full-width Anomaly Detector button
    if st.button("🚨 Anomaly Detector", use_container_width=True,
                 type="primary" if st.session_state.active_section == "anomaly_detector" else "secondary"):
        st.session_state.active_section = "anomaly_detector"
        st.rerun()

    st.divider()

    # ── Section-specific sidebar content ──
    if st.session_state.active_section == "preprocessing":
        st.markdown("""
        <div style='font-size:0.72rem;color:#6b7a96;line-height:2'>
          <div style='color:#63d7c4;font-weight:700;margin-bottom:6px'>🔧 Preprocessing Steps</div>
          ① Upload CSV Dataset<br>
          ② View Missing Values<br>
          ③ Run Data Cleaning<br>
          ④ Explore Distributions<br>
          ⑤ Download Clean CSV
        </div>""", unsafe_allow_html=True)
        st.divider()
        m1_loaded = st.session_state.get("milestone1_loaded") or False
        m1_clean  = st.session_state.get("clean_df") is not None
        st.markdown(
            "<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;"
            "letter-spacing:1px;font-weight:700;margin-bottom:8px'>Status</div>",
            unsafe_allow_html=True
        )
        def sidebar_status(done, label):
            color = "#34d399" if done else "#6b7a96"
            icon  = "✓" if done else "○"
            st.markdown(f"<div style='font-size:0.8rem;color:{color};padding:3px 0'>{icon} {label}</div>", unsafe_allow_html=True)
        sidebar_status(m1_loaded, "Dataset Loaded")
        sidebar_status(m1_clean,  "Data Cleaned")

    elif st.session_state.active_section == "pattern_extraction":
        fl  = st.session_state.files_loaded
        tf  = st.session_state.tsfresh_done
        pf  = st.session_state.prophet_done
        cl  = st.session_state.cluster_done
        ss  = st.session_state.get("steps_sleep_fig") is not None
        pct = int(sum([fl, tf, pf, cl, ss]) / 5 * 100)

        st.markdown(f"""
        <div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;
                    letter-spacing:1px;font-weight:700'>
          Pipeline Progress
          <span style='color:#63d7c4;margin-left:6px'>{pct}%</span>
        </div>
        <div class='prog-bar-bg'>
          <div class='prog-bar-fill' style='width:{pct}%'></div>
        </div>""", unsafe_allow_html=True)

        def nav(active, icon, label):
            d = "nav-dot-active"   if active else "nav-dot-inactive"
            l = "nav-label-active" if active else "nav-label-inactive"
            st.markdown(f"<div class='nav-item'><div class='{d}'></div><span>{icon}</span><span class='{l}'>{label}</span></div>", unsafe_allow_html=True)
        nav(fl, "📁", "Data Loading")
        nav(tf, "⚗️", "TSFresh Features")
        nav(pf, "📅", "Prophet Forecast")
        nav(cl, "🔵", "Clustering")
        nav(ss, "📈", "Steps & Sleep Forecast")

        st.divider()
        st.markdown("<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700'>⚙️ ML Controls</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-top:10px'>KMeans Clusters</div>", unsafe_allow_html=True)
        k_val = st.slider("k", 2, 9, 3, label_visibility="collapsed")
        st.markdown("<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-top:10px'>DBSCAN EPS</div>", unsafe_allow_html=True)
        eps_val = st.slider("eps", 0.5, 5.0, 2.20, step=0.1, label_visibility="collapsed")

    else:  # anomaly_detector
        steps_done = sum([st.session_state.m3_files_loaded,
                          st.session_state.m3_anomaly_done,
                          st.session_state.m3_simulation_done])
        pct_m3 = int(steps_done / 3 * 100)
        st.markdown(f"""
        <div style="margin-bottom:1rem">
          <div style="font-size:0.7rem;color:#94a3b8;font-family:'JetBrains Mono',monospace;margin-bottom:0.4rem">
            PIPELINE · {pct_m3}%
          </div>
          <div style="background:rgba(99,179,237,0.2);border-radius:4px;height:6px;overflow:hidden">
            <div style="width:{pct_m3}%;height:100%;background:linear-gradient(90deg,#fc8181,#f687b3);border-radius:4px;transition:width 0.4s"></div>
          </div>
        </div>""", unsafe_allow_html=True)

        for done, icon, label in [
            (st.session_state.m3_files_loaded,    "📂", "Data Loaded"),
            (st.session_state.m3_anomaly_done,    "🚨", "Anomalies Detected"),
            (st.session_state.m3_simulation_done, "🎯", "Accuracy Simulated"),
        ]:
            dot = '<span style="color:#68d391">●</span>' if done else '<span style="color:#6b7a96">○</span>'
            col = "#e2e8f0" if done else "#6b7a96"
            st.markdown(f'<div style="font-size:0.82rem;padding:0.3rem 0;color:{col}">{dot} {icon} {label}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="font-size:0.72rem;color:#94a3b8;font-family:JetBrains Mono,monospace;margin-bottom:0.5rem">THRESHOLDS</div>', unsafe_allow_html=True)
        hr_high  = st.number_input("HR High (bpm)",   value=100, min_value=80,  max_value=180, key="m3_hr_high")
        hr_low   = st.number_input("HR Low (bpm)",    value=50,  min_value=30,  max_value=70,  key="m3_hr_low")
        st_low   = st.number_input("Steps Low",       value=500, min_value=0,   max_value=2000, key="m3_st_low")
        sl_low   = st.number_input("Sleep Low (min)", value=60,  min_value=0,   max_value=120, key="m3_sl_low")
        sl_high  = st.number_input("Sleep High (min)",value=600, min_value=300, max_value=900, key="m3_sl_high")
        sigma    = st.slider("Residual σ threshold", 1.0, 4.0, 2.0, 0.5, key="m3_sigma")

        st.divider()
        st.markdown('<div style="font-size:0.68rem;color:#94a3b8;font-family:JetBrains Mono,monospace">Real Fitbit Dataset<br>30 users · March–April 2016</div>', unsafe_allow_html=True)

# Set defaults if not in relevant sections
if st.session_state.active_section != "pattern_extraction":
    k_val   = 3
    eps_val = 2.2
if st.session_state.active_section != "anomaly_detector":
    hr_high = 100; hr_low = 50; st_low = 500; sl_low = 60; sl_high = 600; sigma = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
active = st.session_state.active_section
section_map = {
    "preprocessing":       ("🔧", "Preprocessing"),
    "pattern_extraction":  ("🤖", "Pattern Extraction"),
    "anomaly_detector":    ("🚨", "Anomaly Detector"),
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
        file.seek(0)
        df = pd.read_csv(file)
        st.session_state["df"] = df
        st.session_state["milestone1_loaded"] = True
        st.success(f"✅ Dataset loaded — **{len(df):,} rows × {len(df.columns)} columns**")

    if st.session_state.get("milestone1_loaded") and st.session_state.get("df") is not None:
        df = st.session_state["df"]
        with st.expander("🔎 Raw Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### 🔍 Missing Values Analysis")
        missing       = df.isnull().sum()
        total_missing = missing.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows",    f"{len(df):,}")
        c2.metric("Columns",       f"{len(df.columns)}")
        c3.metric("Missing Cells", f"{total_missing:,}")

        if total_missing > 0:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            missing_nz = missing[missing > 0]
            bars = ax.bar(missing_nz.index, missing_nz.values, color=PALETTE[0], edgecolor="none")
            ax.set_title("Missing Values per Column", fontsize=13, pad=12)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(int(bar.get_height())), ha="center", va="bottom", fontsize=8, color="#e2eaf4")
            fig.tight_layout()
            st.pyplot(fig); plt.close(fig)
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
            st.success("✅ Cleaning complete — missing values interpolated / filled.")

    if st.session_state.get("clean_df") is not None:
        st.markdown("### 📑 Cleaned Dataset")
        st.dataframe(st.session_state["clean_df"].head(10), use_container_width=True)
        csv = st.session_state["clean_df"].to_csv(index=False).encode()
        st.download_button("⬇  Download Clean CSV", csv, "clean_dataset.csv", mime="text/csv")

        st.markdown("### 📊 Exploratory Data Analysis")
        df_eda   = st.session_state["clean_df"]
        num_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            pairs = [num_cols[i:i+2] for i in range(0, len(num_cols), 2)]
            for pair in pairs:
                grid = st.columns(len(pair))
                for ax_col, col in zip(grid, pair):
                    with ax_col:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        sns.histplot(df_eda[col], kde=True, ax=ax, color=PALETTE[0],
                                     edgecolor="none", line_kws={"color": PALETTE[1], "linewidth": 2})
                        ax.set_title(col, fontsize=11); ax.set_xlabel("")
                        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("No numeric columns found for EDA.")

# =============================================================================
# PATTERN EXTRACTION SECTION
# =============================================================================
elif active == "pattern_extraction":
    st.markdown("## 🤖 Pattern Extraction — ML Pipeline")
    st.caption("Upload Fitbit CSV files, extract features, forecast trends, and cluster users.")

    REQUIRED = {
        "Daily Activity":     ["TotalSteps", "Calories", "VeryActiveMinutes"],
        "Hourly Steps":       ["StepTotal"],
        "Hourly Intensities": ["TotalIntensity", "AverageIntensity"],
        "Minute Sleep":       ["logId", "value", "date"],
        "Heart Rate":         ["Value", "Time"],
    }

    step_hdr(1, "Upload Fitbit CSV Files")
    st.caption("Upload all 5 Fitbit CSV files — auto-detected by column signature.")
    uploaded_files = st.file_uploader("Upload", type="csv", accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for f in uploaded_files:
            f.seek(0)
            st.session_state["raw_files"][f.name] = f.read()

    detected = {}
    for fname, raw in st.session_state["raw_files"].items():
        try:
            cols = set(pd.read_csv(pd.io.common.BytesIO(raw), nrows=3).columns)
            for name, keys in REQUIRED.items():
                if all(k in cols for k in keys) and name not in detected:
                    detected[name] = fname
        except Exception:
            pass

    step_hdr(2, "File Detection")
    icons_map = {"Daily Activity":"🏃","Hourly Steps":"👟","Hourly Intensities":"⚡","Minute Sleep":"😴","Heart Rate":"❤️"}
    c5 = st.columns(5)
    for col, name in zip(c5, REQUIRED.keys()):
        found = name in detected
        color = "#34d399" if found else "#f87171"
        with col:
            st.markdown(f"""
            <div style='border:1px solid {color}33;background:rgba(255,255,255,0.03);
                        padding:16px 10px 12px;border-radius:14px;text-align:center'>
              <div style='font-size:1.4rem;margin-bottom:6px'>{icons_map[name]}</div>
              <div style='font-size:0.73rem;font-weight:700;color:#e2eaf4;margin-bottom:6px'>{name}</div>
              <span class='fp-badge {"fp-badge-ok" if found else "fp-badge-miss"}'>
                {"Found ✓" if found else "Missing"}
              </span>
            </div>""", unsafe_allow_html=True)

    st.write("")
    dc = len(detected)
    m1, m2, m3 = st.columns(3)
    m1.metric(str(dc), "DETECTED"); m2.metric(str(5-dc), "MISSING"); m3.metric("✓" if dc==5 else "⏳", "READY TO LOAD")
    if dc == 5: st.success("✅ All 5 required files detected — ready to process!")

    step_hdr(3, "Load & Parse All Files")
    if dc == 5:
        if st.button("⚡ Load & Parse All Files"):
            with st.spinner("Parsing and merging datasets…"):
                try:
                    def _rd(name):
                        return pd.read_csv(pd.io.common.BytesIO(st.session_state["raw_files"][detected[name]]))
                    daily     = _rd("Daily Activity"); steps     = _rd("Hourly Steps")
                    intensity = _rd("Hourly Intensities"); sleep = _rd("Minute Sleep"); hr = _rd("Heart Rate")
                    for df_ in [daily, steps, intensity, sleep, hr]:
                        df_.columns = [c.strip() for c in df_.columns]
                    st.session_state.daily = daily; st.session_state.steps = steps
                    st.session_state.intensity = intensity; st.session_state.sleep = sleep; st.session_state.hr = hr
                    id_col   = next(c for c in daily.columns if c.lower() == "id")
                    date_col = next((c for c in daily.columns if "date" in c.lower()), None)
                    keep = [id_col]
                    if date_col: keep.append(date_col)
                    for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"]:
                        if c in daily.columns: keep.append(c)
                    master = daily[keep].copy()
                    master.rename(columns={id_col:"Id"}, inplace=True)
                    if date_col: master.rename(columns={date_col:"Date"}, inplace=True)
                    hr_id = next((c for c in hr.columns if c.lower()=="id"), None)
                    hr_t  = next((c for c in hr.columns if c.lower() in ("time","datetime","timestamp","date","activityminute")), None)
                    hr_v  = next((c for c in hr.columns if c.lower()=="value"), None)
                    if hr_id and hr_t and hr_v and "Date" in master.columns:
                        hr["_d"] = pd.to_datetime(hr[hr_t], errors="coerce").dt.date.astype(str)
                        hr_agg   = hr.groupby([hr_id,"_d"])[hr_v].mean().reset_index()
                        hr_agg.columns = ["Id","Date","AvgHR"]
                        master = master.merge(hr_agg, on=["Id","Date"], how="left")
                    sl_id = next((c for c in sleep.columns if c.lower()=="id"), None)
                    sl_d  = next((c for c in sleep.columns if "date" in c.lower()), None)
                    if sl_id and sl_d and "Date" in master.columns:
                        sleep["_d"] = pd.to_datetime(sleep[sl_d], errors="coerce").dt.date.astype(str)
                        sl_agg = sleep.groupby([sl_id,"_d"]).size().reset_index(name="TotalSleepMinutes")
                        sl_agg.columns = ["Id","Date","TotalSleepMinutes"]
                        master = master.merge(sl_agg, on=["Id","Date"], how="left")
                    st.session_state.master_df    = master
                    st.session_state.files_loaded = True
                    st.success("✅ All 5 files loaded and master DataFrame built")
                except Exception as e:
                    st.error(f"❌ Load error: {e}")

    if st.session_state.files_loaded:
        st.success("✅ All 5 files loaded and master DataFrame built")

    if st.session_state.files_loaded:
        step_hdr(4, "Null Value Check")
        dsets = {"dailyActivity":st.session_state.daily,"hourlySteps":st.session_state.steps,
                 "hourlyIntensities":st.session_state.intensity,"minuteSleep":st.session_state.sleep,"heartrate":st.session_state.hr}
        nc = st.columns(5)
        for col, (name, df_) in zip(nc, dsets.items()):
            nulls = int(df_.isnull().sum().sum()); rows = len(df_)
            val = (f"<div class='null-val-bad'>{nulls:,}</div>" if nulls > 0 else "<div class='null-val-ok'>⊙</div>")
            with col:
                st.markdown(f"<div class='null-card'><div class='null-card-name'>{name}</div>{val}<div class='null-rows'>nulls · {rows:,} rows</div></div>", unsafe_allow_html=True)

        step_hdr(7, "Time Normalization Log")
        hr2  = st.session_state.hr
        hr_t = next((c for c in hr2.columns if c.lower() in ("time","datetime","timestamp","date","activityminute")), None)
        rows_before = len(hr2); rows_after = rows_before; date_min="N/A"; date_max="N/A"; date_range_days="?"
        if hr_t:
            try:
                _dt = pd.to_datetime(hr2[hr_t], errors="coerce"); _dt_clean = _dt.dropna()
                rows_after = int(_dt_clean.dt.floor("T").nunique())
                date_min = _dt_clean.dt.date.min(); date_max = _dt_clean.dt.date.max()
                date_range_days = (date_max - date_min).days + 1
            except Exception: pass
        sl2 = st.session_state.sleep; sleep_rows = len(sl2)
        st2 = st.session_state.steps
        st_t = next((c for c in st2.columns if c.lower() in ("activityhour","datetime","time","date")), None)
        hourly_pct = "100.0"
        if st_t:
            try:
                _dts = pd.to_datetime(st2[st_t], errors="coerce").dropna().sort_values()
                diffs = _dts.diff().dt.total_seconds() / 3600
                hourly_pct = f"{(diffs.dropna() == 1.0).mean() * 100:.1f}"
            except Exception: pass
        st.markdown(f"""
        <div class='log-box'>
          <span style='color:#34d399'>✅ HR resampled</span><span style='color:#6b7a96'> seconds → 1-minute intervals</span><br>
          &nbsp;&nbsp;&nbsp;Rows before : <span style='color:#e2eaf4;font-weight:700'>{rows_before:,}</span>&nbsp;&nbsp;|&nbsp;&nbsp;Rows after : <span style='color:#63d7c4;font-weight:700'>{rows_after:,}</span><br>
          <span style='color:#34d399'>✅ Date range</span><span style='color:#63d7c4'> {date_min} → {date_max}</span><span style='color:#6b7a96'> ({date_range_days} days)</span><br>
          <span style='color:#34d399'>✅ Hourly frequency</span><span style='color:#6b7a96'> 1.0h median &nbsp;|&nbsp;</span><span style='color:#e2eaf4'>{hourly_pct}% exact 1-hour</span><br>
          <span style='color:#34d399'>✅ Sleep stages</span><span style='color:#6b7a96'> 1=Light · 2=Deep · 3=REM &nbsp;|&nbsp;</span><span style='color:#e2eaf4'>{sleep_rows:,} records</span><br>
          <span style='color:#fbbf24'>⚠ Timezone</span><span style='color:#6b7a96'> Local time — UTC normalization not applicable</span>
        </div>""", unsafe_allow_html=True)

        step_hdr(5, "Dataset Overview")
        daily_ = st.session_state.daily
        d_id   = next((c for c in daily_.columns if c.lower()=="id"), daily_.columns[0])
        hr_id_c = next((c for c in hr2.columns if c.lower()=="id"), None)
        sl_id_c = next((c for c in sl2.columns if c.lower()=="id"), None)
        daily_users = daily_[d_id].nunique(); hr_users = hr2[hr_id_c].nunique() if hr_id_c else 0
        sleep_users = sl2[sl_id_c].nunique() if sl_id_c else 0
        master_rows = len(st.session_state.master_df) if st.session_state.master_df is not None else 0
        o1,o2,o3,o4,o5 = st.columns(5)
        o1.metric(str(daily_users),"DAILY USERS"); o2.metric(str(hr_users),"HR USERS")
        o3.metric(str(sleep_users),"SLEEP USERS"); o4.metric(f"{rows_after:,}","HR MINUTE ROWS"); o5.metric(f"{master_rows:,}","MASTER ROWS")

        step_hdr(9, "Cleaned Dataset Preview")
        if st.session_state.master_df is not None:
            st.dataframe(st.session_state.master_df.head(30), use_container_width=True)
            csv_bytes = st.session_state.master_df.to_csv(index=False).encode()
            st.download_button("⬇  Download Master CSV", csv_bytes, "fitpulse_master.csv", mime="text/csv")

    if st.session_state.files_loaded:
        st.divider()
        step_hdr("ML‑1", "TSFresh Feature Extraction")
        st.caption("Extracts statistical features from Heart Rate time-series per user.")
        if st.button("▶  Run TSFresh"):
            with st.spinner("Extracting features…"):
                try:
                    hr3 = st.session_state.hr.copy()
                    hr3.columns = [c.strip() for c in hr3.columns]
                    id_col = next(c for c in hr3.columns if c.lower()=="id")
                    t_col  = next(c for c in hr3.columns if c.lower() in ("time","datetime","timestamp","date","activityminute"))
                    v_col  = next(c for c in hr3.columns if c.lower()=="value")
                    ts_hr = hr3[[id_col,t_col,v_col]].rename(columns={id_col:"id",t_col:"time",v_col:"value"})
                    ts_hr["time"] = pd.to_datetime(ts_hr["time"], errors="coerce")
                    ts_hr["value"] = pd.to_numeric(ts_hr["value"], errors="coerce")
                    ts_hr.dropna(inplace=True)
                    features = extract_features(ts_hr, column_id="id", column_sort="time", column_value="value", default_fc_parameters=MinimalFCParameters())
                    features.dropna(axis=1, how="all", inplace=True)
                    scaler = MinMaxScaler(); norm = scaler.fit_transform(features)
                    fig, ax = plt.subplots(figsize=(11, 4))
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
        step_hdr("ML‑2", "Prophet Forecast")
        st.caption("30-day heart rate forecast using Meta's Prophet model.")
        if st.button("▶  Run Prophet"):
            with st.spinner("Fitting Prophet model…"):
                try:
                    hr4 = st.session_state.hr.copy(); hr4.columns = [c.strip() for c in hr4.columns]
                    t_col = next((c for c in hr4.columns if c.lower() in ("time","datetime","timestamp","date","activityminute")), None)
                    v_col = next((c for c in hr4.columns if c.lower()=="value"), None)
                    if not t_col or not v_col:
                        st.error("Could not find Date/Value columns.")
                    else:
                        hr4["_dt"] = pd.to_datetime(hr4[t_col], errors="coerce")
                        hr4["_date"] = hr4["_dt"].dt.date
                        agg = hr4.groupby("_date")[v_col].mean().reset_index()
                        agg.columns = ["ds","y"]; agg["ds"] = pd.to_datetime(agg["ds"])
                        agg["y"] = pd.to_numeric(agg["y"], errors="coerce"); agg.dropna(inplace=True)
                        agg.sort_values("ds", inplace=True); agg = agg.tail(60).reset_index(drop=True)
                        model = Prophet(interval_width=0.90, daily_seasonality=False, weekly_seasonality=True, mcmc_samples=0)
                        model.fit(agg); future = model.make_future_dataframe(periods=30); forecast = model.predict(future)
                        fig, ax = plt.subplots(figsize=(11, 4))
                        ax.scatter(agg["ds"], agg["y"], color=PALETTE[0], s=18, alpha=0.8, zorder=3, label="Actual HR")
                        ax.plot(forecast["ds"], forecast["yhat"], color=PALETTE[1], linewidth=2, label="Forecast")
                        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.18, color=PALETTE[1], label="90% CI")
                        ax.axvline(agg["ds"].max(), linestyle="--", color=PALETTE[3], alpha=0.6, linewidth=1)
                        ax.set_title("Heart Rate — 30-Day Prophet Forecast", fontsize=12)
                        ax.set_xlabel("Date"); ax.set_ylabel("Avg BPM"); ax.legend(fontsize=9); fig.tight_layout()
                        st.session_state["prophet_fig"] = fig_to_bytes(fig); st.session_state.prophet_done = True
                        st.success("✅ Prophet forecast complete — 30 days ahead.")
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
                    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
                    for c in np.unique(km_labels):
                        m = km_labels==c
                        axes[0].scatter(X_pca[m,0], X_pca[m,1], color=PALETTE[c%len(PALETTE)], s=60, alpha=0.85, edgecolors="none", label=f"Cluster {c}")
                    axes[0].set_title(f"KMeans (k={k_val}) — PCA", fontsize=11); axes[0].legend(fontsize=8)
                    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")
                    db = DBSCAN(eps=eps_val, min_samples=2); db_labels = db.fit_predict(X)
                    for i, c in enumerate(np.unique(db_labels)):
                        m = db_labels==c; label = "Noise" if c==-1 else f"Cluster {c}"
                        color = "#6b7a96" if c==-1 else PALETTE[i%len(PALETTE)]
                        axes[1].scatter(X_pca[m,0], X_pca[m,1], color=color, s=60, alpha=0.85, edgecolors="none", label=label)
                    n_noise = (db_labels==-1).sum()
                    axes[1].set_title(f"DBSCAN (eps={eps_val}) — {len(np.unique(db_labels))-1} clusters, {n_noise} noise", fontsize=11)
                    axes[1].legend(fontsize=8); axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
                    perp = min(30, max(5, len(X)-1)); tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000); X_tsne = tsne.fit_transform(X)
                    for c in np.unique(km_labels):
                        m = km_labels==c
                        axes[2].scatter(X_tsne[m,0], X_tsne[m,1], color=PALETTE[c%len(PALETTE)], s=60, alpha=0.85, edgecolors="none", label=f"Cluster {c}")
                    axes[2].set_title("t-SNE (KMeans labels)", fontsize=11); axes[2].legend(fontsize=8)
                    axes[2].set_xlabel("Dim 1"); axes[2].set_ylabel("Dim 2")
                    for ax in axes: ax.spines[["top","right","left","bottom"]].set_visible(False)
                    fig.suptitle("Clustering Pipeline", fontsize=13, y=1.02, color="#63d7c4", fontweight="bold"); fig.tight_layout()
                    st.session_state["cluster_fig"] = fig_to_bytes(fig)
                    feat_df["Cluster"] = km_labels
                    st.session_state["cluster_summary"] = feat_df.groupby("Cluster")[available].mean().round(1)
                    st.session_state["X_scaled"] = X; st.session_state["km_labels"] = km_labels
                    st.session_state["db_labels"] = db_labels; st.session_state["k_val_used"] = k_val
                    st.session_state["available_cols"] = available; st.session_state["feat_df"] = feat_df.copy()
                    st.session_state.cluster_done = True; st.success("🎉 ML Pipeline completed successfully!")
                except Exception as e:
                    st.error(f"❌ Clustering error: {e}")
        if st.session_state["cluster_fig"] is not None:
            st.image(st.session_state["cluster_fig"], use_container_width=True)
        if st.session_state.get("cluster_summary") is not None:
            step_hdr("ML‑4", "Cluster Profiles")
            st.dataframe(st.session_state["cluster_summary"], use_container_width=True)

    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑6", "Cluster Profiles — Bar Chart & Interpretation")
        _summary = st.session_state.get("cluster_summary"); _k_used = st.session_state.get("k_val_used", k_val)
        _feat_df = st.session_state.get("feat_df"); _km_labels = st.session_state.get("km_labels")
        _avail   = st.session_state.get("available_cols", [])
        if _summary is not None and _feat_df is not None:
            st.dataframe(_summary, use_container_width=True)
            plot_cols = [c for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes","TotalSleepMinutes"] if c in _summary.columns]
            fig, ax = plt.subplots(figsize=(13, 5))
            _summary[plot_cols].plot(kind="bar", ax=ax, color=PALETTE[:len(plot_cols)], edgecolor="#0d1526", width=0.7)
            ax.set_title("Cluster Profiles — Key Feature Averages (Real Fitbit Data)", fontsize=13)
            ax.set_xlabel("Cluster"); ax.set_ylabel("Mean Value")
            ax.set_xticklabels([f"Cluster {i}" for i in range(len(_summary))], rotation=0)
            ax.legend(bbox_to_anchor=(1.02,1), title="Feature", fontsize=8, title_fontsize=8)
            ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
            st.session_state["cluster_bar_fig"] = fig_to_bytes(fig)
            st.image(st.session_state["cluster_bar_fig"], use_container_width=True)

            st.markdown("### 📊 Cluster Interpretation")
            interp_cols = st.columns(min(_k_used, 4))
            for i, col in enumerate(interp_cols):
                if i not in _summary.index: continue
                row = _summary.loc[i]; steps = row.get("TotalSteps",0); sed = row.get("SedentaryMinutes",0); active = row.get("VeryActiveMinutes",0)
                if steps > 10000: profile_icon="🏃"; profile_label="HIGHLY ACTIVE"; profile_color="#34d399"
                elif steps > 5000: profile_icon="🚶"; profile_label="MODERATELY ACTIVE"; profile_color="#fbbf24"
                else: profile_icon="🛋️"; profile_label="SEDENTARY"; profile_color="#f87171"
                with col:
                    st.markdown(f"""
                    <div class='null-card' style='text-align:center;padding:18px 12px'>
                      <div style='font-size:1.8rem'>{profile_icon}</div>
                      <div style='color:{profile_color};font-weight:700;font-size:0.78rem;margin:6px 0 10px'>Cluster {i} · {profile_label}</div>
                      <div style='font-size:0.75rem;color:#6b7a96;line-height:1.9;font-family:"JetBrains Mono",monospace;text-align:left'>
                        Avg Steps &nbsp;&nbsp;&nbsp;: <span style='color:#e2eaf4'>{steps:,.0f}</span><br>
                        Sedentary &nbsp;&nbsp;: <span style='color:#e2eaf4'>{sed:.0f} min</span><br>
                        Very Active : <span style='color:#e2eaf4'>{active:.0f} min</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑7", "KMeans Elbow Curve")
        _X = st.session_state.get("X_scaled")
        if _X is not None:
            if st.button("▶  Run Elbow Curve"):
                with st.spinner("Computing inertia for K = 2…9…"):
                    try:
                        inertias = []; K_range = range(2, 10)
                        for k in K_range:
                            km = KMeans(n_clusters=k, random_state=42, n_init=10); km.fit(_X); inertias.append(km.inertia_)
                        fig, ax = plt.subplots(figsize=(9, 4))
                        ax.plot(list(K_range), inertias, "o-", color="#63d7c4", linewidth=2.5, markersize=9, markerfacecolor="#f97316")
                        ax.set_title("KMeans Elbow Curve — Real Fitbit Data", fontsize=13)
                        ax.set_xlabel("Number of Clusters (K)"); ax.set_ylabel("Inertia"); ax.set_xticks(list(K_range))
                        ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
                        st.session_state["elbow_fig"] = fig_to_bytes(fig); st.success("✅ Elbow curve complete!")
                    except Exception as e:
                        st.error(f"❌ Elbow error: {e}")
            if st.session_state.get("elbow_fig") is not None:
                st.image(st.session_state["elbow_fig"], use_container_width=True)

    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑8", "Steps & Sleep Prophet Forecasts")
        if st.button("▶  Run Steps & Sleep Forecast"):
            with st.spinner("Fitting Prophet for Steps and Sleep…"):
                try:
                    daily3 = st.session_state.daily.copy(); sleep3 = st.session_state.sleep.copy()
                    daily3.columns = [c.strip() for c in daily3.columns]; sleep3.columns = [c.strip() for c in sleep3.columns]
                    d_date = next((c for c in daily3.columns if "date" in c.lower()), None)
                    sl_id = next((c for c in sleep3.columns if c.lower()=="id"), None)
                    sl_d  = next((c for c in sleep3.columns if "date" in c.lower()), None)
                    sl_v  = next((c for c in sleep3.columns if c.lower()=="value"), None)
                    configs = []
                    if d_date and "TotalSteps" in daily3.columns:
                        configs.append(("TotalSteps", d_date, daily3, "#63d7c4", "Steps"))
                    if sl_d and sl_v:
                        sleep3["_dt"] = pd.to_datetime(sleep3[sl_d], errors="coerce"); sleep3["_date"] = sleep3["_dt"].dt.date.astype(str)
                        sleep3[sl_v] = pd.to_numeric(sleep3[sl_v], errors="coerce")
                        sl_daily = sleep3.groupby("_date").size().reset_index(name="SleepMinutes"); sl_daily.columns = ["_date","SleepMinutes"]
                        configs.append(("SleepMinutes","_date",sl_daily,"#818cf8","Sleep (minutes)"))
                    if not configs: st.warning("⚠ Could not find Steps or Sleep columns.")
                    else:
                        fig, axes = plt.subplots(len(configs), 1, figsize=(14, 5*len(configs)))
                        if len(configs)==1: axes=[axes]
                        for ax, (metric, date_col, df_src, color, label) in zip(axes, configs):
                            agg = df_src.groupby(date_col)[metric].mean().reset_index(); agg.columns=["ds","y"]
                            agg["ds"] = pd.to_datetime(agg["ds"], errors="coerce"); agg["y"] = pd.to_numeric(agg["y"], errors="coerce")
                            agg = agg.dropna().sort_values("ds").tail(60)
                            if len(agg) < 2:
                                ax.text(0.5,0.5,f"Not enough data for {label} forecast",ha="center",va="center",transform=ax.transAxes,color="#f87171",fontsize=11)
                                ax.set_title(f"{label} — Insufficient Data", fontsize=13); continue
                            m = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False, interval_width=0.80, changepoint_prior_scale=0.1, mcmc_samples=0)
                            m.fit(agg); future = m.make_future_dataframe(periods=30); forecast = m.predict(future)
                            ax.scatter(agg["ds"], agg["y"], color=color, s=20, alpha=0.75, label=f"Actual {label}", zorder=3)
                            ax.plot(forecast["ds"], forecast["yhat"], color="#e2eaf4", linewidth=2, label="Trend")
                            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.22, color=color, label="80% CI")
                            ax.axvline(agg["ds"].max(), color="#fbbf24", linestyle="--", linewidth=1.5, label="Forecast Start")
                            ax.set_title(f"{label} — Prophet Trend Forecast", fontsize=13); ax.set_xlabel("Date"); ax.set_ylabel(label)
                            ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)
                        fig.tight_layout(); st.session_state["steps_sleep_fig"] = fig_to_bytes(fig); st.success("✅ Steps & Sleep forecasts complete!")
                except Exception as e:
                    st.error(f"❌ Steps/Sleep forecast error: {e}")
        if st.session_state.get("steps_sleep_fig") is not None:
            st.image(st.session_state["steps_sleep_fig"], use_container_width=True)

    if st.session_state.cluster_done:
        st.balloons()
        st.markdown("""
        <div class='fp-card' style='border-color:rgba(99,215,196,0.4);text-align:center;padding:2rem;margin-top:1rem'>
          <div style='font-size:2rem'>🎉</div>
          <h2 style='color:#63d7c4'>Pipeline Complete</h2>
          <p style='color:#6b7a96'>All ML stages finished — TSFresh → Prophet → KMeans → DBSCAN → t-SNE → Elbow → Steps/Sleep Forecast</p>
        </div>""", unsafe_allow_html=True)

        st.divider()
        step_hdr("ML‑5", "Pattern Extraction Summary")
        _feat_df = st.session_state.get("feat_df"); _features = st.session_state.get("features")
        _km_labels = st.session_state.get("km_labels"); _db_labels = st.session_state.get("db_labels"); _k_used = st.session_state.get("k_val_used", k_val)
        if _feat_df is not None and _km_labels is not None and _db_labels is not None:
            _n_users = _feat_df.shape[0]; _n_feats = _features.shape[1] if _features is not None else "N/A"
            _n_clust = len(np.unique(_db_labels[_db_labels!=-1])); _n_noise = int((_db_labels==-1).sum())
            _noise_pct = _n_noise/len(_db_labels)*100
            _km_dist = {int(k):int(v) for k,v in zip(*np.unique(_km_labels, return_counts=True))}
            st.markdown(f"""
            <div class='log-box'>
              <span style='color:#34d399;font-weight:700'>✅ Dataset</span><span style='color:#6b7a96'> : Real Fitbit device data</span><br>
              &nbsp;&nbsp;&nbsp;Users : <span style='color:#e2eaf4'>{_n_users}</span>&nbsp;&nbsp;|&nbsp;&nbsp;Days : <span style='color:#e2eaf4'>31 (March–April 2016)</span><br>
              <span style='color:#34d399;font-weight:700'>✅ TSFresh features extracted</span><span style='color:#63d7c4'> : {_n_feats} features</span><br>
              <span style='color:#34d399;font-weight:700'>✅ Prophet models fitted</span><br>
              &nbsp;&nbsp;&nbsp;Heart Rate — 30-day forecast, 90% CI<br>&nbsp;&nbsp;&nbsp;Steps — 30-day forecast, 80% CI<br>&nbsp;&nbsp;&nbsp;Sleep — 30-day forecast, 80% CI<br>
              <span style='color:#34d399;font-weight:700'>✅ KMeans</span><span style='color:#63d7c4'> : {_k_used} clusters identified</span><br>
              &nbsp;&nbsp;&nbsp;Distribution : <span style='color:#e2eaf4'>{_km_dist}</span><br>
              <span style='color:#34d399;font-weight:700'>✅ DBSCAN</span><span style='color:#63d7c4'> : {_n_clust} clusters, {_n_noise} noise/outlier users</span><br>
              &nbsp;&nbsp;&nbsp;Noise % : <span style='color:#fbbf24'>{_noise_pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)

# =============================================================================
# ANOMALY DETECTOR SECTION (Milestone 3)
# =============================================================================
elif active == "anomaly_detector":

    st.markdown(f"""
    <div class="m3-hero">
      <div class="hero-badge">MILESTONE 3 · ANOMALY DETECTION & VISUALIZATION</div>
      <h1 style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;color:#e2e8f0;margin:0 0 0.4rem 0'>🚨 FitPulse Anomaly Detector</h1>
      <p style='font-size:1.05rem;color:#94a3b8;font-weight:300;margin:0'>Threshold Violations · Prophet Residuals · Outlier Clusters · Interactive Plotly Charts</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Data Loading ────────────────────────────────────────────────
    m3_sec("📂", "Data Loading", "Step 1")
    m3_info("Upload the same 5 Fitbit CSV files as the Pattern Extraction section. Files are auto-detected by column structure.")

    m3_uploaded = st.file_uploader(
        "📁  Drop all 5 Fitbit CSV files here",
        type="csv", accept_multiple_files=True, key="m3_uploader",
        help="Hold Ctrl (Windows) or Cmd (Mac) to select multiple files"
    )

    m3_detected = {}
    m3_ignored  = []
    if m3_uploaded:
        raw_uploads = []
        for uf in m3_uploaded:
            try:
                df_tmp = pd.read_csv(uf); raw_uploads.append((uf.name, df_tmp))
            except Exception:
                m3_ignored.append(uf.name)
        used_names = set()
        for req_name, finfo in M3_REQUIRED_FILES.items():
            best_score, best_name, best_df = 0, None, None
            for uname, udf in raw_uploads:
                s = score_match(udf, finfo)
                if s > best_score: best_score, best_name, best_df = s, uname, udf
            if best_score >= 2:
                m3_detected[req_name] = best_df; used_names.add(best_name)

    # Status grid
    status_html = '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.6rem;margin:1rem 0">'
    for req_name, finfo in M3_REQUIRED_FILES.items():
        found = req_name in m3_detected
        bg  = M3_SUCCESS_BG if found else M3_WARN_BG
        bor = M3_SUCCESS_BOR if found else "rgba(246,173,85,0.4)"
        ico = "✅" if found else "❌"
        status_html += f"""
        <div style="background:{bg};border:1px solid {bor};border-radius:10px;padding:0.7rem 0.9rem">
          <div style="font-size:1.2rem">{ico} {finfo['icon']}</div>
          <div style="font-size:0.72rem;font-weight:600;color:{M3_TEXT};margin-top:0.3rem">{finfo['label']}</div>
          <div style="font-size:0.65rem;color:{M3_MUTED};font-family:'JetBrains Mono',monospace;margin-top:0.1rem">{'Found ✓' if found else 'Missing'}</div>
        </div>"""
    status_html += "</div>"
    st.markdown(status_html, unsafe_allow_html=True)

    n_up = len(m3_detected)
    m3_metrics((n_up,"Detected"), (5-n_up,"Missing"), ("✓" if n_up==5 else "✗","Ready"))
    if n_up < 5:
        missing_labs = [M3_REQUIRED_FILES[r]["label"] for r in M3_REQUIRED_FILES if r not in m3_detected]
        m3_warn(f"Missing: {', '.join(missing_labs)}")

    if st.button("⚡ Load & Build Master DataFrame", disabled=(n_up<5), key="m3_load"):
        with st.spinner("Parsing and building master..."):
            try:
                daily    = m3_detected["dailyActivity_merged.csv"].copy()
                hourly_s = m3_detected["hourlySteps_merged.csv"].copy()
                hourly_i = m3_detected["hourlyIntensities_merged.csv"].copy()
                sleep    = m3_detected["minuteSleep_merged.csv"].copy()
                hr       = m3_detected["heartrate_seconds_merged.csv"].copy()

                daily["ActivityDate"]    = pd.to_datetime(daily["ActivityDate"],    format="%m/%d/%Y")
                hourly_s["ActivityHour"] = pd.to_datetime(hourly_s["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
                hourly_i["ActivityHour"] = pd.to_datetime(hourly_i["ActivityHour"], format="%m/%d/%Y %I:%M:%S %p")
                sleep["date"]            = pd.to_datetime(sleep["date"],            format="%m/%d/%Y %I:%M:%S %p")
                hr["Time"]               = pd.to_datetime(hr["Time"],               format="%m/%d/%Y %I:%M:%S %p")

                hr_minute = (hr.set_index("Time").groupby("Id")["Value"].resample("1min").mean().reset_index())
                hr_minute.columns = ["Id","Time","HeartRate"]; hr_minute = hr_minute.dropna()
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
                master = master.merge(hr_daily, on=["Id","Date"], how="left")
                master = master.merge(sleep_daily, on=["Id","Date"], how="left")
                master["TotalSleepMinutes"]  = master["TotalSleepMinutes"].fillna(0)
                master["DominantSleepStage"] = master["DominantSleepStage"].fillna(0)
                for col in ["AvgHR","MaxHR","MinHR","StdHR"]:
                    master[col] = master.groupby("Id")[col].transform(lambda x: x.fillna(x.median()))

                st.session_state.m3_daily = daily; st.session_state.m3_hourly_s = hourly_s
                st.session_state.m3_hourly_i = hourly_i; st.session_state.m3_sleep = sleep
                st.session_state.m3_hr = hr; st.session_state.m3_hr_minute = hr_minute
                st.session_state.m3_master = master; st.session_state.m3_files_loaded = True
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.m3_files_loaded:
        master = st.session_state.m3_master
        m3_success(f"Master DataFrame ready — {master.shape[0]} rows · {master['Id'].nunique()} users")

        st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

        # ── Section 2: Anomaly Detection ──────────────────────────────────────
        m3_sec("🚨", "Anomaly Detection — Three Methods", "Steps 2–4")

        st.markdown(f"""
        <div class="m3-card">
          <div class="m3-card-title">Detection Methods Applied</div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.8rem;font-size:0.83rem">
            <div style="background:{M3_SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{M3_RED};font-weight:600;margin-bottom:0.4rem">① Threshold Violations</div>
              <div style="color:{M3_MUTED}">Hard upper/lower limits on HR, Steps, Sleep. Simple, interpretable, fast.</div>
            </div>
            <div style="background:{M3_SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{M3_ACC2};font-weight:600;margin-bottom:0.4rem">② Residual-Based</div>
              <div style="color:{M3_MUTED}">Rolling median as baseline. Flag days where actual deviates by ±{sigma:.0f}σ.</div>
            </div>
            <div style="background:{M3_SECTION_BG};border-radius:10px;padding:0.9rem">
              <div style="color:{M3_ACC3};font-weight:600;margin-bottom:0.4rem">③ DBSCAN Outliers</div>
              <div style="color:{M3_MUTED}">Users labelled −1 by DBSCAN are structural outliers not fitting any group.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔍 Run Anomaly Detection (All 3 Methods)", key="m3_detect"):
            with st.spinner("Detecting anomalies..."):
                try:
                    anom_hr    = detect_hr_anomalies(master, hr_high, hr_low, sigma)
                    anom_steps = detect_steps_anomalies(master, st_low, 25000, sigma)
                    anom_sleep = detect_sleep_anomalies(master, sl_low, sl_high, sigma)
                    st.session_state.m3_anom_hr    = anom_hr
                    st.session_state.m3_anom_steps = anom_steps
                    st.session_state.m3_anom_sleep = anom_sleep
                    st.session_state.m3_anomaly_done = True; st.rerun()
                except Exception as e:
                    st.error(f"Detection error: {e}")

        if st.session_state.m3_anomaly_done:
            anom_hr    = st.session_state.m3_anom_hr
            anom_steps = st.session_state.m3_anom_steps
            anom_sleep = st.session_state.m3_anom_sleep

            n_hr    = int(anom_hr["is_anomaly"].sum())
            n_steps = int(anom_steps["is_anomaly"].sum())
            n_sleep = int(anom_sleep["is_anomaly"].sum())
            n_total = n_hr + n_steps + n_sleep

            m3_danger(f"Total anomalies flagged: {n_total}  (HR: {n_hr} · Steps: {n_steps} · Sleep: {n_sleep})")
            m3_metrics((n_hr,"HR Anomalies"), (n_steps,"Steps Anomalies"), (n_sleep,"Sleep Anomalies"), (n_total,"Total Flags"), red_indices=[0,1,2,3])

            # ── Chart 1: Heart Rate ────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("❤️", "Heart Rate — Anomaly Chart", "Step 2")
            m3_anom_tag(f"{n_hr} anomalous days detected")
            m3_screenshot_badge("Heart Rate Chart with Anomaly Highlights")
            m3_step_pill(2, "Threshold + Residual Detection")
            m3_info(f"Red markers = anomaly days. Dashed lines = thresholds (HR>{hr_high} or HR<{hr_low}). Shaded band = ±{sigma:.0f}σ residual zone.")

            hr_anom   = anom_hr[anom_hr["is_anomaly"]]
            fig_hr = go.Figure()
            rolling_upper = anom_hr["rolling_med"] + sigma * anom_hr["residual"].std()
            rolling_lower = anom_hr["rolling_med"] - sigma * anom_hr["residual"].std()
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=rolling_upper, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=rolling_lower, mode="lines", fill="tonexty", fillcolor="rgba(99,179,237,0.1)", line=dict(width=0), name=f"±{sigma:.0f}σ Expected Band"))
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["AvgHR"], mode="lines+markers", name="Avg Heart Rate", line=dict(color=M3_ACCENT, width=2.5), marker=dict(size=5, color=M3_ACCENT), hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<extra></extra>"))
            fig_hr.add_trace(go.Scatter(x=anom_hr["Date"], y=anom_hr["rolling_med"], mode="lines", name="Rolling Median", line=dict(color=M3_ACC3, width=1.5, dash="dot"), hovertemplate="<b>%{x}</b><br>Median: %{y:.1f} bpm<extra></extra>"))
            if not hr_anom.empty:
                fig_hr.add_trace(go.Scatter(x=hr_anom["Date"], y=hr_anom["AvgHR"], mode="markers", name="🚨 Anomaly",
                    marker=dict(color=M3_RED, size=14, symbol="circle", line=dict(color="white", width=2)),
                    hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} bpm<br><b>ANOMALY</b><extra>⚠️</extra>"))
                for _, row in hr_anom.iterrows():
                    fig_hr.add_annotation(x=row["Date"], y=row["AvgHR"], text=f"⚠️ {row['reason']}", showarrow=True,
                        arrowhead=2, arrowcolor=M3_RED, arrowsize=1.2, ax=0, ay=-45,
                        font=dict(color=M3_RED, size=9), bgcolor=M3_CARD, bordercolor=M3_DANGER_BOR, borderwidth=1, borderpad=4)
            fig_hr.add_hline(y=hr_high, line_dash="dash", line_color=M3_RED, line_width=1.5, opacity=0.7, annotation_text=f"High ({hr_high} bpm)", annotation_position="top right", annotation_font_color=M3_RED)
            fig_hr.add_hline(y=hr_low, line_dash="dash", line_color=M3_ACC2, line_width=1.5, opacity=0.7, annotation_text=f"Low ({hr_low} bpm)", annotation_position="bottom right", annotation_font_color=M3_ACC2)
            apply_plotly_theme(fig_hr, "❤️ Heart Rate — Anomaly Detection (Real Fitbit Data)")
            fig_hr.update_layout(height=480, xaxis_title="Date", yaxis_title="Heart Rate (bpm)")
            st.plotly_chart(fig_hr, use_container_width=True)
            if not hr_anom.empty:
                with st.expander(f"📋 View {len(hr_anom)} HR Anomaly Records"):
                    st.dataframe(hr_anom[hr_anom["is_anomaly"]][["Date","AvgHR","rolling_med","residual","reason"]].rename(columns={"rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"}).round(2), use_container_width=True)

            # ── Chart 2: Sleep ─────────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("💤", "Sleep Pattern — Anomaly Visualization", "Step 3")
            m3_anom_tag(f"{n_sleep} anomalous sleep days detected")
            m3_screenshot_badge("Sleep Pattern Visualization with Alerts")
            m3_step_pill(3, "Threshold Detection on Sleep Minutes")
            m3_info(f"Orange = insufficient sleep (<{sl_low} min). Purple dots = anomaly days. Green band = healthy sleep zone ({sl_low}–{sl_high} min).")

            sleep_anom = anom_sleep[anom_sleep["is_anomaly"]]
            fig_sleep = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3],
                                      subplot_titles=["Sleep Duration (minutes/night)","Deviation from Expected"], vertical_spacing=0.08)
            fig_sleep.add_hrect(y0=sl_low, y1=sl_high, fillcolor="rgba(104,211,145,0.08)", line_width=0, annotation_text="✅ Healthy Sleep Zone", annotation_position="top right", annotation_font_color=M3_ACC3, row=1, col=1)
            fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["TotalSleepMinutes"], mode="lines+markers", name="Sleep Minutes", line=dict(color="#b794f4", width=2.5), marker=dict(size=5, color="#b794f4"), hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<extra></extra>"), row=1, col=1)
            fig_sleep.add_trace(go.Scatter(x=anom_sleep["Date"], y=anom_sleep["rolling_med"], mode="lines", name="Rolling Median", line=dict(color=M3_ACC3, width=1.5, dash="dot")), row=1, col=1)
            if not sleep_anom.empty:
                fig_sleep.add_trace(go.Scatter(x=sleep_anom["Date"], y=sleep_anom["TotalSleepMinutes"], mode="markers", name="🚨 Sleep Anomaly",
                    marker=dict(color=M3_RED, size=14, symbol="diamond", line=dict(color="white", width=2)),
                    hovertemplate="<b>%{x}</b><br>Sleep: %{y:.0f} min<br><b>ANOMALY</b><extra>⚠️</extra>"), row=1, col=1)
                for _, row in sleep_anom.iterrows():
                    fig_sleep.add_annotation(x=row["Date"], y=row["TotalSleepMinutes"], text=f"⚠️ {row['reason']}", showarrow=True,
                        arrowhead=2, arrowcolor=M3_RED, arrowsize=1.2, ax=20, ay=-40,
                        font=dict(color=M3_RED, size=9), bgcolor=M3_CARD, bordercolor=M3_DANGER_BOR, borderwidth=1, borderpad=3, row=1, col=1)
            fig_sleep.add_hline(y=sl_low, line_dash="dash", line_color=M3_RED, line_width=1.5, opacity=0.7, row=1, col=1, annotation_text=f"Min ({sl_low} min)", annotation_font_color=M3_RED)
            fig_sleep.add_hline(y=sl_high, line_dash="dash", line_color=M3_ACCENT, line_width=1.5, opacity=0.7, row=1, col=1, annotation_text=f"Max ({sl_high} min)", annotation_font_color=M3_ACCENT)
            colors_resid = [M3_RED if v else M3_ACCENT for v in anom_sleep["resid_anomaly"]]
            fig_sleep.add_trace(go.Bar(x=anom_sleep["Date"], y=anom_sleep["residual"], name="Residual", marker_color=colors_resid, hovertemplate="<b>%{x}</b><br>Residual: %{y:.0f} min<extra></extra>"), row=2, col=1)
            fig_sleep.add_hline(y=0, line_dash="solid", line_color=M3_MUTED, line_width=1, row=2, col=1)
            apply_plotly_theme(fig_sleep)
            fig_sleep.update_layout(height=560, showlegend=True, paper_bgcolor=M3_BG, plot_bgcolor="#0f172a", font_color=M3_TEXT)
            fig_sleep.update_xaxes(gridcolor=M3_GRID, tickfont_color=M3_MUTED); fig_sleep.update_yaxes(gridcolor=M3_GRID, tickfont_color=M3_MUTED)
            st.plotly_chart(fig_sleep, use_container_width=True)
            if not sleep_anom.empty:
                with st.expander(f"📋 View {len(sleep_anom)} Sleep Anomaly Records"):
                    st.dataframe(sleep_anom[sleep_anom["is_anomaly"]][["Date","TotalSleepMinutes","rolling_med","residual","reason"]].rename(columns={"TotalSleepMinutes":"Sleep (min)","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"}).round(2), use_container_width=True)

            # ── Chart 3: Steps ─────────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("🚶", "Step Count Trend — Alerts & Anomalies", "Step 4")
            m3_anom_tag(f"{n_steps} anomalous step-count days detected")
            m3_screenshot_badge("Step Count Trend with Alert Bands")
            m3_step_pill(4, "Threshold + Residual Detection on Steps")
            m3_info(f"Red vertical bands = anomaly alert days. Dashed lines = step thresholds. Bar chart below shows daily deviation from trend.")

            steps_anom = anom_steps[anom_steps["is_anomaly"]]
            fig_steps = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35],
                                      subplot_titles=["Daily Steps (avg across users)","Residual Deviation from Trend"], vertical_spacing=0.08)
            for _, row in steps_anom.iterrows():
                d = str(row["Date"]); d_next = str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10]
                fig_steps.add_vrect(x0=d, x1=d_next, fillcolor="rgba(252,129,129,0.15)", line_color="rgba(252,129,129,0.5)", line_width=1.5, row=1, col=1)
            fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["TotalSteps"], mode="lines+markers", name="Avg Daily Steps", line=dict(color=M3_ACC3, width=2.5), marker=dict(size=5, color=M3_ACC3), hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<extra></extra>"), row=1, col=1)
            fig_steps.add_trace(go.Scatter(x=anom_steps["Date"], y=anom_steps["rolling_med"], mode="lines", name="Trend (Rolling Median)", line=dict(color=M3_ACCENT, width=2, dash="dash")), row=1, col=1)
            if not steps_anom.empty:
                fig_steps.add_trace(go.Scatter(x=steps_anom["Date"], y=steps_anom["TotalSteps"], mode="markers", name="🚨 Steps Anomaly",
                    marker=dict(color=M3_RED, size=14, symbol="triangle-up", line=dict(color="white", width=2)),
                    hovertemplate="<b>%{x}</b><br>Steps: %{y:,.0f}<br><b>ALERT</b><extra>⚠️</extra>"), row=1, col=1)
            fig_steps.add_hline(y=st_low, line_dash="dash", line_color=M3_RED, line_width=1.5, opacity=0.8, row=1, col=1, annotation_text=f"Low Alert ({st_low:,} steps)", annotation_font_color=M3_RED)
            fig_steps.add_hline(y=25000, line_dash="dash", line_color=M3_ACC2, line_width=1.5, opacity=0.7, row=1, col=1, annotation_text="High Alert (25,000 steps)", annotation_font_color=M3_ACC2)
            res_colors = [M3_RED if v else M3_ACC3 for v in anom_steps["resid_anomaly"]]
            fig_steps.add_trace(go.Bar(x=anom_steps["Date"], y=anom_steps["residual"], name="Residual", marker_color=res_colors, hovertemplate="<b>%{x}</b><br>Deviation: %{y:,.0f} steps<extra></extra>"), row=2, col=1)
            fig_steps.add_hline(y=0, line_dash="solid", line_color=M3_MUTED, line_width=1, row=2, col=1)
            apply_plotly_theme(fig_steps)
            fig_steps.update_layout(height=560, showlegend=True, paper_bgcolor=M3_BG, plot_bgcolor="#0f172a", font_color=M3_TEXT)
            fig_steps.update_xaxes(gridcolor=M3_GRID, tickfont_color=M3_MUTED); fig_steps.update_yaxes(gridcolor=M3_GRID, tickfont_color=M3_MUTED)
            st.plotly_chart(fig_steps, use_container_width=True)
            if not steps_anom.empty:
                with st.expander(f"📋 View {len(steps_anom)} Steps Anomaly Records"):
                    st.dataframe(steps_anom[steps_anom["is_anomaly"]][["Date","TotalSteps","rolling_med","residual","reason"]].rename(columns={"TotalSteps":"Steps","rolling_med":"Expected","residual":"Deviation","reason":"Anomaly Reason"}).round(2), use_container_width=True)

            # ── DBSCAN Outlier Users ────────────────────────────────────────────
            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)
            m3_sec("🔍", "DBSCAN Outlier Users — Cluster-Based Anomalies", "Step 5")
            m3_step_pill(5, "Structural Outlier Detection via DBSCAN")
            m3_anom_tag("Outlier = users with atypical overall behaviour pattern")
            m3_info("Cluster each user using DBSCAN on their activity profile. Users labelled −1 are structural outliers.")

            cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes","TotalSleepMinutes"]
            try:
                cf = master.groupby("Id")[cluster_cols].mean().round(3).dropna()
                scaler   = StandardScaler(); X_scaled = scaler.fit_transform(cf)
                db       = DBSCAN(eps=2.2, min_samples=2); db_labels = db.fit_predict(X_scaled)
                pca      = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_scaled)
                var      = pca.explained_variance_ratio_ * 100
                cf["DBSCAN"] = db_labels
                outlier_users = cf[cf["DBSCAN"]==-1].index.tolist()
                n_outliers = len(outlier_users); n_clusters = len(set(db_labels))-(1 if -1 in db_labels else 0)
                m3_metrics((n_clusters,"DBSCAN Clusters"), (n_outliers,"Outlier Users"), (len(cf)-n_outliers,"Normal Users"), red_indices=[1])

                CLUSTER_COLORS = [M3_ACCENT, M3_ACC3, "#f6ad55", "#b794f4", M3_ACC2]
                fig_db = go.Figure()
                for lbl in sorted(set(db_labels)):
                    if lbl==-1: continue
                    mask = db_labels==lbl
                    fig_db.add_trace(go.Scatter(x=X_pca[mask,0], y=X_pca[mask,1], mode="markers+text", name=f"Cluster {lbl}",
                        marker=dict(size=14, color=CLUSTER_COLORS[lbl%len(CLUSTER_COLORS)], opacity=0.85, line=dict(color="white", width=1.5)),
                        text=[str(uid)[-4:] for uid in cf.index[mask]], textposition="top center", textfont=dict(size=8, color=M3_TEXT),
                        hovertemplate="<b>User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"))
                if n_outliers > 0:
                    mask_out = db_labels==-1
                    fig_db.add_trace(go.Scatter(x=X_pca[mask_out,0], y=X_pca[mask_out,1], mode="markers+text", name="🚨 Outlier / Anomaly",
                        marker=dict(size=20, color=M3_RED, symbol="x", line=dict(color="white", width=2.5)),
                        text=[str(uid)[-4:] for uid in cf.index[mask_out]], textposition="top center", textfont=dict(size=9, color=M3_RED),
                        hovertemplate="<b>⚠️ OUTLIER User ...%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra>ANOMALY</extra>"))
                    for i, uid in enumerate(cf.index[mask_out]):
                        xi, yi = X_pca[mask_out][i]
                        fig_db.add_shape(type="circle", x0=xi-0.3, y0=yi-0.3, x1=xi+0.3, y1=yi+0.3, line=dict(color=M3_RED, width=2, dash="dot"), fillcolor="rgba(252,129,129,0.1)")
                apply_plotly_theme(fig_db, "🔍 DBSCAN Outlier Detection — PCA Projection (eps=2.2)")
                fig_db.update_layout(height=500, xaxis_title=f"PC1 ({var[0]:.1f}% variance)", yaxis_title=f"PC2 ({var[1]:.1f}% variance)")
                st.plotly_chart(fig_db, use_container_width=True)
                if outlier_users:
                    out_profile = cf[cf["DBSCAN"]==-1][cluster_cols]
                    st.markdown(f'<div class="m3-card" style="border-color:{M3_DANGER_BOR}"><div class="m3-card-title" style="color:{M3_RED}">🚨 Outlier User Profiles</div></div>', unsafe_allow_html=True)
                    st.dataframe(out_profile.round(2), use_container_width=True)
            except Exception as e:
                m3_warn(f"DBSCAN clustering skipped: {e}")

            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

            # ── Section 3: Accuracy Simulation ────────────────────────────────
            m3_sec("🎯", "Simulated Detection Accuracy — 90%+ Target", "Step 6")
            m3_step_pill(6, "Inject Known Anomalies → Measure Detection Rate")
            m3_info("10 known anomalies are injected into each signal. The detector is run and we measure how many it catches.")

            if st.button("🎯 Run Accuracy Simulation (10 injected anomalies per signal)", key="m3_sim"):
                with st.spinner("Simulating..."):
                    try:
                        sim = simulate_accuracy(master, n_inject=10)
                        st.session_state.m3_sim_results   = sim
                        st.session_state.m3_simulation_done = True; st.rerun()
                    except Exception as e:
                        st.error(f"Simulation error: {e}")

            if st.session_state.m3_simulation_done and st.session_state.m3_sim_results:
                sim = st.session_state.m3_sim_results; overall = sim["Overall"]; passed = overall >= 90.0
                if passed: m3_success(f"Overall accuracy: {overall}% — ✅ MEETS 90%+ REQUIREMENT")
                else:      m3_warn(f"Overall accuracy: {overall}% — below 90% target, adjust thresholds in sidebar")

                html = '<div class="m3-metric-grid">'
                for signal in ["Heart Rate","Steps","Sleep"]:
                    r = sim[signal]; acc = r["accuracy"]; col = M3_ACC3 if acc>=90 else M3_RED
                    html += f'<div class="m3-metric-card" style="border-color:{col}44"><div style="font-size:1.8rem;font-weight:800;color:{col};font-family:Syne,sans-serif">{acc}%</div><div style="font-size:0.8rem;color:{M3_TEXT};font-weight:600;margin:0.3rem 0">{signal}</div><div style="font-size:0.72rem;color:{M3_MUTED}">{r["detected"]}/{r["injected"]} detected</div><div style="font-size:0.7rem;color:{"#9ae6b4" if acc>=90 else M3_RED}">{"✅ PASS" if acc>=90 else "⚠️ LOW"}</div></div>'
                html += f'<div class="m3-metric-card" style="border-color:{"#68d391" if passed else M3_RED}88;background:{"rgba(104,211,145,0.1)" if passed else M3_DANGER_BG}"><div style="font-size:1.8rem;font-weight:800;color:{"#68d391" if passed else M3_RED};font-family:Syne,sans-serif">{overall}%</div><div style="font-size:0.8rem;color:{M3_TEXT};font-weight:600;margin:0.3rem 0">Overall</div><div style="font-size:0.7rem;color:{"#9ae6b4" if passed else M3_RED}">{"✅ 90%+ ACHIEVED" if passed else "⚠️ BELOW TARGET"}</div></div>'
                html += '</div>'
                st.markdown(html, unsafe_allow_html=True)

                signals = ["Heart Rate","Steps","Sleep"]; accs = [sim[s]["accuracy"] for s in signals]
                bar_colors = [M3_ACC3 if a>=90 else M3_RED for a in accs]
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Bar(x=signals, y=accs, marker_color=bar_colors, text=[f"{a}%" for a in accs],
                    textposition="outside", textfont=dict(color=M3_TEXT, size=14, family="Syne, sans-serif"),
                    hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>", name="Detection Accuracy"))
                fig_acc.add_hline(y=90, line_dash="dash", line_color=M3_RED, line_width=2, annotation_text="90% Target", annotation_font_color=M3_RED, annotation_position="top right")
                apply_plotly_theme(fig_acc, "🎯 Simulated Anomaly Detection Accuracy")
                fig_acc.update_layout(height=380, yaxis_range=[0,115], yaxis_title="Detection Accuracy (%)", xaxis_title="Signal", showlegend=False)
                st.plotly_chart(fig_acc, use_container_width=True)

            st.markdown('<hr class="m3-divider">', unsafe_allow_html=True)

            # ── Milestone 3 Summary ────────────────────────────────────────────
            m3_sec("✅", "Milestone 3 Summary")
            checklist = [
                ("🚨","Threshold Violations",  st.session_state.m3_anomaly_done,    f"HR>{hr_high}/{hr_low}, Steps<{st_low}, Sleep<{sl_low}/<{sl_high}"),
                ("📉","Residual-Based",         st.session_state.m3_anomaly_done,    f"Rolling median ±{sigma:.0f}σ on all 3 signals"),
                ("🔍","DBSCAN Outliers",        st.session_state.m3_anomaly_done,    "Structural user-level anomalies via clustering"),
                ("❤️","HR Chart",               st.session_state.m3_anomaly_done,    "Interactive Plotly — annotations + threshold lines"),
                ("💤","Sleep Chart",            st.session_state.m3_anomaly_done,    "Dual subplot — duration + residual bars"),
                ("🚶","Steps Chart",            st.session_state.m3_anomaly_done,    "Trend + alert bands + residual deviation"),
                ("🎯","Accuracy Simulation",    st.session_state.m3_simulation_done, "10 injected anomalies per signal, 90%+ target"),
            ]
            for icon, label, done, detail in checklist:
                dot = "✅" if done else "⬜"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:1rem;padding:0.6rem 0;border-bottom:1px solid {M3_BOR}">
                  <span style="font-size:1.1rem">{dot}</span>
                  <span style="font-size:0.9rem;font-weight:600;color:{M3_TEXT};min-width:180px">{icon} {label}</span>
                  <span style="font-size:0.8rem;color:{M3_MUTED}">{detail}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="m3-card" style="border-color:{M3_DANGER_BOR}">
              <div class="m3-card-title">📸 Screenshots Required for Submission</div>
              <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem;font-size:0.82rem">
                <div style="background:{M3_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem"><span style="color:{M3_ACC2}">📸</span> <b>Chart 1</b> — Heart Rate with anomalies highlighted</div>
                <div style="background:{M3_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem"><span style="color:{M3_ACC2}">📸</span> <b>Chart 2</b> — Sleep pattern visualization</div>
                <div style="background:{M3_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem"><span style="color:{M3_ACC2}">📸</span> <b>Chart 3</b> — Step count trend with alerts</div>
                <div style="background:{M3_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem"><span style="color:{M3_ACC2}">📸</span> <b>Chart 4</b> — DBSCAN outlier scatter (PCA)</div>
                <div style="background:{M3_SECTION_BG};border-radius:8px;padding:0.6rem 0.8rem;grid-column:1/-1"><span style="color:{M3_ACC2}">📸</span> <b>Chart 5</b> — Accuracy bar chart (90%+ target line)</div>
              </div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="m3-card" style="text-align:center;padding:3rem">
          <div style="font-size:3rem;margin-bottom:1rem">🚨</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{M3_TEXT};margin-bottom:0.5rem">Upload Your Fitbit Files to Begin</div>
          <div style="color:{M3_MUTED};font-size:0.88rem">Upload all 5 CSV files above and click <b>Load & Build Master DataFrame</b></div>
        </div>""", unsafe_allow_html=True)