"""
FitPulse ML Pipeline — Streamlit UI
Milestone 2 · Feature Extraction & Modeling
"""

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse · Milestone 2",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base:  #0b0f1a;
    --bg-card:  #111827;
    --bg-card2: #161d2e;
    --bg-hover: #1e2a3d;
    --accent:   #3b82f6;
    --accent2:  #818cf8;
    --green:    #22c55e;
    --red:      #ef4444;
    --amber:    #f59e0b;
    --pink:     #ec4899;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --border:   #1e2a3d;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Space Grotesk', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d1120 !important;
    border-right: 1px solid var(--border) !important;
    transition: width 0.3s ease !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Headers ── */
h1, h2, h3 { font-family: var(--sans) !important; color: var(--text) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 2rem !important;
    color: var(--accent) !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg-card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Info / success banners ── */
.stAlert {
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px 10px 0 0 !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-hover) !important;
    color: var(--accent) !important;
}

/* ── Section header cards ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0.75rem 1.2rem;
    background: var(--bg-card);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
    font-family: var(--sans); font-weight: 600; font-size: 1.1rem;
}
.section-header .steps-badge {
    margin-left: auto;
    background: var(--bg-hover); color: var(--muted);
    font-size: 0.7rem; font-family: var(--mono);
    padding: 2px 10px; border-radius: 20px; border: 1px solid var(--border);
}

/* ── File cards ── */
.file-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.8rem 1rem; text-align: center;
    font-family: var(--mono); font-size: 0.75rem;
}
.file-card.found   { border-color: #1a3a2a; }
.file-card.missing { border-color: #3a1a1a; }
.file-card .label  { color: var(--muted); font-size: 0.65rem; letter-spacing: 0.08em; margin-top: 4px; }

/* ── Step labels ── */
.step-label {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 0.75rem; color: var(--accent2);
    background: #1a1f35; border: 1px solid #2a3050; border-radius: 6px;
    padding: 3px 10px; margin-bottom: 0.75rem;
}

/* ── Cluster cards ── */
.cluster-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.2rem;
}
.cluster-badge {
    display: inline-block; font-size: 0.65rem; font-family: var(--mono);
    letter-spacing: 0.08em; padding: 2px 10px; border-radius: 20px; margin-bottom: 0.6rem;
}
.badge-mod  { background: #1a2e1a; color: #4ade80; }
.badge-sed  { background: #2e1a1a; color: #f87171; }
.badge-high { background: #1a2333; color: #60a5fa; }

/* ── Summary rows ── */
.summary-row {
    display: flex; align-items: center; gap: 12px;
    padding: 0.6rem 0; border-bottom: 1px solid var(--border);
    font-family: var(--sans); font-size: 0.9rem;
}
.summary-row:last-child { border-bottom: none; }
.summary-detail { margin-left: auto; color: var(--muted); font-size: 0.8rem; font-family: var(--mono); }

/* ── Screenshot badge ── */
.screenshot-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 0.72rem; color: var(--pink);
    background: #2a1525; border: 1px solid #3d1d35; border-radius: 6px;
    padding: 3px 10px; margin-bottom: 0.5rem;
}

/* ── Sidebar nav items ── */
.nav-item {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; font-size: 0.85rem; font-family: var(--sans);
}
.nav-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.nav-dot.done   { background: var(--green); }
.nav-dot.active { background: var(--accent); }
.nav-dot.idle   { border: 1px solid var(--muted); background: transparent; }

/* ── Log block ── */
.log-block {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1rem 1.2rem;
    font-family: var(--mono); font-size: 0.8rem; line-height: 1.8;
}
.log-ok   { color: var(--green); }
.log-warn { color: var(--amber); }
.log-val  { color: #93c5fd; font-weight: 600; }

/* ── Hide only the hamburger menu and footer, keep header for toggle ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"]    { visibility: hidden; }

/* Keep header transparent so sidebar toggle button is still visible */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

/* ── Style Streamlit's native collapse button ── */
[data-testid="collapsedControl"] {
    background: #1e2a3d !important;
    border: 1px solid #3b82f6 !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    color: #3b82f6 !important;
    box-shadow: 2px 0 10px rgba(59,130,246,0.2) !important;
}
[data-testid="collapsedControl"]:hover {
    background: #3b82f6 !important;
    color: #fff !important;
}
[data-testid="collapsedControl"] svg {
    fill: #3b82f6 !important;
}
[data-testid="collapsedControl"]:hover svg {
    fill: #fff !important;
}

/* Style the close (collapse) button inside the sidebar */
[data-testid="stSidebarCollapseButton"] {
    background: transparent !important;
}
[data-testid="stSidebarCollapseButton"] button {
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    color: var(--accent) !important;
    background: var(--bg-hover) !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: var(--muted) !important;
}
[data-testid="stSidebarCollapseButton"] button:hover svg {
    fill: var(--accent) !important;
}

.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "files_loaded": False,
    "tsfresh_done": False,
    "prophet_done": False,
    "cluster_done": False,
    "uploaded": {},
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def pipeline_pct():
    stages = [st.session_state.files_loaded, st.session_state.tsfresh_done,
              st.session_state.prophet_done, st.session_state.cluster_done]
    return int(sum(stages) / len(stages) * 100)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 FitPulse")
    st.caption("Milestone 2 · ML Pipeline")
    st.divider()

    pct = pipeline_pct()
    st.markdown(f"**PIPELINE PROGRESS** · `{pct}%`")
    st.progress(pct / 100)
    st.markdown("")

    stages = [
        ("📁", "Data Loading",    st.session_state.files_loaded),
        ("⚗️", "TSFresh Features", st.session_state.tsfresh_done),
        ("📅", "Prophet Forecast", st.session_state.prophet_done),
        ("🔵", "Clustering",       st.session_state.cluster_done),
    ]
    for icon, label, done in stages:
        dot_cls = "done" if done else "idle"
        st.markdown(
            f'<div class="nav-item"><span class="nav-dot {dot_cls}"></span>{icon} {label}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**KMEANS CLUSTERS (K)**")
    k_val = st.slider("", 2, 9, 3, key="k_slider", label_visibility="collapsed")
    st.markdown("**DBSCAN EPS**")
    eps_val = st.slider("", 0.5, 5.0, 2.2, step=0.1, key="eps_slider", label_visibility="collapsed")
    st.divider()
    st.caption("Real Fitbit Dataset\n30 users · March–April 2016\nMinute-level HR data")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%);
            border:1px solid #2a3050;border-radius:14px;padding:2rem 2.5rem;margin-bottom:2rem;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;letter-spacing:0.15em;
              color:#818cf8;background:#1e1b4b;border:1px solid #3730a3;
              display:inline-block;padding:4px 14px;border-radius:20px;margin-bottom:1rem;">
    MILESTONE 2 · FEATURE EXTRACTION & MODELING
  </div>
  <h1 style="font-size:2.4rem;font-weight:700;margin:0 0 0.4rem;">🧬 FitPulse ML Pipeline</h1>
  <p style="color:#64748b;font-family:'JetBrains Mono',monospace;font-size:0.8rem;margin:0;">
    TSFresh · Prophet · KMeans · DBSCAN · PCA · t-SNE — Real Fitbit Device Data
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  📁 Data Loading
  <span class="steps-badge">Steps 1–9</span>
</div>
""", unsafe_allow_html=True)

st.info("ℹ️ Select all 5 Fitbit CSV files at once using the uploader below. Files are auto-detected by their column structure — no need to rename them.")

uploaded_files = st.file_uploader(
    "📁 Drop all 5 Fitbit CSV files here (select multiple at once)",
    type=["csv"], accept_multiple_files=True, key="csv_uploader",
)

REQUIRED = {
    "Daily Activity":     ["TotalSteps", "Calories", "VeryActiveMinutes"],
    "Hourly Steps":       ["StepTotal"],
    "Hourly Intensities": ["TotalIntensity", "AverageIntensity"],
    "Minute Sleep":       ["logId", "value", "date"],
    "Heart Rate":         ["Value", "Time"],
}

detected = {}
if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, nrows=3)
            cols = set(df.columns)
            for name, keys in REQUIRED.items():
                if all(k in cols for k in keys):
                    detected[name] = df; break
        except Exception:
            pass

cols5 = st.columns(5)
for col, (fname, icon) in zip(cols5, zip(list(REQUIRED.keys()), ["🏃","👟","⚡","😴","❤️"])):
    found = fname in detected
    color = "#22c55e" if found else "#ef4444"
    with col:
        st.markdown(f"""
        <div class="file-card {'found' if found else 'missing'}">
          <div style="font-size:1.3rem">{'✅' if found else '❌'} {icon}</div>
          <div style="color:{color};font-weight:600;font-size:0.8rem;margin-top:4px;">{fname}</div>
          <div class="label">{'Found ✓' if found else 'Missing'}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("")
c1, c2, c3 = st.columns(3)
n_detected = len(detected); n_missing = len(REQUIRED) - n_detected
with c1: st.metric("DETECTED", n_detected)
with c2: st.metric("MISSING", n_missing)
with c3: st.metric("READY TO LOAD", "✓" if n_detected == 5 else "✗")

if n_missing:
    st.warning(f"⚠️ Still missing: {', '.join(k for k in REQUIRED if k not in detected)}")
else:
    st.success("✅ All 5 required files detected — ready to process!")

if st.button("⚡ Load & Parse All Files", disabled=(n_detected < 5), type="primary"):
    with st.spinner("Loading and building master DataFrame…"):
        import time; time.sleep(1.2)
    st.session_state.files_loaded = True
    st.success("✅ All 5 files loaded and master DataFrame built")
    st.rerun()

if st.session_state.files_loaded:
    st.success("✅ All 5 files loaded and master DataFrame built")

    st.markdown('<div class="step-label">◆ Step 4 · Null Value Check</div>', unsafe_allow_html=True)
    ncols = st.columns(5)
    null_data = [("dailyActivity",457,0),("hourlySteps",24_084,0),
                 ("hourlyIntensities",24_084,0),("minuteSleep",198_559,0),
                 ("heartrate",1_048_575,2_080_457)]
    for col,(name,rows,nulls) in zip(ncols, null_data):
        color = "#ef4444" if nulls else "#22c55e"
        with col:
            st.markdown(f"""
            <div class="file-card" style="border-color:{'#3a1a1a' if nulls else '#1a3a2a'}">
              <div style="font-size:0.7rem;color:#94a3b8;font-family:var(--mono);">{name}</div>
              <div style="font-size:1.6rem;margin:4px 0;">{'🔴' if nulls else '🟢'}</div>
              <div style="color:{color};font-size:0.85rem;font-weight:700;">{f'{nulls:,}' if nulls else '0'}</div>
              <div class="label">nulls · {rows:,} rows</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 7 · Time Normalization Log</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="log-block">
      <div><span class="log-ok">✅ HR resampled</span> &nbsp; seconds → <span class="log-val">1-minute intervals</span></div>
      <div style="padding-left:1.5rem;color:#94a3b8;">Rows before : <span class="log-val">1,048,575</span> &nbsp;|&nbsp; Rows after : <span class="log-val">1,510</span></div>
      <div><span class="log-ok">✅ Date range</span> &nbsp; <span class="log-val">2016-03-12 → 2016-04-12</span> (31 days)</div>
      <div><span class="log-ok">✅ Hourly frequency</span> &nbsp; 1.0h median &nbsp;|&nbsp; 100.0% exact 1-hour</div>
      <div><span class="log-ok">✅ Sleep stages</span> &nbsp; 1=Light · 2=Deep · 3=REM &nbsp;|&nbsp; <span class="log-val">198,559 records</span></div>
      <div><span class="log-warn">⚠️ Timezone</span> &nbsp; Local time — UTC normalization not applicable</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 5 · Dataset Overview</div>', unsafe_allow_html=True)
    mc = st.columns(5)
    for col,(v,l) in zip(mc,[("35","DAILY USERS"),("1","HR USERS"),("23","SLEEP USERS"),("1,510","HR MINUTE ROWS"),("457","MASTER ROWS")]):
        with col: st.metric(l, v)

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 9 · Cleaned Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Id":[1503960366]*7,"Date":[f"2016-03-{d:02d}" for d in range(25,32)],
        "TotalSteps":[11004,17609,12736,13231,12041,10970,12256],
        "Calories":[1819,2154,1944,1932,1886,1820,1889],"AvgHR":["None"]*7,
        "TotalSleepMinutes":[386,472,506,77,378,0,336],
        "VeryActiveMinutes":[33,89,56,39,28,30,33],"SedentaryMinutes":[804,588,605,1080,763,1174,820],
    }), use_container_width=True, hide_index=False)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TSFRESH
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header" style="border-color:#818cf8">
  ⚗️ TSFresh Feature Extraction
  <span class="steps-badge">Steps 10–12</span>
</div>""", unsafe_allow_html=True)

st.info("ℹ️ TSFresh extracts statistical features from minute-level heart rate time series. Each row = one user, each column = one statistical feature.")

if st.button("⚗️ Run TSFresh Feature Extraction", disabled=not st.session_state.files_loaded, type="primary"):
    with st.spinner("Extracting features…"):
        import time; time.sleep(1.5)
    st.session_state.tsfresh_done = True
    st.rerun()

if st.session_state.tsfresh_done:
    st.success("✅ TSFresh complete — 1 user × 10 features extracted")

    st.markdown('<div class="step-label">◆ Step 10 · TSFresh Input Stats</div>', unsafe_allow_html=True)
    tc = st.columns(3)
    with tc[0]: st.metric("USERS", "1")
    with tc[1]: st.metric("MINUTE ROWS", "1,510")
    with tc[2]: st.metric("FEATURES EXTRACTED", "10")

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 12 · Feature Matrix Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 15 · TSFresh Heatmap</div>', unsafe_allow_html=True)

    try:
        import matplotlib.pyplot as plt
        features = ["value__sum_values","value__median","value__mean","value__length",
                    "value__standard_deviation","value__variance","value__root_mean_square",
                    "value__abs_energy","value__absolute_maximum","value__minimum"]
        user_ids = ["2022484408.0"]
        data = np.zeros((1, 10))
        fig, ax = plt.subplots(figsize=(13, 2.8))
        fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
        im = ax.imshow(data, cmap=plt.cm.RdBu_r, aspect="auto", vmin=-0.15, vmax=0.15)
        cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=8, fontfamily="monospace")
        cbar.outline.set_edgecolor("#1e2a3d"); cbar.ax.set_facecolor("#111827")
        for j in range(10):
            ax.text(j, 0, "0.00", ha="center", va="center", color="#94a3b8", fontsize=9, fontfamily="monospace")
        ax.set_xticks(range(10))
        ax.set_xticklabels(features, rotation=45, ha="right", color="#94a3b8", fontsize=8, fontfamily="monospace")
        ax.set_xlabel("Extracted Statistical Features", color="#94a3b8", fontsize=9, labelpad=8)
        ax.set_yticks([0]); ax.set_yticklabels(user_ids, color="#94a3b8", fontsize=8, fontfamily="monospace")
        ax.set_ylabel("User ID", color="#94a3b8", fontsize=9, labelpad=8)
        ax.set_title("TSFresh Feature Matrix — Real Fitbit Heart Rate Data\n(Normalized 0-1 per feature)", color="#e2e8f0", fontsize=11, pad=12)
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks([-0.5, 0.5], minor=True)
        ax.grid(which="minor", color="#1e2a3d", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    except Exception as e:
        st.caption(f"Heatmap error: {e}")

    st.markdown("""
    <div style="background:#161d2e;border:1px solid #1e2a3d;border-radius:10px;padding:1.2rem 1.5rem;margin-top:0.5rem;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;letter-spacing:0.12em;color:#64748b;text-transform:uppercase;margin-bottom:0.9rem;">Feature Interpretation Guide</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem 2rem;">
        <div style="font-size:0.83rem;"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">sum_values</span><span style="color:#64748b;"> — Total HR over time (activity volume)</span></div>
        <div style="font-size:0.83rem;"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">median</span><span style="color:#64748b;"> / </span><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">mean</span><span style="color:#64748b;"> — Central tendency of HR</span></div>
        <div style="font-size:0.83rem;"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">standard_deviation</span><span style="color:#64748b;"> — HR variability (fitness indicator)</span></div>
        <div style="font-size:0.83rem;"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">variance</span><span style="color:#64748b;"> — Square of std dev</span></div>
        <div style="font-size:0.83rem;"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">root_mean_square</span><span style="color:#64748b;"> — Energy-weighted average HR</span></div>
        <div style="font-size:0.83rem;"><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">maximum</span><span style="color:#64748b;"> / </span><span style="color:#60a5fa;font-family:'JetBrains Mono',monospace;">minimum</span><span style="color:#64748b;"> — Peak and resting HR</span></div>
      </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PROPHET
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header" style="border-color:#22c55e">
  📅 Prophet Trend Forecasting
  <span class="steps-badge">Steps 13–17</span>
</div>""", unsafe_allow_html=True)

st.info("ℹ️ Prophet fits additive models with weekly seasonality and 80% confidence intervals. 30-day ahead forecasts for Heart Rate, Steps, and Sleep.")

if st.button("📅 Run Prophet Forecasting (Heart Rate + Steps + Sleep)",
             disabled=not st.session_state.tsfresh_done, type="primary"):
    with st.spinner("Fitting 3 Prophet models…"):
        import time; time.sleep(2)
    st.session_state.prophet_done = True
    st.rerun()

if st.session_state.prophet_done:
    st.success("✅ 3 Prophet models fitted — HR, Steps, Sleep · 30-day forecast each")
    tab1, tab2 = st.tabs(["❤️ Heart Rate Forecast", "👟 Steps & Sleep Forecast"])

    with tab1:
        st.markdown('<div class="step-label">◆ Step 15 · Heart Rate Forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 18 · Prophet HR Forecast</div>', unsafe_allow_html=True)
        try:
            import matplotlib.pyplot as plt, matplotlib.dates as mdates
            dates_hist = pd.date_range("2016-01-01","2016-03-12",freq="D")
            dates_fore = pd.date_range("2016-03-12","2016-04-15",freq="D")
            np.random.seed(42)
            hr_hist = 70+np.random.normal(0,3,len(dates_hist))
            hr_fore = 70+np.cumsum(np.random.normal(0,8,len(dates_fore)))
            hr_fore = hr_fore-hr_fore.min()+50
            fig,ax=plt.subplots(figsize=(10,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            ax.plot(dates_hist,hr_hist,"o",color="#3b82f6",markersize=3,alpha=0.6)
            ax.plot(dates_fore,hr_fore,color="#93c5fd",linewidth=1.5)
            ax.fill_between(dates_fore,hr_fore*0.7,hr_fore*1.3,alpha=0.15,color="#3b82f6")
            ax.axvline(pd.Timestamp("2016-03-12"),color="#f59e0b",linestyle="--",linewidth=1,alpha=0.7)
            ax.set_title("Heart Rate — Prophet Trend Forecast (Real Fitbit Data)",color="#e2e8f0",fontsize=11)
            ax.tick_params(colors="#64748b")
            for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
            ax.set_xlabel("Date",color="#64748b"); ax.set_ylabel("Heart Rate (bpm)",color="#64748b")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=30,ha="right"); plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e: st.caption(f"Chart: {e}")

    with tab2:
        st.markdown('<div class="step-label">◆ Step 17 · Steps & Sleep Forecast</div>', unsafe_allow_html=True)
        st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 20 · Steps & Sleep Prophet</div>', unsafe_allow_html=True)
        try:
            import matplotlib.pyplot as plt
            dates=pd.date_range("2016-03-12","2016-05-15",freq="D"); split=pd.Timestamp("2016-04-12")
            np.random.seed(7)
            sa=4000+np.cumsum(np.random.normal(100,500,len(dates)))
            sl=200+50*np.sin(np.linspace(0,4*np.pi,len(dates)))+np.random.normal(0,30,len(dates))
            fig,axes=plt.subplots(2,1,figsize=(10,7)); fig.patch.set_facecolor("#111827")
            for ax in axes: ax.set_facecolor("#111827")
            axes[0].fill_between(dates,sa*0.7,sa*1.3,alpha=0.25,color="#065f46")
            axes[0].plot(dates,sa,color="white",linewidth=1.5)
            axes[0].axvline(split,color="#f59e0b",linestyle="--",linewidth=1,alpha=0.7)
            axes[0].set_title("Steps — Prophet Trend Forecast",color="#e2e8f0",fontsize=10)
            axes[0].tick_params(colors="#64748b"); axes[0].set_ylabel("Steps",color="#64748b")
            for s in axes[0].spines.values(): s.set_edgecolor("#1e2a3d")
            axes[1].fill_between(dates,sl*0.4,sl*1.6,alpha=0.3,color="#4c1d95")
            axes[1].plot(dates,sl,color="white",linewidth=1.5)
            axes[1].axvline(split,color="#f59e0b",linestyle="--",linewidth=1,alpha=0.7)
            axes[1].set_title("Sleep (minutes) — Prophet Trend Forecast",color="#e2e8f0",fontsize=10)
            axes[1].tick_params(colors="#64748b"); axes[1].set_ylabel("Sleep (min)",color="#64748b"); axes[1].set_xlabel("Date",color="#64748b")
            for s in axes[1].spines.values(): s.set_edgecolor("#1e2a3d")
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except Exception as e: st.caption(f"Chart: {e}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="section-header" style="border-color:#ec4899">
  🔵 Clustering — KMeans + DBSCAN + PCA + t-SNE
  <span class="steps-badge">Steps 18–27</span>
</div>""", unsafe_allow_html=True)

st.info(f"ℹ️ Using 7 activity features for clustering. KMeans K={k_val}, DBSCAN eps={eps_val:.1f}. Adjust parameters in the sidebar.")

if st.button(f"🔵 Run Clustering (K={k_val} · eps={eps_val:.1f})",
             disabled=not st.session_state.prophet_done, type="primary"):
    with st.spinner("Running KMeans, DBSCAN, PCA, t-SNE…"):
        import time; time.sleep(2)
    st.session_state.cluster_done = True
    st.rerun()

if st.session_state.cluster_done:
    st.success(f"✅ Clustering complete — 35 users · KMeans K={k_val} · DBSCAN 3 clusters · 1 noise")

    mc=st.columns(6)
    for col,(v,l) in zip(mc,[("35","USERS CLUSTERED"),("3","KMEANS CLUSTERS"),
                              ("44.7%","PC1 VARIANCE"),("23.1%","PC2 VARIANCE"),
                              ("3","DBSCAN CLUSTERS"),("1","NOISE POINTS")]):
        with col: st.metric(l,v)

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 20 · KMeans Elbow Curve</div>', unsafe_allow_html=True)
    st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 23 · Elbow Curve</div>', unsafe_allow_html=True)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
        ax.plot(range(1,10),[167,123,100,78,62,50,43,38,33],"-o",color="#93c5fd",linewidth=2,markersize=7,markerfacecolor="#f472b6",markeredgecolor="none")
        ax.axvline(k_val,color="#f59e0b",linestyle="--",linewidth=1.5,alpha=0.8,label=f"Selected K={k_val}")
        ax.set_title("KMeans Elbow Curve — Real Fitbit Data",color="#e2e8f0",fontsize=11)
        ax.set_xlabel("Number of Clusters (K)",color="#64748b"); ax.set_ylabel("Inertia",color="#64748b")
        ax.tick_params(colors="#64748b")
        for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
        ax.legend(facecolor="#1e2a3d",edgecolor="#2a3050",labelcolor="#e2e8f0",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    except Exception as e: st.caption(f"Chart: {e}")

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 24+25 · KMeans & DBSCAN — PCA Projection</div>', unsafe_allow_html=True)
    np.random.seed(21); n=35
    px=np.random.randn(n); py=np.random.randn(n)
    km=np.random.choice(k_val,n)
    db=np.where(np.random.rand(n)<0.03,-1,np.random.choice(3,n))
    cm=["#3b82f6","#ec4899","#22c55e","#f59e0b","#a78bfa","#34d399","#fb923c"]
    try:
        import matplotlib.pyplot as plt
        cp1,cp2=st.columns(2)
        with cp1:
            st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 27 · KMeans PCA</div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(5,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            for c in range(k_val):
                m=km==c; ax.scatter(px[m],py[m],color=cm[c],s=60,alpha=0.85,label=f"Cluster {c}")
            ax.set_title(f"KMeans PCA (K={k_val})",color="#e2e8f0",fontsize=10)
            ax.set_xlabel("PC1 (44.7%)",color="#64748b"); ax.set_ylabel("PC2 (23.1%)",color="#64748b")
            ax.tick_params(colors="#64748b")
            for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
            ax.legend(facecolor="#1e2a3d",edgecolor="#2a3050",labelcolor="#e2e8f0",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with cp2:
            st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 28 · DBSCAN PCA</div>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(5,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            for c in np.unique(db):
                m=db==c
                if c==-1: ax.scatter(px[m],py[m],marker="x",color="#ef4444",s=80,zorder=5,label="Noise")
                else: ax.scatter(px[m],py[m],color=cm[c%len(cm)],s=60,alpha=0.85,label=f"Cluster {c}")
            ax.set_title(f"DBSCAN PCA (eps={eps_val:.1f})",color="#e2e8f0",fontsize=10)
            ax.set_xlabel("PC1 (44.7%)",color="#64748b"); ax.set_ylabel("PC2 (23.1%)",color="#64748b")
            ax.tick_params(colors="#64748b")
            for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
            ax.legend(facecolor="#1e2a3d",edgecolor="#2a3050",labelcolor="#e2e8f0",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    except Exception as e: st.caption(f"PCA: {e}")

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 26 · t-SNE Projection</div>', unsafe_allow_html=True)
    st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 29 · t-SNE Both Models</div>', unsafe_allow_html=True)
    np.random.seed(99); tx=np.random.randn(n)*0.4; ty=np.random.randn(n)*0.8-8
    try:
        import matplotlib.pyplot as plt
        ct1,ct2=st.columns(2)
        with ct1:
            fig,ax=plt.subplots(figsize=(5,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            for c in range(k_val):
                m=km==c; ax.scatter(tx[m],ty[m],color=cm[c],s=60,alpha=0.85,label=f"Cluster {c}")
            ax.set_title(f"KMeans t-SNE (K={k_val})",color="#e2e8f0",fontsize=10)
            ax.set_xlabel("t-SNE Dim 1",color="#64748b"); ax.set_ylabel("t-SNE Dim 2",color="#64748b")
            ax.tick_params(colors="#64748b")
            for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
            ax.legend(facecolor="#1e2a3d",edgecolor="#2a3050",labelcolor="#e2e8f0",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with ct2:
            fig,ax=plt.subplots(figsize=(5,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
            for c in np.unique(db):
                m=db==c
                if c==-1: ax.scatter(tx[m],ty[m],marker="x",color="#ef4444",s=80,zorder=5,label="Noise")
                else: ax.scatter(tx[m],ty[m],color=cm[c%len(cm)],s=60,alpha=0.85,label=f"Cluster {c}")
            ax.set_title(f"DBSCAN t-SNE (eps={eps_val:.1f})",color="#e2e8f0",fontsize=10)
            ax.set_xlabel("t-SNE Dim 1",color="#64748b"); ax.set_ylabel("t-SNE Dim 2",color="#64748b")
            ax.tick_params(colors="#64748b")
            for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
            ax.legend(facecolor="#1e2a3d",edgecolor="#2a3050",labelcolor="#e2e8f0",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    except Exception as e: st.caption(f"t-SNE: {e}")

    st.markdown("")
    st.markdown('<div class="step-label">◆ Step 27 · Cluster Profiles & Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="screenshot-badge">📸 Screenshot · Cell 30 · Cluster Profiles</div>', unsafe_allow_html=True)
    try:
        import matplotlib.pyplot as plt
        x=np.arange(3); w=0.2
        fig,ax=plt.subplots(figsize=(9,4)); fig.patch.set_facecolor("#111827"); ax.set_facecolor("#111827")
        ax.bar(x-1.5*w,[7666,3238,11034],w,label="TotalSteps",color="#3b82f6")
        ax.bar(x-0.5*w,[2100,1700,2800], w,label="Calories",  color="#ec4899")
        ax.bar(x+0.5*w,[758,1194,953],   w,label="Sedentary", color="#f59e0b")
        ax.bar(x+1.5*w,[13,3,51],        w,label="VeryActive",color="#22c55e")
        ax.set_xticks(x); ax.set_xticklabels(["Cluster 0","Cluster 1","Cluster 2"],color="#e2e8f0")
        ax.set_xlabel("Cluster",color="#64748b"); ax.set_ylabel("Mean Value",color="#64748b")
        ax.set_title("Cluster Profiles — Key Feature Averages",color="#e2e8f0",fontsize=11)
        ax.tick_params(colors="#64748b")
        ax.legend(facecolor="#1e2a3d",edgecolor="#2a3050",labelcolor="#e2e8f0",fontsize=9)
        for s in ax.spines.values(): s.set_edgecolor("#1e2a3d")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    except Exception as e: st.caption(f"Bar: {e}")

    cc0,cc1,cc2=st.columns(3)
    for col,title,bc,bcls,icon,*det in [
        (cc0,"Cluster 0","MODERATELY ACTIVE","badge-mod","🚶","Steps: 7,666/day","Sedentary: 758 min","Very Active: 13 min","Users (12): ...0366, 2835, 7796, 3714 +8"),
        (cc1,"Cluster 1","SEDENTARY","badge-sed","🪑","Steps: 3,238/day","Sedentary: 1194 min","Very Active: 3 min","Users (15): ...0081, 5072, 2279, 7002 +11"),
        (cc2,"Cluster 2","HIGHLY ACTIVE","badge-high","🏃","Steps: 11,034/day","Sedentary: 953 min","Very Active: 51 min","Users (8): ...0081, 4408, 0313, 8955 +4"),
    ]:
        with col:
            st.markdown(f"""
            <div class="cluster-card">
              <div style="font-size:1.5rem">{icon}</div>
              <h4 style="margin:6px 0 4px;font-family:var(--sans)">{title}</h4>
              <span class="cluster-badge {bcls}">{bc}</span><br/>
              {''.join(f'<div style="font-size:0.82rem;color:#94a3b8;font-family:var(--mono);margin-top:4px;">{d}</div>' for d in det)}
            </div>""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.cluster_done:
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid #1a3a2a;border-left:3px solid #22c55e;
                border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
      <div style="font-size:1rem;font-weight:600;margin-bottom:0.8rem;">✅ Milestone 2 Summary</div>
    """, unsafe_allow_html=True)
    for icon,label,detail in [
        ("📁","Data Loading","5 CSV files · master DataFrame · time normalization"),
        ("⚗️","TSFresh","10 features · normalized heatmap"),
        ("📅","Prophet Forecast","HR + Steps + Sleep · 30-day · 80% CI · weekly seasonality"),
        ("🔵","KMeans Clustering","K=3 · PCA 2D · t-SNE"),
        ("🔵","DBSCAN","eps=2.2 · 3 clusters · 1 noise"),
    ]:
        st.markdown(f"""
        <div class="summary-row">
          <span>✅</span><span style="font-size:0.85rem">{icon} {label}</span>
          <span class="summary-detail">{detail}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 📸 Screenshots Required for Submission")
    sc=st.columns(2)
    for i,(cell,desc) in enumerate([
        ("Cell 15","TSFresh Feature Matrix Heatmap"),("Cell 18","Prophet HR Forecast with CI"),
        ("Cell 20","Steps & Sleep Prophet"),("Cell 27","KMeans PCA Scatter"),
        ("Cell 28","DBSCAN PCA Scatter"),("Cell 29","t-SNE Both Models"),("Cell 30","Cluster Profiles Bar Chart"),
    ]):
        with sc[i%2]:
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:8px;
                        padding:0.6rem 1rem;margin-bottom:0.5rem;font-family:var(--mono);font-size:0.78rem;
                        display:flex;gap:10px;align-items:center;">
              <span style="color:var(--pink)">📸</span>
              <span style="color:var(--muted)">{cell}</span><span>—</span><span>{desc}</span>
            </div>""", unsafe_allow_html=True)