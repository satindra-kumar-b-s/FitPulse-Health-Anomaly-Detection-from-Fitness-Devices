import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

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

/* NAV BUTTON STYLES */
.nav-btn-active {
  display:block; width:100%; padding:12px 16px;
  background:rgba(99,215,196,0.12);
  border:1px solid rgba(99,215,196,0.35);
  border-radius:12px; margin-bottom:8px;
  color:#e2eaf4 !important; font-weight:700;
  font-family:'Syne',sans-serif;
  font-size:0.9rem; cursor:pointer;
  text-align:left;
}
.nav-btn-inactive {
  display:block; width:100%; padding:12px 16px;
  background:rgba(255,255,255,0.03);
  border:1px solid rgba(99,215,196,0.1);
  border-radius:12px; margin-bottom:8px;
  color:#6b7a96 !important; font-weight:600;
  font-family:'Syne',sans-serif;
  font-size:0.9rem; cursor:pointer;
  text-align:left;
}
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

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
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

# Active section: "preprocessing" or "pattern_extraction"
if "active_section" not in st.session_state:
    st.session_state["active_section"] = "preprocessing"

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:4px 0 6px 0'>
      <span style='font-size:1.4rem'>🩺</span>
      <div>
        <div style='font-size:1.05rem;font-weight:800;color:#e2eaf4'>FitPulse</div>
        <div style='font-size:0.7rem;color:#6b7a96'>Fitness ML Analytics Pipeline</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Navigation Buttons ──
    st.markdown(
        "<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;"
        "letter-spacing:1px;font-weight:700;margin-bottom:10px'>📌 Pipeline Sections</div>",
        unsafe_allow_html=True
    )

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

        # Preprocessing status
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
            st.markdown(
                f"<div style='font-size:0.8rem;color:{color};padding:3px 0'>"
                f"{icon} {label}</div>",
                unsafe_allow_html=True
            )
        sidebar_status(m1_loaded, "Dataset Loaded")
        sidebar_status(m1_clean,  "Data Cleaned")

    else:
        # Pattern Extraction sidebar
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
            st.markdown(
                f"<div class='nav-item'><div class='{d}'></div>"
                f"<span>{icon}</span><span class='{l}'>{label}</span></div>",
                unsafe_allow_html=True
            )

        nav(fl, "📁", "Data Loading")
        nav(tf, "⚗️", "TSFresh Features")
        nav(pf, "📅", "Prophet Forecast")
        nav(cl, "🔵", "Clustering")
        nav(ss, "📈", "Steps & Sleep Forecast")

        st.divider()

        st.markdown(
            "<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;"
            "letter-spacing:1px;font-weight:700'>⚙️ ML Controls</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;"
            "letter-spacing:1px;font-weight:700;margin-top:10px'>KMeans Clusters</div>",
            unsafe_allow_html=True
        )
        k_val = st.slider("k", 2, 9, 3, label_visibility="collapsed")

        st.markdown(
            "<div style='font-size:0.7rem;color:#6b7a96;text-transform:uppercase;"
            "letter-spacing:1px;font-weight:700;margin-top:10px'>DBSCAN EPS</div>",
            unsafe_allow_html=True
        )
        eps_val = st.slider("eps", 0.5, 5.0, 2.20, step=0.1, label_visibility="collapsed")

        st.divider()
        st.markdown("""
        <div style='font-size:0.72rem;color:#6b7a96;line-height:1.8'>
          Pipeline Steps:<br>
          ① TSFresh Features<br>
          ② Prophet Forecast<br>
          ③ KMeans Clustering<br>
          ④ DBSCAN Clustering<br>
          ⑤ PCA Projection<br>
          ⑥ t-SNE Visualization
        </div>""", unsafe_allow_html=True)

# Set defaults if not in pattern extraction
if st.session_state.active_section != "pattern_extraction":
    k_val   = 3
    eps_val = 2.2

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
active = st.session_state.active_section
section_label = "Preprocessing" if active == "preprocessing" else "Pattern Extraction"
section_icon  = "🔧" if active == "preprocessing" else "🤖"

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
# PREPROCESSING SECTION (was Milestone 1)
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
            bars = ax.bar(missing_nz.index, missing_nz.values,
                          color=PALETTE[0], edgecolor="none")
            ax.set_title("Missing Values per Column", fontsize=13, pad=12)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        str(int(bar.get_height())),
                        ha="center", va="bottom", fontsize=8, color="#e2eaf4")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
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
                        sns.histplot(df_eda[col], kde=True, ax=ax,
                                     color=PALETTE[0], edgecolor="none",
                                     line_kws={"color": PALETTE[1], "linewidth": 2})
                        ax.set_title(col, fontsize=11)
                        ax.set_xlabel("")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
        else:
            st.info("No numeric columns found for EDA.")

# =============================================================================
# PATTERN EXTRACTION SECTION (was Milestone 2)
# =============================================================================
if active == "pattern_extraction":

    st.markdown("## 🤖 Pattern Extraction — ML Pipeline")
    st.caption("Upload Fitbit CSV files, extract features, forecast trends, and cluster users.")

    # ── Required file signatures ──
    REQUIRED = {
        "Daily Activity":     ["TotalSteps", "Calories", "VeryActiveMinutes"],
        "Hourly Steps":       ["StepTotal"],
        "Hourly Intensities": ["TotalIntensity", "AverageIntensity"],
        "Minute Sleep":       ["logId", "value", "date"],
        "Heart Rate":         ["Value", "Time"],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — UPLOAD
    # ─────────────────────────────────────────────────────────────────────────
    step_hdr(1, "Upload Fitbit CSV Files")
    st.caption("Upload all 5 Fitbit CSV files — auto-detected by column signature.")

    uploaded_files = st.file_uploader(
        "Upload", type="csv", accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        for f in uploaded_files:
            f.seek(0)
            st.session_state["raw_files"][f.name] = f.read()

    # Auto-detect
    detected = {}
    for fname, raw in st.session_state["raw_files"].items():
        try:
            cols = set(pd.read_csv(pd.io.common.BytesIO(raw), nrows=3).columns)
            for name, keys in REQUIRED.items():
                if all(k in cols for k in keys) and name not in detected:
                    detected[name] = fname
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — DETECTION CARDS
    # ─────────────────────────────────────────────────────────────────────────
    step_hdr(2, "File Detection")

    icons_map = {
        "Daily Activity": "🏃", "Hourly Steps": "👟",
        "Hourly Intensities": "⚡", "Minute Sleep": "😴", "Heart Rate": "❤️"
    }
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
    m1.metric(str(dc),                  "DETECTED")
    m2.metric(str(5 - dc),              "MISSING")
    m3.metric("✓" if dc == 5 else "⏳", "READY TO LOAD")

    if dc == 5:
        st.success("✅ All 5 required files detected — ready to process!")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — LOAD & PARSE
    # ─────────────────────────────────────────────────────────────────────────
    step_hdr(3, "Load & Parse All Files")

    if dc == 5:
        if st.button("⚡ Load & Parse All Files"):
            with st.spinner("Parsing and merging datasets…"):
                try:
                    def _rd(name):
                        return pd.read_csv(pd.io.common.BytesIO(
                            st.session_state["raw_files"][detected[name]]))

                    daily     = _rd("Daily Activity")
                    steps     = _rd("Hourly Steps")
                    intensity = _rd("Hourly Intensities")
                    sleep     = _rd("Minute Sleep")
                    hr        = _rd("Heart Rate")

                    for df_ in [daily, steps, intensity, sleep, hr]:
                        df_.columns = [c.strip() for c in df_.columns]

                    st.session_state.daily     = daily
                    st.session_state.steps     = steps
                    st.session_state.intensity = intensity
                    st.session_state.sleep     = sleep
                    st.session_state.hr        = hr

                    # ── Build master DataFrame ──
                    id_col   = next(c for c in daily.columns if c.lower() == "id")
                    date_col = next((c for c in daily.columns if "date" in c.lower()), None)

                    keep = [id_col]
                    if date_col: keep.append(date_col)
                    for c in ["TotalSteps","Calories","VeryActiveMinutes","SedentaryMinutes"]:
                        if c in daily.columns: keep.append(c)
                    master = daily[keep].copy()
                    master.rename(columns={id_col: "Id"}, inplace=True)
                    if date_col: master.rename(columns={date_col: "Date"}, inplace=True)

                    hr_id = next((c for c in hr.columns if c.lower() == "id"), None)
                    hr_t  = next((c for c in hr.columns if c.lower() in
                                  ("time","datetime","timestamp","date","activityminute")), None)
                    hr_v  = next((c for c in hr.columns if c.lower() == "value"), None)
                    if hr_id and hr_t and hr_v and "Date" in master.columns:
                        hr["_d"] = pd.to_datetime(hr[hr_t], errors="coerce").dt.date.astype(str)
                        hr_agg   = hr.groupby([hr_id, "_d"])[hr_v].mean().reset_index()
                        hr_agg.columns = ["Id","Date","AvgHR"]
                        master = master.merge(hr_agg, on=["Id","Date"], how="left")

                    sl_id = next((c for c in sleep.columns if c.lower() == "id"), None)
                    sl_d  = next((c for c in sleep.columns if "date" in c.lower()), None)
                    if sl_id and sl_d and "Date" in master.columns:
                        sleep["_d"] = pd.to_datetime(sleep[sl_d], errors="coerce").dt.date.astype(str)
                        sl_agg = sleep.groupby([sl_id, "_d"]).size().reset_index(name="TotalSleepMinutes")
                        sl_agg.columns = ["Id","Date","TotalSleepMinutes"]
                        master = master.merge(sl_agg, on=["Id","Date"], how="left")

                    st.session_state.master_df    = master
                    st.session_state.files_loaded = True
                    st.success("✅ All 5 files loaded and master DataFrame built")

                except Exception as e:
                    st.error(f"❌ Load error: {e}")

    if st.session_state.files_loaded:
        st.success("✅ All 5 files loaded and master DataFrame built")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — NULL VALUE CHECK
    # ─────────────────────────────────────────────────────────────────────────
    if st.session_state.files_loaded:
        step_hdr(4, "Null Value Check")

        dsets = {
            "dailyActivity":     st.session_state.daily,
            "hourlySteps":       st.session_state.steps,
            "hourlyIntensities": st.session_state.intensity,
            "minuteSleep":       st.session_state.sleep,
            "heartrate":         st.session_state.hr,
        }
        nc = st.columns(5)
        for col, (name, df_) in zip(nc, dsets.items()):
            nulls = int(df_.isnull().sum().sum())
            rows  = len(df_)
            val   = (f"<div class='null-val-bad'>{nulls:,}</div>"
                     if nulls > 0 else "<div class='null-val-ok'>⊙</div>")
            with col:
                st.markdown(f"""
                <div class='null-card'>
                  <div class='null-card-name'>{name}</div>
                  {val}
                  <div class='null-rows'>nulls · {rows:,} rows</div>
                </div>""", unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 7 — TIME NORMALISATION LOG
        # ─────────────────────────────────────────────────────────────────────
        step_hdr(7, "Time Normalization Log")

        hr2  = st.session_state.hr
        hr_t = next((c for c in hr2.columns if c.lower() in
                     ("time","datetime","timestamp","date","activityminute")), None)

        rows_before     = len(hr2)
        rows_after      = rows_before
        date_min        = "N/A"
        date_max        = "N/A"
        date_range_days = "?"

        if hr_t:
            try:
                _dt          = pd.to_datetime(hr2[hr_t], errors="coerce")
                _dt_clean    = _dt.dropna()
                rows_after       = int(_dt_clean.dt.floor("T").nunique())
                date_min         = _dt_clean.dt.date.min()
                date_max         = _dt_clean.dt.date.max()
                date_range_days  = (date_max - date_min).days + 1
            except Exception:
                pass

        sl2        = st.session_state.sleep
        sleep_rows = len(sl2)

        st2  = st.session_state.steps
        st_t = next((c for c in st2.columns if c.lower() in
                     ("activityhour","datetime","time","date")), None)
        hourly_pct = "100.0"
        if st_t:
            try:
                _dts   = pd.to_datetime(st2[st_t], errors="coerce").dropna().sort_values()
                diffs  = _dts.diff().dt.total_seconds() / 3600
                hourly_pct = f"{(diffs.dropna() == 1.0).mean() * 100:.1f}"
            except Exception:
                pass

        st.markdown(f"""
        <div class='log-box'>
          <span style='color:#34d399'>✅ HR resampled</span>
          <span style='color:#6b7a96'> seconds → 1-minute intervals</span><br>
          &nbsp;&nbsp;&nbsp;Rows before : <span style='color:#e2eaf4;font-weight:700'>{rows_before:,}</span>
          &nbsp;&nbsp;|&nbsp;&nbsp;Rows after : <span style='color:#63d7c4;font-weight:700'>{rows_after:,}</span><br>
          <span style='color:#34d399'>✅ Date range</span>
          <span style='color:#63d7c4'> {date_min} → {date_max}</span>
          <span style='color:#6b7a96'> ({date_range_days} days)</span><br>
          <span style='color:#34d399'>✅ Hourly frequency</span>
          <span style='color:#6b7a96'> 1.0h median &nbsp;|&nbsp;</span>
          <span style='color:#e2eaf4'>{hourly_pct}% exact 1-hour</span><br>
          <span style='color:#34d399'>✅ Sleep stages</span>
          <span style='color:#6b7a96'> 1=Light · 2=Deep · 3=REM &nbsp;|&nbsp;</span>
          <span style='color:#e2eaf4'>{sleep_rows:,} records</span><br>
          <span style='color:#fbbf24'>⚠ Timezone</span>
          <span style='color:#6b7a96'> Local time — UTC normalization not applicable</span>
        </div>""", unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5 — DATASET OVERVIEW
        # ─────────────────────────────────────────────────────────────────────
        step_hdr(5, "Dataset Overview")

        daily_  = st.session_state.daily
        d_id    = next((c for c in daily_.columns if c.lower() == "id"), daily_.columns[0])
        hr_id_c = next((c for c in hr2.columns  if c.lower() == "id"), None)
        sl_id_c = next((c for c in sl2.columns  if c.lower() == "id"), None)

        daily_users = daily_[d_id].nunique()
        hr_users    = hr2[hr_id_c].nunique()  if hr_id_c  else 0
        sleep_users = sl2[sl_id_c].nunique()  if sl_id_c  else 0
        master_rows = len(st.session_state.master_df) if st.session_state.master_df is not None else 0

        o1,o2,o3,o4,o5 = st.columns(5)
        o1.metric(str(daily_users),     "DAILY USERS")
        o2.metric(str(hr_users),        "HR USERS")
        o3.metric(str(sleep_users),     "SLEEP USERS")
        o4.metric(f"{rows_after:,}",    "HR MINUTE ROWS")
        o5.metric(f"{master_rows:,}",   "MASTER ROWS")

        # ─────────────────────────────────────────────────────────────────────
        # STEP 9 — CLEANED DATASET PREVIEW
        # ─────────────────────────────────────────────────────────────────────
        step_hdr(9, "Cleaned Dataset Preview")

        if st.session_state.master_df is not None:
            st.dataframe(st.session_state.master_df.head(30), use_container_width=True)
            csv_bytes = st.session_state.master_df.to_csv(index=False).encode()
            st.download_button("⬇  Download Master CSV", csv_bytes,
                               "fitpulse_master.csv", mime="text/csv")

    # =========================================================================
    # ML-1 — TSFresh
    # =========================================================================
    if st.session_state.files_loaded:
        st.divider()
        step_hdr("ML‑1", "TSFresh Feature Extraction")
        st.caption("Extracts statistical features from Heart Rate time-series per user.")

        if st.button("▶  Run TSFresh"):
            with st.spinner("Extracting features…"):
                try:
                    hr3 = st.session_state.hr.copy()
                    hr3.columns = [c.strip() for c in hr3.columns]

                    id_col = next(c for c in hr3.columns if c.lower() == "id")
                    t_col  = next(c for c in hr3.columns if c.lower() in
                                  ("time","datetime","timestamp","date","activityminute"))
                    v_col  = next(c for c in hr3.columns if c.lower() == "value")

                    ts_hr = hr3[[id_col, t_col, v_col]].rename(
                        columns={id_col:"id", t_col:"time", v_col:"value"})
                    ts_hr["time"]  = pd.to_datetime(ts_hr["time"], errors="coerce")
                    ts_hr["value"] = pd.to_numeric(ts_hr["value"],  errors="coerce")
                    ts_hr.dropna(inplace=True)

                    features = extract_features(
                        ts_hr,
                        column_id="id",
                        column_sort="time",
                        column_value="value",
                        default_fc_parameters=MinimalFCParameters()
                    )
                    features.dropna(axis=1, how="all", inplace=True)

                    scaler = MinMaxScaler()
                    norm   = scaler.fit_transform(features)

                    fig, ax = plt.subplots(figsize=(11, 4))
                    sns.heatmap(norm, cmap="YlOrRd", ax=ax, linewidths=0,
                                cbar_kws={"shrink": 0.7})
                    ax.set_title("TSFresh Feature Heatmap (normalised)", fontsize=12)
                    ax.set_xlabel("Feature Index")
                    ax.set_ylabel("User ID")
                    fig.tight_layout()

                    st.session_state["tsfresh_fig"] = fig_to_bytes(fig)
                    st.session_state.features       = features
                    st.session_state.tsfresh_done   = True
                    st.success(f"✅ Extracted **{features.shape[1]}** features "
                               f"for **{features.shape[0]}** users.")
                except Exception as e:
                    st.error(f"❌ TSFresh error: {e}")

        if st.session_state["tsfresh_fig"] is not None:
            st.image(st.session_state["tsfresh_fig"], use_container_width=True)

    # =========================================================================
    # ML-2 — Prophet
    # =========================================================================
    if st.session_state.tsfresh_done:
        st.divider()
        step_hdr("ML‑2", "Prophet Forecast")
        st.caption("30-day heart rate forecast using Meta's Prophet model.")

        if st.button("▶  Run Prophet"):
            with st.spinner("Fitting Prophet model… (usually < 10 seconds)"):
                try:
                    hr4 = st.session_state.hr.copy()
                    hr4.columns = [c.strip() for c in hr4.columns]

                    t_col = next((c for c in hr4.columns if c.lower() in
                                  ("time","datetime","timestamp","date","activityminute")), None)
                    v_col = next((c for c in hr4.columns if c.lower() == "value"), None)

                    if not t_col or not v_col:
                        st.error("Could not find Date/Value columns in Heart Rate dataset.")
                    else:
                        hr4["_dt"]   = pd.to_datetime(hr4[t_col], errors="coerce")
                        hr4["_date"] = hr4["_dt"].dt.date
                        agg = (hr4.groupby("_date")[v_col]
                                  .mean()
                                  .reset_index())
                        agg.columns = ["ds", "y"]
                        agg["ds"] = pd.to_datetime(agg["ds"])
                        agg["y"]  = pd.to_numeric(agg["y"], errors="coerce")
                        agg.dropna(inplace=True)
                        agg.sort_values("ds", inplace=True)
                        agg = agg.tail(60).reset_index(drop=True)

                        model = Prophet(
                            interval_width=0.90,
                            daily_seasonality=False,
                            weekly_seasonality=True,
                            mcmc_samples=0
                        )
                        model.fit(agg)
                        future   = model.make_future_dataframe(periods=30)
                        forecast = model.predict(future)

                        fig, ax = plt.subplots(figsize=(11, 4))
                        ax.scatter(agg["ds"], agg["y"],
                                   color=PALETTE[0], s=18, alpha=0.8,
                                   zorder=3, label="Actual HR")
                        ax.plot(forecast["ds"], forecast["yhat"],
                                color=PALETTE[1], linewidth=2, label="Forecast")
                        ax.fill_between(forecast["ds"],
                                        forecast["yhat_lower"],
                                        forecast["yhat_upper"],
                                        alpha=0.18, color=PALETTE[1], label="90% CI")
                        ax.axvline(agg["ds"].max(), linestyle="--",
                                   color=PALETTE[3], alpha=0.6, linewidth=1)
                        ax.set_title("Heart Rate — 30-Day Prophet Forecast", fontsize=12)
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Avg BPM")
                        ax.legend(fontsize=9)
                        fig.tight_layout()

                        st.session_state["prophet_fig"] = fig_to_bytes(fig)
                        st.session_state.prophet_done   = True
                        st.success("✅ Prophet forecast complete — 30 days ahead.")

                except Exception as e:
                    st.error(f"❌ Prophet error: {e}")

        if st.session_state["prophet_fig"] is not None:
            st.image(st.session_state["prophet_fig"], use_container_width=True)

    # =========================================================================
    # ML-3 — Clustering
    # =========================================================================
    if st.session_state.prophet_done:
        st.divider()
        step_hdr("ML‑3", "Clustering & Dimensionality Reduction")
        st.caption(
            f"KMeans (k={k_val}), DBSCAN (eps={eps_val}), "
            f"PCA & t-SNE on Daily Activity features."
        )

        if st.button("▶  Run Clustering Pipeline"):
            with st.spinner("Running clustering…"):
                try:
                    daily2 = st.session_state.daily.copy()
                    daily2.columns = [c.strip() for c in daily2.columns]

                    cluster_cols = ["TotalSteps","Calories","VeryActiveMinutes",
                                    "FairlyActiveMinutes","LightlyActiveMinutes",
                                    "SedentaryMinutes"]
                    available = [c for c in cluster_cols if c in daily2.columns]
                    id_col2   = next((c for c in daily2.columns if c.lower() == "id"),
                                     daily2.columns[0])
                    feat_df   = daily2.groupby(id_col2)[available].mean().dropna()

                    scaler2 = StandardScaler()
                    X       = scaler2.fit_transform(feat_df)

                    pca   = PCA(n_components=2)
                    X_pca = pca.fit_transform(X)

                    kmeans    = KMeans(n_clusters=k_val, n_init=10, random_state=42)
                    km_labels = kmeans.fit_predict(X)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

                    for c in np.unique(km_labels):
                        m = km_labels == c
                        axes[0].scatter(X_pca[m,0], X_pca[m,1],
                                        color=PALETTE[c % len(PALETTE)],
                                        s=60, alpha=0.85, edgecolors="none",
                                        label=f"Cluster {c}")
                    axes[0].set_title(f"KMeans (k={k_val}) — PCA", fontsize=11)
                    axes[0].legend(fontsize=8)
                    axes[0].set_xlabel("PC 1"); axes[0].set_ylabel("PC 2")

                    db        = DBSCAN(eps=eps_val, min_samples=2)
                    db_labels = db.fit_predict(X)
                    for i, c in enumerate(np.unique(db_labels)):
                        m     = db_labels == c
                        label = "Noise" if c == -1 else f"Cluster {c}"
                        color = "#6b7a96" if c == -1 else PALETTE[i % len(PALETTE)]
                        axes[1].scatter(X_pca[m,0], X_pca[m,1],
                                        color=color, s=60, alpha=0.85,
                                        edgecolors="none", label=label)
                    n_noise = (db_labels == -1).sum()
                    axes[1].set_title(
                        f"DBSCAN (eps={eps_val}) — "
                        f"{len(np.unique(db_labels))-1} clusters, {n_noise} noise",
                        fontsize=11)
                    axes[1].legend(fontsize=8)
                    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")

                    perp   = min(30, max(5, len(X) - 1))
                    tsne   = TSNE(n_components=2, perplexity=perp,
                                  random_state=42, max_iter=1000)
                    X_tsne = tsne.fit_transform(X)
                    for c in np.unique(km_labels):
                        m = km_labels == c
                        axes[2].scatter(X_tsne[m,0], X_tsne[m,1],
                                        color=PALETTE[c % len(PALETTE)],
                                        s=60, alpha=0.85, edgecolors="none",
                                        label=f"Cluster {c}")
                    axes[2].set_title("t-SNE (KMeans labels)", fontsize=11)
                    axes[2].legend(fontsize=8)
                    axes[2].set_xlabel("Dim 1"); axes[2].set_ylabel("Dim 2")

                    for ax in axes:
                        ax.spines[["top","right","left","bottom"]].set_visible(False)

                    fig.suptitle("Clustering Pipeline", fontsize=13, y=1.02,
                                 color="#63d7c4", fontweight="bold")
                    fig.tight_layout()

                    st.session_state["cluster_fig"] = fig_to_bytes(fig)

                    feat_df["Cluster"] = km_labels
                    st.session_state["cluster_summary"] = (
                        feat_df.groupby("Cluster")[available].mean().round(1)
                    )
                    st.session_state["X_scaled"]       = X
                    st.session_state["km_labels"]      = km_labels
                    st.session_state["db_labels"]      = db_labels
                    st.session_state["k_val_used"]     = k_val
                    st.session_state["available_cols"] = available
                    st.session_state["feat_df"]        = feat_df.copy()
                    st.session_state.cluster_done = True
                    st.success("🎉 ML Pipeline completed successfully!")

                except Exception as e:
                    st.error(f"❌ Clustering error: {e}")

        if st.session_state["cluster_fig"] is not None:
            st.image(st.session_state["cluster_fig"], use_container_width=True)

        if st.session_state.get("cluster_summary") is not None:
            step_hdr("ML‑4", "Cluster Profiles")
            st.dataframe(st.session_state["cluster_summary"], use_container_width=True)

    # =========================================================================
    # ML-6 — CLUSTER PROFILES BAR CHART + INTERPRETATION
    # =========================================================================
    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑6", "Cluster Profiles — Bar Chart & Interpretation")

        _summary   = st.session_state.get("cluster_summary")
        _k_used    = st.session_state.get("k_val_used", k_val)
        _feat_df   = st.session_state.get("feat_df")
        _km_labels = st.session_state.get("km_labels")
        _avail     = st.session_state.get("available_cols", [])

        if _summary is not None and _feat_df is not None:
            st.dataframe(_summary, use_container_width=True)

            plot_cols = [c for c in
                         ["TotalSteps","Calories","VeryActiveMinutes",
                          "SedentaryMinutes","TotalSleepMinutes"]
                         if c in _summary.columns]

            fig, ax = plt.subplots(figsize=(13, 5))
            _summary[plot_cols].plot(
                kind="bar", ax=ax,
                color=PALETTE[:len(plot_cols)],
                edgecolor="#0d1526", width=0.7
            )
            ax.set_title(
                "Cluster Profiles — Key Feature Averages (Real Fitbit Data)",
                fontsize=13)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Mean Value")
            ax.set_xticklabels(
                [f"Cluster {i}" for i in range(len(_summary))], rotation=0)
            ax.legend(bbox_to_anchor=(1.02, 1), title="Feature",
                      fontsize=8, title_fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
            fig.tight_layout()
            st.session_state["cluster_bar_fig"] = fig_to_bytes(fig)
            st.image(st.session_state["cluster_bar_fig"], use_container_width=True)

            st.markdown("### 📊 Cluster Interpretation")
            interp_cols = st.columns(min(_k_used, 4))
            for i, col in enumerate(interp_cols):
                if i not in _summary.index:
                    continue
                row    = _summary.loc[i]
                steps  = row.get("TotalSteps", 0)
                sed    = row.get("SedentaryMinutes", 0)
                active = row.get("VeryActiveMinutes", 0)

                if steps > 10000:
                    profile_icon  = "🏃"
                    profile_label = "HIGHLY ACTIVE"
                    profile_color = "#34d399"
                elif steps > 5000:
                    profile_icon  = "🚶"
                    profile_label = "MODERATELY ACTIVE"
                    profile_color = "#fbbf24"
                else:
                    profile_icon  = "🛋️"
                    profile_label = "SEDENTARY"
                    profile_color = "#f87171"

                with col:
                    st.markdown(f"""
                    <div class='null-card' style='text-align:center;padding:18px 12px'>
                      <div style='font-size:1.8rem'>{profile_icon}</div>
                      <div style='color:{profile_color};font-weight:700;
                                  font-size:0.78rem;margin:6px 0 10px'>
                        Cluster {i} · {profile_label}
                      </div>
                      <div style='font-size:0.75rem;color:#6b7a96;line-height:1.9;
                                  font-family:"JetBrains Mono",monospace;text-align:left'>
                        Avg Steps &nbsp;&nbsp;&nbsp;: <span style='color:#e2eaf4'>{steps:,.0f}</span><br>
                        Sedentary &nbsp;&nbsp;: <span style='color:#e2eaf4'>{sed:.0f} min</span><br>
                        Very Active : <span style='color:#e2eaf4'>{active:.0f} min</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

    # =========================================================================
    # ML-7 — ELBOW CURVE
    # =========================================================================
    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑7", "KMeans Elbow Curve")
        st.caption("Inertia vs K — use the elbow point to choose optimal clusters.")

        _X = st.session_state.get("X_scaled")
        if _X is not None:
            if st.button("▶  Run Elbow Curve"):
                with st.spinner("Computing inertia for K = 2…9…"):
                    try:
                        inertias = []
                        K_range  = range(2, 10)
                        for k in K_range:
                            km = KMeans(n_clusters=k, random_state=42, n_init=10)
                            km.fit(_X)
                            inertias.append(km.inertia_)

                        fig, ax = plt.subplots(figsize=(9, 4))
                        ax.plot(list(K_range), inertias, "o-",
                                color="#63d7c4", linewidth=2.5, markersize=9,
                                markerfacecolor="#f97316")
                        ax.set_title("KMeans Elbow Curve — Real Fitbit Data", fontsize=13)
                        ax.set_xlabel("Number of Clusters (K)")
                        ax.set_ylabel("Inertia")
                        ax.set_xticks(list(K_range))
                        ax.spines[["top","right"]].set_visible(False)
                        fig.tight_layout()
                        st.session_state["elbow_fig"] = fig_to_bytes(fig)
                        st.success("✅ Elbow curve complete — screenshot this!")
                    except Exception as e:
                        st.error(f"❌ Elbow error: {e}")

            if st.session_state.get("elbow_fig") is not None:
                st.image(st.session_state["elbow_fig"], use_container_width=True)

    # =========================================================================
    # ML-8 — STEPS & SLEEP PROPHET FORECASTS
    # =========================================================================
    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑8", "Steps & Sleep Prophet Forecasts")
        st.caption("30-day forecast for daily steps and sleep minutes (80% CI).")

        if st.button("▶  Run Steps & Sleep Forecast"):
            with st.spinner("Fitting Prophet for Steps and Sleep… (usually < 15 seconds)"):
                try:
                    daily3 = st.session_state.daily.copy()
                    sleep3 = st.session_state.sleep.copy()
                    daily3.columns = [c.strip() for c in daily3.columns]
                    sleep3.columns = [c.strip() for c in sleep3.columns]

                    d_date = next((c for c in daily3.columns if "date" in c.lower()), None)

                    sl_id  = next((c for c in sleep3.columns if c.lower() == "id"), None)
                    sl_d   = next((c for c in sleep3.columns if "date" in c.lower()), None)
                    sl_v   = next((c for c in sleep3.columns if c.lower() == "value"), None)

                    configs = []
                    if d_date and "TotalSteps" in daily3.columns:
                        configs.append(("TotalSteps", d_date, daily3, "#63d7c4", "Steps"))

                    if sl_d and sl_v:
                        sleep3["_dt"]    = pd.to_datetime(sleep3[sl_d], errors="coerce")
                        sleep3["_date"]  = sleep3["_dt"].dt.date.astype(str)
                        sleep3[sl_v]     = pd.to_numeric(sleep3[sl_v], errors="coerce")
                        sl_daily = (sleep3.groupby("_date")
                                          .size()
                                          .reset_index(name="SleepMinutes"))
                        sl_daily.columns = ["_date", "SleepMinutes"]
                        configs.append(("SleepMinutes", "_date", sl_daily, "#818cf8", "Sleep (minutes)"))

                    if not configs:
                        st.warning("⚠ Could not find Steps or Sleep columns.")
                    else:
                        fig, axes = plt.subplots(len(configs), 1,
                                                 figsize=(14, 5 * len(configs)))
                        if len(configs) == 1:
                            axes = [axes]

                        for ax, (metric, date_col, df_src, color, label) in zip(axes, configs):
                            agg = (df_src.groupby(date_col)[metric]
                                         .mean().reset_index())
                            agg.columns = ["ds", "y"]
                            agg["ds"] = pd.to_datetime(agg["ds"], errors="coerce")
                            agg["y"]  = pd.to_numeric(agg["y"], errors="coerce")
                            agg = agg.dropna().sort_values("ds").tail(60)

                            if len(agg) < 2:
                                ax.text(0.5, 0.5, f"Not enough data for {label} forecast",
                                        ha="center", va="center", transform=ax.transAxes,
                                        color="#f87171", fontsize=11)
                                ax.set_title(f"{label} — Insufficient Data", fontsize=13)
                                continue

                            m = Prophet(
                                weekly_seasonality=True,
                                yearly_seasonality=False,
                                daily_seasonality=False,
                                interval_width=0.80,
                                changepoint_prior_scale=0.1,
                                mcmc_samples=0
                            )
                            m.fit(agg)
                            future   = m.make_future_dataframe(periods=30)
                            forecast = m.predict(future)

                            ax.scatter(agg["ds"], agg["y"],
                                       color=color, s=20, alpha=0.75,
                                       label=f"Actual {label}", zorder=3)
                            ax.plot(forecast["ds"], forecast["yhat"],
                                    color="#e2eaf4", linewidth=2, label="Trend")
                            ax.fill_between(forecast["ds"],
                                            forecast["yhat_lower"],
                                            forecast["yhat_upper"],
                                            alpha=0.22, color=color, label="80% CI")
                            ax.axvline(agg["ds"].max(), color="#fbbf24",
                                       linestyle="--", linewidth=1.5,
                                       label="Forecast Start")
                            ax.set_title(f"{label} — Prophet Trend Forecast", fontsize=13)
                            ax.set_xlabel("Date")
                            ax.set_ylabel(label)
                            ax.legend(fontsize=9)
                            ax.spines[["top","right"]].set_visible(False)

                        fig.tight_layout()
                        st.session_state["steps_sleep_fig"] = fig_to_bytes(fig)
                        st.success("✅ Steps & Sleep forecasts complete — screenshot this!")

                except Exception as e:
                    st.error(f"❌ Steps/Sleep forecast error: {e}")

        if st.session_state.get("steps_sleep_fig") is not None:
            st.image(st.session_state["steps_sleep_fig"], use_container_width=True)

    # =========================================================================
    # COMPLETION CARD
    # =========================================================================
    if st.session_state.cluster_done:
        st.balloons()
        st.markdown("""
        <div class='fp-card'
             style='border-color:rgba(99,215,196,0.4);text-align:center;padding:2rem;margin-top:1rem'>
          <div style='font-size:2rem'>🎉</div>
          <h2 style='color:#63d7c4'>Pipeline Complete</h2>
          <p style='color:#6b7a96'>
            All ML stages finished — TSFresh → Prophet → KMeans → DBSCAN → t-SNE → Elbow → Steps/Sleep Forecast
          </p>
        </div>""", unsafe_allow_html=True)

    # =========================================================================
    # ML-5 — MILESTONE 2 SUMMARY
    # =========================================================================
    if st.session_state.cluster_done:
        st.divider()
        step_hdr("ML‑5", "Pattern Extraction Summary")

        _feat_df   = st.session_state.get("feat_df")
        _features  = st.session_state.get("features")
        _km_labels = st.session_state.get("km_labels")
        _db_labels = st.session_state.get("db_labels")
        _k_used    = st.session_state.get("k_val_used", k_val)

        if _feat_df is not None and _km_labels is not None and _db_labels is not None:
            _n_users   = _feat_df.shape[0]
            _n_feats   = _features.shape[1] if _features is not None else "N/A"
            _n_clust   = len(np.unique(_db_labels[_db_labels != -1]))
            _n_noise   = int((_db_labels == -1).sum())
            _noise_pct = _n_noise / len(_db_labels) * 100
            _km_dist   = {int(k): int(v) for k, v in
                          zip(*np.unique(_km_labels, return_counts=True))}

            st.markdown(f"""
            <div class='log-box'>
              <span style='color:#34d399;font-weight:700'>✅ Dataset</span>
              <span style='color:#6b7a96'> : Real Fitbit device data</span><br>
              &nbsp;&nbsp;&nbsp;Users &nbsp;&nbsp;: <span style='color:#e2eaf4'>{_n_users}</span>
              &nbsp;&nbsp;|&nbsp;&nbsp;
              Days &nbsp;&nbsp;: <span style='color:#e2eaf4'>31 (March–April 2016)</span><br>

              <span style='color:#34d399;font-weight:700'>✅ TSFresh features extracted</span>
              <span style='color:#63d7c4'> : {_n_feats} features</span><br>
              &nbsp;&nbsp;&nbsp;Source : <span style='color:#6b7a96'>Real minute-level heart rate data</span><br>

              <span style='color:#34d399;font-weight:700'>✅ Prophet models fitted</span><br>
              &nbsp;&nbsp;&nbsp;Heart Rate &nbsp;— 30-day forecast, 90% CI<br>
              &nbsp;&nbsp;&nbsp;Steps &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— 30-day forecast, 80% CI<br>
              &nbsp;&nbsp;&nbsp;Sleep &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— 30-day forecast, 80% CI<br>

              <span style='color:#34d399;font-weight:700'>✅ KMeans</span>
              <span style='color:#63d7c4'> : {_k_used} clusters identified</span><br>
              &nbsp;&nbsp;&nbsp;Distribution : <span style='color:#e2eaf4'>{_km_dist}</span><br>

              <span style='color:#34d399;font-weight:700'>✅ DBSCAN</span>
              <span style='color:#63d7c4'> : {_n_clust} clusters, {_n_noise} noise/outlier users</span><br>
              &nbsp;&nbsp;&nbsp;Noise % : <span style='color:#fbbf24'>{_noise_pct:.1f}%</span><br>

              <span style='color:#818cf8;font-weight:700'>📸 Screenshots to submit:</span><br>
              &nbsp;&nbsp;&nbsp;1. TSFresh feature matrix heatmap<br>
              &nbsp;&nbsp;&nbsp;2. Prophet HR forecast with confidence interval<br>
              &nbsp;&nbsp;&nbsp;3. Steps and Sleep Prophet forecasts<br>
              &nbsp;&nbsp;&nbsp;4. KMeans PCA scatter plot<br>
              &nbsp;&nbsp;&nbsp;5. DBSCAN PCA scatter plot<br>
              &nbsp;&nbsp;&nbsp;6. t-SNE projection (both models)<br>
              &nbsp;&nbsp;&nbsp;7. Cluster profiles bar chart
            </div>""", unsafe_allow_html=True)