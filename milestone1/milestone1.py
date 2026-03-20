import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitPulse — Data Processing",
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

.prog-bar-bg {
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

.stat-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(99,215,196,0.14);
  border-radius: 14px;
  padding: 18px 16px 14px;
  text-align: center;
}
.stat-card-label {
  font-size:0.7rem; color:#6b7a96; text-transform:uppercase;
  letter-spacing:1px; font-weight:700; margin-bottom:8px;
  font-family:'JetBrains Mono',monospace;
}
.stat-card-value { font-size:1.7rem; font-weight:800; color:#63d7c4; }
.stat-card-sub   { font-size:0.7rem; color:#6b7a96; margin-top:4px; }

.summary-box {
  background:rgba(255,255,255,0.02);
  border:1px solid rgba(99,215,196,0.12);
  border-radius:12px;
  padding:1.1rem 1.4rem;
  font-family:'JetBrains Mono',monospace;
  font-size:0.8rem;
  line-height:2;
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
for k in ["df", "clean_df", "milestone1_loaded", "cleaning_done"]:
    if k not in st.session_state:
        st.session_state[k] = None

if "milestone1_loaded" not in st.session_state:
    st.session_state["milestone1_loaded"] = False
if "cleaning_done" not in st.session_state:
    st.session_state["cleaning_done"] = False

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:4px 0 6px 0'>
      <span style='font-size:1.4rem'>🩺</span>
      <div>
        <div style='font-size:1.05rem;font-weight:800;color:#e2eaf4'>FitPulse</div>
        <div style='font-size:0.7rem;color:#6b7a96'>Milestone 1 — Data Processing</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Pipeline progress
    s1 = st.session_state.get("milestone1_loaded") or False
    s2 = st.session_state.get("cleaning_done") or False
    s3 = s2  # EDA unlocks after cleaning
    pct = int(sum([bool(s1), bool(s2), bool(s3)]) / 3 * 100)

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

    nav(s1, "📂", "Data Upload")
    nav(s1, "🔍", "Missing Values Analysis")
    nav(s2, "🛠️", "Data Cleaning")
    nav(s3, "📊", "EDA — Distributions")
    nav(s3, "🔗", "EDA — Correlations")

    st.divider()

    st.markdown("""
    <div style='font-size:0.72rem;color:#6b7a96;line-height:1.9'>
      Pipeline Steps:<br>
      ① Upload CSV dataset<br>
      ② Inspect missing values<br>
      ③ Interpolate &amp; clean<br>
      ④ Explore distributions<br>
      ⑤ Correlation heatmap<br>
      ⑥ Download clean data
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:1rem 0 0.3rem 0'>
  <h1 style='margin:0'>🩺 FitPulse <span style='color:#63d7c4'>Analytics</span></h1>
  <p style='color:#6b7a96;font-size:0.85rem;margin-top:4px'>
    Milestone 1 — Data Processing Pipeline
  </p>
</div>""", unsafe_allow_html=True)
st.divider()

# =============================================================================
# STEP 1 — UPLOAD
# =============================================================================
step_hdr(1, "Upload Dataset")
st.caption("Upload any fitness or activity CSV file to begin the pipeline.")

file = st.file_uploader("Upload your fitness CSV", type=["csv"])

if file:
    file.seek(0)
    df = pd.read_csv(file)
    st.session_state["df"] = df
    st.session_state["milestone1_loaded"] = True

if st.session_state.get("milestone1_loaded") and st.session_state["df"] is not None:
    df = st.session_state["df"]
    st.success(f"✅ Dataset loaded — **{len(df):,} rows × {len(df.columns)} columns**")

    # ── Overview metrics ──
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    total_missing = int(df.isnull().sum().sum())
    missing_pct = round(total_missing / (len(df) * len(df.columns)) * 100, 2)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"{len(df):,}",        "TOTAL ROWS")
    m2.metric(f"{len(df.columns)}",  "COLUMNS")
    m3.metric(f"{len(num_cols)}",    "NUMERIC COLS")
    m4.metric(f"{len(cat_cols)}",    "CATEGORICAL COLS")
    m5.metric(f"{total_missing:,}",  "MISSING CELLS")

    # ── Raw preview ──
    with st.expander("🔎 Raw Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Column Data Types"):
        dtype_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
            "Null %": (df.isnull().sum() / len(df) * 100).round(2).values
        })
        st.dataframe(dtype_df, use_container_width=True)

    # =========================================================================
    # STEP 2 — MISSING VALUES ANALYSIS
    # =========================================================================
    st.divider()
    step_hdr(2, "Missing Values Analysis")

    missing       = df.isnull().sum()
    total_missing = missing.sum()

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{total_missing:,}",   "TOTAL MISSING CELLS")
    c2.metric(f"{missing_pct}%",      "MISSING RATE")
    c3.metric(f"{(missing > 0).sum()}", "COLUMNS WITH NULLS")

    if total_missing > 0:
        missing_nz = missing[missing > 0].sort_values(ascending=False)

        # Bar chart
        fig, ax = plt.subplots(figsize=(max(8, len(missing_nz) * 0.8), 3.8))
        bars = ax.bar(missing_nz.index, missing_nz.values,
                      color=PALETTE[0], edgecolor="none",
                      width=0.6)
        ax.set_title("Missing Values per Column", fontsize=13, pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=8, color="#e2eaf4")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Per-column breakdown
        st.markdown("#### 🗂 Column-Level Breakdown")
        cols_per_row = 4
        missing_cols = missing_nz.index.tolist()
        for i in range(0, len(missing_cols), cols_per_row):
            batch = missing_cols[i:i + cols_per_row]
            grid  = st.columns(cols_per_row)
            for col_ui, col_name in zip(grid, batch):
                n   = int(missing_nz[col_name])
                pct = round(n / len(df) * 100, 1)
                with col_ui:
                    st.markdown(f"""
                    <div class='stat-card'>
                      <div class='stat-card-label'>{col_name}</div>
                      <div class='stat-card-value' style='color:#f87171'>{n:,}</div>
                      <div class='stat-card-sub'>{pct}% missing</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.success("🎉 No missing values found in the dataset!")

    # =========================================================================
    # STEP 3 — DATA CLEANING
    # =========================================================================
    st.divider()
    step_hdr(3, "Data Cleaning")
    st.caption("Interpolates numeric columns and fills remaining gaps with 'Unknown'.")

    if st.button("▶  Run Preprocessing"):
        with st.spinner("Cleaning data…"):
            df_clean = df.copy()

            # Numeric interpolation
            numeric_cols_clean = df_clean.select_dtypes(include=np.number).columns
            for col in numeric_cols_clean:
                df_clean[col] = df_clean[col].interpolate(method="linear",
                                                           limit_direction="both")

            # Fill remaining
            df_clean = df_clean.fillna("Unknown")

            st.session_state["clean_df"]     = df_clean
            st.session_state["cleaning_done"] = True

        st.success("✅ Cleaning complete — missing values interpolated / filled.")

    if st.session_state.get("cleaning_done") and st.session_state["clean_df"] is not None:
        df_clean = st.session_state["clean_df"]

        # Before / After comparison
        before_null = st.session_state["df"].isnull().sum().sum()
        after_null  = df_clean.isnull().sum().sum()

        b1, b2, b3 = st.columns(3)
        b1.metric(f"{before_null:,}",      "NULLS BEFORE")
        b2.metric(f"{after_null:,}",       "NULLS AFTER")
        b3.metric(f"{before_null - after_null:,}", "CELLS FIXED")

        # =========================================================================
        # STEP 4 — CLEANED DATASET
        # =========================================================================
        st.divider()
        step_hdr(4, "Cleaned Dataset Preview")

        st.dataframe(df_clean.head(30), use_container_width=True)

        csv_bytes = df_clean.to_csv(index=False).encode()
        st.download_button(
            "⬇  Download Clean CSV",
            csv_bytes,
            "clean_dataset.csv",
            mime="text/csv"
        )

        # =========================================================================
        # STEP 5 — EDA — DISTRIBUTIONS
        # =========================================================================
        st.divider()
        step_hdr(5, "Exploratory Data Analysis — Distributions")

        num_cols_clean = df_clean.select_dtypes(include=np.number).columns.tolist()

        if num_cols_clean:
            # Descriptive stats table
            with st.expander("📐 Descriptive Statistics", expanded=False):
                st.dataframe(df_clean[num_cols_clean].describe().round(2),
                             use_container_width=True)

            # Histograms
            st.markdown("#### 📊 Feature Distributions")
            pairs = [num_cols_clean[i:i+2] for i in range(0, len(num_cols_clean), 2)]
            for pair in pairs:
                grid = st.columns(len(pair))
                for ax_col, col in zip(grid, pair):
                    with ax_col:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        sns.histplot(df_clean[col].dropna(), kde=True, ax=ax,
                                     color=PALETTE[0], edgecolor="none",
                                     line_kws={"color": PALETTE[1], "linewidth": 2})
                        ax.set_title(col, fontsize=11)
                        ax.set_xlabel("")
                        ax.spines[["top","right"]].set_visible(False)
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

        else:
            st.info("No numeric columns found for distribution analysis.")

        # =========================================================================
        # STEP 6 — EDA — CORRELATIONS
        # =========================================================================
        st.divider()
        step_hdr(6, "Exploratory Data Analysis — Correlations")

        num_cols_corr = df_clean.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols_corr) >= 2:
            corr_matrix = df_clean[num_cols_corr].corr()

            fig, ax = plt.subplots(figsize=(max(8, len(num_cols_corr) * 0.9),
                                            max(6, len(num_cols_corr) * 0.75)))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix, mask=mask, ax=ax,
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, linecolor="#0d1526",
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Feature Correlation Matrix", fontsize=13, pad=14)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Top correlations table
            st.markdown("#### 🔗 Top Feature Correlations")
            corr_pairs = (corr_matrix.where(~mask)
                                     .stack()
                                     .reset_index())
            corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
            corr_pairs["Abs Corr"] = corr_pairs["Correlation"].abs()
            corr_pairs = (corr_pairs
                          .sort_values("Abs Corr", ascending=False)
                          .drop(columns="Abs Corr")
                          .reset_index(drop=True))
            st.dataframe(corr_pairs.head(15), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

        # =========================================================================
        # COMPLETION SUMMARY
        # =========================================================================
        st.divider()
        st.markdown(f"""
        <div class='summary-box'>
          <span style='color:#34d399;font-weight:700'>✅ Upload</span>
          <span style='color:#6b7a96'> : Dataset loaded successfully</span><br>
          &nbsp;&nbsp;&nbsp;Rows &nbsp;&nbsp;&nbsp;&nbsp; : <span style='color:#e2eaf4'>{len(df_clean):,}</span>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Columns : <span style='color:#e2eaf4'>{len(df_clean.columns)}</span><br>

          <span style='color:#34d399;font-weight:700'>✅ Missing Values</span>
          <span style='color:#6b7a96'> : Analysed &amp; visualised</span><br>
          &nbsp;&nbsp;&nbsp;Before : <span style='color:#f87171'>{before_null:,} nulls</span>
          &nbsp;&nbsp;→&nbsp;&nbsp;
          After : <span style='color:#34d399'>{after_null:,} nulls</span><br>

          <span style='color:#34d399;font-weight:700'>✅ Cleaning</span>
          <span style='color:#6b7a96'> : Interpolation + fillna applied</span><br>
          &nbsp;&nbsp;&nbsp;Numeric columns : <span style='color:#e2eaf4'>{len(num_cols_clean)}</span>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Cells fixed : <span style='color:#63d7c4'>{before_null - after_null:,}</span><br>

          <span style='color:#34d399;font-weight:700'>✅ EDA</span>
          <span style='color:#6b7a96'> : Distributions &amp; correlation heatmap generated</span><br>

          <span style='color:#818cf8;font-weight:700'>📸 Deliverables:</span><br>
          &nbsp;&nbsp;&nbsp;1. Missing values bar chart<br>
          &nbsp;&nbsp;&nbsp;2. Feature distribution histograms<br>
          &nbsp;&nbsp;&nbsp;3. Correlation heatmap<br>
          &nbsp;&nbsp;&nbsp;4. Clean CSV download
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class='fp-card'
             style='border-color:rgba(99,215,196,0.4);text-align:center;
                    padding:2rem;margin-top:1.2rem'>
          <div style='font-size:2rem'>🎉</div>
          <h2 style='color:#63d7c4'>Milestone 1 Complete</h2>
          <p style='color:#6b7a96'>
            Data uploaded → missing values analysed → cleaned → EDA complete
          </p>
        </div>""", unsafe_allow_html=True)

        st.balloons()