import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Fitness Data Pro",
    layout="wide"
)

# -------------------------------------------------
# GLOBAL DARK THEME + SIDEBAR STYLE FIX
# -------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
    --indigo:   #6366f1;
    --violet:   #a855f7;
    --sky:      #38bdf8;
    --surface:  rgba(255,255,255,0.04);
    --border:   rgba(255,255,255,0.08);
    --text:     #e2e8f0;
    --muted:    #94a3b8;
}

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #1e1b4b 50%, #1a0533);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* Main block container padding */
[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080814 0%, #11112a 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] a { text-decoration: none !important; }

/* Headings */
h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem !important;
    background: linear-gradient(90deg, #a5b4fc, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    color: #c4b5fd !important;
    letter-spacing: -0.3px;
}

/* Step headers — pill style */
[data-testid="stHeader"] {
    background: transparent;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, var(--indigo), var(--violet));
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    font-size: 0.875rem;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(168,85,247,0.5);
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(90deg, var(--sky), var(--indigo));
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    padding: 0.55rem 1.4rem;
    box-shadow: 0 4px 15px rgba(56,189,248,0.3);
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56,189,248,0.45);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed rgba(99,102,241,0.45);
    border-radius: 14px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--violet);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* Alert / success / info */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: none;
}

/* Badge */
.badge {
    display: inline-block;
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    border: 1px solid rgba(248,113,113,0.3);
    padding: 4px 12px;
    border-radius: 20px;
    color: #fca5a5;
    font-size: 13px;
    font-weight: 500;
    margin: 3px 4px 3px 0;
}

/* Success box */
.success-box {
    background: linear-gradient(90deg, rgba(6,95,70,0.6), rgba(5,150,105,0.3));
    border: 1px solid rgba(52,211,153,0.25);
    padding: 12px 16px;
    border-radius: 12px;
    color: #6ee7b7;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
}

/* Step card wrapper */
.step-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    backdrop-filter: blur(8px);
}

/* Section divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* Sidebar nav links */
.pipeline-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 14px;
    border-radius: 10px;
    margin-bottom: 5px;
    transition: all 0.2s ease;
    color: var(--muted) !important;
    text-decoration: none !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    border: 1px solid transparent;
}
.pipeline-link:hover {
    background: linear-gradient(90deg, rgba(99,102,241,0.18), rgba(168,85,247,0.18));
    border-color: rgba(99,102,241,0.3);
    color: #e0e7ff !important;
    transform: translateX(3px);
}

/* Sidebar brand */
.sidebar-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.sidebar-subtitle {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
}

/* Metric-style stat blocks */
.stat-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 0.75rem 0;
}
.stat-chip {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    padding: 6px 14px;
    border-radius: 8px;
    font-size: 13px;
    color: #c7d2fe;
}

/* Column header label */
.col-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 6px;
}

/* Graph container card */
.graph-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR (Navigation Only)
# -------------------------------------------------
st.sidebar.markdown('<div class="sidebar-brand">🏋️ Fitness Data Pro</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Health Analytics Pipeline</div>', unsafe_allow_html=True)

st.sidebar.markdown('<a href="#upload"   class="pipeline-link">📂 &nbsp;Upload CSV</a>',           unsafe_allow_html=True)
st.sidebar.markdown('<a href="#check"    class="pipeline-link">🔍 &nbsp;Check Missing Values</a>',  unsafe_allow_html=True)
st.sidebar.markdown('<a href="#preprocess" class="pipeline-link">🛠 &nbsp;Data Preprocessing</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#preview"  class="pipeline-link">📑 &nbsp;Preview Dataset</a>',       unsafe_allow_html=True)
st.sidebar.markdown('<a href="#eda"      class="pipeline-link">📊 &nbsp;Run EDA</a>',               unsafe_allow_html=True)

st.sidebar.markdown('<hr style="border-color:rgba(255,255,255,0.07);margin:1.5rem 0">', unsafe_allow_html=True)
st.sidebar.markdown('<p style="font-size:0.72rem;color:#475569;text-align:center;">v1.0 · Fitness Data Pro</p>', unsafe_allow_html=True)

# -------------------------------------------------
# MAIN TITLE
# -------------------------------------------------
st.title("🏋️ Fitness Health Data — Pro Pipeline")
st.markdown('<p style="color:#94a3b8;font-size:0.95rem;margin-top:-0.5rem;margin-bottom:1.5rem;">Upload your CSV and move through the pipeline step by step.</p>', unsafe_allow_html=True)

# -------------------------------------------------
# STEP 1 — UPLOAD
# -------------------------------------------------
st.markdown('<div id="upload"></div>', unsafe_allow_html=True)
st.header("📂 Step 1 · Upload CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["original_df"] = df.copy()
    st.success("✅ Dataset loaded successfully!")
    st.markdown(
        f'<div class="stat-row">'
        f'<span class="stat-chip">🗂 {df.shape[0]:,} rows</span>'
        f'<span class="stat-chip">📐 {df.shape[1]} columns</span>'
        f'<span class="stat-chip">💾 {uploaded_file.size / 1024:.1f} KB</span>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# -------------------------------------------------
# STEP 2 — CHECK MISSING VALUES
# -------------------------------------------------
st.markdown('<div id="check"></div>', unsafe_allow_html=True)
st.header("🔍 Step 2 · Check Missing Values")

if "original_df" in st.session_state:
    df = st.session_state["original_df"]

    null_counts  = df.isnull().sum()
    total_rows   = len(df)
    cols_with_nulls = null_counts[null_counts > 0]

    if len(cols_with_nulls) > 0:
        badge_html = ""
        for col, val in cols_with_nulls.items():
            percent = (val / total_rows) * 100
            badge_html += f'<span class="badge">⚠ {col}: {val} ({percent:.1f}%)</span>'
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 2.2))          # ← smaller
        fig.patch.set_facecolor("#0f0c29")
        ax.set_facecolor("#1e1b4b")
        cols_with_nulls.sort_values().plot(kind="barh", ax=ax, color="#818cf8")
        ax.set_xlabel("Missing Count", color="#94a3b8", fontsize=8)
        ax.set_title("Missing Values per Column", color="#c4b5fd", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor((1.0, 1.0, 1.0, 0.08))
        fig.tight_layout(pad=0.6)
        st.pyplot(fig)
    else:
        st.markdown('<div class="success-box">🎉 No missing values detected!</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# -------------------------------------------------
# STEP 3 — PREPROCESS
# -------------------------------------------------
st.markdown('<div id="preprocess"></div>', unsafe_allow_html=True)
st.header("🛠 Step 3 · Data Preprocessing")

if "original_df" in st.session_state:

    if st.button("▶ Run Preprocessing"):

        original_df = st.session_state["original_df"]
        df   = original_df.copy()
        logs = []

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            logs.append("📅 Parsed Date column.")

        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].interpolate(method="linear")
                df[col] = df[col].bfill().ffill()
        logs.append("🔢 Interpolated numeric columns.")

        for col in df.select_dtypes(include="object").columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna("No Workout")
        logs.append("🏷 Filled categorical nulls.")

        st.session_state["cleaned_df"] = df
        st.success("✅ Preprocessing Complete")

        st.subheader("Missing Value Comparison")

        before = original_df.isnull().sum()
        after  = df.isnull().sum()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="col-label">Before</p>', unsafe_allow_html=True)
            st.dataframe(before[before > 0], use_container_width=True)
        with col2:
            st.markdown('<p class="col-label">After</p>', unsafe_allow_html=True)
            if after.sum() == 0:
                st.markdown('<div class="success-box">🎉 Zero missing values remaining!</div>', unsafe_allow_html=True)
            else:
                st.dataframe(after[after > 0], use_container_width=True)

        st.subheader("Processing Log")
        for log in logs:
            st.success(log)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# -------------------------------------------------
# STEP 4 — PREVIEW
# -------------------------------------------------
st.markdown('<div id="preview"></div>', unsafe_allow_html=True)
st.header("📑 Step 4 · Preview Cleaned Dataset")

if "cleaned_df" in st.session_state:
    st.dataframe(st.session_state["cleaned_df"].head(10), use_container_width=True)

    csv = st.session_state["cleaned_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_fitness_data.csv",
        mime="text/csv"
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# -------------------------------------------------
# STEP 5 — EDA (Smaller Graphs)
# -------------------------------------------------
st.markdown('<div id="eda"></div>', unsafe_allow_html=True)
st.header("📊 Step 5 · Exploratory Data Analysis")

if "cleaned_df" in st.session_state:

    df           = st.session_state["cleaned_df"]
    numeric_cols = df.select_dtypes(include=np.number).columns
    cols_per_row = 2

    # Shared matplotlib dark style for all EDA plots
    plt.rcParams.update({
        "figure.facecolor":  "#0f0c29",
        "axes.facecolor":    "#1a1740",
        "axes.edgecolor":    "#2d2b55",
        "axes.labelcolor":   "#94a3b8",
        "xtick.color":       "#64748b",
        "ytick.color":       "#64748b",
        "text.color":        "#94a3b8",
        "grid.color":        "#1e1b4b",
        "grid.linewidth":    0.5,
    })

    for i in range(0, len(numeric_cols), cols_per_row):
        row = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(numeric_cols):
                col = numeric_cols[i + j]
                fig, ax = plt.subplots(figsize=(3, 2))    # ← smaller graphs
                sns.histplot(
                    df[col], kde=True, ax=ax,
                    color="#818cf8",
                    line_kws={"color": "#e879f9", "linewidth": 1.2},
                )
                ax.set_title(col, fontsize=8, fontweight="bold", color="#c4b5fd", pad=4)
                ax.set_xlabel("")
                ax.tick_params(labelsize=6)
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout(pad=0.5)
                row[j].pyplot(fig)