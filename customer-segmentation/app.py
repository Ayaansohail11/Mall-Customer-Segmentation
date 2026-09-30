import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
#  ROBUST FILE PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _p(filename):
    return os.path.join(BASE_DIR, filename)

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Mall Customer Segmentation | Ayaan Sohail",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CUSTOM CSS – Navy + Gold Palette
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 60%, #0d1b2a 100%);
        color: #e2e8f0;
    }

    /* Hide default header */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(120deg, #1a2744 0%, #162032 50%, #0f1e35 100%);
        border: 1px solid #2a3a5c;
        border-radius: 18px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    }
    .hero-banner h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        color: #f5c842;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #1e2d45;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 9px;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.92rem;
        padding: 10px 22px;
        border: none;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f5c842, #e0a820) !important;
        color: #0a0e1a !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 12px rgba(245,200,66,0.35);
    }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a2744, #162032);
        border: 1px solid #2a3a5c;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card .label {
        color: #64748b;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #f5c842;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card .sub {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* ── Input styling ── */
    .stNumberInput > div > div > input,
    .stSlider { color: #e2e8f0 !important; }

    label { color: #cbd5e1 !important; font-size: 0.9rem !important; }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #f5c842;
        border-left: 4px solid #f5c842;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Developer card ── */
    .dev-card {
        background: linear-gradient(135deg, #1a2744, #162032);
        border: 1px solid #2a3a5c;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        max-width: 460px;
        margin: 0 auto;
        box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    }
    .dev-avatar {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f5c842, #e0a820);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        margin: 0 auto 1.2rem auto;
        box-shadow: 0 4px 20px rgba(245,200,66,0.3);
    }
    .dev-name {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #f5c842;
        margin-bottom: 0.2rem;
    }
    .dev-role {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #2a5080;
        color: #7dd3fc;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        margin: 3px;
    }

    /* ── Info box ── */
    .info-box {
        background: #1a2744;
        border-left: 4px solid #f5c842;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
        color: #cbd5e1;
        font-size: 0.93rem;
        line-height: 1.6;
    }
    .info-box strong { color: #f5c842; }

    /* ── Warning/Disclaimer ── */
    .disclaimer-box {
        background: #2d1515;
        border: 1px solid #7f1d1d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #fca5a5;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 1rem;
    }

    /* ── Segment Badge ── */
    .segment-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #0a0e1a;
        background: linear-gradient(90deg, #f5c842, #e0a820);
        box-shadow: 0 2px 12px rgba(245,200,66,0.3);
    }

    /* ── Card ── */
    .card {
        background: linear-gradient(135deg, #1a2744, #162032);
        border: 1px solid #2a3a5c;
        padding: 22px 26px;
        border-radius: 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* ── Prediction Result ── */
    .predict-result {
        background: linear-gradient(135deg, #1a3a1a, #0f2a0f);
        border: 2px solid #22c55e;
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(34,197,94,0.15);
        margin-top: 1.5rem;
    }
    .predict-result .price {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        color: #22c55e;
        font-weight: 700;
    }
    .predict-result .label {
        color: #86efac;
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
#  LOAD MODEL / SCALER
# ============================================================
MODEL_PATH = _p("kmeans_model.pkl")
SCALER_PATH = _p("scaler.pkl")
RAW_CSV_PATH = _p("shopping_Custmer_Data.csv")
STATS_PATH = _p("model_stats.json")
CLUSTERED_CSV_PATH = _p("clustered_data.csv")

missing = [p for p in (MODEL_PATH, SCALER_PATH, RAW_CSV_PATH) if not os.path.exists(p)]
if missing:
    st.error(
        "❌ Missing required file(s):\n\n"
        + "\n".join(f"- {os.path.basename(p)}" for p in missing)
        + "\n\nMake sure `kmeans_model.pkl`, `scaler.pkl`, and "
        "`shopping_Custmer_Data.csv` are in the **same folder** as `app.py`, then rerun the app."
    )
    st.stop()


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


model, scaler = load_model()


def compute_segment_names(cluster_means: pd.DataFrame) -> dict:
    """Build human-readable segment labels from cluster mean income/spending."""
    income_median = cluster_means["Annual Income"].median()
    spending_median = cluster_means["Spending Score"].median()
    names = {}
    for c, row in cluster_means.iterrows():
        inc_level = "High Income" if row["Annual Income"] >= income_median else "Moderate Income"
        spend_level = "High Spending" if row["Spending Score"] >= spending_median else "Low Spending"
        names[int(c)] = f"{inc_level}, {spend_level} Customers"
    return names


@st.cache_data
def load_dataset_and_stats():
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.cluster import KMeans

    raw_df = pd.read_csv(RAW_CSV_PATH)
    X = raw_df.iloc[:, [3, 4]]
    X_scaled = scaler.transform(X)
    clusters = model.predict(X_scaled)

    if os.path.exists(CLUSTERED_CSV_PATH):
        df = pd.read_csv(CLUSTERED_CSV_PATH)
    else:
        df = raw_df.copy()
        df["Cluster"] = clusters

    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "r") as f:
            stats = json.load(f)
        stats["segment_names"] = {int(k): v for k, v in stats["segment_names"].items()}
    else:
        cluster_means = (
            pd.DataFrame(X, columns=["Annual Income", "Spending Score"])
            .assign(Cluster=clusters)
            .groupby("Cluster")
            .mean()
        )
        segment_names = compute_segment_names(cluster_means)

        wcss = []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
            km.fit(X_scaled)
            wcss.append(float(km.inertia_))

        stats = {
            "silhouette_score": float(silhouette_score(X_scaled, clusters)),
            "davies_bouldin_score": float(davies_bouldin_score(X_scaled, clusters)),
            "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, clusters)),
            "wcss": wcss,
            "n_clusters": int(model.n_clusters) if hasattr(model, "n_clusters") else len(cluster_means),
            "segment_names": segment_names,
            "cluster_means": cluster_means.to_dict(orient="index"),
            "n_rows": int(len(raw_df)),
            "n_features": 2,
        }

    if "Customer Segment" not in df.columns:
        df["Customer Segment"] = df["Cluster"].map(stats["segment_names"])

    return df, stats


df, stats = load_dataset_and_stats()
SEGMENT_NAMES = {int(k): v for k, v in stats["segment_names"].items()}

SEGMENT_ADVICE = {
    "Moderate Income, Low Spending Customers": (
        "🏷️", "Discount & Value Seekers",
        "Ye customers madhyam aay wale hain lekin soch samajh ke kharcha karte hain. "
        "Inhe discount offers, combo deals aur loyalty points se attract karein.",
    ),
    "Moderate Income, High Spending Customers": (
        "💜", "Loyal Everyday Shoppers",
        "Ye apni aay ke hisaab se zyada kharcha karte hain — ek loyal aur engaged segment. "
        "Inhe loyalty rewards, festival offers aur early sale access dein.",
    ),
    "High Income, High Spending Customers": (
        "👑", "Premium / VIP Customers",
        "Zyada kamaane wale aur khulke kharcha karne wale customers. "
        "Inke liye premium brands, exclusive memberships aur personalized VIP experience best hai.",
    ),
    "High Income, Low Spending Customers": (
        "🎯", "Potential / Untapped Customers",
        "Zyada aay hai lekin kharcha kam karte hain. Yahan bahut potential hai — "
        "targeted marketing aur curated recommendations se inhe engage karein.",
    ),
}

# Color constants for Plotly charts
PLOT_BG = "rgba(15,24,41,0)"
PAPER_BG = "rgba(0,0,0,0)"
GOLD = "#f5c842"
GREEN = "#22c55e"
BLUE = "#38bdf8"
PURPLE = "#a78bfa"
PINK = "#f472b6"
FONT_CLR = "#cbd5e1"
GRID_CLR = "#1e2d45"


def base_layout(title):
    return dict(
        title=dict(text=title, font=dict(color=GOLD, size=15, family="Inter")),
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, family="Inter"),
        margin=dict(l=20, r=20, t=50, b=20),
    )


# ============================================================
#  HERO BANNER
# ============================================================
st.markdown(
    """
    <div class="hero-banner">
        <h1>🛍️ Mall Customer Segmentation</h1>
        <p>AI-powered K-Means clustering · Discover Indian customer segments · Built by Ayaan Sohail</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
#  TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮  Predict Segment", "📊  Data Visualization", "🎯  Model Accuracy & Dataset", "👨‍💻  About & Developer"]
)

# ------------------------------------------------------------
# TAB 1 — PREDICTION
# ------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Enter Customer Details</div>', unsafe_allow_html=True)
    st.caption(
        "Name, Age, and Gender are collected for a personalized report only — "
        "the model predicts the segment using **Annual Income** and **Spending Score** alone, "
        "exactly like the trained K-Means model."
    )

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("🧑 Customer Name", placeholder="e.g. Ayaan Sohail")
        age = st.number_input("🎂 Age", min_value=10, max_value=100, step=1)
        gender = st.radio("⚧ Gender", ["Male", "Female"], horizontal=True)

    with col2:
        income = st.number_input(
            "💰 Annual Income (₹ in thousands)",
            min_value=0.0,
            step=1.0,
            help="Enter income in thousands. E.g. 50 means ₹50,000. Used by the model for prediction.",
        )
        spending = st.slider(
            "🛒 Spending Score (1-100)",
            1,
            100,
            50,
            help="Used by the model for prediction.",
        )

    predict_clicked = st.button("✨ Predict Customer Segment", use_container_width=True, type="primary")

    if predict_clicked:
        user_data = np.array([[income, spending]])
        user_data_scaled = scaler.transform(user_data)
        cluster = int(model.predict(user_data_scaled)[0])
        segment = SEGMENT_NAMES.get(cluster, "Unknown Segment")
        emoji, short_label, advice = SEGMENT_ADVICE.get(
            segment, ("🔎", "Segment", "No additional info available.")
        )

        st.markdown("---")
        greet_name = name.strip() if name and name.strip() else "Customer"
        st.markdown(
            f"""
            <div class="card">
                <h3>{emoji} Hello, {greet_name}!</h3>
                <p style="font-size:1rem; color:#cbd5e1;">
                    Age: <b>{age}</b> &nbsp;|&nbsp; Gender: <b>{gender}</b> &nbsp;|&nbsp;
                    Annual Income: <b>₹{income:,.0f}k</b> &nbsp;|&nbsp; Spending Score: <b>{spending}</b>
                </p>
                <p>Cluster assigned: <b>{cluster}</b></p>
                <span class="segment-badge">{short_label}</span>
                <p style="margin-top:14px;">{advice}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Visual: where does this customer fall among all clusters?
        fig = px.scatter(
            df,
            x="Annual Income",
            y="Spending Score",
            color="Customer Segment",
            opacity=0.35,
            title="Where this customer falls among existing clusters",
            template="plotly_dark",
        )
        fig.update_layout(**base_layout("Where this customer falls among existing clusters"))
        fig.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
        fig.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
        fig.add_trace(
            go.Scatter(
                x=[income],
                y=[spending],
                mode="markers",
                marker=dict(size=22, color=GOLD, symbol="star", line=dict(width=2, color="#0a0e1a")),
                name=f"{greet_name} (You)",
            )
        )
        fig.update_layout(legend=dict(orientation="h", y=-0.2), height=480)
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# TAB 2 — DATA VISUALIZATION
# ------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Explore the Dataset</div>', unsafe_allow_html=True)

    plot_df = df.copy()
    if len(plot_df) > 3000:
        plot_df_sample = plot_df.sample(3000, random_state=42)
    else:
        plot_df_sample = plot_df

    c1, c2 = st.columns(2)

    with c1:
        seg_counts = df["Customer Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_pie = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            title="Customer Segment Distribution",
            hole=0.45,
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma,
        )
        fig_pie.update_layout(**base_layout("Customer Segment Distribution"))
        fig_pie.update_traces(textinfo="percent+label", textfont_color=FONT_CLR)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_gender = px.pie(
            df,
            names="Gender",
            title="Gender Distribution",
            hole=0.45,
            template="plotly_dark",
            color_discrete_sequence=["#f472b6", "#60a5fa", "#a78bfa"],
        )
        fig_gender.update_layout(**base_layout("Gender Distribution"))
        fig_gender.update_traces(textinfo="percent+label", textfont_color=FONT_CLR)
        st.plotly_chart(fig_gender, use_container_width=True)

    fig_scatter = px.scatter(
        plot_df_sample,
        x="Annual Income",
        y="Spending Score",
        color="Customer Segment",
        title="Annual Income vs Spending Score (by Segment)",
        template="plotly_dark",
        opacity=0.75,
    )
    fig_scatter.update_layout(**base_layout("Annual Income vs Spending Score (by Segment)"))
    fig_scatter.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig_scatter.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig_age = px.histogram(
            df,
            x="Age",
            color="Customer Segment",
            nbins=30,
            title="Age Distribution by Segment",
            template="plotly_dark",
        )
        fig_age.update_layout(**base_layout("Age Distribution by Segment"))
        fig_age.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
        fig_age.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
        st.plotly_chart(fig_age, use_container_width=True)

    with c4:
        fig_box = px.box(
            df,
            x="Customer Segment",
            y="Spending Score",
            color="Customer Segment",
            title="Spending Score Spread per Segment",
            template="plotly_dark",
        )
        fig_box.update_layout(**base_layout("Spending Score Spread per Segment"))
        fig_box.update_layout(showlegend=False, xaxis_title="")
        fig_box.update_xaxes(gridcolor=GRID_CLR)
        fig_box.update_yaxes(gridcolor=GRID_CLR)
        st.plotly_chart(fig_box, use_container_width=True)

    fig_income_box = px.violin(
        df,
        x="Customer Segment",
        y="Annual Income",
        color="Customer Segment",
        box=True,
        points=False,
        title="Annual Income Distribution per Segment",
        template="plotly_dark",
    )
    fig_income_box.update_layout(**base_layout("Annual Income Distribution per Segment"))
    fig_income_box.update_layout(showlegend=False, xaxis_title="", height=480)
    fig_income_box.update_xaxes(gridcolor=GRID_CLR)
    fig_income_box.update_yaxes(gridcolor=GRID_CLR)
    st.plotly_chart(fig_income_box, use_container_width=True)

# ------------------------------------------------------------
# TAB 3 — MODEL ACCURACY & DATASET KNOWLEDGE
# ------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Clustering Quality Metrics</div>', unsafe_allow_html=True)
    st.caption(
        "K-Means is unsupervised, so there is no 'accuracy' in the classification sense. "
        "Instead, clustering quality is judged with the metrics below."
    )

    m1, m2, m3 = st.columns(3)
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">
            <div class="metric-card">
                <div class="label">Silhouette Score</div>
                <div class="value">{stats['silhouette_score']:.3f}</div>
                <div class="sub">Closer to 1 is better</div>
            </div>
            <div class="metric-card">
                <div class="label">Davies–Bouldin Index</div>
                <div class="value">{stats['davies_bouldin_score']:.3f}</div>
                <div class="sub">Lower is better</div>
            </div>
            <div class="metric-card">
                <div class="label">Calinski–Harabasz Score</div>
                <div class="value">{stats['calinski_harabasz_score']:.0f}</div>
                <div class="sub">Higher is better</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Elbow Method — Choosing k</div>', unsafe_allow_html=True)
    wcss = stats["wcss"]
    fig_elbow = go.Figure()
    fig_elbow.add_trace(
        go.Scatter(
            x=list(range(1, len(wcss) + 1)),
            y=wcss,
            mode="lines+markers",
            line=dict(color=GOLD, width=3),
            marker=dict(size=9, color=PINK),
        )
    )
    fig_elbow.add_vline(x=stats["n_clusters"], line_dash="dash", line_color=BLUE)
    fig_elbow.update_layout(
        **base_layout("WCSS vs Number of Clusters (k)"),
        xaxis_title="Number of clusters (k)",
        yaxis_title="WCSS (Inertia)",
        height=430,
    )
    fig_elbow.update_xaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    fig_elbow.update_yaxes(gridcolor=GRID_CLR, zerolinecolor=GRID_CLR)
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.info(f"The model uses **k = {stats['n_clusters']}** clusters, chosen at the elbow point above.")

    st.markdown('<div class="section-title">Cluster Centers (Original Scale)</div>', unsafe_allow_html=True)
    means_df = pd.DataFrame(stats["cluster_means"]).T
    means_df.index.name = "Cluster"
    means_df["Segment"] = [SEGMENT_NAMES.get(int(i), "") for i in means_df.index]
    st.dataframe(means_df.style.format({"Annual Income": "{:.0f}", "Spending Score": "{:.1f}"}), use_container_width=True)

    st.markdown('<div class="section-title">Dataset Knowledge</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;">
            <div class="metric-card">
                <div class="label">Total Records</div>
                <div class="value">{stats['n_rows']:,}</div>
            </div>
            <div class="metric-card">
                <div class="label">Features Used</div>
                <div class="value">{stats['n_features']}</div>
            </div>
            <div class="metric-card">
                <div class="label">Clusters Formed</div>
                <div class="value">{stats['n_clusters']}</div>
            </div>
            <div class="metric-card">
                <div class="label">Columns</div>
                <div class="value">{df.shape[1]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Preview raw dataset"):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("📊 Statistical summary (describe)"):
        st.dataframe(df.describe(), use_container_width=True)

    with st.expander("ℹ️ Column info"):
        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [str(t) for t in df.dtypes],
                "Missing Values": df.isnull().sum().values,
            }
        )
        st.dataframe(info_df, use_container_width=True)

# ------------------------------------------------------------
# TAB 4 — ABOUT / DEVELOPER / TERMS
# ------------------------------------------------------------
with tab4:
    st.markdown(
        f"""
        <div class="dev-card">
            <div class="dev-avatar">👨‍💻</div>
            <div class="dev-name">Ayaan Sohail</div>
            <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.6;margin-bottom:1rem;">
                Built this app to demonstrate unsupervised learning (K-Means Clustering) for customer
                segmentation, wrapped in an interactive Streamlit interface.
            </div>
            <div>
                <span class="badge">Python</span>
                <span class="badge">Streamlit</span>
                <span class="badge">scikit-learn</span>
                <span class="badge">Plotly</span>
                <span class="badge">K-Means</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="info-box">
            <strong>📘 About This App</strong><br>
            This application uses a <b>K-Means Clustering</b> model trained on mall customer data
            to group customers into meaningful segments based on their <b>Annual Income (in ₹ thousands)</b> and
                    <b>Spending Score</b>. Businesses across India can use these insights to design targeted marketing
                    strategies, loyalty programs, and personalized offers for each customer segment.
        </div>

        <div class="info-box">
            <strong>🧭 How to Use</strong><br>
            1. Go to the <b>Predict Segment</b> tab.<br>
            2. Enter the customer's Name, Age, and Gender (for personalization).<br>
            3. Enter the Annual Income (in ₹ thousands) and Spending Score (used for the actual prediction).<br>
            4. Click <b>Predict Customer Segment</b> to view the assigned cluster and segment.<br>
            5. Explore the <b>Data Visualization</b> tab for interactive charts on the full dataset.<br>
            6. Check the <b>Model Accuracy & Dataset</b> tab to understand how well the model performs.
        </div>

        <div class="info-box">
            <strong>📜 Terms & Conditions</strong><br>
            By using this application, you agree that it is provided for
            <b>educational and demonstrational purposes only</b>. The developer is not liable for
            any decisions made based on the outputs of this app. Do not use this app to store or
            process sensitive personal data. All data entered during a session stays within that
            session and is not saved on any server by this app.
        </div>

        <div class="disclaimer-box">
            <strong>⚠️ Disclaimer</strong><br>
            The predictions generated by this app are based on a statistical clustering model
            trained on a sample dataset and <b>do not represent real financial, marketing, or
            business advice</b>. Segment labels and recommendations are illustrative. Always
            validate insights with real business data and domain experts before making
            decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#64748b;font-size:0.82rem;padding:1rem 0;border-top:1px solid #1e2d45;">
            © 2026 Ayaan Sohail · Mall Customer Segmentation v1.0 · Built with ❤️ using Python & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )