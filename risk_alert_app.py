import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Alert Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%);
    border-right: 1px solid rgba(99, 179, 237, 0.15);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Header Banner */
.hero-banner {
    background: linear-gradient(135deg, #1a2744 0%, #0f2027 40%, #1a1a2e 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #90cdf4, #bee3f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    font-size: 1rem;
    color: #718096;
    font-weight: 400;
    margin: 0;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1a2744, #1e2d45);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover {
    border-color: rgba(99, 179, 237, 0.5);
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #63b3ed;
}
.metric-label {
    font-size: 0.8rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Section Headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #90cdf4;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99, 179, 237, 0.2);
    margin: 1.5rem 0 1rem 0;
}

/* Risk Badge */
.risk-high {
    background: linear-gradient(135deg, #742a2a, #c53030);
    color: #fed7d7;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.2rem;
    display: inline-block;
    border: 1px solid rgba(252,129,129,0.4);
}
.risk-low {
    background: linear-gradient(135deg, #1a4731, #276749);
    color: #c6f6d5;
    padding: 0.6rem 1.4rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.2rem;
    display: inline-block;
    border: 1px solid rgba(104,211,145,0.4);
}

/* Prediction Box */
.pred-box {
    background: linear-gradient(135deg, #1a2744, #1e2d45);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(26, 39, 68, 0.6);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(99, 179, 237, 0.15);
}
.stTabs [data-baseweb="tab"] {
    color: #718096 !important;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a4a7a, #1e5799) !important;
    color: #bee3f8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a4a7a, #2b6cb0);
    color: #bee3f8;
    border: 1px solid rgba(99,179,237,0.4);
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2b6cb0, #3182ce);
    border-color: #63b3ed;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(99,179,237,0.2);
}

/* Sliders & inputs */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem;
}

/* Info boxes */
.info-box {
    background: rgba(26, 74, 122, 0.2);
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #a0aec0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── ML Imports ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, roc_curve, roc_auc_score
)
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# ─── Plotly Theme ───────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,22,41,0.6)",
    font=dict(family="Inter", color="#a0aec0", size=12),
    title_font=dict(family="Space Grotesk", color="#90cdf4", size=15),
    legend=dict(
        bgcolor="rgba(15,22,41,0.8)",
        bordercolor="rgba(99,179,237,0.2)",
        borderwidth=1
    ),
    colorway=["#63b3ed", "#fc8181", "#68d391", "#f6ad55", "#b794f4", "#76e4f7"],
)

# Reusable helpers — applied per-chart to avoid duplicate-key conflicts
AXIS_STYLE    = dict(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.2)")
MARGIN_DEF    = dict(t=50, b=40, l=40, r=20)

COLORS = {
    "primary":   "#63b3ed",
    "danger":    "#fc8181",
    "success":   "#68d391",
    "warning":   "#f6ad55",
    "purple":    "#b794f4",
    "teal":      "#76e4f7",
    "grid":      "rgba(99,179,237,0.08)",
}

# ─── Data Loading & Preprocessing ───────────────────────────────────────────
@st.cache_data
def load_and_prepare(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # KNN impute numeric
    num_cols = ["age", "annual_income_inr", "credit_score",
                "credit_utilization_ratio", "monthly_spend_inr"]
    present_num = [c for c in num_cols if c in df.columns]
    if present_num:
        imputer = KNNImputer(n_neighbors=5)
        df[present_num] = imputer.fit_transform(df[present_num])

    # Mode-fill categoricals
    for col in ["region", "employment_type"]:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df

@st.cache_resource
def train_all_models(df):
    df_m = df.copy()
    cat_cols = [c for c in ["gender", "region", "employment_type", "last_transaction_date"]
                if c in df_m.columns]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_m[col] = le.fit_transform(df_m[col].astype(str))
        encoders[col] = le

    X = df_m.drop(["risk_status", "customer_id"], axis=1, errors="ignore")
    y = df_m["risk_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Balance with SMOTE (best per notebook)
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree":       DecisionTreeClassifier(random_state=42, max_depth=7),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_bal, y_bal)
        yp = m.predict(X_test)
        yprob = m.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, yp), 4),
            "precision": round(precision_score(y_test, yp), 4),
            "recall":    round(recall_score(y_test, yp), 4),
            "f1":        round(f1_score(y_test, yp), 4),
            "auc":       round(roc_auc_score(y_test, yprob), 4),
            "cm":        confusion_matrix(y_test, yp),
            "fpr":       roc_curve(y_test, yprob)[0],
            "tpr":       roc_curve(y_test, yprob)[1],
            "y_pred":    yp,
        }
        trained[name] = m

    # Balancing comparison
    samplers = {
        "No Balancing":    None,
        "Under-Sampling":  RandomUnderSampler(random_state=42),
        "Over-Sampling":   RandomOverSampler(random_state=42),
        "SMOTE":           SMOTE(random_state=42),
        "ADASYN":          ADASYN(random_state=42),
    }
    balance_results = []
    for bname, samp in samplers.items():
        if samp is None:
            Xr, yr = X_train, y_train
        else:
            Xr, yr = samp.fit_resample(X_train, y_train)
        m = LogisticRegression(max_iter=1000)
        m.fit(Xr, yr)
        yp = m.predict(X_test)
        yprob = m.predict_proba(X_test)[:, 1]
        balance_results.append({
            "Method":  bname,
            "Recall":  round(recall_score(y_test, yp), 4),
            "F1":      round(f1_score(y_test, yp), 4),
            "AUC-ROC": round(roc_auc_score(y_test, yprob), 4),
        })

    feat_imp = pd.DataFrame({
        "Feature":   X.columns,
        "Importance": trained["Random Forest"].feature_importances_
    }).sort_values("Importance", ascending=False)

    return {
        "results":         results,
        "trained":         trained,
        "X_test":          X_test,
        "y_test":          y_test,
        "balance_results": balance_results,
        "feat_imp":        feat_imp,
        "X_columns":       X.columns.tolist(),
        "encoders":        encoders,
        "cat_cols":        cat_cols,
        "df_raw":          df,
    }

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding-bottom:1.5rem;'>
        <span style='font-size:2.5rem;'>🛡️</span>
        <p style='font-family:Space Grotesk; font-size:1.1rem; font-weight:700;
                  color:#90cdf4; margin:0.3rem 0 0 0;'>Risk Alert Classifier</p>
        <p style='font-size:0.75rem; color:#4a5568; margin:0;'>Customer Risk Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        help="Upload Risk_Alert_Classifier_Dataset CSV"
    )

    st.markdown("---")
    st.markdown("""<p style='font-size:0.78rem; color:#4a5568;'>
        <b style='color:#63b3ed'>Models:</b> Logistic Regression · Decision Tree · Random Forest<br><br>
        <b style='color:#63b3ed'>Balancing:</b> Under/Over-Sampling · SMOTE · ADASYN<br><br>
        <b style='color:#63b3ed'>Metrics:</b> Accuracy · Precision · Recall · F1 · AUC-ROC
    </p>""", unsafe_allow_html=True)

# ─── Hero Banner ────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <p class='hero-title'>🛡️ Risk Alert Classifier</p>
    <p class='hero-sub'>
        Identify high-risk customers using ML — Logistic Regression, Decision Tree &amp; Random Forest
        with imbalance handling and full model explainability.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Gate: need data ────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class='info-box'>
        📂 <b>Upload your dataset</b> using the sidebar to begin analysis.
        Expected: <code>Risk_Alert_Classifier_Dataset_4600.csv</code>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for c, icon, title, desc in [
        (col1, "🔬", "EDA", "Explore distributions, correlations, class balance, and key feature statistics."),
        (col2, "⚙️", "Model Training", "Train LR, DT, and RF models with SMOTE balancing. Compare metrics."),
        (col3, "🎯", "Live Prediction", "Input customer features and get an instant risk classification."),
    ]:
        with c:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; padding:1.5rem;'>
                <span style='font-size:1.8rem'>{icon}</span>
                <p style='font-family:Space Grotesk; font-size:1rem; font-weight:600;
                          color:#90cdf4; margin:0.5rem 0 0.3rem 0;'>{title}</p>
                <p style='font-size:0.82rem; color:#718096; margin:0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ─── Load Data ──────────────────────────────────────────────────────────────
with st.spinner("Loading & training models…"):
    df = load_and_prepare(uploaded)
    cache = train_all_models(df)

results   = cache["results"]
trained   = cache["trained"]
feat_imp  = cache["feat_imp"]
bal_res   = cache["balance_results"]
X_columns = cache["X_columns"]
encoders  = cache["encoders"]
cat_cols  = cache["cat_cols"]

# ─── Top KPIs ────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]

col1, col2, col3, col4, col5 = st.columns(5)
for col, label, val in zip(
    [col1, col2, col3, col4, col5],
    ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
    [best["accuracy"], best["precision"], best["recall"], best["f1"], best["auc"]]
):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val:.2%}</div>
            <div class='metric-label'>{label}</div>
            <div style='font-size:0.7rem; color:#4a5568; margin-top:0.15rem;'>{best_name}</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Analysis",
    "🤖 Model Performance",
    "⚖️ Imbalance Handling",
    "🎯 Live Prediction",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    # Raw shape stats
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Total Records", "Features", "High Risk", "Low Risk"],
        [
            len(df),
            len(df.columns) - 2,
            int(df["risk_status"].sum()),
            int((df["risk_status"] == 0).sum()),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val:,}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Target Class Distribution</div>", unsafe_allow_html=True)
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        vc = df["risk_status"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Low Risk (0)", "High Risk (1)"],
            values=[vc.get(0, 0), vc.get(1, 0)],
            hole=0.55,
            marker=dict(colors=[COLORS["success"], COLORS["danger"]],
                        line=dict(color="#0a0e1a", width=2)),
            textfont=dict(color="#e2e8f0"),
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, title="Class Split", height=300,
                              showlegend=True,
                              margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with cc2:
        num_feats = ["age", "credit_score", "annual_income_inr",
                     "credit_utilization_ratio", "missed_payments_12m", "monthly_spend_inr"]
        num_feats = [f for f in num_feats if f in df.columns]
        sel_feat  = st.selectbox("Feature distribution", num_feats, index=1)
        fig_hist  = px.histogram(
            df, x=sel_feat, color="risk_status",
            barmode="overlay", nbins=40,
            color_discrete_map={0: COLORS["success"], 1: COLORS["danger"]},
            labels={"risk_status": "Risk"},
            opacity=0.75,
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, title=f"{sel_feat} by Risk Class", height=300,
                               xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, margin=MARGIN_DEF)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div class='section-header'>Feature Importances (Random Forest)</div>", unsafe_allow_html=True)
    fig_fi = px.bar(
        feat_imp.head(12), x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=[[0, "#1a3a5c"], [0.5, "#2b6cb0"], [1, "#63b3ed"]],
    )
    fig_fi.update_layout(**PLOTLY_LAYOUT, title="Top 12 Feature Importances", height=380,
                         coloraxis_showscale=False,
                         xaxis=AXIS_STYLE,
                         yaxis=dict(autorange="reversed", **AXIS_STYLE),
                         margin=MARGIN_DEF)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
    num_df = df.select_dtypes(include=np.number).drop(
        ["customer_id"], axis=1, errors="ignore"
    )
    corr = num_df.corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="Blues",
        zmin=-1, zmax=1, aspect="auto",
        color_continuous_midpoint=0,
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, title="Pearson Correlation Matrix", height=420,
                           margin=MARGIN_DEF)
    st.plotly_chart(fig_corr, use_container_width=True)

    if "region" in df.columns:
        st.markdown("<div class='section-header'>Risk by Region</div>", unsafe_allow_html=True)
        reg_df = df.groupby("region")["risk_status"].mean().reset_index()
        reg_df.columns = ["Region", "High Risk Rate"]
        fig_reg = px.bar(
            reg_df, x="Region", y="High Risk Rate",
            color="High Risk Rate",
            color_continuous_scale=[[0, "#276749"], [0.5, "#f6ad55"], [1, "#c53030"]],
        )
        fig_reg.update_layout(**PLOTLY_LAYOUT, title="High Risk Rate by Region",
                              coloraxis_showscale=False, height=320,
                              xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, margin=MARGIN_DEF)
        st.plotly_chart(fig_reg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Model Comparison</div>", unsafe_allow_html=True)

    metrics_df = pd.DataFrame([
        {"Model": k, **{m: results[k][m.lower().replace("-", "")]
                         if m.lower().replace("-", "") in results[k]
                         else results[k].get({"AUC-ROC":"auc","F1":"f1",
                                               "Accuracy":"accuracy","Precision":"precision",
                                               "Recall":"recall"}[m]) for m in
                        ["Accuracy","Precision","Recall","F1","AUC-ROC"]}}
        for k in results
    ])

    fig_bar = px.bar(
        metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=[COLORS["primary"], COLORS["warning"], COLORS["success"]],
    )
    fig_bar.update_layout(**PLOTLY_LAYOUT, title="All Models — All Metrics", height=380,
                          yaxis=dict(range=[0, 1.05], **AXIS_STYLE),
                          xaxis=AXIS_STYLE, margin=MARGIN_DEF)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ROC Curves
    st.markdown("<div class='section-header'>ROC Curves</div>", unsafe_allow_html=True)
    roc_colors = [COLORS["primary"], COLORS["warning"], COLORS["success"]]
    fig_roc = go.Figure()
    for (name, r), color in zip(results.items(), roc_colors):
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{name} (AUC={r['auc']:.3f})",
            line=dict(color=color, width=2.5),
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color="#4a5568", dash="dash", width=1.5),
    ))
    fig_roc.update_layout(
        **PLOTLY_LAYOUT, title="ROC Curve Comparison", height=400,
        xaxis=dict(title="False Positive Rate", **AXIS_STYLE),
        yaxis=dict(title="True Positive Rate", **AXIS_STYLE),
        margin=MARGIN_DEF,
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion Matrices
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    cm_cols = st.columns(3)
    for i, (name, r) in enumerate(results.items()):
        cm = r["cm"]
        with cm_cols[i]:
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                x=["Low Risk", "High Risk"],
                y=["Low Risk", "High Risk"],
                color_continuous_scale=[[0, "#0f2027"], [0.5, "#1a4a7a"], [1, "#63b3ed"]],
            )
            fig_cm.update_layout(
                **PLOTLY_LAYOUT,
                title=name, height=280,
                coloraxis_showscale=False,
                xaxis=AXIS_STYLE, yaxis=AXIS_STYLE,
                margin=dict(t=45, b=30, l=30, r=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    # Metrics table
    st.markdown("<div class='section-header'>Performance Summary Table</div>", unsafe_allow_html=True)
    st.dataframe(
        metrics_df.style
            .highlight_max(subset=["Accuracy","Precision","Recall","F1","AUC-ROC"],
                           color="#1a4731", axis=0)
            .format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1","AUC-ROC"]}),
        use_container_width=True, hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Imbalance Handling
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Effect of Balancing Techniques on Logistic Regression</div>",
                unsafe_allow_html=True)

    bal_df = pd.DataFrame(bal_res)

    fig_bal = px.bar(
        bal_df.melt(id_vars="Method", var_name="Metric", value_name="Score"),
        x="Method", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=[COLORS["primary"], COLORS["warning"], COLORS["purple"]],
    )
    fig_bal.update_layout(**PLOTLY_LAYOUT,
                          title="Recall · F1 · AUC-ROC by Balancing Method",
                          height=380,
                          xaxis=AXIS_STYLE,
                          yaxis=dict(range=[0, 1.05], **AXIS_STYLE),
                          margin=MARGIN_DEF)
    st.plotly_chart(fig_bal, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_recall = px.line(
            bal_df, x="Method", y="Recall", markers=True,
            color_discrete_sequence=[COLORS["danger"]],
        )
        fig_recall.update_traces(marker=dict(size=9, symbol="circle"))
        fig_recall.update_layout(**PLOTLY_LAYOUT, title="Recall Improvement", height=300,
                                  xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, margin=MARGIN_DEF)
        st.plotly_chart(fig_recall, use_container_width=True)

    with c2:
        fig_auc = px.line(
            bal_df, x="Method", y="AUC-ROC", markers=True,
            color_discrete_sequence=[COLORS["teal"]],
        )
        fig_auc.update_traces(marker=dict(size=9, symbol="diamond"))
        fig_auc.update_layout(**PLOTLY_LAYOUT, title="AUC-ROC Improvement", height=300,
                               xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, margin=MARGIN_DEF)
        st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown("<div class='section-header'>Class Distribution After Balancing</div>",
                unsafe_allow_html=True)
    orig_low  = int((df["risk_status"] == 0).sum())
    orig_high = int(df["risk_status"].sum())
    smote_n   = max(orig_low, orig_high)
    dist_data = {
        "Method":    ["Original", "Under-Sampling", "Over-Sampling", "SMOTE", "ADASYN"],
        "Low Risk":  [orig_low, orig_high, orig_low, smote_n, smote_n],
        "High Risk": [orig_high, orig_high, orig_low, smote_n, smote_n],
    }
    dist_df   = pd.DataFrame(dist_data)
    fig_dist  = px.bar(
        dist_df.melt(id_vars="Method", var_name="Class", value_name="Count"),
        x="Method", y="Count", color="Class", barmode="group",
        color_discrete_map={"Low Risk": COLORS["success"], "High Risk": COLORS["danger"]},
    )
    fig_dist.update_layout(**PLOTLY_LAYOUT,
                           title="Approximate Class Sizes After Each Method", height=340,
                           xaxis=AXIS_STYLE, yaxis=AXIS_STYLE, margin=MARGIN_DEF)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.dataframe(bal_df.style.highlight_max(
        subset=["Recall","F1","AUC-ROC"], color="#1a4731"
    ).format({c:"{:.4f}" for c in ["Recall","F1","AUC-ROC"]}),
    use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Customer Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Fill in the customer details below and click <b>Predict Risk</b> to get an
        instant classification from all three trained models.
    </div>""", unsafe_allow_html=True)

    with st.form("pred_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**👤 Demographics**")
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            region = st.selectbox("Region", sorted(df["region"].dropna().unique().tolist()))
            employment_type = st.selectbox(
                "Employment Type",
                sorted(df["employment_type"].dropna().unique().tolist())
            )

        with col_b:
            st.markdown("**💰 Financial Profile**")
            annual_income_inr = st.number_input(
                "Annual Income (INR)", 50000, 5000000, 600000, step=10000
            )
            credit_score = st.slider("Credit Score", 300, 900, 680)
            credit_utilization_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.35, 0.01)
            monthly_spend_inr = st.number_input(
                "Monthly Spend (INR)", 1000, 200000, 25000, step=1000
            )
            debt_balance_inr = st.number_input(
                "Debt Balance (INR)", 0, 2000000, 80000, step=5000
            )

        with col_c:
            st.markdown("**⚠️ Risk Indicators**")
            missed_payments_12m   = st.slider("Missed Payments (12m)", 0, 12, 1)
            avg_late_payment_days = st.slider("Avg Late Payment Days", 0, 90, 5)
            cash_advance_count_6m = st.slider("Cash Advances (6m)", 0, 10, 0)
            complaints_last_6m    = st.slider("Complaints (6m)", 0, 10, 0)
            failed_login_attempts_3m = st.slider("Failed Logins (3m)", 0, 20, 1)
            monthly_transaction_count = st.slider("Monthly Transactions", 0, 100, 15)
            account_tenure_months     = st.slider("Account Tenure (months)", 1, 240, 36)

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True)

    if submitted:
        # Build feature row
        input_dict = {
            "age":                        age,
            "gender":                     gender,
            "region":                     region,
            "employment_type":            employment_type,
            "annual_income_inr":          annual_income_inr,
            "credit_score":               credit_score,
            "credit_utilization_ratio":   credit_utilization_ratio,
            "missed_payments_12m":        missed_payments_12m,
            "avg_late_payment_days":      avg_late_payment_days,
            "monthly_transaction_count":  monthly_transaction_count,
            "monthly_spend_inr":          monthly_spend_inr,
            "cash_advance_count_6m":      cash_advance_count_6m,
            "complaints_last_6m":         complaints_last_6m,
            "failed_login_attempts_3m":   failed_login_attempts_3m,
            "account_tenure_months":      account_tenure_months,
            "last_transaction_date":      df["last_transaction_date"].mode()[0]
                                          if "last_transaction_date" in df.columns else "2024-01-01",
            "debt_balance_inr":           debt_balance_inr,
        }
        input_df = pd.DataFrame([input_dict])

        # Encode
        for col in cat_cols:
            if col in input_df.columns and col in encoders:
                le = encoders[col]
                val = input_df[col].values[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])
                else:
                    input_df[col] = le.transform([le.classes_[0]])

        # Align columns
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_columns]

        # Predictions
        preds = {}
        for mname, model in trained.items():
            pred   = model.predict(input_df)[0]
            prob   = model.predict_proba(input_df)[0][1]
            preds[mname] = {"pred": pred, "prob": prob}

        # Display
        st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)
        res_cols = st.columns(3)
        for i, (mname, p) in enumerate(preds.items()):
            with res_cols[i]:
                risk_label = "🚨 HIGH RISK" if p["pred"] == 1 else "✅ LOW RISK"
                badge_class = "risk-high" if p["pred"] == 1 else "risk-low"
                st.markdown(f"""
                <div class='pred-box'>
                    <p style='font-family:Space Grotesk; font-size:0.9rem; color:#718096;
                              margin:0 0 0.7rem 0;'>{mname}</p>
                    <span class='{badge_class}'>{risk_label}</span>
                    <p style='font-size:1.5rem; font-weight:700; color:#63b3ed;
                              margin:0.8rem 0 0.2rem 0;'>{p["prob"]:.1%}</p>
                    <p style='font-size:0.75rem; color:#718096; margin:0;'>High Risk Probability</p>
                </div>
                """, unsafe_allow_html=True)

        # Gauge chart for RF probability
        rf_prob = preds["Random Forest"]["prob"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rf_prob * 100,
            delta={"reference": 50, "valueformat": ".1f"},
            number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0"}},
            title={"text": "Random Forest Risk Probability", "font": {"color": "#90cdf4", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#4a5568", "tickfont": {"color": "#718096"}},
                "bar": {"color": COLORS["danger"] if rf_prob > 0.5 else COLORS["success"],
                        "thickness": 0.25},
                "bgcolor": "rgba(15,22,41,0.8)",
                "borderwidth": 1,
                "bordercolor": "rgba(99,179,237,0.2)",
                "steps": [
                    {"range": [0, 30],  "color": "rgba(104,211,145,0.1)"},
                    {"range": [30, 60], "color": "rgba(246,173,85,0.1)"},
                    {"range": [60, 100],"color": "rgba(252,129,129,0.1)"},
                ],
                "threshold": {
                    "line": {"color": "#f6ad55", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#a0aec0"),
            height=280,
            margin=dict(t=30, b=10, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature contribution bar
        st.markdown("<div class='section-header'>Key Risk Factors for This Customer</div>",
                    unsafe_allow_html=True)
        fi_vals = pd.Series(
            trained["Random Forest"].feature_importances_,
            index=X_columns
        )
        inp_arr     = input_df.values[0]
        contribution = fi_vals * inp_arr
        top_contrib  = pd.DataFrame({
            "Feature": X_columns,
            "Contribution": contribution,
        }).sort_values("Contribution", ascending=False).head(10)

        fig_contrib = px.bar(
            top_contrib, x="Contribution", y="Feature", orientation="h",
            color="Contribution",
            color_continuous_scale=[[0,"#276749"],[0.5,"#f6ad55"],[1,"#c53030"]],
        )
        fig_contrib.update_layout(
            **PLOTLY_LAYOUT,
            title="Top 10 Contributing Features",
            height=340,
            coloraxis_showscale=False,
            xaxis=AXIS_STYLE,
            yaxis=dict(autorange="reversed", **AXIS_STYLE),
            margin=MARGIN_DEF,
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:rgba(99,179,237,0.1); margin:2rem 0 1rem 0;'>
<p style='text-align:center; font-size:0.78rem; color:#4a5568;'>
    Risk Alert Classifier · Logistic Regression · Decision Tree · Random Forest ·
    SMOTE Imbalance Handling · Built with Streamlit &amp; Plotly
</p>
""", unsafe_allow_html=True)
