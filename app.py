import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="FraudShield", page_icon="shield", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0A0A0F !important; color: #E0E0E8 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.block-container { padding: 0 2rem 3rem 2rem !important; }
#MainMenu, footer { display: none !important; }
[data-testid="stSidebar"] { background: #0F0F1A !important; border-right: 2px solid #FF6400 !important; }
[data-testid="stSidebar"] * { color: #E0E0E8 !important; }
.hero { background: linear-gradient(135deg,#0A0A0F,#110D02); border-bottom: 2px solid #FF6400; padding: 2rem 0 1.5rem; margin-bottom: 1.5rem; }
.hero-tag { background: rgba(255,100,0,0.1); border: 1px solid rgba(255,100,0,0.3); color: #FF6400; font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase; display: inline-block; padding: 0.25rem 0.9rem; margin-bottom: 1rem; }
.hero-title { font-size: 2.5rem; font-weight: 900; color: #fff; line-height: 1.05; margin: 0 0 0.4rem; }
.hero-title span { color: #FF6400; }
.hero-desc { font-size: 0.9rem; color: #555568; max-width: 500px; line-height: 1.7; margin-bottom: 1.5rem; }
.stat-row { display: flex; gap: 2.5rem; flex-wrap: wrap; }
.stat-num { font-size: 1.8rem; font-weight: 800; color: #FF6400; line-height: 1; }
.stat-lbl { font-size: 0.6rem; color: #444456; letter-spacing: 0.12em; text-transform: uppercase; margin-top: 0.2rem; }
.sec-tag { font-size: 0.62rem; color: #FF6400; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.3rem; }
.sec-title { font-size: 1.4rem; font-weight: 700; color: #fff; border-left: 3px solid #FF6400; padding-left: 0.7rem; margin-bottom: 1.2rem; }
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi { background: #0F0F1A; border: 1px solid #1E1E2E; border-top: 3px solid #FF6400; padding: 1.2rem; }
.kpi-lbl { font-size: 0.6rem; color: #444456; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem; }
.kpi-val { font-size: 1.8rem; font-weight: 800; line-height: 1; color: white; }
.kpi-val.orange { color: #FF6400; } .kpi-val.red { color: #FF453A; } .kpi-val.green { color: #32D74B; }
.kpi-sub { font-size: 0.68rem; color: #333345; margin-top: 0.3rem; }
.info-box { background: #0F0F1A; border: 1px solid #1E1E2E; border-left: 3px solid #FF6400; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.info-title { font-size: 0.75rem; color: #FF6400; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
.info-val { font-size: 1.4rem; font-weight: 700; color: white; }
.info-desc { font-size: 0.72rem; color: #444456; margin-top: 0.2rem; }
.res-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; margin: 1rem 0; }
.res-card { background: #0F0F1A; border: 1px solid #1E1E2E; padding: 1.2rem; text-align: center; }
.res-num { font-size: 1.8rem; font-weight: 800; color: #FF6400; }
.res-lbl { font-size: 0.6rem; color: #444456; text-transform: uppercase; letter-spacing: 0.1em; }
.div { border: none; border-top: 1px solid #1A1A2A; margin: 1.5rem 0; }
.foot { text-align: center; padding: 1.5rem; border-top: 1px solid #1A1A2A; margin-top: 2rem; font-size: 0.65rem; color: #222233; }
.foot b { color: #FF6400; }
.stButton > button { background: #FF6400 !important; color: #000 !important; border: none !important; font-weight: 700 !important; width: 100% !important; padding: 0.6rem !important; }
.stButton > button:hover { background: #FF8533 !important; }
[data-testid="stSelectbox"] > div > div { background: #0F0F1A !important; border: 1px solid #2A2A3E !important; color: #E0E0E8 !important; }
label { color: #555568 !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#0F0F1A', 'axes.facecolor': '#0F0F1A',
    'axes.edgecolor': '#1E1E2E', 'axes.labelcolor': '#666678',
    'xtick.color': '#444456', 'ytick.color': '#444456',
    'grid.color': '#1A1A2A', 'text.color': '#888898',
    'font.family': 'monospace',
    'axes.spines.top': False, 'axes.spines.right': False,
})

@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv", nrows=50000)

df = load_data()
total = len(df)
fraud = int(df['Class'].sum())
legit = total - fraud
fraud_pct = fraud / total * 100
avg_fraud_amt = df[df['Class']==1]['Amount'].mean()
avg_legit_amt = df[df['Class']==0]['Amount'].mean()
max_fraud_amt = df[df['Class']==1]['Amount'].max()

# SIDEBAR
with st.sidebar:
    st.markdown("### 🛡️ FraudShield")
    st.markdown("---")
    page = st.radio("Navigation", ["Overview", "Fraud Analysis", "EDA", "Model Training", "Dataset"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Quick Stats**")
    st.markdown(f"🔴 Fraud: **{fraud}** cases")
    st.markdown(f"🟢 Legit: **{legit:,}** cases")
    st.markdown(f"📊 Rate: **{fraud_pct:.2f}%**")
    st.markdown(f"💰 Avg Fraud: **€{avg_fraud_amt:.1f}**")
    st.markdown("---")
    st.markdown("**Developer**")
    st.markdown("👤 Aravind Damera")
    st.markdown("🎓 2nd Year CSE DS")
    st.markdown("🏫 QIS College, Ongole")
    st.markdown("---")
    st.markdown("**Internship**")
    st.markdown("🏢 Codec Technologies")
    st.markdown("📅 Industrial Project 2026")

# HERO
st.markdown(f"""
<div class="hero">
  <div class="hero-tag">🛡️ FraudShield · Codec Technologies · 2026</div>
  <div class="hero-title">CREDIT CARD <span>FRAUD</span><br>DETECTION DASHBOARD</div>
  <div class="hero-desc">Real-time ML-powered analysis of 50,000 anonymised European credit card transactions. Identify fraud patterns, train models, evaluate performance.</div>
  <div class="stat-row">
    <div><div class="stat-num">{total:,}</div><div class="stat-lbl">Transactions</div></div>
    <div><div class="stat-num">{fraud}</div><div class="stat-lbl">Fraud Cases</div></div>
    <div><div class="stat-num">{fraud_pct:.2f}%</div><div class="stat-lbl">Fraud Rate</div></div>
    <div><div class="stat-num">€{avg_fraud_amt:.0f}</div><div class="stat-lbl">Avg Fraud Amt</div></div>
    <div><div class="stat-num">2</div><div class="stat-lbl">ML Models</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.markdown('<div class="sec-tag">// 01 — Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">DATASET OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-lbl">Total Transactions</div><div class="kpi-val">{total:,}</div><div class="kpi-sub">European cardholders · Sep 2013</div></div>
      <div class="kpi"><div class="kpi-lbl">Fraudulent</div><div class="kpi-val red">{fraud:,}</div><div class="kpi-sub">High priority alerts</div></div>
      <div class="kpi"><div class="kpi-lbl">Legitimate</div><div class="kpi-val green">{legit:,}</div><div class="kpi-sub">Safe transactions</div></div>
      <div class="kpi"><div class="kpi-lbl">Fraud Rate</div><div class="kpi-val orange">{fraud_pct:.2f}%</div><div class="kpi-sub">Extreme class imbalance</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.markdown(f'<div class="info-box"><div class="info-title">Avg Fraud Amount</div><div class="info-val">€{avg_fraud_amt:.2f}</div><div class="info-desc">Per fraudulent transaction</div></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="info-box"><div class="info-title">Avg Legitimate Amount</div><div class="info-val">€{avg_legit_amt:.2f}</div><div class="info-desc">Per legitimate transaction</div></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="info-box"><div class="info-title">Max Fraud Amount</div><div class="info-val">€{max_fraud_amt:.2f}</div><div class="info-desc">Highest single fraud</div></div>', unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("**About this Dataset**")
    st.info("This dataset contains credit card transactions by European cardholders in September 2013. Features V1-V28 are PCA-transformed to protect privacy. 'Amount' is the transaction value in Euros. 'Class' is 1 for fraud, 0 for legitimate.")

elif page == "Fraud Analysis":
    st.markdown('<div class="sec-tag">// 02 — Fraud Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">FRAUD PATTERN ANALYSIS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(['Legitimate', 'Fraudulent'], [legit, fraud], color=['#32D74B', '#FF453A'], width=0.4, zorder=3)
        ax.yaxis.grid(True, linewidth=0.4, zorder=0)
        ax.set_title('Transaction Class Distribution', color='white', pad=10)
        for i, v in enumerate([legit, fraud]):
            ax.text(i, v+200, f'{v:,}', ha='center', fontsize=9, color='#CCCCDD', fontweight='bold')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fraud_amt = df[df['Class']==1]['Amount']
        legit_amt = df[df['Class']==0]['Amount']
        categories = ['Min', 'Mean', 'Median', 'Max']
        fraud_vals = [fraud_amt.min(), fraud_amt.mean(), fraud_amt.median(), min(fraud_amt.max(), 2500)]
        legit_vals = [legit_amt.min(), legit_amt.mean(), legit_amt.median(), min(legit_amt.max(), 2500)]
        x = np.arange(len(categories))
        ax.bar(x-0.2, fraud_vals, 0.35, color='#FF453A', label='Fraud', zorder=3)
        ax.bar(x+0.2, legit_vals, 0.35, color='#32D74B', label='Legitimate', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(categories)
        ax.yaxis.grid(True, linewidth=0.4, zorder=0)
        ax.set_title('Amount Comparison (EUR)', color='white', pad=10)
        ax.legend(fontsize=8, facecolor='#1A1A2A', labelcolor='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bins = [0, 10, 50, 100, 500, 1000, 5000]
        labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+']
        fraud_binned = pd.cut(df[df['Class']==1]['Amount'], bins=bins, labels=labels).value_counts().sort_index()
        ax.bar(fraud_binned.index, fraud_binned.values, color='#FF6400', zorder=3)
        ax.yaxis.grid(True, linewidth=0.4, zorder=0)
        ax.set_xlabel('Amount Range (EUR)', fontsize=8)
        ax.set_ylabel('Fraud Count', fontsize=8)
        ax.set_title('Fraud by Amount Range', color='white', pad=10)
        plt.xticks(rotation=30, fontsize=7)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        time_bins = pd.cut(df['Time'], bins=24, labels=False)
        fraud_by_time = df.groupby(time_bins)['Class'].sum()
        ax.fill_between(fraud_by_time.index, fraud_by_time.values, color='#FF6400', alpha=0.7)
        ax.plot(fraud_by_time.index, fraud_by_time.values, color='#FF8533', lw=1.5)
        ax.yaxis.grid(True, linewidth=0.4, zorder=0)
        ax.set_xlabel('Time Period', fontsize=8)
        ax.set_ylabel('Fraud Cases', fontsize=8)
        ax.set_title('Fraud Over Time', color='white', pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("**Key Fraud Insights**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Fraud Amount", f"€{avg_fraud_amt:.1f}", delta=f"vs €{avg_legit_amt:.1f} legit")
    c2.metric("Max Fraud", f"€{max_fraud_amt:.0f}")
    c3.metric("Small Frauds (<€10)", f"{len(df[(df['Class']==1) & (df['Amount']<10)])}")
    c4.metric("Large Frauds (>€1000)", f"{len(df[(df['Class']==1) & (df['Amount']>1000)])}")

elif page == "EDA":
    st.markdown('<div class="sec-tag">// 03 — EDA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[df['Class']==0]['Amount'].clip(upper=500), bins=50, color='#32D74B', alpha=0.6, label='Legitimate', zorder=3)
        ax.hist(df[df['Class']==1]['Amount'].clip(upper=500), bins=50, color='#FF6400', alpha=0.8, label='Fraudulent', zorder=3)
        ax.yaxis.grid(True, linewidth=0.4, zorder=0)
        ax.set_xlabel('Amount (EUR)', fontsize=8)
        ax.legend(fontsize=8, facecolor='#1A1A2A', labelcolor='white')
        ax.set_title('Amount Distribution', color='white', pad=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        top_features = ['V4','V11','V2','V19','V21','V3','V10','V14','V17']
        fraud_means = df[df['Class']==1][top_features].mean()
        legit_means = df[df['Class']==0][top_features].mean()
        x = np.arange(len(top_features))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(x-0.2, fraud_means, 0.35, color='#FF453A', label='Fraud', zorder=3)
        ax.bar(x+0.2, legit_means, 0.35, color='#32D74B', label='Legitimate', zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(top_features, fontsize=7)
        ax.yaxis.grid(True, linewidth=0.4, zorder=0)
        ax.set_title('Key Feature Averages', color='white', pad=10)
        ax.legend(fontsize=8, facecolor='#1A1A2A', labelcolor='white')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    fig, ax = plt.subplots(figsize=(14, 3.8))
    cols = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','Amount','Class']
    sns.heatmap(df[cols].corr(), annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
                linewidths=0.3, linecolor='#0A0A0F', annot_kws={'size': 7})
    ax.set_title('Feature Correlation Heatmap', color='white', pad=10)
    fig.patch.set_facecolor('#0F0F1A'); fig.tight_layout()
    st.pyplot(fig); plt.close()

elif page == "Model Training":
    st.markdown('<div class="sec-tag">// 04 — ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">MODEL TRAINING & EVALUATION</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,2,1], gap="medium")
    with col1:
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])
    with col2:
        test_size = st.slider("Test Split %", 10, 40, 20, step=5)
    with col3:
        st.write(""); st.write("")
        train = st.button("TRAIN MODEL")
    if train:
        with st.spinner("Training..."):
            X = df.drop('Class', axis=1); y = df['Class']
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)
            X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=test_size/100, random_state=42, stratify=y)
            model = LogisticRegression(max_iter=1000, random_state=42) if model_choice == "Logistic Regression" else RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:,1]
            acc = accuracy_score(y_te, y_pred)
            roc = roc_auc_score(y_te, y_pred)
            rec = classification_report(y_te, y_pred, output_dict=True)['1']['recall']
        st.success(f"Training complete — {model_choice}!")
        st.markdown(f"""
        <div class="res-row">
          <div class="res-card"><div class="res-num">{acc*100:.1f}%</div><div class="res-lbl">Accuracy</div></div>
          <div class="res-card"><div class="res-num">{roc:.3f}</div><div class="res-lbl">ROC-AUC</div></div>
          <div class="res-card"><div class="res-num">{rec*100:.1f}%</div><div class="res-lbl">Fraud Recall</div></div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            cm = confusion_matrix(y_te, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap=sns.light_palette('#FF6400', as_cmap=True),
                        xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'], ax=ax,
                        annot_kws={'size':13,'weight':'bold','color':'white'})
            ax.set_title('Confusion Matrix', color='white')
            fig.patch.set_facecolor('#0F0F1A'); fig.tight_layout()
            st.pyplot(fig); plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            fpr, tpr, _ = roc_curve(y_te, y_prob)
            ax.plot(fpr, tpr, color='#FF6400', lw=2.5, label=f'AUC={roc:.3f}')
            ax.plot([0,1],[0,1], color='gray', lw=1, linestyle='--')
            ax.fill_between(fpr, tpr, alpha=0.08, color='#FF6400')
            ax.set_title('ROC Curve', color='white')
            ax.legend(fontsize=8, facecolor='#1A1A2A', labelcolor='white')
            fig.patch.set_facecolor('#0F0F1A'); fig.tight_layout()
            st.pyplot(fig); plt.close()
        if model_choice == "Random Forest":
            feat = pd.Series(model.feature_importances_, index=df.drop('Class',axis=1).columns).nlargest(15)
            colors = ['#FF6400' if i<3 else '#CC5200' if i<8 else '#883300' for i in range(len(feat))]
            fig, ax = plt.subplots(figsize=(12,3.5))
            ax.barh(feat.index[::-1], feat.values[::-1], color=colors[::-1], height=0.6, zorder=3)
            ax.xaxis.grid(True, linewidth=0.4, zorder=0)
            ax.set_title('Top 15 Feature Importances', color='white')
            fig.patch.set_facecolor('#0F0F1A'); fig.tight_layout()
            st.pyplot(fig); plt.close()

elif page == "Dataset":
    st.markdown('<div class="sec-tag">// 05 — Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">DATASET PREVIEW</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fraud Transactions**")
        st.dataframe(df[df['Class']==1].head(20), use_container_width=True, height=300)
    with col2:
        st.markdown("**Legitimate Transactions**")
        st.dataframe(df[df['Class']==0].head(20), use_container_width=True, height=300)
    st.markdown("**Full Dataset**")
    st.dataframe(df.head(100), use_container_width=True, height=300)

st.markdown("""
<div class="foot">
  FRAUDSHIELD · Built by <b>Aravind Damera</b> · Codec Technologies Industrial Project · 2026<br>
  Python · Streamlit · scikit-learn · pandas · matplotlib · Kaggle Dataset
</div>
""", unsafe_allow_html=True)
