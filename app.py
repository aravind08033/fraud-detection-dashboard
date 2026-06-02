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
[data-testid="stSidebar"] { background: #0D0D1A !important; border-right: 1px solid #1E1E3F !important; }
.metric-card {
    background: linear-gradient(135deg, #0D0D1A, #1A1A2E);
    border: 1px solid #2D2D5E;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 5px;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #00D4FF; }
.metric-label { font-size: 0.85rem; color: #888; margin-top: 5px; }
.sec-tag { color: #00D4FF; font-size: 0.75rem; letter-spacing: 3px; margin-bottom: 5px; }
.sec-title { font-size: 1.6rem; font-weight: 800; color: #E0E0E8; margin-bottom: 20px; }
.alert-fraud { background: rgba(255,50,50,0.1); border-left: 4px solid #FF3232; border-radius: 8px; padding: 15px; margin: 10px 0; }
.alert-legit { background: rgba(0,212,127,0.1); border-left: 4px solid #00D47F; border-radius: 8px; padding: 15px; margin: 10px 0; }
.stButton > button {
    background: linear-gradient(135deg, #00D4FF, #7B61FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] { background: #0D0D1A; border: 1px solid #2D2D5E; border-radius: 10px; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
  <div style='font-size:3rem;'>🛡️</div>
  <h1 style='font-size:2.8rem; font-weight:900; background: linear-gradient(135deg,#00D4FF,#7B61FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;'>FRAUDSHIELD</h1>
  <p style='color:#888; font-size:1rem; margin-top:8px;'>Credit Card Fraud Detection Dashboard · Codec Technologies Internship 2026</p>
</div>
""", unsafe_allow_html=True)

# ── DATA GENERATION ──
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 50000
    # Generate realistic-looking transaction data
    df = pd.DataFrame(np.random.randn(n, 28), columns=[f'V{i}' for i in range(1, 29)])
    df['Time'] = np.sort(np.random.uniform(0, 172800, n))
    df['Amount'] = np.abs(np.random.exponential(88, n))
    # Fraud: ~0.17% like real dataset
    fraud_indices = np.random.choice(n, int(n * 0.0017), replace=False)
    df['Class'] = 0
    df.loc[fraud_indices, 'Class'] = 1
    # Make fraud transactions slightly different
    df.loc[df['Class']==1, 'Amount'] = np.abs(np.random.exponential(200, df['Class'].sum()))
    df.loc[df['Class']==1, 'V1'] -= 2
    df.loc[df['Class']==1, 'V3'] -= 1.5
    df.loc[df['Class']==1, 'V4'] += 1.8
    return df

df = load_data()

# ── SIDEBAR ──
st.sidebar.markdown("<h2 style='color:#00D4FF;'>⚙️ Controls</h2>", unsafe_allow_html=True)
selected_model = st.sidebar.selectbox("🤖 ML Model", ["Random Forest", "Logistic Regression"])
threshold = st.sidebar.slider("🎯 Detection Threshold", 0.1, 0.9, 0.5, 0.05)
sample_size = st.sidebar.slider("📊 Sample Size", 5000, 50000, 20000, 5000)
show_raw = st.sidebar.checkbox("Show Raw Data", False)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color:#888; font-size:0.8rem;'>Built by <b style='color:#00D4FF;'>Aravind Damera</b><br>Codec Technologies · 2026</p>", unsafe_allow_html=True)

df_sample = df.sample(sample_size, random_state=42)

# ── TABS ──
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 EDA", "🤖 ML Model", "⚡ Live Detector", "📁 Data"])

# ══ TAB 1: OVERVIEW ══
with tab1:
    st.markdown('<div class="sec-tag">// 01 — OVERVIEW</div><div class="sec-title">DASHBOARD OVERVIEW</div>', unsafe_allow_html=True)

    total = len(df)
    fraud_count = df['Class'].sum()
    legit_count = total - fraud_count
    fraud_pct = (fraud_count / total) * 100
    avg_amount = df['Amount'].mean()
    fraud_amount = df[df['Class']==1]['Amount'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💳 Total Transactions", f"{total:,}")
    c2.metric("🚨 Fraud Cases", f"{fraud_count:,}")
    c3.metric("✅ Legitimate", f"{legit_count:,}")
    c4.metric("📈 Fraud Rate", f"{fraud_pct:.3f}%")
    c5.metric("💰 Avg Amount", f"${avg_amount:.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🍩 Transaction Distribution")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0D0D1A')
        ax.set_facecolor('#0D0D1A')
        sizes = [legit_count, fraud_count]
        colors = ['#00D47F', '#FF3232']
        explode = (0, 0.1)
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=['Legitimate', 'Fraud'],
                                           colors=colors, autopct='%1.3f%%', startangle=90,
                                           textprops={'color': '#E0E0E8'})
        for at in autotexts:
            at.set_color('#E0E0E8')
            at.set_fontsize(10)
        ax.set_title('Fraud vs Legitimate', color='#E0E0E8', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### ⏱️ Transactions Over Time")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0D0D1A')
        ax.set_facecolor('#0D0D1A')
        df['hour'] = (df['Time'] / 3600).astype(int) % 24
        hourly = df.groupby(['hour', 'Class']).size().unstack(fill_value=0)
        if 0 in hourly.columns:
            ax.plot(hourly.index, hourly[0], color='#00D47F', label='Legitimate', linewidth=2)
        if 1 in hourly.columns:
            ax.plot(hourly.index, hourly[1], color='#FF3232', label='Fraud', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Hour of Day', color='#888')
        ax.set_ylabel('Transaction Count', color='#888')
        ax.set_title('Hourly Transaction Pattern', color='#E0E0E8', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#888')
        ax.legend(facecolor='#1A1A2E', labelcolor='#E0E0E8')
        for spine in ax.spines.values(): spine.set_color('#2D2D5E')
        st.pyplot(fig)
        plt.close()

# ══ TAB 2: EDA ══
with tab2:
    st.markdown('<div class="sec-tag">// 02 — EDA</div><div class="sec-title">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Amount Distribution")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0D0D1A')
        ax.set_facecolor('#0D0D1A')
        fraud_amt = df[df['Class']==1]['Amount'].clip(upper=1000)
        legit_amt = df[df['Class']==0]['Amount'].clip(upper=1000).sample(1000)
        ax.hist(legit_amt, bins=50, color='#00D47F', alpha=0.6, label='Legitimate', density=True)
        ax.hist(fraud_amt, bins=50, color='#FF3232', alpha=0.8, label='Fraud', density=True)
        ax.set_xlabel('Amount ($)', color='#888')
        ax.set_ylabel('Density', color='#888')
        ax.set_title('Transaction Amount Distribution', color='#E0E0E8', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#888')
        ax.legend(facecolor='#1A1A2E', labelcolor='#E0E0E8')
        for spine in ax.spines.values(): spine.set_color('#2D2D5E')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0D0D1A')
        ax.set_facecolor('#0D0D1A')
        corr_features = ['V1','V2','V3','V4','V5','Amount','Class']
        corr = df[corr_features].corr()
        sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=True, fmt='.2f',
                    annot_kws={'size': 8}, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation', color='#E0E0E8', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#888', labelsize=8)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### 📦 V1–V4 Feature Boxplots by Class")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), facecolor='#0D0D1A')
    for i, feat in enumerate(['V1','V2','V3','V4']):
        axes[i].set_facecolor('#0D0D1A')
        data_legit = df[df['Class']==0][feat].sample(500)
        data_fraud = df[df['Class']==1][feat]
        bp = axes[i].boxplot([data_legit, data_fraud], labels=['Legit','Fraud'],
                              patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor('#00D47F')
        bp['boxes'][0].set_alpha(0.7)
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('#FF3232')
            bp['boxes'][1].set_alpha(0.7)
        axes[i].set_title(feat, color='#E0E0E8', fontweight='bold')
        axes[i].tick_params(colors='#888')
        for spine in axes[i].spines.values(): spine.set_color('#2D2D5E')
    fig.patch.set_facecolor('#0D0D1A')
    st.pyplot(fig)
    plt.close()

# ══ TAB 3: ML MODEL ══
with tab3:
    st.markdown('<div class="sec-tag">// 03 — ML</div><div class="sec-title">MACHINE LEARNING MODEL</div>', unsafe_allow_html=True)

    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
            X = df_sample[features]
            y = df_sample['Class']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

            if selected_model == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = LogisticRegression(random_state=42, max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred_thresh = (y_prob >= threshold).astype(int)

            acc = accuracy_score(y_test, y_pred_thresh)
            auc = roc_auc_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred_thresh)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("✅ Accuracy", f"{acc:.4f}")
            c2.metric("📈 ROC-AUC", f"{auc:.4f}")
            c3.metric("🎯 Threshold", f"{threshold}")
            c4.metric("🔢 Test Size", f"{len(y_test):,}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0D0D1A')
                ax.set_facecolor('#0D0D1A')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Legit','Fraud'], yticklabels=['Legit','Fraud'],
                            linewidths=1)
                ax.set_title('Confusion Matrix', color='#E0E0E8', fontweight='bold')
                ax.tick_params(colors='#888')
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("#### ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0D0D1A')
                ax.set_facecolor('#0D0D1A')
                ax.plot(fpr, tpr, color='#00D4FF', lw=2, label=f'AUC = {auc:.4f}')
                ax.plot([0,1],[0,1], color='#FF3232', lw=1, linestyle='--', label='Random')
                ax.set_xlabel('False Positive Rate', color='#888')
                ax.set_ylabel('True Positive Rate', color='#888')
                ax.set_title('ROC Curve', color='#E0E0E8', fontweight='bold')
                ax.tick_params(colors='#888')
                ax.legend(facecolor='#1A1A2E', labelcolor='#E0E0E8')
                for spine in ax.spines.values(): spine.set_color('#2D2D5E')
                st.pyplot(fig)
                plt.close()

            if selected_model == "Random Forest":
                st.markdown("#### 🏆 Feature Importance")
                feat_imp = pd.Series(model.feature_importances_, index=features).nlargest(10)
                fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0D0D1A')
                ax.set_facecolor('#0D0D1A')
                feat_imp.plot(kind='barh', ax=ax, color='#7B61FF')
                ax.set_title('Top 10 Important Features', color='#E0E0E8', fontweight='bold')
                ax.tick_params(colors='#888')
                for spine in ax.spines.values(): spine.set_color('#2D2D5E')
                st.pyplot(fig)
                plt.close()
    else:
        st.info("👆 'Train Model' button click cheyyi!")

# ══ TAB 4: LIVE DETECTOR ══
with tab4:
    st.markdown('<div class="sec-tag">// 04 — DETECTOR</div><div class="sec-title">LIVE FRAUD DETECTOR</div>', unsafe_allow_html=True)

    st.markdown("#### Enter Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("💰 Amount ($)", min_value=0.0, max_value=50000.0, value=150.0, step=10.0)
        v1 = st.slider("V1 Feature", -5.0, 5.0, 0.0, 0.1)
        v2 = st.slider("V2 Feature", -5.0, 5.0, 0.0, 0.1)
    with col2:
        v3 = st.slider("V3 Feature", -5.0, 5.0, 0.0, 0.1)
        v4 = st.slider("V4 Feature", -5.0, 5.0, 0.0, 0.1)
        v5 = st.slider("V5 Feature", -5.0, 5.0, 0.0, 0.1)
    with col3:
        time_val = st.number_input("⏱️ Time (seconds)", 0, 172800, 43200)
        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button("🔍 DETECT FRAUD", use_container_width=True)

    if detect_btn:
        input_data = np.zeros(30)
        input_data[0] = v1; input_data[1] = v2; input_data[2] = v3
        input_data[3] = v4; input_data[4] = v5
        input_data[28] = amount; input_data[29] = time_val

        features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        X_train_quick = df_sample[features]
        y_train_quick = df_sample['Class']
        scaler_q = StandardScaler()
        X_scaled_q = scaler_q.fit_transform(X_train_quick)
        model_q = LogisticRegression(random_state=42, max_iter=500)
        model_q.fit(X_scaled_q, y_train_quick)

        input_scaled = scaler_q.transform([input_data])
        prob = model_q.predict_proba(input_scaled)[0][1]
        prediction = prob >= threshold

        if prediction:
            st.markdown(f"""
            <div class="alert-fraud">
                <h2>🚨 FRAUD DETECTED!</h2>
                <p>Fraud Probability: <b style='color:#FF3232; font-size:1.5rem;'>{prob:.1%}</b></p>
                <p>Amount: ${amount:.2f} | Time: {time_val}s</p>
                <p>⚠️ This transaction has been flagged for review.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-legit">
                <h2>✅ TRANSACTION LEGITIMATE</h2>
                <p>Fraud Probability: <b style='color:#00D47F; font-size:1.5rem;'>{prob:.1%}</b></p>
                <p>Amount: ${amount:.2f} | Time: {time_val}s</p>
                <p>✅ Transaction approved.</p>
            </div>""", unsafe_allow_html=True)

# ══ TAB 5: DATA ══
with tab5:
    st.markdown('<div class="sec-tag">// 05 — Data</div><div class="sec-title">DATASET PREVIEW</div>', unsafe_allow_html=True)
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
    Python · Streamlit · scikit-learn · pandas · matplotlib · Generated Dataset
</div>
""", unsafe_allow_html=True)
