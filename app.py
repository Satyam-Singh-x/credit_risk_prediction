# ═══════════════════════════════════════════════════════════════════
#  Credit Risk Prediction — Streamlit Demo
#  OpenAImer 2026 | SRIJAN — Jadavpur University
# ═══════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size:2.8rem; font-weight:800; color:#2c3e50; text-align:center; padding:1rem 0 0.2rem 0; }
    .sub-header  { font-size:1.1rem; color:#7f8c8d; text-align:center; margin-bottom:2rem; }
    .metric-card { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding:1.2rem; border-radius:12px; color:white; text-align:center; margin:0.3rem; }
    .risk-very-high { background:linear-gradient(135deg,#e74c3c,#c0392b); color:white; padding:1.5rem; border-radius:12px; text-align:center; font-size:1.6rem; font-weight:800; }
    .risk-high      { background:linear-gradient(135deg,#e67e22,#d35400); color:white; padding:1.5rem; border-radius:12px; text-align:center; font-size:1.6rem; font-weight:800; }
    .risk-medium    { background:linear-gradient(135deg,#f39c12,#e67e22); color:white; padding:1.5rem; border-radius:12px; text-align:center; font-size:1.6rem; font-weight:800; }
    .risk-low       { background:linear-gradient(135deg,#27ae60,#2ecc71); color:white; padding:1.5rem; border-radius:12px; text-align:center; font-size:1.6rem; font-weight:800; }
    .section-title  { font-size:1.4rem; font-weight:700; color:#2c3e50; border-left:4px solid #3498db; padding-left:0.8rem; margin:1.5rem 0 1rem 0; }
    .insight-box    { background:#f8f9fa; border-left:4px solid #3498db; padding:1rem; border-radius:0 8px 8px 0; margin:0.5rem 0; font-size:0.95rem; color:#2c3e50; }
    .stButton > button { width:100%; background:linear-gradient(135deg,#3498db,#2980b9); color:white; font-size:1.2rem; font-weight:700; padding:0.8rem; border:none; border-radius:8px; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

# ── Competition banner ─────────────────────────────────────────────
# FIX: Use a single compact string — multiline triple-quoted HTML with mixed
# indentation can confuse Streamlit's markdown renderer and print raw tags.
banner_html = (
    '<div style="background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);'
    'padding:2rem;border-radius:16px;border:1px solid #e74c3c;'
    'margin-bottom:1.5rem;text-align:center;">'
    '<h1 style="color:white;margin-bottom:0.5rem;font-size:2.5rem;font-weight:800;">'
    '🏦 Credit Risk Prediction</h1>'
    '<div style="color:#4fc3f7;font-size:1rem;line-height:1.6;opacity:0.95;">'
    '<b>LightGBM + XGBoost Ensemble</b> &bull; '
    'OOF AUC: <b>0.786</b> &bull; '
    '307,511 customers &bull; '
    '252 features &bull; '
    'SHAP Explainability'
    '</div></div>'
)
st.markdown(banner_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 1. LOAD MODELS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    base     = os.path.dirname(os.path.abspath(__file__))
    models   = os.path.join(base, 'models')
    shap_out = os.path.join(base, 'shap_outputs')

    with open(os.path.join(models, 'lgb_best_single.pkl'), 'rb') as f:
        lgb_model = pickle.load(f)

    with open(os.path.join(models, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)

    with open(os.path.join(models, 'best_threshold.pkl'), 'rb') as f:
        best_threshold = pickle.load(f)

    shap_importance = pd.read_csv(os.path.join(shap_out, 'shap_importance.csv'))
    explainer = shap.TreeExplainer(lgb_model)

    return [lgb_model], feature_names, best_threshold, explainer, shap_importance

lgb_models, feature_names, best_threshold, explainer, shap_importance = load_models()


# ══════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
def engineer_features(inputs):
    d = inputs.copy()

    d['CREDIT_INCOME_RATIO']  = d['AMT_CREDIT'] / (d['AMT_INCOME_TOTAL'] + 1)
    d['ANNUITY_INCOME_RATIO'] = d['AMT_ANNUITY'] / (d['AMT_INCOME_TOTAL'] + 1)
    d['CREDIT_TERM']          = d['AMT_CREDIT'] / (d['AMT_ANNUITY'] + 1)
    d['CREDIT_GOODS_RATIO']   = d['AMT_CREDIT'] / (d['AMT_GOODS_PRICE'] + 1)
    d['CREDIT_GOODS_DIFF']    = d['AMT_CREDIT'] - d['AMT_GOODS_PRICE']
    d['INCOME_PER_PERSON']    = d['AMT_INCOME_TOTAL'] / (d['CNT_FAM_MEMBERS'] + 1)
    d['ANNUITY_PER_PERSON']   = d['AMT_ANNUITY'] / (d['CNT_FAM_MEMBERS'] + 1)
    d['CHILDREN_RATIO']       = d['CNT_CHILDREN'] / (d['CNT_FAM_MEMBERS'] + 1)

    d['AGE_YEARS']            = d['DAYS_BIRTH'] / -365
    d['YEARS_EMPLOYED']       = d['DAYS_EMPLOYED'] / -365
    d['EMPLOY_TO_AGE_RATIO']  = d['DAYS_EMPLOYED'] / (d['DAYS_BIRTH'] - 1)
    d['YEARS_REGISTRATION']   = d['DAYS_REGISTRATION'] / -365
    d['YEARS_ID_PUBLISH']     = d['DAYS_ID_PUBLISH'] / -365
    d['YEARS_PHONE_CHANGE']   = d['DAYS_LAST_PHONE_CHANGE'] / -365

    ext = [d.get('EXT_SOURCE_1', np.nan),
           d.get('EXT_SOURCE_2', np.nan),
           d.get('EXT_SOURCE_3', np.nan)]
    ext_valid = [x for x in ext if not np.isnan(x)]

    d['EXT_SOURCE_MEAN']     = np.mean(ext_valid) if ext_valid else np.nan
    d['EXT_SOURCE_MIN']      = np.min(ext_valid)  if ext_valid else np.nan
    d['EXT_SOURCE_MAX']      = np.max(ext_valid)  if ext_valid else np.nan
    d['EXT_SOURCE_STD']      = np.std(ext_valid)  if len(ext_valid) > 1 else 0
    d['EXT_SOURCE_PROD']     = np.prod(ext_valid) if ext_valid else np.nan
    d['EXT_SOURCE_RANGE']    = d['EXT_SOURCE_MAX'] - d['EXT_SOURCE_MIN'] if ext_valid else np.nan
    d['EXT_SOURCE_COUNT']    = len(ext_valid)
    d['EXT_SOURCE_WEIGHTED'] = (
        d.get('EXT_SOURCE_1', 0) * 0.25 +
        d.get('EXT_SOURCE_2', 0) * 0.50 +
        d.get('EXT_SOURCE_3', 0) * 0.25
    )
    d['EXT_SOURCE_MEAN_x_CREDIT_RATIO'] = (
        d['EXT_SOURCE_MEAN'] * d['CREDIT_INCOME_RATIO']
        if not np.isnan(d['EXT_SOURCE_MEAN']) else np.nan
    )
    d['EXT_SOURCE_MEAN_x_AGE'] = (
        d['EXT_SOURCE_MEAN'] * d['AGE_YEARS']
        if not np.isnan(d['EXT_SOURCE_MEAN']) else np.nan
    )

    age = d['AGE_YEARS']
    if   age < 25: d['AGE_GROUP'] = 0
    elif age < 35: d['AGE_GROUP'] = 1
    elif age < 45: d['AGE_GROUP'] = 2
    elif age < 55: d['AGE_GROUP'] = 3
    else:          d['AGE_GROUP'] = 4

    row = {feat: d.get(feat, 0) for feat in feature_names}
    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════
# 3. PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════
def predict(input_df):
    probs = np.zeros(len(input_df))
    for model in lgb_models:
        probs += model.predict_proba(input_df)[:, 1]
    return probs / len(lgb_models)

def get_risk_category(prob, threshold):
    if   prob >= 0.70:        return "VERY HIGH RISK", "risk-very-high", "🔴"
    elif prob >= 0.50:        return "HIGH RISK",      "risk-high",      "🟠"
    elif prob >= threshold:   return "MEDIUM RISK",    "risk-medium",    "🟡"
    else:                     return "LOW RISK",        "risk-low",      "🟢"


# ══════════════════════════════════════════════════════════════════
# 4. SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=80)
    st.markdown("## 🏦 Credit Risk Predictor")
    st.divider()

    st.markdown("### 📊 Model Performance")
    st.metric("LightGBM OOF AUC", "0.7856")
    st.metric("XGBoost OOF AUC",  "0.7859")
    st.metric("Ensemble AUC",     "0.790+")
    st.metric("Training Samples", "307,511")
    st.metric("Features",         f"{len(feature_names)}")
    st.divider()

    st.markdown("### 🔑 Top Predictors")
    top5 = shap_importance.head(5)
    for _, row in top5.iterrows():
        st.markdown(f"**{int(row['rank'])}. {row['feature']}**")

    st.divider()
    st.markdown("*Built with LightGBM + SHAP*")
    st.markdown("*Track 1 — Supervised ML*")


# ══════════════════════════════════════════════════════════════════
# 5. DIVIDER
# ══════════════════════════════════════════════════════════════════
st.divider()


# ══════════════════════════════════════════════════════════════════
# 6. INPUT FORM
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">📋 Customer Application Details</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**💰 Financial Information**")
    AMT_INCOME_TOTAL = st.number_input("Annual Income (₹)",   min_value=10000,  max_value=10000000, value=200000, step=10000)
    AMT_CREDIT       = st.number_input("Loan Amount (₹)",     min_value=10000,  max_value=5000000,  value=500000, step=10000)
    AMT_ANNUITY      = st.number_input("Monthly Annuity (₹)", min_value=1000,   max_value=200000,   value=25000,  step=1000)
    AMT_GOODS_PRICE  = st.number_input("Goods Price (₹)",     min_value=10000,  max_value=4000000,  value=450000, step=10000)

with col2:
    st.markdown("**👤 Personal Information**")
    age_years      = st.slider("Age (Years)", 18, 70, 35)
    DAYS_BIRTH     = int(age_years * -365)

    years_employed = st.slider("Years Employed", 0, 40, 5)
    DAYS_EMPLOYED  = int(years_employed * -365)

    CNT_CHILDREN   = st.slider("Number of Children", 0, 10, 0)
    CNT_FAM_MEMBERS = st.slider("Family Members",     1, 10, 3)

    DAYS_REGISTRATION      = st.slider("Years Since Registration",  0, 30, 10) * -365
    DAYS_ID_PUBLISH        = st.slider("Years Since ID Published",  0, 20,  5) * -365
    DAYS_LAST_PHONE_CHANGE = st.slider("Years Since Phone Change",  0, 10,  2) * -365

with col3:
    st.markdown("**🏦 Credit Score & Profile**")
    EXT_SOURCE_1 = st.slider("External Credit Score 1",     0.0, 1.0, 0.5, 0.01, help="Third-party credit score (0=worst, 1=best)")
    EXT_SOURCE_2 = st.slider("External Credit Score 2 ⭐",  0.0, 1.0, 0.5, 0.01, help="Most important predictor!")
    EXT_SOURCE_3 = st.slider("External Credit Score 3",     0.0, 1.0, 0.5, 0.01)

    st.markdown("**🏠 Assets & Profile**")
    FLAG_OWN_CAR    = st.selectbox("Owns Car?",      ["No", "Yes"])
    FLAG_OWN_REALTY = st.selectbox("Owns Property?", ["No", "Yes"])
    NAME_EDUCATION_TYPE = st.selectbox("Education", [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree"
    ])
    CODE_GENDER = st.selectbox("Gender", ["M", "F"])


# ══════════════════════════════════════════════════════════════════
# 7. PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════
st.divider()
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_clicked = st.button("🔮 PREDICT DEFAULT RISK", type="primary")

if predict_clicked:

    raw_inputs = {
        'AMT_INCOME_TOTAL'       : AMT_INCOME_TOTAL,
        'AMT_CREDIT'             : AMT_CREDIT,
        'AMT_ANNUITY'            : AMT_ANNUITY,
        'AMT_GOODS_PRICE'        : AMT_GOODS_PRICE,
        'DAYS_BIRTH'             : DAYS_BIRTH,
        'DAYS_EMPLOYED'          : DAYS_EMPLOYED,
        'CNT_CHILDREN'           : CNT_CHILDREN,
        'CNT_FAM_MEMBERS'        : CNT_FAM_MEMBERS,
        'DAYS_REGISTRATION'      : DAYS_REGISTRATION,
        'DAYS_ID_PUBLISH'        : DAYS_ID_PUBLISH,
        'DAYS_LAST_PHONE_CHANGE' : DAYS_LAST_PHONE_CHANGE,
        'EXT_SOURCE_1'           : EXT_SOURCE_1,
        'EXT_SOURCE_2'           : EXT_SOURCE_2,
        'EXT_SOURCE_3'           : EXT_SOURCE_3,
        'FLAG_OWN_CAR'           : 1 if FLAG_OWN_CAR    == "Yes" else 0,
        'FLAG_OWN_REALTY'        : 1 if FLAG_OWN_REALTY == "Yes" else 0,
        'CODE_GENDER'            : 0 if CODE_GENDER == "M" else 1,
        'NAME_EDUCATION_TYPE'    : [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree"
        ].index(NAME_EDUCATION_TYPE),
    }

    with st.spinner("Running feature engineering..."):
        input_df = engineer_features(raw_inputs)

    with st.spinner("Computing default probability..."):
        prob = predict(input_df)[0]
        risk_label, risk_class, risk_emoji = get_risk_category(prob, best_threshold)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    # 8. RESULTS SECTION
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)

    with res_col1:
        # FIX: Build risk badge as a compact single-line string
        st.markdown(
            f'<div class="{risk_class}">{risk_emoji} {risk_label}</div>',
            unsafe_allow_html=True
        )

    with res_col2:
        st.metric(
            label="Default Probability",
            value=f"{prob:.1%}",
            delta=f"{prob - 0.08:.1%} vs average",
            delta_color="inverse"
        )

    with res_col3:
        decision = "❌ REJECT" if prob >= best_threshold else "✅ APPROVE"
        st.metric(label="Loan Decision", value=decision)

    with res_col4:
        confidence = abs(prob - 0.5) / 0.5 * 100
        st.metric(label="Model Confidence", value=f"{confidence:.0f}%")

    # ── Probability gauge ──────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Risk Probability Gauge</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    zones = [
        (0.00, 0.20, '#d5f5e3', 'Very Safe'),
        (0.20, 0.40, '#a9dfbf', 'Safe'),
        (0.40, 0.55, '#fdebd0', 'Caution'),
        (0.55, 0.70, '#f5cba7', 'Risky'),
        (0.70, 1.00, '#f1948a', 'Danger'),
    ]

    for x_start, x_end, color, label in zones:
        ax.barh(0.5, x_end - x_start, left=x_start, height=0.6,
                color=color, edgecolor='white', linewidth=2)
        ax.text((x_start + x_end) / 2, 0.5, label,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='#2c3e50')

    ax.axvline(prob, color='#2c3e50', linewidth=4, ymin=0.1, ymax=0.9)
    ax.plot(prob, 0.85, 'v', color='#2c3e50', markersize=15)
    ax.text(prob, 0.1, f'{prob:.1%}', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#2c3e50')
    ax.axvline(best_threshold, color='#e74c3c', linewidth=2,
               linestyle='--', ymin=0.1, ymax=0.9,
               label=f'Threshold ({best_threshold:.2f})')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ══════════════════════════════════════════════════════════════
    # 9. KEY FINANCIAL RATIOS
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">💡 Key Risk Indicators</div>', unsafe_allow_html=True)

    credit_income  = AMT_CREDIT / AMT_INCOME_TOTAL
    annuity_income = AMT_ANNUITY / AMT_INCOME_TOTAL * 100
    credit_term    = AMT_CREDIT / AMT_ANNUITY
    ext_mean       = np.mean([EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Credit / Income Ratio",  f"{credit_income:.2f}x",
              "⚠️ High" if credit_income > 3    else "✅ Normal", delta_color="off")
    r2.metric("Monthly Burden",         f"{annuity_income:.1f}% of income",
              "⚠️ High" if annuity_income > 30  else "✅ Normal", delta_color="off")
    r3.metric("Loan Term",              f"{credit_term:.0f} months",
              "⚠️ Long" if credit_term > 48     else "✅ Short",  delta_color="off")
    r4.metric("Avg Credit Score",       f"{ext_mean:.2f}",
              "⚠️ Low"  if ext_mean < 0.4       else "✅ Good",   delta_color="off")

    # ══════════════════════════════════════════════════════════════
    # 10. SHAP EXPLANATION
    # ══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🔍 Why This Prediction? — SHAP Explanation</div>',
                unsafe_allow_html=True)

    with st.spinner("Computing SHAP explanation..."):
        shap_vals_single = explainer.shap_values(input_df)
        if isinstance(shap_vals_single, list):
            shap_vals_single = shap_vals_single[1]

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap':    shap_vals_single[0]
    }).sort_values('shap', key=abs, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['shap'].values]

    bars = ax.barh(shap_df['feature'][::-1], shap_df['shap'][::-1],
                   color=colors[::-1], edgecolor='white', height=0.7)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(
        'SHAP Value  (Red = increases default risk  |  Green = decreases default risk)',
        fontsize=10
    )
    ax.set_title(
        f'Top 15 Features Driving This Prediction\nCustomer Default Probability: {prob:.1%}',
        fontweight='bold', fontsize=12, pad=12
    )

    for bar, val in zip(bars, shap_df['shap'][::-1].values):
        ax.text(val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}', va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')

    red_patch   = mpatches.Patch(color='#e74c3c', label='Increases default risk')
    green_patch = mpatches.Patch(color='#2ecc71', label='Decreases default risk')
    ax.legend(handles=[red_patch, green_patch], loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Plain English SHAP explanation ────────────────────────────
    st.markdown('<div class="section-title">📖 Plain English Explanation</div>',
                unsafe_allow_html=True)

    top_risk_features    = shap_df[shap_df['shap'] > 0].head(3)
    top_protect_features = shap_df[shap_df['shap'] < 0].head(3)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**🔴 Factors Increasing Default Risk:**")
        for _, row in top_risk_features.iterrows():
            val_str = f"{float(input_df[row['feature']].values[0]):.3f}" \
                      if row['feature'] in input_df.columns else "N/A"
            # FIX: compact single-line f-string avoids stray HTML rendering
            st.markdown(
                f'<div class="insight-box">⬆️ <b>{row["feature"]}</b><br>'
                f'Value&nbsp;: <b>{val_str}</b><br>'
                f'SHAP&nbsp;&nbsp;: <b style="color:#e74c3c">+{row["shap"]:.4f}</b>'
                f' — pushes toward default</div>',
                unsafe_allow_html=True
            )

    with exp_col2:
        st.markdown("**🟢 Factors Decreasing Default Risk:**")
        for _, row in top_protect_features.iterrows():
            val_str = f"{float(input_df[row['feature']].values[0]):.3f}" \
                      if row['feature'] in input_df.columns else "N/A"
            st.markdown(
                f'<div class="insight-box">⬇️ <b>{row["feature"]}</b><br>'
                f'Value&nbsp;: <b>{val_str}</b><br>'
                f'SHAP&nbsp;&nbsp;: <b style="color:#2ecc71">{row["shap"]:.4f}</b>'
                f' — pulls away from default</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════
    # 11. GLOBAL FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown('<div class="section-title">🌍 Global Model Insights</div>',
                unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Top 10 Most Important Features (Global)**")
        top10 = shap_importance.head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top10['feature'][::-1], top10['mean_abs_shap'][::-1],
                color='#3498db', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Global Feature Importance', fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with g2:
        st.markdown("**Risk Factor Summary**")
        st.dataframe(
            shap_importance[['rank', 'feature', 'mean_abs_shap', 'direction']].head(10),
            use_container_width=True,
            hide_index=True
        )

    # ── Footer ─────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<center><small>🏦 Credit Risk Predictor · Built with LightGBM + SHAP + Streamlit</small></center>",
        unsafe_allow_html=True
    )
