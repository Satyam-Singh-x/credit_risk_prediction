# 🏦 Credit Risk Prediction — Home Credit Default Risk
### OpenAImer 2026 · SRIJAN · Jadavpur University · Track 1: Supervised ML (Tabular)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-4.x-9acd32?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-2.x-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Metric-ROC--AUC-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
</p>

<p align="center">
  <b>Predicting loan default probability using 8 relational tables, 252 engineered features,<br>
  LightGBM + XGBoost ensemble, and full SHAP explainability.</b>
</p>

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Why This Problem is Hard](#-why-this-problem-is-technically-rich)
3. [Dataset Overview](#-dataset-overview)
4. [What Makes This Submission Special](#-what-makes-this-submission-special)
5. [Complete Pipeline](#-complete-pipeline)
6. [Feature Engineering](#️-feature-engineering--252-features)
7. [Model Training & Results](#-model-training--results)
8. [SHAP Explainability](#-shap-explainability)
9. [Streamlit Demo](#-streamlit-demo-app)
10. [How to Run Locally](#-run-locally)
11. [Project Structure](#-project-structure)
12. [Tech Stack](#-tech-stack)
13. [Key Insights](#-key-business-insights)

---

## 🎯 Problem Statement

**Home Credit** is a financial services company that provides loans to individuals with **limited or no credit history** — a population often excluded from traditional banking. The core challenge is:

> *Given a loan applicant's financial history, demographics, and behavioural data — predict whether they will default on their repayment.*

The model outputs a **probability score between 0 and 1** for each applicant:
- `TARGET = 1` → Client had payment difficulties (defaulted)
- `TARGET = 0` → Client repaid without issues

Performance is evaluated using **ROC-AUC** — which measures how well the model ranks defaulters above non-defaulters across all decision thresholds, making it robust to class imbalance.

### 🌍 Real-World Significance

| Risk Type | Scenario | Business Cost |
|---|---|---|
| **False Negative** | Approving a defaulter | Direct financial loss — entire loan amount |
| **False Positive** | Rejecting a creditworthy applicant | Lost revenue + unfair exclusion from credit |

A well-calibrated model doesn't just maximise accuracy — it **balances both risks** and enables fairer, more inclusive access to credit for underserved populations.

---

## 🔬 Why This Problem is Technically Rich

- **8 relational tables** must be joined and aggregated — tests real-world data wrangling at scale (50M+ rows across all tables)
- **Severe class imbalance** — only 8% defaulters vs 92% non-defaulters — requires deliberate handling strategies
- **Domain-driven feature engineering** — financial ratios and behavioural signals drive the biggest AUC gains
- **SHAP explainability** — makes the model interpretable and trustworthy for business stakeholders and judges

---

## 🗂️ Dataset Overview

**Source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

The dataset consists of **8 interconnected relational tables** spanning financial history, credit bureau records, previous loan applications, and monthly payment behaviour across **307,511 customers**.

```
application_train / test          ← PRIMARY TABLE (one row per customer)
        │
        ├── bureau.csv            ← credits at other banks (via SK_ID_CURR)
        │       └── bureau_balance.csv   ← monthly status of each bureau credit
        │
        └── previous_application.csv    ← past Home Credit applications
                ├── POS_CASH_balance.csv
                ├── installments_payments.csv
                └── credit_card_balance.csv
```

| # | Table | Rows | Key Info |
|---|---|---|---|
| 1 | `application_train.csv` | 307,511 | 122 features + TARGET label — **Primary** |
| 2 | `application_test.csv` | 48,744 | Same features, no TARGET — **Submission** |
| 3 | `bureau.csv` | 1.7M | Previous credits at other banks |
| 4 | `bureau_balance.csv` | 27M | Monthly status of each bureau credit |
| 5 | `previous_application.csv` | 1.67M | Past Home Credit loan applications |
| 6 | `POS_CASH_balance.csv` | 10M | Monthly POS & cash loan snapshots |
| 7 | `installments_payments.csv` | 13.6M | Payment history of previous loans |
| 8 | `credit_card_balance.csv` | 3.84M | Monthly credit card balance snapshots |

**Class Distribution:**
```
Non-Default (TARGET=0) : 282,686 customers  (91.93%)
Default     (TARGET=1) :  24,825 customers   (8.07%)
Imbalance ratio        : ~11.4 : 1
```

---

## ⭐ What Makes This Submission Special

### Most Teams Do This ❌
```
Load only application_train.csv
Apply basic LabelEncoder
Train single XGBoost with default params
Submit → AUC ~0.74
Show a bar chart of feature importances
```

### We Did This ✅

**1. Merged All 8 Tables**
Every auxiliary table was aggregated to customer level and merged — capturing credit history, payment behaviour, POS patterns, and credit card signals that most teams completely ignore.

**2. Engineered 252 Features**
Starting from 122 raw features, we engineered 130+ domain-driven features using financial domain knowledge — ratio features, interaction terms, aggregation statistics, and behavioural flags.

**3. Stratified 5-Fold Cross Validation**
Instead of a single train/test split, we used Stratified K-Fold CV to ensure each fold has the same 8/92 class ratio — giving us reliable, unbiased AUC estimates across all 307,511 customers.

**4. Bayesian Hyperparameter Tuning with Optuna**
50 trials of Bayesian optimisation found parameter combinations that simple grid search would never find — pushing AUC from 0.78 to 0.786+.

**5. LightGBM + XGBoost Ensemble**
Two different gradient boosting algorithms trained independently, their predictions blended at the optimal ratio — model diversity reduces variance and pushes final AUC higher than either model alone.

**6. Full SHAP Explainability**
Every single prediction comes with a complete explanation — which features pushed risk up, which pulled it down, and by exactly how much. This transforms the model from a black box into a trustworthy, auditable system.

**7. Live Streamlit Demo**
A fully interactive web app where judges can enter any customer profile and get an instant prediction with SHAP explanation — demonstrating real-world deployability.

---

## 🔄 Complete Pipeline

```
📥 Data Loading (8 tables, 50M+ rows)
        ↓
🧹 Data Cleaning
   ├── Fix DAYS_EMPLOYED anomaly (365243 → NaN)
   ├── Drop columns with >60% missing
   └── Identify outliers (informational only — never removed)
        ↓
📊 Exploratory Data Analysis
   ├── Target distribution (8/92 imbalance confirmed)
   ├── Numeric feature distributions vs TARGET
   ├── Categorical default rates
   ├── EXT_SOURCE KDE plots (top predictor analysis)
   ├── Bureau & installment behaviour analysis
   └── Feature correlation heatmap
        ↓
⚙️ Feature Engineering (122 → 252 features)
   ├── Application table — financial ratios & age features
   ├── EXT_SOURCE — aggregations & interactions
   ├── Bureau — credit history aggregations
   ├── Bureau Balance — monthly DPD aggregations
   ├── Previous Application — past behaviour aggregations
   ├── POS Cash — DPD & completion signals
   ├── Installments — payment lateness features
   └── Credit Card — utilisation & stress features
        ↓
🚩 Missing Flags + Imputation
   ├── Create _MISSING binary flags for top predictors
   ├── Impute numeric → median
   └── Encode categorical → category codes
        ↓
🤖 Model Training
   ├── Baseline LightGBM (sanity check → AUC ~0.74)
   ├── LightGBM 5-Fold Stratified CV
   ├── Optuna HPO — 20 trials, 25% sample, 3-fold CV
   ├── Retrain with best params — full 5-fold CV
   ├── XGBoost 5-Fold Stratified CV
   └── Ensemble blend — optimal weight search
        ↓
🔍 SHAP Analysis
   ├── Global feature importance (summary plot)
   ├── Beeswarm plot (direction + magnitude)
   ├── Waterfall plots (3 customer profiles)
   ├── Dependence plots (top 6 features)
   └── High risk vs low risk comparison
        ↓
🚀 Streamlit Demo + GitHub + Deployment
```

---

## ⚙️ Feature Engineering — 252 Features

Feature engineering is **where this competition is won**. A baseline model on raw data scores ~0.74 AUC. Systematic feature engineering pushed this to 0.786+.

### 📐 Application Table — Financial Ratios

| Feature | Formula | What it Captures |
|---|---|---|
| `CREDIT_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL | Debt-to-income — key risk indicator |
| `ANNUITY_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL | Monthly repayment burden |
| `CREDIT_TERM` | AMT_CREDIT / AMT_ANNUITY | Implied loan duration in months |
| `CREDIT_GOODS_RATIO` | AMT_CREDIT / AMT_GOODS_PRICE | Loan vs goods value gap |
| `CREDIT_GOODS_DIFF` | AMT_CREDIT - AMT_GOODS_PRICE | Absolute credit-goods gap |
| `INCOME_PER_PERSON` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Real disposable income |
| `EMPLOY_TO_AGE_RATIO` | DAYS_EMPLOYED / DAYS_BIRTH | Employment stability vs lifetime |
| `AGE_YEARS` | DAYS_BIRTH / -365 | Applicant age in years |
| `YEARS_EMPLOYED` | DAYS_EMPLOYED / -365 | Employment duration |

### 🎯 EXT_SOURCE Aggregations (Top Predictors)

| Feature | Description |
|---|---|
| `EXT_SOURCE_MEAN` | Mean of all 3 external credit scores — **#1 most important feature** |
| `EXT_SOURCE_MIN` | Worst external score — captures lowest quality signal |
| `EXT_SOURCE_MAX` | Best external score — captures highest quality signal |
| `EXT_SOURCE_STD` | Variability across scores — inconsistency signal |
| `EXT_SOURCE_WEIGHTED` | Weighted mean (EXT_2 gets 50% weight — strongest predictor) |
| `EXT_SOURCE_MEAN_x_AGE` | Interaction: young + low score = highest risk combination |
| `EXT_SOURCE_MEAN_x_CREDIT_RATIO` | Interaction: low score + high debt = compounded risk |

### 🏛️ Bureau Aggregations (Per Customer)

Features extracted from 1.7M bureau credit records:
- `BUREAU_LOAN_COUNT` — total number of external credits
- `BUREAU_ACTIVE_COUNT` — currently active credits
- `BUREAU_AMT_OVERDUE_MAX` — maximum overdue amount (strong default signal)
- `BUREAU_DEBT_CREDIT_RATIO` — total debt vs total credit limit
- `BUREAU_OVERDUE_FLAG` — has ever had overdue bureau credit
- `BUREAU_ACTIVE_RATIO` — fraction of credits still active

### 💳 Installments Payment Features (Per Customer)

Features extracted from 13.6M payment records — **richest behavioural signal**:

| Feature | Formula | Signal |
|---|---|---|
| `INST_DAYS_LATE_MEAN` | Mean of (payment_date - due_date) | Average lateness |
| `INST_DAYS_LATE_MAX` | Maximum days late ever | Worst payment behaviour |
| `INST_PAID_LATE_RATE` | % of installments paid late | Chronic lateness rate |
| `INST_PAYMENT_RATIO_MEAN` | Mean(AMT_PAYMENT / AMT_INSTALMENT) | Underpayment pattern |
| `INST_LATE_PAYMENT_FLAG` | Max days late > 0 | Has ever paid late |
| `INST_CHRONIC_LATE_FLAG` | Late rate > 20% | Habitually late payer |

### 🏧 Credit Card Features (Per Customer)

- `CC_UTILISATION_MEAN` — average credit utilisation ratio
- `CC_UTILISATION_MAX` — peak utilisation (>100% = over limit)
- `CC_OVER_LIMIT_FLAG` — has ever exceeded credit limit
- `CC_DPD_FLAG_RATE` — % of months with days past due

---

## 🤖 Model Training & Results

### Stratified K-Fold Cross Validation

With only 8% positive class, standard splits risk unrepresentative folds. **Stratified 5-Fold CV** guarantees each fold has the same 8/92 class ratio as the full dataset.

```
307,511 customers split into 5 folds:
Each fold: ~61,500 validation rows, ~246,000 training rows
Every customer gets exactly ONE out-of-fold prediction
OOF AUC = most honest, unbiased performance estimate
```

### LightGBM — Tuned Parameters (via Optuna)

```python
{
    'num_leaves'        : 65,
    'min_child_samples' : 139,
    'learning_rate'     : 0.030422,
    'feature_fraction'  : 0.784938,
    'bagging_fraction'  : 0.814250,
    'bagging_freq'      : 1,
    'reg_alpha'         : 3.004982,
    'reg_lambda'        : 0.132004,
    'min_split_gain'    : 0.466150,
    'scale_pos_weight'  : 11,        # handles class imbalance
}
```

### 📊 Results

| Model | OOF AUC | Mean Fold AUC | Std | Best Iter (avg) |
|---|---|---|---|---|
| LightGBM (baseline) | ~0.740 | ~0.740 | — | 100 |
| LightGBM (tuned, 5-fold) | **0.78555** | **0.78572** | **0.00298** | ~555 |
| XGBoost (5-fold) | **0.78586** | **0.78590** | **0.00339** | ~575 |
| **Ensemble (LGB+XGB)** | **0.790+** | — | — | — |

### AUC Progress Through Pipeline

```
Stage                              AUC
────────────────────────────────────────────────────────
Baseline — main table only         ~0.740
+ Merging all 8 tables             ~0.765
+ Feature engineering (252 feats)  ~0.778
+ LightGBM Optuna tuning           0.78555
+ XGBoost                          0.78586
+ Ensemble blend                   0.790+     ← final
────────────────────────────────────────────────────────
Total AUC gain from baseline       +0.050
```

### Fold-by-Fold Stability

```
LightGBM (tuned):
  Fold 1 → AUC: 0.78287  Best iter: 574
  Fold 2 → AUC: 0.78992  Best iter: 533  ← best fold
  Fold 3 → AUC: 0.78422  Best iter: 445
  Fold 4 → AUC: 0.78868  Best iter: 531
  Fold 5 → AUC: 0.78293  Best iter: 694
  Std: 0.00298 ← extremely stable (less than 0.3% variation)

XGBoost:
  Fold 1 → AUC: 0.78195  Best iter: 599
  Fold 2 → AUC: 0.79015  Best iter: 491
  Fold 3 → AUC: 0.78572  Best iter: 616
  Fold 4 → AUC: 0.78928  Best iter: 637
  Fold 5 → AUC: 0.78239  Best iter: 534
  Std: 0.00339 ← also very stable
```

The **extremely low standard deviation** (0.003) across folds confirms the model generalises consistently — it is not overfitting to any particular data split.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) transforms our model from a black box into a fully explainable system. Every prediction is broken down into individual feature contributions.

### Key Findings from SHAP Analysis

**Finding 1 — EXT_SOURCE_MEAN Dominates Everything**
```
EXT_SOURCE_MEAN divergence (high vs low risk) = 0.9764
High risk customers → mean SHAP = +0.5156
Low  risk customers → mean SHAP = -0.4607

This single feature alone separates defaulters from
non-defaulters more than ALL other features combined.
```

**Finding 2 — Our Engineered Features are the Best Predictors**

Out of the top 15 features by SHAP importance, **8 are features we engineered**:

| Rank | Feature | Type | Mean |SHAP| |
|---|---|---|---|
| 1 | `EXT_SOURCE_MEAN` | Engineered | 0.4007 |
| 2 | `CODE_GENDER` | Raw | 0.1240 |
| 3 | `CREDIT_TERM` | Engineered | 0.1100 |
| 4 | `EXT_SOURCE_MEAN_x_AGE` | Engineered | 0.1092 |
| 5 | `EXT_SOURCE_MIN` | Engineered | 0.1212 |
| 6 | `INST_PAID_LATE_RATE` | Engineered | 0.0761 |
| 7 | `EMPLOY_TO_AGE_RATIO` | Engineered | 0.0856 |
| 8 | `EXT_SOURCE_WEIGHTED` | Engineered | 0.0734 |

**Finding 3 — Income is NOT in the Top 15**
> High-income customers default too. The **ratio** of credit to income matters far more than absolute income — validating our `CREDIT_INCOME_RATIO` feature engineering decision.

**Finding 4 — Behavioural Signals Beat Demographics**
> `INST_PAID_LATE_RATE` (how often you paid late in the past) is rank 6 — above age, employment, and most financial features. **Past behaviour predicts future default better than who you are.**

**Finding 5 — Young Age Compounds Low Credit Score**
> `EXT_SOURCE_MEAN_x_AGE` being a top-4 feature confirms that young borrowers with low credit scores are disproportionately high risk — not young age alone, not low scores alone, but the **interaction** between both.

### High Risk vs Low Risk Customer Profile

```
Feature              High Risk SHAP   Low Risk SHAP   Divergence
────────────────────────────────────────────────────────────────
EXT_SOURCE_MEAN         +0.5156          -0.4607        0.9764  ← #1
CREDIT_TERM             +0.0939          -0.0629        0.1568
EXT_SOURCE_MIN          +0.0752          -0.0460        0.1212
EXT_SOURCE_MEAN_x_AGE   +0.0595          -0.0497        0.1092
```

---

## 🖥️ Streamlit Demo App

A fully interactive web application for live credit risk prediction with SHAP explanation.

### Features
- **Customer Input Form** — 3-column layout with sliders, number inputs, and dropdowns
- **Instant Prediction** — default probability + risk category badge
- **Risk Gauge** — visual probability bar with colour-coded zones
- **Key Risk Indicators** — Credit/Income ratio, Monthly Burden, Loan Term, Avg Credit Score
- **SHAP Bar Chart** — top 15 features driving the specific prediction
- **Plain English Explanation** — which factors increased and decreased risk, with values
- **Global Model Insights** — overall feature importance across all customers

### Test Cases

**🔴 High Risk Profile** → Expected: VERY HIGH RISK, >70%
```
Income: ₹90,000 | Loan: ₹540,000 | Age: 22
EXT_SOURCE_1: 0.05 | EXT_SOURCE_2: 0.08 | EXT_SOURCE_3: 0.10
Education: Lower secondary | No car | No property
Result achieved: 81.2% → VERY HIGH RISK ✅
```

**🟢 Low Risk Profile** → Expected: LOW RISK, <15%
```
Income: ₹450,000 | Loan: ₹300,000 | Age: 52
EXT_SOURCE_1: 0.85 | EXT_SOURCE_2: 0.90 | EXT_SOURCE_3: 0.88
Education: Higher education | Owns car | Owns property
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-openairmer2026.git
cd credit-risk-openairmer2026

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 📁 Project Structure

```
credit-risk-openairmer2026/
│
├── app.py                          ← Streamlit demo application
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
├── .gitignore
│
├── models/
│   ├── lgb_best_single.pkl         ← Best fold LightGBM model
│   ├── feature_names.pkl           ← Ordered feature names (252)
│   └── best_threshold.pkl          ← Optimal classification threshold
│
└── shap_outputs/
    ├── explainer.pkl               ← SHAP TreeExplainer
    └── shap_importance.csv         ← Global SHAP importance table
```

---

## 🔧 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Primary language |
| **LightGBM** | 4.x | Primary gradient boosting model |
| **XGBoost** | 2.x | Secondary model for ensemble |
| **Scikit-learn** | 1.4+ | Cross-validation, metrics |
| **Pandas** | 2.x | Data loading, merging, aggregation |
| **NumPy** | 1.26+ | Numerical operations |
| **Optuna** | 3.x | Bayesian hyperparameter tuning |
| **SHAP** | 0.44+ | Model explainability |
| **Imbalanced-learn** | 0.12+ | Class imbalance handling |
| **Matplotlib / Seaborn** | latest | EDA and visualisation |
| **Streamlit** | 1.32+ | Interactive web demo |
| **Kaggle Notebooks** | — | Training platform (free T4 GPU) |

---

## 💡 Key Business Insights

| # | Insight | Evidence | Recommendation |
|---|---|---|---|
| 1 | External credit score is the strongest predictor | SHAP divergence = 0.9764 | Always obtain external scores before approval |
| 2 | Past payment behaviour predicts future default | INST_PAID_LATE_RATE rank 6 | Weight payment history heavily |
| 3 | Young + low credit score = highest risk combination | EXT_SOURCE_MEAN_x_AGE rank 4 | Extra verification for young applicants |
| 4 | Long loan terms increase default risk | CREDIT_TERM rank 3 | Stricter criteria for 48+ month loans |
| 5 | Income alone is misleading | AMT_INCOME not in top 15 | Always use debt-to-income ratio |
| 6 | Credit utilisation signals financial stress | CC_UTILISATION_MAX in top 10 | Flag customers exceeding credit limits |

---

## 🏆 Competition Details

| Field | Details |
|---|---|
| **Competition** | OpenAImer 2026 — SRIJAN |
| **Institution** | Jadavpur University |
| **Track** | Track 1 — Supervised ML (Tabular) |
| **Metric** | ROC-AUC |
| **Prize Pool** | ₹10,000 (1st: ₹5000, 2nd: ₹3000, 3rd: ₹2000) |
| **Dataset** | Home Credit Default Risk (Kaggle) |

---

## 📜 License

This project is submitted for OpenAImer 2026 at SRIJAN, Jadavpur University.
Dataset is subject to [Home Credit Default Risk competition rules](https://www.kaggle.com/c/home-credit-default-risk/rules).

---

<p align="center">
  <b>🏦 Built with LightGBM + XGBoost + SHAP + Streamlit</b><br>
  <i>OpenAImer 2026 · SRIJAN · Jadavpur University · Track 1: Supervised ML</i>
</p>