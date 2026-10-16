# 💳 PROJECT 1: End-to-End Credit Risk Scoring & Default Prediction Pipeline
**Author:** John Apollos | Apollos Digital Solutions  
**Series:** 10 Data Science Projects That Get You Hired (Project 1 of 10)  
**Tech Stack:** Python 3.11+, Scikit-Learn, Pandas, NumPy, XGBoost, Streamlit  
**Domain:** Financial Services / FinTech Lending  

---

## 📌 Executive Summary & Business Objective

In commercial banking and digital fintech lending, accurately predicting credit default is the single highest-leverage machine learning application. A 1% improvement in default prediction accuracy translates to millions of dollars saved in non-performing loans (NPLs).

This project delivers a **leak-free, production-grade credit risk scoring system** that:
1. Ingests raw multi-attribute loan applicant data.
2. Applies automated, leak-free feature preprocessing via Scikit-Learn `ColumnTransformer`.
3. Trains and benchmarks ensemble classifiers using Stratified 5-Fold Cross-Validation.
4. Evaluates predictions using threshold-invariant metrics (ROC-AUC, PR-AUC, and Brier Score Calibration).
5. Maps model probabilities into an industry-standard **300–850 Credit Scorecard** using the log-odds transformation.

---

## 1. Problem Framing & Cost Matrix

Traditional accuracy is a fatal metric in credit risk because defaults are heavily imbalanced. In our dataset:
* **Approved / Performing Loans ($y=1$):** ~69%
* **Defaults / Rejections ($y=0$):** ~31%

In fintech lending, the cost of an error is asymmetrical:
* **False Positive (Type I Error):** Predicting an applicant will repay when they actually default. Loss: Total principal lost (e.g., $10,000).
* **False Negative (Type II Error):** Rejecting a creditworthy applicant. Loss: Opportunity cost of interest margin (e.g., $400).

Therefore, model selection prioritizes **Precision-Recall AUC and Probability Calibration** rather than raw classification accuracy.

---

## 2. Leak-Free Preprocessing Architecture

Data leakage occurs when information from the validation or test fold inadvertently influences feature transformations. To ensure zero data leakage:

```python
# 1. Numerical Preprocessing: Median Imputation + Standardization
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 2. Categorical Preprocessing: Mode Imputation + One-Hot Encoding
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

# 3. Combined ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)
```

---

## 3. Benchmark Model Performance

Using 5-Fold Stratified Cross-Validation on the training partition:

| Model Architecture | 5-Fold CV ROC-AUC | Std Dev | Brier Calibration Score |
|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.7420 | ± 0.038 | 0.165 |
| **Random Forest (200 trees, depth=6)** | 0.7915 | ± 0.029 | 0.148 |
| **Gradient Boosting Classifier (Champion)** | **0.8140** | **± 0.024** | **0.139** |

---

## 4. Converting Probabilities to a 300–850 Credit Scorecard

Fintech risk committees do not inspect raw probabilities; they require a scaled credit score analogous to FICO (ranging from 300 to 850).

The mathematical transformation maps model calibrated probability $p$ to a score using log-odds:

$$\text{Odds} = \frac{p}{1 - p}$$
$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

Where Offset and Factor are calibrated so that:
* Base score = 600 at odds of 50:1
* 20 points double the odds (PDO = 20)

In our pipeline, scaled scores partition applicants into 5 credit risk tiers:
* **750 – 850 (Tier A):** Instant Approval, lowest interest rate
* **700 – 749 (Tier B):** Automated Approval, standard rate
* **650 – 699 (Tier C):** Conditional Approval, secondary income verification
* **600 – 649 (Tier D):** Manual Underwriting Review
* **300 – 599 (Tier E):** Automated Decline

---

## 5. Technical Interview Questions & Talking Points

When presenting this project to interviewers at top tech and finance companies, be prepared to answer:

### Q1: "Why did you use ColumnTransformer instead of transforming the Pandas DataFrame directly before splitting?"
> **High-Scoring Answer:**  
> "Transforming the entire DataFrame before the train/test split introduces data leakage. For example, computing the mean or median on the whole dataset allows information from the test set to bleed into the training partition. By embedding `ColumnTransformer` inside a Scikit-Learn `Pipeline`, all fit operations happen strictly on the training folds during cross-validation, guaranteeing completely unbiased holdout evaluation."

### Q2: "How would you monitor this model once deployed to production?"
> **High-Scoring Answer:**  
> "I would implement monitoring across two dimensions: data drift and concept drift. For data drift, I'd track Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests on key numerical inputs like `ApplicantIncome` and `LoanAmount`. A PSI > 0.2 indicates significant distribution shift. For concept drift, I would track rolling Brier scores and 90-day delinquency rates against initial probability predictions."

---

## 🚀 Get the Complete Book (10 Portfolio-Ready Projects)

This is **Project 1 of 10** from the book:  
**"10 Data Science Projects That Get You Hired"**

### What are the remaining 9 projects?
* **Project 2:** Customer Churn & LTV Value Optimization
* **Project 3:** Real Estate Geospatial Price Forecasting
* **Project 4:** E-Mobility Fleet Energy & Battery Degradation Analysis
* **Project 5:** Financial Fraud & Anomaly Detection Pipeline
* **Project 6:** Startup Venture Capital Growth Valuation
* **Project 7:** E-Commerce Multi-Touch Attribution Engine
* **Project 8:** Content Safety & Toxic Language Classification (NLP)
* **Project 9:** Supply Chain Demand & Inventory Lead-Time Forecasting
* **Project 10:** Agricultural Crop Yield & Climate Change Impact

👉 **[Get the Full Book on Amazon KDP ($9.99) & Gumroad ($19)](https://gumroad.com/l/10-ds-projects)**
