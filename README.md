# 💳 Production Credit Risk Scoring & Default Prediction Pipeline

> **A leak-free, production-grade machine learning system that transforms raw applicant features into calibrated default probabilities and an industry-standard 300–850 credit scorecard.**

---

## 📌 Business Overview & Architecture

In retail banking and FinTech lending, predicting credit default ($y \in \{0, 1\}$) is an asymmetrical risk problem. A False Positive (approving an applicant who defaults) costs the entire principal balance, while a False Negative (rejecting a creditworthy borrower) only loses the expected interest margin.

This repository provides an end-to-end, reproducible implementation designed according to Tier-1 financial engineering standards:

* **Zero-Leakage Preprocessing:** Numerical and categorical transformations are strictly fit on training folds using Scikit-Learn `ColumnTransformer`.
* **Stratified 5-Fold Cross-Validation:** Robust evaluation preventing optimistic bias on imbalanced targets (~69% approved, ~31% defaulted).
* **Calibrated Probability Modeling:** Benchmarks Logistic Regression and XGBoost with Brier score calibration.
* **300–850 Credit Scorecard Conversion:** Maps predicted default probabilities into standard FICO-scale scores via log-odds:
  $$\text{Score} = 600 + 50 \times \ln\left(\frac{1 - P(\text{default})}{P(\text{default})}\right)$$

---

## 🛠️ Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/JohnApollos/credit-risk-scoring-pipeline.git
cd credit-risk-scoring-pipeline
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
Run the complete training, cross-validation, and scoring script:
```bash
python credit_scoring_pipeline.py
```

### 3. Pipeline Output
The pipeline produces:
* Cross-validation ROC-AUC, Precision, Recall, and F1 metrics.
* Calibrated default probability distributions.
* Sample applicant credit scorecard outputs (300 to 850 range).
* Serialized production model pipeline saved to disk (`credit_scoring_model.joblib`).

---

## 📚 Part of the "10 Data Science Projects That Get You Hired" Series

This repository is **Project 01** from the comprehensive book and codebase:
**10 Data Science Projects That Get You Hired** by John Apollos.

If you are preparing for senior Data Science, ML Engineering, or Quant roles, get the complete collection:
* 10 complete end-to-end industrial pipelines (FinTech, Geospatial, Churn, Fraud, E-Mobility Fleet, Computer Vision).
* 150+ Technical interview preparation questions and answers.
* Production Docker and FastAPI deployment configurations.

👉 **[Get the Complete Book & Codebase on Gumroad ($19)](https://gumroad.com)**  
👉 **[Get the Kindle Edition on Amazon ($9.99)](https://amazon.com)**

---

## 📄 License
MIT License. Free to use for personal, academic, and commercial evaluation.  
Created by **John Apollos** | Apollos Digital Solutions.
