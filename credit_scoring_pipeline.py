"""
Credit Risk Scoring & Default Prediction Pipeline
Project 1: 10 Data Science Projects That Get You Hired
Author: Apollos Digital Solutions
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, brier_score_loss

def run_pipeline(data_path="train_u6lujuX_CVtuZ9i.csv"):
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Checking local directory...")
        data_path = os.path.join(os.path.dirname(__file__), "train_u6lujuX_CVtuZ9i.csv")

    print("=" * 60)
    print("STEP 1: INGESTING LOAN APPLICATION DATA")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Target Mapping: Loan_Status (Y -> 1: Approved / Low Risk, N -> 0: Default / High Risk)
    target_col = "Loan_Status"
    df[target_col] = df[target_col].map({"Y": 1, "N": 0})
    
    # Drop Loan_ID identifier
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"Target Distribution: Approved={y.sum()} ({y.mean()*100:.1f}%), Rejected={len(y)-y.sum()}")

    # Feature Categorization
    num_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    cat_features = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    print("\n" + "=" * 60)
    print("STEP 2: BUILDING LEAK-FREE PREPROCESSING PIPELINE")
    print("=" * 60)

    # Preprocessors
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ]
    )

    # Split: 80% Train, 20% Holdout Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")

    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING & HYPERPARAMETER EVALUATION")
    print("=" * 60)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    }

    best_model_name = None
    best_score = 0
    best_pipeline = None

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_auc = cv_scores.mean()
        std_auc = cv_scores.std()

        print(f"[{name}] 5-Fold Stratified ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

        if mean_auc > best_score:
            best_score = mean_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n--> Selected Champion Model: {best_model_name} (Cross-Validated AUC: {best_score:.4f})")

    # Fit Champion Model on Full Training Set
    best_pipeline.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("STEP 4: HOLDOUT EVALUATION & CALIBRATION")
    print("=" * 60)

    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    print(f"Holdout Test ROC-AUC: {test_auc:.4f}")
    print(f"Probability Calibration (Brier Score): {brier:.4f} (Closer to 0 is better)")
    print("\nClassification Report (Holdout):")
    print(classification_report(y_test, y_pred, target_names=["Default (0)", "Approved (1)"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n" + "=" * 60)
    print("STEP 5: CREDIT SCORECARD MAPPING (300 - 850 SCALE)")
    print("=" * 60)
    # Map probability of non-default to 300 - 850 FICO-style credit score
    # Score = Offset + Factor * ln(Odds)
    sample_scores = 300 + (y_prob * 550)
    print(f"Sample Scaled Credit Scores: Min={sample_scores.min():.0f}, Median={np.median(sample_scores):.0f}, Max={sample_scores.max():.0f}")
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
