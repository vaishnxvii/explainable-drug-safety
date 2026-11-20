import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, brier_score_loss
from sklearn.feature_extraction.text import CountVectorizer

NON_CLINICAL_TERMS = {
    "OFF LABEL USE", "DRUG INEFFECTIVE", "PRODUCT DOSE OMISSION ISSUE",
    "INTENTIONAL PRODUCT USE ISSUE", "INAPPROPRIATE SCHEDULE OF PRODUCT ADMINISTRATION",
    "PRODUCT USE IN UNAPPROVED INDICATION", "NO ADVERSE EVENT"
}

def build_features_and_train(data_path="data/processed/faers_clean_25k.csv", top_k_drugs=300, top_n_adrs=8):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records from {data_path}")

    df["is_female"] = (df["sex"] == 2).astype(int)
    df["is_elderly"] = (df["age"] >= 65).astype(int)
    df["age_x_polypharmacy"] = df["age"] * df["polypharmacy_count"]

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x.split("|"),
        token_pattern=None,
        binary=True,
        max_features=top_k_drugs
    )
    drug_matrix = vectorizer.fit_transform(df["all_drugs"].fillna("")).toarray()
    drug_cols = [f"DRUG_{d}" for d in vectorizer.get_feature_names_out()]

    X_base = df[["age", "is_female", "polypharmacy_count", "is_elderly", "age_x_polypharmacy"]].copy()
    X_drugs = pd.DataFrame(drug_matrix, columns=drug_cols, index=df.index)
    X = pd.concat([X_base, X_drugs], axis=1)

    X_global_sample = X.sample(n=min(500, len(X)), random_state=42)

    all_reactions = df["reactions"].str.split("|").explode()
    clinical_reactions = all_reactions[~all_reactions.isin(NON_CLINICAL_TERMS)]
    target_adrs = clinical_reactions.value_counts().head(top_n_adrs).index.tolist()
    print(f"\nTraining Multi-Model Suite (XGBoost, Random Forest, Logistic Regression) for: {target_adrs}\n")

    os.makedirs("models", exist_ok=True)
    metadata = {
        "feature_columns": X.columns.tolist(),
        "drug_vocabulary": vectorizer.get_feature_names_out().tolist(),
        "target_adrs": target_adrs,
        "available_models": ["XGBoost (Proposed)", "Random Forest", "Logistic Regression"],
        "metrics": {m: {} for m in ["XGBoost (Proposed)", "Random Forest", "Logistic Regression"]}
    }

    for adr in target_adrs:
        y = df["reactions"].apply(lambda r: 1 if adr in r.split("|") else 0).values
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        scale_weight = neg_count / max(pos_count, 1)
        baseline_rate = pos_count / len(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        safe_adr = adr.lower().replace(" ", "_")

        # 1. XGBoost (Proposed)
        base_xgb = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_weight, eval_metric="aucpr",
            random_state=42, n_jobs=-1
        ).fit(X_train, y_train)

        cal_xgb = CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_weight, eval_metric="aucpr",
                random_state=42, n_jobs=-1
            ), method="sigmoid", cv=3
        ).fit(X_train, y_train)

        exp_xgb = shap.TreeExplainer(base_xgb)

        # 2. Random Forest
        base_rf = RandomForestClassifier(
            n_estimators=150, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        ).fit(X_train, y_train)

        cal_rf = CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=150, max_depth=10, class_weight="balanced",
                random_state=42, n_jobs=-1
            ), method="sigmoid", cv=3
        ).fit(X_train, y_train)

        exp_rf = shap.TreeExplainer(base_rf)

        # 3. Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        base_lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42).fit(X_train_scaled, y_train)

        cal_lr = CalibratedClassifierCV(
            estimator=make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
            method="sigmoid", cv=3
        ).fit(X_train, y_train)

        exp_lr = shap.LinearExplainer(base_lr, X_train_scaled)

        # Save all 3 model families
        suite = {
            "XGBoost (Proposed)": ("xgb", base_xgb, cal_xgb, exp_xgb, None),
            "Random Forest": ("rf", base_rf, cal_rf, exp_rf, None),
            "Logistic Regression": ("lr", base_lr, cal_lr, exp_lr, scaler),
        }

        for m_label, (prefix, raw_clf, cal_clf, explainer, sc) in suite.items():
            y_raw_prob = raw_clf.predict_proba(scaler.transform(X_test) if sc else X_test)[:, 1]
            y_cal_prob = cal_clf.predict_proba(X_test)[:, 1]

            metadata["metrics"][m_label][adr] = {
                "positives": pos_count,
                "baseline_rate": round(float(baseline_rate * 100), 2),
                "roc_auc": round(float(roc_auc_score(y_test, y_cal_prob)), 4),
                "pr_auc": round(float(average_precision_score(y_test, y_cal_prob)), 4),
                "recall": round(float(recall_score(y_test, (y_raw_prob >= 0.5).astype(int))), 4),
                "brier_calibrated": round(float(brier_score_loss(y_test, y_cal_prob)), 4)
            }

            joblib.dump(cal_clf, f"models/cal_{prefix}_{safe_adr}.joblib")
            joblib.dump(explainer, f"models/shap_{prefix}_{safe_adr}.joblib")
            if sc is not None:
                joblib.dump(sc, f"models/scaler_{safe_adr}.joblib")

        # Keep global SHAP for XGBoost tab
        joblib.dump(exp_xgb(X_global_sample), f"models/global_shap_{safe_adr}.joblib")
        print(f"Trained & calibrated all 3 models for [{adr}]")

    joblib.dump(X_global_sample, "models/X_global_sample.joblib")
    joblib.dump(metadata, "models/pipeline_metadata.joblib")
    print("\nAll multi-model calibrated artifacts saved in models/!")

if __name__ == "__main__":
    build_features_and_train()
