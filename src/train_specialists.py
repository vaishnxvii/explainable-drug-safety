import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
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

    # 1. Clinical Feature Engineering
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

    # Save background cohort sample (500 patients) for Global SHAP Summary Plots
    X_global_sample = X.sample(n=min(500, len(X)), random_state=42)

    all_reactions = df["reactions"].str.split("|").explode()
    clinical_reactions = all_reactions[~all_reactions.isin(NON_CLINICAL_TERMS)]
    target_adrs = clinical_reactions.value_counts().head(top_n_adrs).index.tolist()
    print(f"\nSelected Target ADRs: {target_adrs}\n")

    os.makedirs("models", exist_ok=True)
    metadata = {
        "feature_columns": X.columns.tolist(),
        "drug_vocabulary": vectorizer.get_feature_names_out().tolist(),
        "target_adrs": target_adrs,
        "metrics": {}
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

        # Base XGBoost for tree structure & SHAP TreeExplainer
        base_xgb = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )
        base_xgb.fit(X_train, y_train)

        # Calibrated Classifier (Platt Sigmoid Scaling) for true clinical probabilities
        calibrator = CalibratedClassifierCV(
            estimator=xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1
            ),
            method="sigmoid",
            cv=3
        )
        calibrator.fit(X_train, y_train)

        # Evaluate raw recall vs calibrated probability Brier score
        y_raw_prob = base_xgb.predict_proba(X_test)[:, 1]
        y_cal_prob = calibrator.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_cal_prob)
        pr_auc = average_precision_score(y_test, y_cal_prob)
        recall = recall_score(y_test, (y_raw_prob >= 0.5).astype(int))
        brier_raw = brier_score_loss(y_test, y_raw_prob)
        brier_cal = brier_score_loss(y_test, y_cal_prob)

        print(
            f"[{adr:<19}] BaseRate: {baseline_rate*100:4.1f}% | AUC: {roc_auc:.3f} | "
            f"Recall: {recall:.3f} | Brier: {brier_raw:.3f} -> {brier_cal:.3f}"
        )

        # Build SHAP TreeExplainer & precompute Global SHAP values on 500-patient cohort
        explainer = shap.TreeExplainer(base_xgb)
        global_shap_vals = explainer(X_global_sample)

        safe_name = adr.lower().replace(" ", "_")
        joblib.dump(calibrator, f"models/cal_{safe_name}.joblib")
        joblib.dump(explainer, f"models/shap_{safe_name}.joblib")
        joblib.dump(global_shap_vals, f"models/global_shap_{safe_name}.joblib")

        metadata["metrics"][adr] = {
            "positives": pos_count,
            "baseline_rate": round(float(baseline_rate * 100), 2),
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "recall": round(float(recall), 4),
            "brier_calibrated": round(float(brier_cal), 4)
        }

    joblib.dump(X_global_sample, "models/X_global_sample.joblib")
    joblib.dump(metadata, "models/pipeline_metadata.joblib")
    print("\nAll Calibrated Specialist Models & Global SHAP artifacts saved in models/!")

if __name__ == "__main__":
    build_features_and_train()
