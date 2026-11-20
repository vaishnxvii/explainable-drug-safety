import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

NON_CLINICAL_TERMS = {
    "OFF LABEL USE", "DRUG INEFFECTIVE", "PRODUCT DOSE OMISSION ISSUE",
    "INTENTIONAL PRODUCT USE ISSUE", "INAPPROPRIATE SCHEDULE OF PRODUCT ADMINISTRATION",
    "PRODUCT USE IN UNAPPROVED INDICATION", "NO ADVERSE EVENT"
}

def run_benchmark(data_path="data/processed/faers_clean_25k.csv", top_k_drugs=300, top_n_adrs=8):
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records for multi-model benchmarking...\n")

    # Feature Engineering (identical to train_specialists.py)
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

    all_reactions = df["reactions"].str.split("|").explode()
    clinical_reactions = all_reactions[~all_reactions.isin(NON_CLINICAL_TERMS)]
    target_adrs = clinical_reactions.value_counts().head(top_n_adrs).index.tolist()

    benchmark_rows = []

    for adr in target_adrs:
        y = df["reactions"].apply(lambda r: 1 if adr in r.split("|") else 0).values
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        scale_weight = neg_count / max(pos_count, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        models = {
            "Logistic Regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost (Proposed)": xgb.XGBClassifier(
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
        }

        for model_name, clf in models.items():
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = round(time.time() - t0, 2)

            y_prob = clf.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            benchmark_rows.append({
                "ADR": adr,
                "Model": model_name,
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
                "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1-Score": round(f1_score(y_test, y_pred), 4),
                "Train_Sec": train_time
            })

    res_df = pd.DataFrame(benchmark_rows)
    os.makedirs("models", exist_ok=True)
    res_df.to_csv("models/model_benchmark_results.csv", index=False)

    # Print Macro-Averaged Summary Across All 8 ADRs
    summary = (
        res_df.groupby("Model")[["ROC-AUC", "PR-AUC", "Recall", "F1-Score", "Train_Sec"]]
        .mean()
        .round(4)
        .sort_values(by="ROC-AUC", ascending=False)
    )
    print("=== MACRO-AVERAGED PERFORMANCE ACROSS ALL 8 ADRs ===")
    print(summary.to_string())
    print("\nDetailed per-ADR results saved to models/model_benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
