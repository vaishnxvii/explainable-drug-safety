import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

metadata = joblib.load("models/pipeline_metadata.joblib")
FEATURE_COLS = metadata["feature_columns"]
DRUG_VOCAB = metadata["drug_vocabulary"]
TARGET_ADRS = metadata["target_adrs"]
AVAILABLE_MODELS = metadata["available_models"]
METRICS = metadata["metrics"]
X_GLOBAL_SAMPLE = joblib.load("models/X_global_sample.joblib")
DISPLAY_NAMES = [c.replace("DRUG_", "Drug: ").replace("_", " ").title() for c in FEATURE_COLS]

MODEL_PREFIX = {
    "XGBoost (Proposed)": "xgb",
    "Random Forest": "rf",
    "Logistic Regression": "lr"
}

CAL_MODELS = {m: {} for m in AVAILABLE_MODELS}
EXPLAINERS = {m: {} for m in AVAILABLE_MODELS}
SCALERS = {}
GLOBAL_SHAP = {}

for adr in TARGET_ADRS:
    safe_adr = adr.lower().replace(" ", "_")
    GLOBAL_SHAP[adr] = joblib.load(f"models/global_shap_{safe_adr}.joblib")
    SCALERS[adr] = joblib.load(f"models/scaler_{safe_adr}.joblib")
    for m_label, prefix in MODEL_PREFIX.items():
        CAL_MODELS[m_label][adr] = joblib.load(f"models/cal_{prefix}_{safe_adr}.joblib")
        EXPLAINERS[m_label][adr] = joblib.load(f"models/shap_{prefix}_{safe_adr}.joblib")

BENCHMARK_DF = (
    pd.read_csv("models/model_benchmark_results.csv")
    if os.path.exists("models/model_benchmark_results.csv")
    else pd.DataFrame()
)

def build_patient_vector(age, sex_label, selected_drugs):
    is_female = 1 if sex_label == "Female" else 0
    drugs_clean = sorted(set([d.strip().upper() for d in selected_drugs if d]))
    poly_count = max(len(drugs_clean), 1)
    is_elderly = 1 if age >= 65 else 0
    age_x_poly = age * poly_count

    row = {col: 0.0 for col in FEATURE_COLS}
    row["age"] = float(age)
    row["is_female"] = float(is_female)
    row["polypharmacy_count"] = float(poly_count)
    row["is_elderly"] = float(is_elderly)
    row["age_x_polypharmacy"] = float(age_x_poly)

    for d in drugs_clean:
        col_name = f"DRUG_{d}"
        if col_name in row:
            row[col_name] = 1.0

    return pd.DataFrame([row])[FEATURE_COLS]

def relative_risk_badge(prob_pct, baseline_pct):
    ratio = prob_pct / max(baseline_pct, 0.1)
    if ratio >= 2.0 or prob_pct >= 15.0:
        return f"HIGH ({ratio:.1f}x baseline)"
    elif ratio >= 1.2 or prob_pct >= 6.0:
        return f"MODERATE ({ratio:.1f}x baseline)"
    return f"LOW ({ratio:.1f}x baseline)"

def get_single_patient_shap(model_choice, focus_adr, X_patient):
    explainer = EXPLAINERS[model_choice][focus_adr]
    if model_choice == "Logistic Regression":
        X_scaled = SCALERS[focus_adr].transform(X_patient)
        sv = explainer(X_scaled)
        vals = sv.values[0]
        base_val = float(sv.base_values[0]) if np.ndim(sv.base_values) > 0 else float(sv.base_values)
    else:
        sv = explainer(X_patient)
        # Random Forest TreeExplainer returns shape (1, n_features, 2) for binary classification
        if sv.values.ndim == 3:
            vals = sv.values[0, :, 1]
            base_val = float(sv.base_values[0, 1])
        else:
            vals = sv.values[0]
            base_val = float(sv.base_values[0])
    return vals, base_val

def predict_and_explain(model_choice, age, sex_label, selected_drugs, focus_adr):
    if not selected_drugs:
        return pd.DataFrame({"Message": ["Please select at least one drug."]}), None, "Select a drug to generate explanation."

    X_patient = build_patient_vector(age, sex_label, selected_drugs)

    results = []
    for adr in TARGET_ADRS:
        cal_prob_pct = float(CAL_MODELS[model_choice][adr].predict_proba(X_patient)[0, 1]) * 100
        m = METRICS[model_choice][adr]
        base_pct = m["baseline_rate"]
        results.append({
            "Adverse Event (MedDRA)": adr,
            "Calibrated Risk (%)": round(cal_prob_pct, 2),
            "Cohort Baseline (%)": base_pct,
            "Relative Risk Level": relative_risk_badge(cal_prob_pct, base_pct),
            "ROC-AUC": m["roc_auc"],
            "Brier Score": m["brier_calibrated"]
        })

    results_df = pd.DataFrame(results).sort_values(by="Calibrated Risk (%)", ascending=False)

    # Local SHAP Waterfall Plot for chosen model & focus ADR
    vals, base_val = get_single_patient_shap(model_choice, focus_adr, X_patient)
    shap_exp = shap.Explanation(
        values=vals,
        base_values=base_val,
        data=X_patient.iloc[0].values,
        feature_names=DISPLAY_NAMES
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    plt.title(f"[{model_choice}] Local SHAP Waterfall: {focus_adr}", fontsize=11, pad=12)
    plt.tight_layout()

    top_idx = np.argsort(np.abs(vals))[::-1][:4]
    drivers = []
    for idx in top_idx:
        impact = vals[idx]
        direction = "increases" if impact > 0 else "decreases"
        drivers.append(f"- **{DISPLAY_NAMES[idx]}** (value={X_patient.iloc[0, idx]:.1f}) {direction} risk contribution by `{impact:+.3f}`")

    focus_prob = float(CAL_MODELS[model_choice][focus_adr].predict_proba(X_patient)[0, 1]) * 100
    focus_base = METRICS[model_choice][focus_adr]["baseline_rate"]
    summary_md = (
        f"### [{model_choice}] Clinical Risk Breakdown: **{focus_adr}**\n"
        f"- **Calibrated Patient Risk:** `{focus_prob:.2f}%` vs. **Cohort Baseline:** `{focus_base:.2f}%` "
        f"(**{relative_risk_badge(focus_prob, focus_base)}**)\n"
        f"- **Active Medications Count (Polypharmacy):** `{len(selected_drugs)}`\n\n"
        f"**Top Patient-Specific Risk Drivers (SHAP):**\n" + "\n".join(drivers)
    )

    return results_df, fig, summary_md

def generate_global_shap_plots(adr_name):
    g_shap = GLOBAL_SHAP[adr_name]
    g_exp = shap.Explanation(
        values=g_shap.values,
        base_values=g_shap.base_values,
        data=X_GLOBAL_SAMPLE.values,
        feature_names=DISPLAY_NAMES
    )

    fig_bee, _ = plt.subplots(figsize=(9, 5.5))
    shap.plots.beeswarm(g_exp, max_display=12, show=False)
    plt.title(f"Global SHAP Beeswarm Summary (n=500 FAERS Cohort) — {adr_name}", fontsize=11, pad=12)
    plt.tight_layout()

    fig_bar, _ = plt.subplots(figsize=(9, 5))
    shap.plots.bar(g_exp, max_display=12, show=False)
    plt.title(f"Global Mean |SHAP| Feature Importance — {adr_name}", fontsize=11, pad=12)
    plt.tight_layout()

    m = METRICS["XGBoost (Proposed)"][adr_name]
    stats_md = (
        f"### Cohort-Level Model & Signal Summary: **{adr_name}**\n"
        f"- **Total Positive FAERS Reports:** `{m['positives']}` (`{m['baseline_rate']}%` incidence in cohort)\n"
        f"- **XGBoost ROC-AUC:** `{m['roc_auc']}` | **Recall:** `{m['recall']}` | **Calibrated Brier Score:** `{m['brier_calibrated']}`"
    )
    return fig_bee, fig_bar, stats_md

def filter_benchmark_table(adr_filter):
    if BENCHMARK_DF.empty:
        return pd.DataFrame({"Info": ["Run src/benchmark_models.py to populate benchmark metrics."]})
    if adr_filter == "All ADRs (Macro Average)":
        return (
            BENCHMARK_DF.groupby("Model")[["ROC-AUC", "PR-AUC", "Recall", "F1-Score", "Train_Sec"]]
            .mean()
            .round(4)
            .reset_index()
            .sort_values(by="ROC-AUC", ascending=False)
        )
    return BENCHMARK_DF[BENCHMARK_DF["ADR"] == adr_filter].sort_values(by="ROC-AUC", ascending=False)

with gr.Blocks(theme=gr.themes.Soft(), title="Explainable Drug Safety Predictor") as demo:
    gr.Markdown(
        "# Explainable Drug Safety & Adverse Event Risk Predictor\n"
        "Multi-Model Calibrated Pharmacovigilance Engine (`XGBoost`, `Random Forest`, `Logistic Regression`) + SHAP Explainability trained on FDA FAERS."
    )

    with gr.Tabs():
        with gr.Tab("1. Patient-Level Risk & Model Comparison"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_selector = gr.Dropdown(
                        choices=AVAILABLE_MODELS,
                        value="XGBoost (Proposed)",
                        label="Select Classifier Model"
                    )
                    age_input = gr.Slider(minimum=1, maximum=100, value=68, step=1, label="Patient Age (Years)")
                    sex_input = gr.Radio(["Male", "Female"], value="Female", label="Biological Sex")
                    drugs_input = gr.Dropdown(
                        choices=sorted(DRUG_VOCAB),
                        multiselect=True,
                        value=[DRUG_VOCAB[0]] if DRUG_VOCAB else [],
                        label="Suspect & Concomitant Medications (Select 1 or more)"
                    )
                    focus_adr_input = gr.Dropdown(
                        choices=TARGET_ADRS,
                        value=TARGET_ADRS[0],
                        label="Target Adverse Event for Local SHAP Explanation"
                    )
                    predict_btn = gr.Button("Analyze Calibrated Adverse Event Risk", variant="primary")

                with gr.Column(scale=2):
                    risk_table = gr.Dataframe(label="Calibrated Multi-ADR Risk Screening Profile", interactive=False)
                    explanation_text = gr.Markdown()
                    shap_plot = gr.Plot(label="Local Patient SHAP Waterfall Explanation")

            predict_btn.click(
                fn=predict_and_explain,
                inputs=[model_selector, age_input, sex_input, drugs_input, focus_adr_input],
                outputs=[risk_table, shap_plot, explanation_text]
            )

        with gr.Tab("2. Global SHAP Population Insights"):
            gr.Markdown(
                "Explore cohort-wide pharmacovigilance drivers across 500 sampled FAERS patients using the proposed XGBoost specialist models."
            )
            with gr.Row():
                global_adr_dropdown = gr.Dropdown(
                    choices=TARGET_ADRS,
                    value=TARGET_ADRS[0],
                    label="Select Adverse Event Specialist Model"
                )
                global_btn = gr.Button("Generate Global SHAP Summary", variant="primary")

            global_stats = gr.Markdown()
            with gr.Row():
                beeswarm_plot = gr.Plot(label="Global SHAP Beeswarm Plot")
                bar_plot = gr.Plot(label="Global Mean |SHAP| Feature Importance")

            global_btn.click(
                fn=generate_global_shap_plots,
                inputs=[global_adr_dropdown],
                outputs=[beeswarm_plot, bar_plot, global_stats]
            )

        with gr.Tab("3. Empirical Model Benchmarks"):
            gr.Markdown("Compare **XGBoost (Proposed)** against **Random Forest** and **Logistic Regression** baselines across identical 80/20 FAERS test splits.")
            bench_filter = gr.Dropdown(
                choices=["All ADRs (Macro Average)"] + TARGET_ADRS,
                value="All ADRs (Macro Average)",
                label="Filter Benchmark View"
            )
            bench_table = gr.Dataframe(value=filter_benchmark_table("All ADRs (Macro Average)"), interactive=False)
            bench_filter.change(fn=filter_benchmark_table, inputs=[bench_filter], outputs=[bench_table])

if __name__ == "__main__":
    demo.launch()
