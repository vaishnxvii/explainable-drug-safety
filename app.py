import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Load saved metadata, calibrated models, and SHAP artifacts
metadata = joblib.load("models/pipeline_metadata.joblib")
FEATURE_COLS = metadata["feature_columns"]
DRUG_VOCAB = metadata["drug_vocabulary"]
TARGET_ADRS = metadata["target_adrs"]
METRICS = metadata["metrics"]
X_GLOBAL_SAMPLE = joblib.load("models/X_global_sample.joblib")
DISPLAY_NAMES = [c.replace("DRUG_", "Drug: ").replace("_", " ").title() for c in FEATURE_COLS]

CAL_MODELS = {}
EXPLAINERS = {}
GLOBAL_SHAP = {}

for adr in TARGET_ADRS:
    safe_name = adr.lower().replace(" ", "_")
    CAL_MODELS[adr] = joblib.load(f"models/cal_{safe_name}.joblib")
    EXPLAINERS[adr] = joblib.load(f"models/shap_{safe_name}.joblib")
    GLOBAL_SHAP[adr] = joblib.load(f"models/global_shap_{safe_name}.joblib")

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

def predict_and_explain(age, sex_label, selected_drugs, focus_adr):
    if not selected_drugs:
        return pd.DataFrame({"Message": ["Please select at least one drug."]}), None, "Select a drug to generate explanation."

    X_patient = build_patient_vector(age, sex_label, selected_drugs)

    # 1. Calibrated Risk across all Specialist ADR Models
    results = []
    for adr in TARGET_ADRS:
        cal_prob_pct = float(CAL_MODELS[adr].predict_proba(X_patient)[0, 1]) * 100
        m = METRICS[adr]
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

    # 2. Local SHAP Waterfall Plot for selected Focus ADR
    explainer = EXPLAINERS[focus_adr]
    shap_values = explainer(X_patient)

    shap_exp = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=X_patient.iloc[0].values,
        feature_names=DISPLAY_NAMES
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    plt.title(f"Patient-Level SHAP Waterfall: {focus_adr}", fontsize=12, pad=12)
    plt.tight_layout()

    # 3. Clinical Reasoning Summary
    vals = shap_values.values[0]
    top_idx = np.argsort(np.abs(vals))[::-1][:4]
    drivers = []
    for idx in top_idx:
        impact = vals[idx]
        direction = "increases" if impact > 0 else "decreases"
        drivers.append(f"- **{DISPLAY_NAMES[idx]}** (value={X_patient.iloc[0, idx]:.1f}) {direction} risk log-odds by `{impact:+.3f}`")

    focus_prob = float(CAL_MODELS[focus_adr].predict_proba(X_patient)[0, 1]) * 100
    focus_base = METRICS[focus_adr]["baseline_rate"]
    summary_md = (
        f"### Calibrated Clinical Risk Breakdown: **{focus_adr}**\n"
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

    # Plot 1: Beeswarm Summary Plot
    fig_bee, _ = plt.subplots(figsize=(9, 5.5))
    shap.plots.beeswarm(g_exp, max_display=12, show=False)
    plt.title(f"Global SHAP Beeswarm Summary (n=500 FAERS Cohort) — {adr_name}", fontsize=11, pad=12)
    plt.tight_layout()

    # Plot 2: Mean |SHAP| Bar Importance Plot
    fig_bar, _ = plt.subplots(figsize=(9, 5))
    shap.plots.bar(g_exp, max_display=12, show=False)
    plt.title(f"Global Mean |SHAP| Feature Importance — {adr_name}", fontsize=11, pad=12)
    plt.tight_layout()

    m = METRICS[adr_name]
    stats_md = (
        f"### Cohort-Level Model & Signal Summary: **{adr_name}**\n"
        f"- **Total Positive FAERS Reports:** `{m['positives']}` (`{m['baseline_rate']}%` incidence in cohort)\n"
        f"- **Discrimination (ROC-AUC):** `{m['roc_auc']}` | **Sensitivity (Recall):** `{m['recall']}` | **Calibrated Brier Score:** `{m['brier_calibrated']}`"
    )
    return fig_bee, fig_bar, stats_md

with gr.Blocks(theme=gr.themes.Soft(), title="Explainable Drug Safety Predictor") as demo:
    gr.Markdown(
        "# Explainable Drug Safety & Adverse Event Risk Predictor\n"
        "Calibrated Multi-Specialist XGBoost + SHAP Pharmacovigilance Screening Engine trained on FDA FAERS postmarketing reports."
    )

    with gr.Tabs():
        with gr.Tab("1. Patient-Level Risk & Local SHAP"):
            with gr.Row():
                with gr.Column(scale=1):
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
                inputs=[age_input, sex_input, drugs_input, focus_adr_input],
                outputs=[risk_table, shap_plot, explanation_text]
            )

        with gr.Tab("2. Global SHAP Population Insights"):
            gr.Markdown(
                "Explore cohort-wide pharmacovigilance drivers across 500 sampled FAERS patients. "
                "**Beeswarm plots** show how high (red) vs. low (blue) feature values shift adverse event risk."
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

if __name__ == "__main__":
    demo.launch()
