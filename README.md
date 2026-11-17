# Explainable Drug Safety & Adverse Event Risk Predictor

An end-to-end, clinically calibrated Machine Learning and Explainable AI (XAI) pharmacovigilance pipeline trained on real-world **FDA Adverse Event Reporting System (FAERS)** Individual Case Safety Reports (ICSRs) via the openFDA API.

## System Architecture

1. **Automated Data Ingestion & Deduplication (`src/data_pipeline.py`):**
   - Cursor-paginated extraction (`search_after`) from openFDA Drug Event API.
   - ICH E2B age unit normalization (converting decades, months, weeks, days, hours into continuous years).
   - Two-stage deduplication (`safetyreportid` + clinical signature matching) and polypharmacy feature derivation.
2. **Calibrated Multi-Specialist Modeling (`src/train_specialists.py`):**
   - Modular **One-Model-per-ADR** architecture using `XGBClassifier` with dynamic class-imbalance weighting (`scale_pos_weight`).
   - **Platt Sigmoid Calibration (`CalibratedClassifierCV`)** to align predicted risk probabilities with true cohort incidence rates (verified via Brier Score reduction).
3. **Dual-Level Explainability & Interactive Dashboard (`app.py`):**
   - **Local Interpretability:** Patient-specific SHAP Waterfall plots decomposing individual risk into demographic, polypharmacy, and drug-level log-odds contributions.
   - **Global Interpretability:** Cohort-level SHAP Beeswarm and Mean `|SHAP|` feature importance plots for population pharmacovigilance signal verification.

## Quickstart

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    export OPENFDA_API_KEY="your_openfda_api_key"
    python src/data_pipeline.py
    python src/train_specialists.py
    python app.py

## Disclaimer
This software is a research and pharmacovigilance decision-support prototype trained on spontaneous postmarketing reports (FAERS). It does not establish causal clinical risk and is not a substitute for professional medical advice.

