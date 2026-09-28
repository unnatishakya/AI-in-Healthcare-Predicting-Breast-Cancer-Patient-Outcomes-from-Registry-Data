"""Streamlit front end for the breast cancer outcome-risk model (educational demo).

Run the notebook first so that models/risk_model.joblib and models/model_meta.json exist, then:
    streamlit run app.py

The input form is generated from model_meta.json, so it always matches the trained model
(allowed categories, numeric ranges, decision threshold).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/risk_model.joblib")
META_PATH = Path("models/model_meta.json")

st.set_page_config(page_title="Breast Cancer Outcome Risk (Educational Demo)", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline and its metadata once per session."""
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


st.title("Breast Cancer Outcome Risk Explorer")
st.warning(
    "Educational demo only. It is trained on 2006-2010 registry data, has not been clinically "
    "validated, and must not be used for medical decisions or patient counseling."
)

if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Run the notebook first to create the models/ folder.")
    st.stop()

model, meta = load_artifacts()

# --- Sidebar: transparency about the model ---------------------------------------
with st.sidebar:
    st.header("About the model")
    st.write(f"**Type:** {meta['model_name']}")
    st.write(f"**Trained:** {meta['created']}")
    tm = meta["test_metrics"]
    st.write(f"**Test ROC-AUC:** {tm['roc_auc']:.2f}")
    st.write(f"**Test PR-AUC:** {tm['pr_auc']:.2f} (chance = {tm['prevalence_test']:.2f})")
    st.write(f"**Flag threshold:** {meta['threshold']:.2f}")
    st.write(f"**Recall at threshold (test):** {tm['recall_at_threshold']:.0%}")
    st.write(f"**Precision at threshold (test):** {tm['precision_at_threshold']:.0%}")
    st.caption("Low precision means many flagged patients did not have the outcome. "
               "The threshold favors catching high-risk patients over avoiding false alarms.")

# --- Input form ---------------------------------------------------------------------
st.subheader("Patient characteristics")
ranges = meta["numeric_ranges"]
patient = {}

col1, col2 = st.columns(2)
with col1:
    patient["age"] = st.number_input(
        "Age", int(ranges["age"]["min"]), int(ranges["age"]["max"]), int(ranges["age"]["median"]))
    patient["tumor_size"] = st.number_input(
        "Tumor size (mm)", int(ranges["tumor_size"]["min"]), int(ranges["tumor_size"]["max"]),
        int(ranges["tumor_size"]["median"]))
    patient["regional_node_examined"] = st.number_input(
        "Regional lymph nodes examined", int(ranges["regional_node_examined"]["min"]),
        int(ranges["regional_node_examined"]["max"]), int(ranges["regional_node_examined"]["median"]))
    patient["regional_node_positive"] = st.number_input(
        "Regional lymph nodes positive", int(ranges["regional_node_positive"]["min"]),
        int(ranges["regional_node_positive"]["max"]), int(ranges["regional_node_positive"]["median"]))
with col2:
    # Ordinal fields are listed lowest to highest risk, exactly as the model expects.
    for field, order in meta["ordinal_categories"].items():
        patient[field] = st.selectbox(field.replace("_", " ").title(), order)
    for field, options in meta["categorical_options"].items():
        patient[field] = st.selectbox(field.replace("_", " ").title(), options)

# --- Prediction ---------------------------------------------------------------------
if st.button("Estimate risk", type="primary"):
    if patient["regional_node_positive"] > patient["regional_node_examined"]:
        st.error("Positive lymph nodes cannot exceed the number of nodes examined.")
    else:
        row = pd.DataFrame([patient])[meta["features"]]  # column order must match training
        probability = float(model.predict_proba(row)[0, 1])
        flagged = probability >= meta["threshold"]

        st.metric("Estimated probability of death (registry follow-up)", f"{probability:.0%}")
        if flagged:
            st.error("Higher-risk profile: above the screening threshold. In a clinical setting this "
                     "would prompt closer review by a clinician.")
        else:
            st.success("Lower-risk profile relative to the screening threshold.")

        with st.expander("How to read this result"):
            st.write(
                "The number is the share of similar patients in the 2006-2010 registry cohort who were "
                "recorded as deceased by the end of follow-up (any cause). It is not a personal "
                "prognosis. The flag uses a threshold chosen to catch most deaths in training data, so "
                "many flagged patients will do well. Performance was not equal across all subgroups; "
                "see Section 10 of the notebook."
            )
