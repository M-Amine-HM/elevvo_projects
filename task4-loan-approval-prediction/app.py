"""Standalone Gradio app for Task 4 — Loan Approval Prediction.

Loads the trained model, scaler, and label encoders saved by the training notebook
(notebook.ipynb) into ./models/, then exposes the exact same Gradio interface as the notebook's
final section. Fully self-contained — no notebook dependency.

Run standalone locally from the task4 folder with:
    python app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))


def predict_loan(no_of_dependents, education, self_employed, income_annum, loan_amount,
                 loan_term, cibil_score, residential_assets_value, commercial_assets_value,
                 luxury_assets_value, bank_asset_value):
    """Build the 13-value row in the exact feature order the model was trained on, then predict."""
    total_asset_value = (residential_assets_value + commercial_assets_value
                         + luxury_assets_value + bank_asset_value)
    loan_to_income_ratio = loan_amount / income_annum
    row = [
        encoders["education"].transform([education])[0],
        encoders["self_employed"].transform([self_employed])[0],
        no_of_dependents, income_annum, loan_amount, loan_term, cibil_score,
        residential_assets_value, commercial_assets_value, luxury_assets_value,
        bank_asset_value, total_asset_value, loan_to_income_ratio,
    ]
    assert len(row) == len(
        feature_names), "row does not match trained feature order"
    # column names match training -> no warnings
    X = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    status = "Approved" if pred == 1 else "Rejected"
    return status, {"Approved": round(float(proba), 4), "Rejected": round(1 - float(proba), 4)}


# One input per feature, in the exact training order. Slider ranges are derived from the current
# dataset (loan_approval_dataset.csv) so the widgets sit on realistic values.
inputs = [
    gr.Slider(0, 5, value=3, step=1, label="No of Dependents"),
    gr.Dropdown(["Graduate", "Not Graduate"],
                value="Graduate", label="Education"),
    gr.Dropdown(["No", "Yes"], value="No", label="Self Employed"),
    gr.Slider(200000, 9900000, value=5100000,
              step=100000, label="Annual Income (INR)"),
    gr.Slider(300000, 39500000, value=14500000,
              step=100000, label="Loan Amount (INR)"),
    gr.Slider(2, 20, value=10, step=1, label="Loan Term (years)"),
    gr.Slider(300, 900, value=600, step=1, label="CIBIL Score"),
    gr.Slider(-100000, 29100000, value=5600000, step=100000,
              label="Residential Assets Value (INR)"),
    gr.Slider(0, 19400000, value=3700000, step=100000,
              label="Commercial Assets Value (INR)"),
    gr.Slider(300000, 39200000, value=14600000, step=100000,
              label="Luxury Assets Value (INR)"),
    gr.Slider(0, 14700000, value=4600000, step=100000,
              label="Bank Assets Value (INR)"),
]

outputs = [
    gr.Label(label="Prediction"),
    gr.Label(label="Approval Probability"),
]

demo = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs=outputs,
    title="Task 4: Loan Approval Prediction — Live Prediction",
    description="Enter the applicant's details; the model predicts loan approval and the approval probability.",
)

if __name__ == "__main__":
    demo.launch(share=True)
