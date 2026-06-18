
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered"
)

# ─────────────────────────────────────────────
# Load Artifacts (cached so they load only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model           = joblib.load("artifacts/model.pkl")
    scaler          = joblib.load("artifacts/scaler.pkl")
    selected_cols   = joblib.load("artifacts/selected_features.pkl")
    return model, scaler, selected_cols

try:
    model, scaler, selected_cols = load_artifacts()
except FileNotFoundError:
    st.error(" Model artifacts not found. Please run `model.py` first to train and save the model.")
    st.stop()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.markdown("Fill in the customer details below and click **Predict** to see if they are likely to churn.")
st.divider()

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age             = st.number_input("Age",                    min_value=18,   max_value=100,  value=35)
    tenure          = st.number_input("Tenure (months)",        min_value=0,    max_value=120,  value=24)
    monthly_charges = st.number_input("Monthly Charges ($)",    min_value=0.0,  max_value=500.0, value=65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",      min_value=0.0,  max_value=10000.0, value=1500.0, step=10.0)

with col2:
    gender          = st.selectbox("Gender",            ["Male", "Female"])
    internet        = st.selectbox("Internet Service",  ["DSL", "Fiber Optic", "NA"])
    tech_support    = st.selectbox("Tech Support",      ["Yes", "No"])
    contract        = st.selectbox("Contract Type",     ["Month-to-Month", "One-Year", "Two-Year"])

st.divider()

# ─────────────────────────────────────────────
# Feature Engineering (must match model.py logic)
# ─────────────────────────────────────────────
def build_input(age, tenure, monthly_charges, total_charges,
                gender, internet, tech_support, contract):
    """
    Recreates the same feature engineering applied during training:
      - One-hot encoding for Gender, InternetService, TechSupport (drop_first=True)
      - Ordinal encoding for ContractType
    Returns a DataFrame with columns matching selected_cols.
    """

    # One-hot encoded columns (drop_first=True means Male→0, Female→1 for Gender etc.)
    gender_female           = 1 if gender == "Female" else 0
    internet_fiber          = 1 if internet == "Fiber Optic" else 0
    internet_na             = 1 if internet == "NA" else 0
    tech_support_yes        = 1 if tech_support == "Yes" else 0

    # Ordinal: Month-to-Month=0, One-Year=1, Two-Year=2
    contract_map = {"Month-to-Month": 0, "One-Year": 1, "Two-Year": 2}
    contract_enc = contract_map[contract]

    # Build full feature dict (same columns as training data before feature selection)
    data = {
        "Age":                      age,
        "Tenure":                   tenure,
        "MonthlyCharges":           monthly_charges,
        "TotalCharges":             total_charges,
        "ContractType":             contract_enc,
        "Gender_Male":              1 - gender_female,   # drop_first drops Female → Male is the kept column
        "InternetService_Fiber optic": internet_fiber,
        "InternetService_NA":       internet_na,
        "TechSupport_Yes":          tech_support_yes,
    }

    df_input = pd.DataFrame([data])

    # Keep only the features the model was trained on
    # (selected_cols determines which columns survived importance threshold)
    available = [c for c in selected_cols if c in df_input.columns]
    missing   = [c for c in selected_cols if c not in df_input.columns]

    if missing:
        # Fill any missing derived columns with 0
        for col in missing:
            df_input[col] = 0

    return df_input[selected_cols]

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    input_df      = build_input(age, tenure, monthly_charges, total_charges,
                                gender, internet, tech_support, contract)
    input_scaled  = scaler.transform(input_df)
    prediction    = model.predict(input_scaled)[0]
    probability   = model.predict_proba(input_scaled)[0]

    churn_prob    = probability[1] * 100
    no_churn_prob = probability[0] * 100

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn** ({churn_prob:.1f}% probability)")
    else:
        st.success(f"✅ This customer is **not likely to churn** ({no_churn_prob:.1f}% probability)")

    # Probability breakdown
    st.markdown("#### Probability Breakdown")
    col_a, col_b = st.columns(2)
    col_a.metric("Churn Probability",    f"{churn_prob:.1f}%")
    col_b.metric("Retention Probability", f"{no_churn_prob:.1f}%")

    st.progress(int(churn_prob))

    # Show the input features that were used
    with st.expander("🔎 See features used for prediction"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Model: Logistic Regression • Balancing: SMOTE • Scaling: StandardScaler")
