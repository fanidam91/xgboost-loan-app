import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="XGBoost Loan Prediction App",
    page_icon="💰",
    layout="wide"
)

# =========================
# Project Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_loan_model.pkl")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# App Title
# =========================
st.title("💰 XGBoost Loan Approval Prediction App")
st.write("Enter customer details below and the AI model will predict whether the loan may be approved.")

# =========================
# Input Fields
# =========================
st.subheader("📝 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", min_value=0, value=50000, step=1000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700, step=10)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=200000, step=5000)

with col2:
    employment_years = st.number_input("Employment Years", min_value=0, max_value=50, value=3, step=1)
    existing_debt = st.number_input("Existing Debt", min_value=0, value=10000, step=1000)

# =========================
# Prediction Input
# IMPORTANT: Column names must match training data exactly
# =========================
input_data = pd.DataFrame([{
    "income": income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "employment_years": employment_years,
    "existing_debt": existing_debt
}])

st.subheader("📊 Input Data Preview")
st.dataframe(input_data, width="stretch")

# =========================
# Predict Button
# =========================
if st.button("🔮 Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    approved_probability = probability[1] * 100
    rejected_probability = probability[0] * 100

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"✅ Approval Probability: **{approved_probability:.2f}%**")
    st.write(f"❌ Rejection Probability: **{rejected_probability:.2f}%**")

# =========================
# Feature Importance
# =========================
st.subheader("📌 Feature Importance")

try:
    booster = model.get_booster()
    importance = booster.get_score(importance_type="weight")

    if importance:
        importance_df = pd.DataFrame({
            "Feature": list(importance.keys()),
            "Importance": list(importance.values())
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df, width="stretch")
        st.bar_chart(importance_df.set_index("Feature"))
    else:
        st.info("Feature importance is not available for this model.")

except Exception as e:
    st.warning(f"Feature importance not available: {e}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built with Streamlit, XGBoost and Python 🚀")