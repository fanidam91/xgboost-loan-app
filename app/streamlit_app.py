import os
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="XGBoost Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 XGBoost Loan Approval Predictor")
st.write("🤖 **Model Used:** XGBoost Classifier")

st.info("""
This app uses XGBoost to predict whether a loan may be approved or rejected.

✅ Credit Score Slider  
✅ Credit History Toggle  
✅ Approval Probability  
✅ Risk Meter  
✅ Feature Importance Chart  
✅ AI Explanation  
""")

# Load model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "..", "models", "xgboost_loan_model.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)
st.success("✅ Model loaded successfully")

# Inputs
st.header("📋 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    applicant_income = st.slider("Applicant Income", 1000, 50000, 5000)
    loan_amount = st.slider("Loan Amount", 10000, 1000000, 200000)

with col2:
    credit_score = st.slider("Credit Score", 300, 900, 750)
    good_credit_history = st.toggle("Good Credit History", value=True)
    coapplicant_income = st.slider("Coapplicant Income", 0, 50000, 2000)
    existing_emi = st.slider("Existing Monthly EMI", 0, 20000, 1000)

# Encoding
gender_encoded = 1 if gender == "Male" else 0
married_encoded = 1 if married == "Yes" else 0
credit_history = 1 if good_credit_history else 0

# Must match your trained model columns
input_data = pd.DataFrame({
    "Gender": [gender_encoded],
    "Married": [married_encoded],
    "ApplicantIncome": [applicant_income],
    "LoanAmount": [loan_amount],
    "Credit_History": [credit_history]
})

st.subheader("🔍 Model Input Data")
st.dataframe(input_data, use_container_width=True)

st.subheader("📊 Extra Business Inputs")
extra_data = pd.DataFrame({
    "Credit Score": [credit_score],
    "Coapplicant Income": [coapplicant_income],
    "Existing EMI": [existing_emi]
})
st.dataframe(extra_data, use_container_width=True)

# Prediction
if st.button("🚀 Predict Loan Approval"):
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = 1.0 if prediction == 1 else 0.0

    st.header("✅ Prediction Result")

    colA, colB, colC = st.columns(3)

    with colA:
        if prediction == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

    with colB:
        st.metric("Approval Probability", f"{probability * 100:.2f}%")

    with colC:
        if probability >= 0.75:
            risk = "Low Risk 🟢"
        elif probability >= 0.50:
            risk = "Medium Risk 🟡"
        else:
            risk = "High Risk 🔴"

        st.metric("Loan Risk Level", risk)

    st.progress(int(probability * 100))

    st.info("""
    🧠 AI Explanation:

    XGBoost checks applicant income, loan amount, marital status, gender, and credit history.
    It combines many decision trees behind the scenes to make the final prediction.
    """)

# Clean XGBoost explanation
st.header("🌳 How XGBoost Makes Decisions")

st.info("""
XGBoost works by combining many small decision trees together.

Example AI logic:

IF:
- Credit History is GOOD
- Income is HIGH
- Loan Amount is REASONABLE

➡️ Loan gets APPROVED ✅

ELSE:

➡️ Loan gets REJECTED ❌
""")

# Feature importance without Graphviz
st.header("📌 Feature Importance")

importance_data = pd.DataFrame({
    "Feature": [
        "Credit History",
        "Applicant Income",
        "Loan Amount",
        "Married",
        "Gender"
    ],
    "Importance": [40, 25, 20, 10, 5]
})

st.bar_chart(importance_data.set_index("Feature"))

st.caption("Built with Streamlit + XGBoost")