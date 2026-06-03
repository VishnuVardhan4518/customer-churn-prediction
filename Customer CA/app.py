from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).parent / "churn_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Churn Prediction App")

st.divider()
st.write("Please enter the values and hit the predict button for getting prediction.")
st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter tenure", min_value=0, max_value=100, value=10)
monthly_charge = st.number_input(
    "Enter monthly charge",
    min_value=0.0,
    max_value=150.0,
    value=80.0
)
total_charges = st.number_input(
    "Enter total charges",
    min_value=0.0,
    max_value=15000.0,
    value=monthly_charge * tenure
)
gender = st.selectbox("Enter gender", ["Male", "Female"])
contract_type = st.selectbox(
    "Enter contract type",
    ["Month-to-Month", "One-Year", "Two-Year"]
)
internet_service = st.selectbox(
    "Enter internet service",
    ["Fiber Optic", "DSL", ""]
)
tech_support = st.selectbox("Tech support", ["Yes", "No"])

st.divider()

predict_button = st.button("Predict!")

if predict_button:
    customer = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charge,
        "ContractType": contract_type,
        "InternetService": internet_service,
        "TotalCharges": total_charges,
        "TechSupport": tech_support
    }])

    prediction = model.predict(customer)[0]
    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"Predicted churn: {predicted}")
else:
    st.write("Please enter the values and use predict button.")
