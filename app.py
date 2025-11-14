import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.title("🩺 DiseasePredict AI - Diabetes Risk Classifier")
st.write("Enter medical details to predict diabetes")

# Input fields for 10 diabetes features
feature_names = [
    "Age", "Sex", "BMI", "Blood Pressure", "S1", "S2", "S3", "S4", "S5", "S6"
]
values = []
for feature in feature_names:
    val = st.number_input(f"{feature}:", value=0.0)
    values.append(val)

if st.button("Predict"):
    scaled = scaler.transform([values])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred] * 100

    if pred == 1:
        st.error(f"⚠️ High Risk of Diabetes ({prob:.2f}% confidence)")
    else:
        st.success(f"🟢 Low Risk of Diabetes ({prob:.2f}% confidence)")
