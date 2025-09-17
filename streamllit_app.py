# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load model
MODEL = joblib.load("xgb_model.pkl")

# Features (must match training)
features = [
    "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)",
    "PT08.S5(O3)", "T", "RH", "AH"
]

st.set_page_config(page_title="AirQuality Predictor", layout="wide")

st.title("🌍 AirQuality CO(GT) Predictor — XGBoost")
st.write("This Streamlit app loads a trained XGBoost model and shows evaluation plots + prediction UI.")

# Show saved plots
st.subheader("📊 Model Evaluation Plots")
col1, col2 = st.columns(2)
with col1:
    st.image("actual_vs_predicted.png", caption="Actual vs Predicted", use_column_width=True)
with col2:
    st.image("residuals_hist.png", caption="Residuals Histogram", use_column_width=True)

st.image("feature_importances.png", caption="Top Feature Importances", use_column_width=True)

# Manual input for prediction
st.sidebar.header("Manual Prediction Input")
input_data = {}
for f in features:
    input_data[f] = st.sidebar.number_input(f, value=0.0)

if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_data])
    pred = MODEL.predict(input_df)[0]
    st.sidebar.success(f"Predicted CO(GT): {pred:.4f}")
