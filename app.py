import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Demand Forecasting", page_icon="📈")
st.title("Supply Chain Demand Forecasting")
st.write("Random Forest baseline for demand forecasting.")

checkpoint = joblib.load("demand_model.joblib")
model = checkpoint["model"]
features = checkpoint["features"]

lag1 = st.number_input("Previous-day demand", 0.0, 100000.0, 100.0)
lag7 = st.number_input("Demand 7 days ago", 0.0, 100000.0, 95.0)
rolling = st.number_input("7-day rolling average", 0.0, 100000.0, 98.0)
day = st.slider("Day of week (0=Monday, 6=Sunday)", 0, 6, 0)
month = st.slider("Month", 1, 12, 1)

if st.button("Forecast Demand"):
    x = pd.DataFrame([[lag1,lag7,rolling,day,month]], columns=features)
    forecast = model.predict(x)[0]
    st.success(f"Predicted demand: {forecast:.0f} units")
    st.write("Use the forecast as a planning aid and compare it with actual demand.")
