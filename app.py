import streamlit as st
import hopsworks
import joblib
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Islamabad 3-Day AQI Forecast", layout="wide")

@st.cache_resource
def load_resources():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # Best Model: RandomForest Version 2
    model_obj = mr.get_model("islamabad_aqi_randomforest", version=2)
    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    
    # Feature View Version 2
    fv = fs.get_feature_view(name="islamabad_aqi_v12_view", version=2)
    
    return model, fv

try:
    st.title("🇵🇰 Islamabad 3-Day Air Quality Forecast")
    model, fv = load_resources()

    # 1. Get current data for the baseline
    batch_data = fv.get_batch_data()
    latest_record = batch_data.sort_values(by='datetime').iloc[-1]

    st.subheader("Upcoming 3-Day Predictions")
    forecast_cols = st.columns(3)

    # 2. Logic for Next 3 Days
    # Note: In a production app, you'd fetch real weather forecasts here.
    # For now, we simulate the next 3 days based on current trends.
    for i in range(1, 4):
        future_date = pd.to_datetime(latest_record['datetime']) + timedelta(days=i)
        
        # Create a feature row for the future date
        # We keep weather features similar but update the time features
        future_features = latest_record.copy()
        future_features['hour'] = 12  # Predicting for noon each day
        future_features['weekday'] = future_date.weekday()
        future_features['month'] = future_date.month
        
        # Prepare for model (Drop target and non-numeric)
        input_data = pd.DataFrame([future_features]).drop(columns=['city', 'datetime', 'aqi'], errors='ignore')
        
        # Predict
        pred_aqi = model.predict(input_data)[0]
        
        with forecast_cols[i-1]:
            st.metric(label=f"📅 {future_date.strftime('%A')}", value=f"{pred_aqi:.2f} AQI")
            st.write(f"Date: {future_date.strftime('%d %b')}")
            
            if pred_aqi < 50:
                st.success("Good")
            elif pred_aqi < 100:
                st.warning("Moderate")
            else:
                st.error("Unhealthy")

    st.info("Note: Predictions are generated using the Random Forest v2 model based on seasonal trends.")

except Exception as e:
    st.error(f"Error: {e}")