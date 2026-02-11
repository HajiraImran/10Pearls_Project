import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Path Fix: Ensure utils is found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import fetch_weather_forecast, fetch_historical_aqi_data

load_dotenv()
st.set_page_config(page_title="Islamabad AQI", layout="wide")

@st.cache_resource(ttl=3600)
def load_assets():
    api_key = os.getenv("HOPSWORKS_KEY")
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    # Model Version Logic
    models = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(models, key=lambda m: m.version)
    model = joblib.load(os.path.join(model_meta.download(), "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    # UI logic starts here
    st.title("🌬️ Islamabad Air Quality Index")
    
    forecast_weather = fetch_weather_forecast(days=4)
    
    # METRICS DISPLAY
    if not forecast_weather.empty:
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Temp", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
        m2.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
        m3.metric("Wind", f"{forecast_weather.iloc[0]['wind_speed']:.1f} km/h")
    
    # FORECAST LOOP
    st.subheader("📅 3-Day Forecast")
    cols = st.columns(3)
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    
    for i, row in future_df.iterrows():
        # Feature vector alignment
        # (Yahan aapne check karna hai ke training features ka order same ho)
        feat = np.array([[row['temperature'], row['humidity'], row['wind_speed'], 12.0, row['datetime'].weekday(), row['datetime'].month, 3.0, 15.0, 0.0]])
        pred = model.predict(feat)[0]
        with cols[i]:
            st.info(f"**{row['datetime'].strftime('%A')}**\n\nAQI: {int(round(pred))}")

except Exception as e:
    st.error(f"App Error: {e}")