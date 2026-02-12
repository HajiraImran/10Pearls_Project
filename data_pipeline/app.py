import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import altair as alt
from utils import fetch_weather_forecast, fetch_historical_aqi_data
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Islamabad AQI", layout="wide")

@st.cache_resource(ttl=3600)
def load_assets():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    model_meta = mr.get_model("best_islamabad_aqi_model", version=1) # Version check karein
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    return model, model_meta, fs

try:
    model, meta, fs = load_assets()
    
    # 1. Latest Data (Hive enabled)
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_hive": True}).sort_values("datetime", ascending=False).head(1)
    
    if latest_df.empty:
        st.warning("Feature store is empty. Check ingestion.")
        st.stop()

    last_aqi = float(latest_df['aqi'].iloc[0])
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].iloc[0])
    
    # 2. Forecast
    weather_df = fetch_weather_forecast(4)
    
    # 3. UI - Metrics
    st.title("🌬️ Islamabad AQI Dashboard")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current AQI", int(last_aqi))
    m2.metric("Temp", f"{weather_df.iloc[0]['temperature']:.1f}°C")
    m3.metric("Humidity", f"{weather_df.iloc[0]['humidity']:.1f}%")

    # 4. Graph
    st.subheader("📈 7-Day History")
    hist_data = fetch_historical_aqi_data(fs)
    if not hist_data.empty:
        st.line_chart(hist_data.set_index('Date'))

    # 5. Prediction Logic
    st.subheader("📅 3-Day Forecast")
    cols = st.columns(3)
    FEATURE_ORDER = ['temperature', 'humidity', 'wind_speed', 'hour', 'weekday', 'month', 'aqi_lag_1', 'pm2_5_rolling_6h', 'wind_stagnant']
    
    curr_lag = last_aqi
    curr_pm = last_pm25
    
    for i in range(1, 4):
        row = weather_df.iloc[i]
        feat = pd.DataFrame([{
            'temperature': row['temperature'], 'humidity': row['humidity'], 'wind_speed': row['wind_speed'],
            'hour': 12.0, 'weekday': float(datetime.now().weekday()), 'month': float(datetime.now().month),
            'aqi_lag_1': curr_lag, 'pm2_5_rolling_6h': curr_pm, 'wind_stagnant': 1.0 if row['wind_speed'] < 2.0 else 0.0
        }])[FEATURE_ORDER]
        
        pred = model.predict(feat)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        curr_lag = pred # Evolving lag
        
        with cols[i-1]:
            st.info(f"Day {i}: AQI {aqi_val}")

except Exception as e:
    st.error(f"System Error: {e}")