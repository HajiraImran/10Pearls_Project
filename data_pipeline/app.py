import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import altair as alt
from utils import fetch_weather_forecast, fetch_historical_aqi_data

# Page Config
st.set_page_config(page_title="Islamabad AQI Insight", layout="wide", page_icon="🌬️")

# Keys from Secrets (Cloud) or Env (Local)
HOPSWORKS_KEY = st.secrets.get("HOPSWORKS_KEY") or os.getenv("HOPSWORKS_KEY")

# Custom CSS for better look
st.markdown("""
    <style>
    .metric-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
    .aqi-val { font-size: 3rem; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(ttl=3600)
def load_assets():
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    # Ensure version 5 is correct in your registry
    model_meta = mr.get_model("best_islamabad_aqi_model", version=5) 
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    return model, fs

try:
    model, fs = load_assets()

    # 1. Fetch Latest State from Hopsworks (Using Hive)
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_hive": True}).sort_values("datetime", ascending=False).head(1)

    if latest_df.empty:
        st.error("No data found in Feature Store. Ingestion might be failing.")
        st.stop()

    last_aqi = float(latest_df['aqi'].iloc[0])
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].iloc[0])

    # 2. Display Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current AQI", int(last_aqi))
    with col2:
        st.metric("Last PM2.5 (Rolling)", f"{last_pm25:.1f}")
    with col3:
        st.write(f"Last Update: {latest_df['datetime'].iloc[0]}")

    # 3. History Chart
    st.subheader("📈 Past 7 Days Trend")
    hist_df = fetch_historical_aqi_data(fs)
    if not hist_df.empty:
        st.line_chart(hist_df.set_index('Date'))

    # 4. Forecast Prediction
    st.subheader("📅 3-Day Prediction")
    weather_df = fetch_weather_forecast(4)
    f_cols = st.columns(3)
    
    # CRITICAL: Feature order must match your model's training
    FEATURE_ORDER = ['temperature', 'humidity', 'wind_speed', 'hour', 'weekday', 'month', 'aqi_lag_1', 'pm2_5_rolling_6h', 'wind_stagnant']
    
    current_lag = last_aqi
    current_pm = last_pm25

    for i in range(1, 4):
        row = weather_df.iloc[i]
        feat = pd.DataFrame([{
            'temperature': float(row['temperature']),
            'humidity': float(row['humidity']),
            'wind_speed': float(row['wind_speed']),
            'hour': 12.0, # Noon prediction
            'weekday': float(row['datetime'].weekday()),
            'month': float(row['datetime'].month),
            'aqi_lag_1': float(current_lag),
            'pm2_5_rolling_6h': float(current_pm),
            'wind_stagnant': 1.0 if row['wind_speed'] < 2.0 else 0.0
        }])[FEATURE_ORDER]

        pred = model.predict(feat)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Update lag for next iteration
        current_lag = pred
        # Simple PM2.5 evolution
        current_pm = current_pm * 0.9 + (aqi_val * 5) 

        with f_cols[i-1]:
            st.markdown(f"""
                <div class="metric-card">
                    <p>{row['datetime'].strftime('%A')}</p>
                    <p class="aqi-val">{aqi_val}</p>
                    <p>Temp: {row['temperature']}°C</p>
                </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Error: {e}")
    st.info("Check your Hopsworks model version and Feature Group name.")