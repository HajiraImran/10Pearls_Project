import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import pytz 
import altair as alt
from dotenv import load_dotenv

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from utils import fetch_weather_forecast, fetch_historical_aqi_data
except ImportError:
    st.error("❌ 'utils.py' missing in data_pipeline folder!")
    st.stop()

load_dotenv()
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

st.set_page_config(page_title="Islamabad AQI", layout="wide", page_icon="🌬️")

@st.cache_resource(ttl=3600)
def load_assets():
    api_key = os.getenv("HOPSWORKS_KEY")
    if not api_key and "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    model_list = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(model_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    st.title("🌬️ Islamabad Air Quality Index")
    st.write(f"🛰️ **Live:** {now_pk.strftime('%A, %d %b %Y | %I:%M %p')}")

    # Fetch Data
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
    
    # Fallback if FG is empty
    last_aqi = float(latest_df['aqi'].values[0]) if not latest_df.empty else 2.0
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0]) if not latest_df.empty else 15.0
    
    forecast_df = fetch_weather_forecast(days=4)

    # 3-Day Forecast Section
    st.markdown("---")
    st.subheader("📅 Smart Forecast & Health Alerts")
    f_cols = st.columns(3)
    
    future_df = forecast_df.iloc[1:4].reset_index(drop=True)
    curr_lag = last_aqi
    
    status_map = {1:("Good","#00cc96"), 2:("Fair","#fec032"), 3:("Moderate","#ffa15a"), 4:("Poor","#ef553b"), 5:("Hazardous","#ab63fa")}

    for i, row in future_df.iterrows():
        # Match training feature order exactly
        feat_array = np.array([[
            max(float(curr_lag), 1.0), float(row['humidity']), 12.0, float(row['datetime'].month),
            max(float(last_pm25), 5.0), float(row['temperature']), float(row['datetime'].weekday()),
            float(row['wind_speed']), 1.0 if row['wind_speed'] < 2.5 else 0.0
        ]])
        
        pred = model.predict(feat_array)[0]
        aqi_v = int(np.clip(round(pred), 1, 5))
        label, color = status_map.get(aqi_v, ("Good", "#00cc96"))
        
        curr_lag = pred 
        
        with f_cols[i]:
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; border-top:5px solid {color}; text-align:center;">
                    <p>{row['datetime'].strftime('%A')}</p>
                    <h2 style="color:{color};">{label}</h2>
                    <h1 style="font-size:3rem;">{aqi_v}</h1>
                </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ Dashboard Error: {e}")