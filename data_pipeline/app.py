import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz  
import altair as alt
from utils import fetch_weather_forecast, fetch_historical_aqi_data
from dotenv import load_dotenv

# --- CONFIG & TIMEZONE ---
load_dotenv()
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #ffffff; }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .aqi-val { font-size: 3rem; font-weight: 800; margin: 0; }
    .footer { text-align: center; color: #666; padding: 20px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource(ttl=3600) 
def load_assets():
    # Secrets handling for Local and Cloud
    if "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    else:
        api_key = os.getenv("HOPSWORKS_KEY")
        
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    # Latest Best Model Fetching
    best_models_list = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(best_models_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    # ✅ EXACT FEATURE ORDER & NAMES (Based on your training mismatch error)
    # 1. Names: 'temp' used instead of 'temperature'
    # 2. Order: aqi_lag_1 is first
    TRAINING_FEATURES = [
        'aqi_lag_1', 
        'humidity', 
        'hour', 
        'month', 
        'pm2_5_rolling_6h', 
        'temp', 
        'weekday', 
        'wind_speed', 
        'wind_stagnant'
    ]

    # --- CURRENT DATA FETCH ---
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    # Use_trend=False taake hamesha live data aaye
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    if not latest_df.empty:
        last_aqi = float(latest_df['aqi'].values[0])
        last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0])
    else:
        last_aqi, last_pm25 = 2.0, 15.0 # Fallbacks

    forecast_weather = fetch_weather_forecast(days=4)

    # --- UI HEADER ---
    st.title("🌬️ Islamabad Air Quality Index")
    st.markdown(f"🛰️ **Live Monitoring (PKT):** {now_pk.strftime('%I:%M %p | %A, %d %b')}")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current AQI", int(last_aqi))
    with m2: st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    with m3: st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    with m4: st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f}")

    # --- FORECAST SECTION ---
    st.markdown("---")
    st.subheader("📅 3-Day Smart Forecast")
    f_cols = st.columns(3)
    
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    
    # State tracking for recursive forecasting
    current_aqi_lag = last_aqi
    current_pm25_rolling = last_pm25
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Hazardous", "#ab63fa", "🚨")
    }

    

    for i, row in future_df.iterrows():
        # Feature dictionary matching TRAINING_FEATURES exactly
        feat_dict = {
            'aqi_lag_1': float(current_aqi_lag),
            'humidity': float(row['humidity']),
            'hour': float(row['datetime'].hour),
            'month': float(row['datetime'].month),
            'pm2_5_rolling_6h': float(current_pm25_rolling),
            'temp': float(row['temperature']), # Key name updated to 'temp'
            'weekday': float(row['datetime'].weekday()),
            'wind_speed': float(row['wind_speed']),
            'wind_stagnant': 1.0 if float(row['wind_speed']) < 2.5 else 0.0
        }
        
        # Enforce exact training column order
        feat_df = pd.DataFrame([feat_dict])[TRAINING_FEATURES]
        
        # Prediction
        pred = model.predict(feat_df)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Recursive Updates: Agle din ke liye prediction ko input banayein
        current_aqi_lag = float(pred)
        # Dynamic PM2.5 proxy update taake features change hon
        current_pm25_rolling = current_pm25_rolling * 0.9 + (pred * 5.0) * 0.1

        label, color, icon = status_map.get(aqi_val, ("Good", "#00cc96", "🌿"))

        with f_cols[i]:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <p style="color: #666;">{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color: {color};">{icon} {label}</h2>
                    <h1 class="aqi-val" style="color: {color};">{aqi_val}</h1>
                    <p style="font-size: 0.8rem; opacity: 0.8;">🌡️ {row['temperature']:.0f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
                </div>
            """, unsafe_allow_html=True)

    with st.sidebar:
        st.success(f"📌 Model: Version {best_meta.version}")
        st.info("Feature names: aligned ✅")

except Exception as e:
    st.error(f"Prediction Error: {e}")

st.markdown('<div class="footer">Islamabad AQI MLOps • Hopsworks 4.2 • Streamlit Cloud</div>', unsafe_allow_html=True)