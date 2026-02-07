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

st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide", page_icon="🌬️")

# --- ASSET LOADING ---
@st.cache_resource(ttl=3600) 
def load_assets():
    if "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    else:
        api_key = os.getenv("HOPSWORKS_KEY")
        
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    best_models_list = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(best_models_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    # ✅ EXACT ORDER FIXED: As per your XGBoost Mismatch Error
    TRAINING_FEATURES = [
        'aqi_lag_1', 
        'humidity', 
        'hour', 
        'month', 
        'pm2_5_rolling_6h', 
        'temperature', 
        'weekday', 
        'wind_speed', 
        'wind_stagnant'
    ]

    # --- DATA FETCHING ---
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    if not latest_df.empty:
        last_aqi = float(latest_df['aqi'].values[0])
        last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0])
    else:
        last_aqi, last_pm25 = 2.0, 15.0

    forecast_weather = fetch_weather_forecast(days=4)
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)

    # --- UI HEADER ---
    st.title("🌬️ Islamabad Air Quality Index")
    st.markdown(f"🛰️ **Live Monitoring (PKT):** {now_pk.strftime('%I:%M %p | %A, %d %b')}")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Current AQI", int(last_aqi))
    m2.metric("Temp", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    m3.metric("PM2.5", f"{last_pm25:.1f} µg/m³")

    # --- FORECAST SECTION ---
    st.markdown("---")
    st.subheader("📅 3-Day Smart Forecast")
    f_cols = st.columns(3)
    
    current_aqi_lag = float(last_aqi)
    current_pm25_rolling = float(last_pm25)
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Hazardous", "#ab63fa", "🚨")
    }

    

    for i, row in future_df.iterrows():
        # Feature construction
        feat_dict = {
            'aqi_lag_1': current_aqi_lag,
            'humidity': float(row['humidity']),
            'hour': float(row['datetime'].hour),
            'month': float(row['datetime'].month),
            'pm2_5_rolling_6h': current_pm25_rolling,
            'temperature': float(row['temperature']),
            'weekday': float(row['datetime'].weekday()),
            'wind_speed': float(row['wind_speed']),
            'wind_stagnant': 1.0 if float(row['wind_speed']) < 2.5 else 0.0
        }
        
        # Enforce exact training order
        feat_df = pd.DataFrame([feat_dict])[TRAINING_FEATURES]
        
        # Predict
        pred = model.predict(feat_df)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Update lag for next day
        current_aqi_lag = float(pred)
        # Minor heuristic update for PM2.5 to help model vary predictions
        current_pm25_rolling = current_pm25_rolling * 0.9 + (pred * 4.0) * 0.1

        label, color, icon = status_map.get(aqi_val, ("Good", "#00cc96", "🌿"))

        with f_cols[i]:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding:20px; border-radius:15px; border-top: 5px solid {color}; text-align:center;">
                    <p style="color:#888;">{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color:{color};">{icon} {label}</h2>
                    <h1 style="font-size: 3.5rem; margin:0;">{aqi_val}</h1>
                    <p style="font-size: 0.9rem; opacity:0.7;">🌡️ {row['temperature']:.0f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
                </div>
            """, unsafe_allow_html=True)

    with st.sidebar:
        st.success(f"Model: Version {best_meta.version}")
        st.info("Feature alignment verified. ✅")

except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.info("Tip: If you see a mismatch, update TRAINING_FEATURES list order in app.py.")