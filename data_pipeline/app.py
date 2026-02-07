import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pytz  
from utils import fetch_weather_forecast
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide")

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
    
    # ✅ EXACT ORDER FIXED: Matching your model's requirement perfectly
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

    # --- DATA FETCH ---
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    last_aqi = float(latest_df['aqi'].values[0]) if not latest_df.empty else 2.0
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0]) if not latest_df.empty else 15.0

    forecast_weather = fetch_weather_forecast(days=4)
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)

    # --- UI ---
    st.title("🌬️ Islamabad Air Quality Index")
    st.write(f"Last Updated: {now_pk.strftime('%I:%M %p')}")

    # --- FORECAST LOOP ---
    st.markdown("---")
    f_cols = st.columns(3)
    
    current_aqi_lag = last_aqi
    current_pm25 = last_pm25

    

    for i, row in future_df.iterrows():
        # Data dictionary matching TRAINING_FEATURES order
        feat_dict = {
            'aqi_lag_1': float(current_aqi_lag),
            'humidity': float(row['humidity']),
            'hour': float(row['datetime'].hour),
            'month': float(row['datetime'].month),
            'pm2_5_rolling_6h': float(current_pm25),
            'temperature': float(row['temperature']),
            'weekday': float(row['datetime'].weekday()),
            'wind_speed': float(row['wind_speed']),
            'wind_stagnant': 1.0 if float(row['wind_speed']) < 2.5 else 0.0
        }
        
        # Enforce column order
        feat_df = pd.DataFrame([feat_dict])[TRAINING_FEATURES]
        
        # Prediction
        pred = model.predict(feat_df)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Update lag for next iteration
        current_aqi_lag = float(pred)
        current_pm25 = current_pm25 * 0.9 + (pred * 5.0) * 0.1

        with f_cols[i]:
            st.metric(f"{row['datetime'].strftime('%A')}", f"AQI {aqi_val}")

except Exception as e:
    st.error(f"Prediction Error: {e}")

st.markdown('<div style="text-align:center; color:gray;">Islamabad AQI MLOps • Hopsworks 4.2</div>', unsafe_allow_html=True)