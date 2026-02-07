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

# --- CONFIG ---
load_dotenv()
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

st.set_page_config(page_title="Islamabad AQI", layout="wide")

# --- ASSETS (Force Refresh) ---
@st.cache_resource(ttl=60) # TTL kam kar diya taake purana model cache na ho
def load_assets():
    # Secret handling for Local vs Cloud
    if "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    else:
        api_key = os.getenv("HOPSWORKS_KEY")
        
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    # Model Version check
    best_models_list = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(best_models_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    # --- EXACT FEATURE ORDER ---
    # Aapki training script clean_features use karti hai. 
    # Us logic ko hamesha alphabetical rakhna best hota hai order mismatch se bachne ke liye.
    TRAINING_FEATURES = [
        'aqi_lag_1', 'humidity', 'hour', 'month', 
        'pm2_5_rolling_6h', 'temperature', 'weekday', 
        'wind_speed', 'wind_stagnant'
    ]

    # --- DATA FETCH ---
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    last_aqi = float(latest_df['aqi'].values[0]) if not latest_df.empty else 2.0
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0]) if not latest_df.empty else 15.0

    forecast_weather = fetch_weather_forecast(days=4)
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)

    # UI
    st.title("🌬️ Islamabad Air Quality Index")
    st.sidebar.write(f"Model Version: {best_meta.version}")
    
    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Current AQI (Islamabad)", int(last_aqi))

    # --- THE FORECAST CHAIN ---
    st.markdown("---")
    f_cols = st.columns(3)
    
    current_aqi_lag = last_aqi
    current_pm25_rolling = last_pm25

    for i, row in future_df.iterrows():
        # Data dictionary
        feat_dict = {
            'temperature': float(row['temperature']), 
            'humidity': float(row['humidity']), 
            'wind_speed': float(row['wind_speed']),
            'hour': float(row['datetime'].hour), 
            'weekday': float(row['datetime'].weekday()), 
            'month': float(row['datetime'].month),
            'aqi_lag_1': float(current_aqi_lag), 
            'pm2_5_rolling_6h': float(current_pm25_rolling), 
            'wind_stagnant': 1.0 if float(row['wind_speed']) < 2.0 else 0.0
        }
        
        # Enforce column order (Matching Training script columns)
        # TIP: Agar model hamesha 1 de raha hai, to TrainingFeatures ka order 
        # wahi rakhein jo aapne '🚀 Features being used' mein dekha tha.
        feat_df = pd.DataFrame([feat_dict])[TRAINING_FEATURES]
        
        # Predict
        pred = model.predict(feat_df)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # UPDATE FOR NEXT DAY (Recursive)
        current_aqi_lag = pred
        # Predict hone par PM2.5 thora change karein taake model bias na kare
        current_pm25_rolling = current_pm25_rolling * 0.95 + (pred * 4) * 0.05

        with f_cols[i]:
            st.metric(f"{row['datetime'].strftime('%A')}", aqi_val)

except Exception as e:
    st.error(f"Error: {e}")