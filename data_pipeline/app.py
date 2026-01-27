import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from utils import fetch_weather_forecast, apply_feature_engineering
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide", page_icon="🌬️")

st.title("🌬️ Islamabad Real-Time AQI Forecast")
st.markdown("---")

@st.cache_resource
def load_assets():
    """Hopsworks se model aur latest AQI data load karta hai."""
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    mr = project.get_model_registry()
    
    # Model download
    model_meta = mr.get_model("best_islamabad_aqi_model", version=1)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    fs = project.get_feature_store()
    return model, fs

try:
    with st.spinner("Connecting to Hopsworks & Fetching Forecast..."):
        model, fs = load_assets()
        
        # 1. Get Latest known state from Feature Group (for Lag features)
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=3)
        # Latest record uthayen taake pichla AQI mil sakay
        latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
        
        last_aqi = float(latest_df['aqi'].values[0])
        last_pm25_rolling = float(latest_df['pm2_5_rolling_6h'].values[0])

        # 2. Fetch Actual Weather Forecast for next 3 days
        # Ye function Meteostat se real forecast data lata hai
        forecast_weather = fetch_weather_forecast(days=3)

    if forecast_weather.empty:
        st.error("Could not fetch weather forecast. Please check Meteostat connection.")
    else:
        st.subheader("📅 Next 3 Days Prediction")
        cols = st.columns(3)
        
        current_aqi_lag = last_aqi
        
        for i, row in forecast_weather.iterrows():
            f_date = row['datetime']
            temp = row['temperature']
            hum = row['humidity']
            wspd = row['wind_speed']

            # Prepare Feature Vector (Must match training order exactly)
            # Features: ['temperature', 'humidity', 'wind_speed', 'hour', 'weekday', 'month', 'aqi_lag_1', 'pm2_5_rolling_6h', 'wind_stagnant']
            features = pd.DataFrame([{
                'temperature': temp,
                'humidity': hum,
                'wind_speed': wspd,
                'hour': 12.0, # Noon prediction
                'weekday': float(f_date.weekday()),
                'month': float(f_date.month),
                'aqi_lag_1': current_aqi_lag,
                'pm2_5_rolling_6h': last_pm25_rolling, # Using last known rolling avg
                'wind_stagnant': 1.0 if wspd < 2.0 else 0.0
            }])

            # Make Prediction
            raw_pred = model.predict(features)[0]
            aqi_val = int(np.clip(round(raw_pred), 1, 5))
            
            # Recursive update: Aaj ki prediction kal ka 'lag' banegi
            current_aqi_lag = raw_pred 

            # UI Display
            with cols[i]:
                st.info(f"**{f_date.strftime('%A, %d %b')}**")
                
                # AQI Styling
                status_map = {
                    1: ("Good", "🟢"), 
                    2: ("Fair", "🟡"), 
                    3: ("Moderate", "🟠"), 
                    4: ("Poor", "🔴"), 
                    5: ("Very Poor", "🟣")
                }
                label, emoji = status_map.get(aqi_val, ("Unknown", "⚪"))
                
                st.metric(label="Predicted AQI Level", value=f"{aqi_val} - {label}")
                st.write(f"### {emoji}")
                
                # Weather context table
                st.write("**Forecasted Weather:**")
                st.write(f"🌡️ Temp: `{temp:.1f}°C`")
                st.write(f"💧 Humidity: `{hum:.1f}%`")
                st.write(f"💨 Wind: `{wspd:.1f} km/h`")

    st.markdown("---")
    st.caption(f"Last updated from Feature Store at: {latest_df['datetime'].values[0]}")

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.warning("Ensure your .env file has valid Hopsworks and OpenWeather keys.")