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

# Environment & Config
load_dotenv()
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- ADVANCED CSS ---
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
    if "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    else:
        api_key = os.getenv("HOPSWORKS_KEY")
        
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    best_models_list = mr.get_models("best_islamabad_aqi_model")
    if not best_models_list:
        st.error("Model not found!")
        st.stop()
    
    model_meta = max(best_models_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    # 1. Header Section
    st.title("🌬️ Islamabad Air Quality Index")
    st.markdown(f"🛰️ **Live Monitoring (PKT):** {now_pk.strftime('%A, %d %b %Y | %I:%M %p')}")
    st.markdown("---")

    # 2. Key Metrics Row
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    last_aqi = float(latest_df['aqi'].values[0]) if not latest_df.empty else 2.0
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0]) if not latest_df.empty else 15.0
    forecast_weather = fetch_weather_forecast(days=4)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current AQI", int(last_aqi))
    with m2: st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    with m3: st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    with m4: st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

    # 3. Graph Section
    st.subheader("📈 Historical AQI Trends")
    hist_df = fetch_historical_aqi_data(fs, num_days=7)
    if not hist_df.empty:
        chart = alt.Chart(hist_df).mark_area(line={'color':'#00cc96'}).encode(
            x='Date:T', y=alt.Y('Average AQI:Q', scale=alt.Scale(domain=[1, 5])), tooltip=['Date', 'Average AQI']
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    # 4. Forecast Section
    st.markdown("---")
    st.subheader("📅 3-Day Forecast")
    f_cols = st.columns(3)
    
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    current_aqi_lag = last_aqi
    current_pm25 = last_pm25
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Hazardous", "#ab63fa", "🚨")
    }

    

    for i, row in future_df.iterrows():
        # ✅ FIX: Bypassing Name Mismatch with NumPy Array in EXACT Training Order
        # Order: aqi_lag_1, humidity, hour, month, pm2_5_rolling_6h, temperature, weekday, wind_speed, wind_stagnant
        input_values = [
            max(float(current_aqi_lag), 1.0),
            float(row['humidity']),
            12.0, # noon as proxy
            float(row['datetime'].month),
            max(float(current_pm25), 5.0),
            float(row['temperature']),
            float(row['datetime'].weekday()),
            float(row['wind_speed']),
            1.0 if float(row['wind_speed']) < 2.5 else 0.0
        ]
        
        # Predict using NumPy array to ignore feature names conflict
        pred = model.predict(np.array([input_values]))[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Recursive updates
        current_aqi_lag = pred
        current_pm25 = current_pm25 * 0.9 + (pred * 5.0) * 0.1
        
        label, color, icon = status_map.get(aqi_val, ("Good", "#00cc96", "🌿"))

        with f_cols[i]:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <p style="color: #888;">{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color: {color};">{icon} {label}</h2>
                    <p class="aqi-val" style="color: {color};">{aqi_val}</p>
                    <p style="font-size: 0.8rem; opacity: 0.8;">🌡️ {row['temperature']:.0f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
                </div>
            """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.success(f"📌 Model: Version {best_meta.version}")
        if hasattr(model, 'feature_importances_'):
            st.subheader("💡 Drivers")
            feats = ['Lag AQI', 'Humid', 'Hour', 'Month', 'PM2.5', 'Temp', 'W-Day', 'Wind', 'Stag']
            imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(5)
            st.bar_chart(imp)

except Exception as e:
    st.error(f"System Error: {e}")

st.markdown('<div class="footer">Islamabad AQI Dashboard • Hopsworks 4.2</div>', unsafe_allow_html=True)