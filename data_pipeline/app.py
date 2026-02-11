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

# --- PATH FIX FOR UTILS ---
# Ye line ensure karti hai ke app.py apne hi folder mein utils ko dhoonde
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils import fetch_weather_forecast, fetch_historical_aqi_data

load_dotenv()
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- CUSTOM CSS ---
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
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(ttl=3600) 
def load_assets():
    api_key = os.getenv("HOPSWORKS_KEY")
    if not api_key and "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    
    if not api_key:
        st.error("❌ API Key nahi mili!")
        st.stop()

    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    # Latest Model fetching
    model_list = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(model_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    return model, model_meta, fs

try:
    model, best_meta, fs = load_assets()
    
    st.title("🌬️ Islamabad Air Quality Index")
    st.write(f"🛰️ **Live:** {now_pk.strftime('%A, %d %b %Y | %I:%M %p')}")

    # Latest Data from Hopsworks
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
    
    last_aqi = float(latest_df['aqi'].values[0]) if not latest_df.empty else 1.0
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0]) if not latest_df.empty else 10.0
    
    forecast_weather = fetch_weather_forecast(days=4)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current AQI", int(last_aqi))
    with m2: st.metric("Temp", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    with m3: st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    with m4: st.metric("PM2.5", f"{last_pm25:.1f}")

    # Historical Graph
    st.markdown("---")
    st.subheader("📈 Historical Trends")
    hist_df = fetch_historical_aqi_data(fs, num_days=7)
    if not hist_df.empty:
        chart = alt.Chart(hist_df).mark_line(color='#00cc96').encode(
            x='Date:T', y='Average AQI:Q'
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    # 3-Day Forecast Boxes
    st.markdown("---")
    st.subheader("📅 Smart 3-Day Forecast")
    f_cols = st.columns(3)
    
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    curr_lag = last_aqi
    
    status_map = {1:("Good","#00cc96"), 2:("Fair","#fec032"), 3:("Moderate","#ffa15a"), 4:("Poor","#ef553b"), 5:("Hazardous","#ab63fa")}

    for i, row in future_df.iterrows():
        # Correct feature order
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
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <p>{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color:{color};">{label}</h2>
                    <p class="aqi-val" style="color:{color};">{aqi_v}</p>
                </div>
            """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error Loading Dashboard: {e}")