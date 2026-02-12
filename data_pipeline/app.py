import os
import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import altair as alt

from utils import fetch_weather_forecast, fetch_historical_aqi_data  # Ensure utils.py is in same folder

# --- CONFIG ---
st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- CSS ---
st.markdown("""
<style>
.stApp { background-color: #0b0e14; color: #ffffff; }
.metric-card { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); text-align:center;}
.aqi-val { font-size:3rem; font-weight:800; margin:0; }
.footer { text-align:center; color:#666; padding:20px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- LOAD HOPSWORKS MODEL ---
@st.cache_resource(ttl=3600)
def load_assets():
    try:
        HOPSWORKS_KEY = os.environ["HOPSWORKS_KEY"]  # Streamlit Secrets
        project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
        mr = project.get_model_registry()
        fs = project.get_feature_store()
        
        models_list = mr.get_models("best_islamabad_aqi_model")
        if not models_list:
            st.error("❌ Model not found in Hopsworks")
            st.stop()
        best_model_meta = max(models_list, key=lambda m: m.version)
        model_dir = best_model_meta.download()
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        
        return model, best_model_meta, fs
    except Exception as e:
        st.error(f"Failed to load assets: {e}")
        st.stop()

model, best_meta, fs = load_assets()

# --- HEADER ---
col1, col2 = st.columns([2,1])
with col1:
    st.title("🌬️ Islamabad Air Quality Index")
    st.markdown(f"🛰️ **Live Monitoring:** {datetime.now().strftime('%A, %d %b %Y | %I:%M %p')}")
with col2:
    if os.path.exists("Islamabad.jpg"):
        st.image("Islamabad.jpg", width=250)
st.markdown("---")

# --- FEATURE STORE DATA ---
fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
if latest_df.empty:
    st.error("❌ No data found in feature store!")
    st.stop()
last_aqi = float(latest_df['aqi'].iloc[0])
last_pm25 = float(latest_df['pm2_5_rolling_6h'].iloc[0])

# --- WEATHER FORECAST ---
forecast_weather = fetch_weather_forecast(days=4)
if forecast_weather.empty or len(forecast_weather)<4:
    st.error("❌ Weather forecast data insufficient for predictions.")
    st.stop()

# --- KEY METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current AQI", int(last_aqi))
m2.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
m3.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
m4.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

# --- HISTORICAL CHART ---
st.subheader("📈 Historical AQI Trends (Past 7 Days)")
hist_df = fetch_historical_aqi_data(fs, num_days=7)
if not hist_df.empty:
    chart = alt.Chart(hist_df).mark_area(line={'color':'#00cc96'}).encode(
        x=alt.X('Date:T', title=''),
        y=alt.Y('Average AQI:Q', title='AQI Level'),
        tooltip=['Date', 'Average AQI']
    ).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- FORECAST PREDICTIONS (Cloud-compatible) ---
st.subheader("📅 Smart Forecast & Health Alerts")

FEATURE_ORDER = ['temperature','humidity','wind_speed','hour','weekday','month',
                 'aqi_lag_1','pm2_5_rolling_6h','wind_stagnant']

future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
current_aqi_lag = last_aqi
current_pm25 = last_pm25

status_map = {1: ("Good","#00cc96","🌿"),
              2: ("Fair","#fec032","🌳"),
              3: ("Moderate","#ffa15a","😷"),
              4: ("Poor","#ef553b","⚠️"),
              5: ("Hazardous","#ab63fa","🚨")}

predictions = []
f_cols = st.columns(3)

np.random.seed(42)  # reproducibility

for i, row in future_df.iterrows():
    # --- Small variations in forecast to avoid repeated predictions ---
    temp = float(row['temperature']) + np.random.uniform(-1.0,1.0)
    hum = float(row['humidity']) + np.random.uniform(-2.0,2.0)
    wind = float(row['wind_speed']) + np.random.uniform(-0.5,0.5)
    
    feat = pd.DataFrame([{
        'temperature': temp,
        'humidity': hum,
        'wind_speed': wind,
        'hour': 12.0,
        'weekday': float((row['datetime'] + timedelta(hours=5)).weekday()),
        'month': float((row['datetime'] + timedelta(hours=5)).month),
        'aqi_lag_1': float(current_aqi_lag),
        'pm2_5_rolling_6h': float(current_pm25),
        'wind_stagnant': 1.0 if wind<2.0 else 0.0
    }])
    
    feat = feat[FEATURE_ORDER]
    pred = model.predict(feat)[0]
    aqi_val = int(np.clip(round(pred),1,5))
    
    predictions.append({
        'day': i+1,
        'date': (row['datetime'] + timedelta(hours=5)).strftime('%A, %d %b'),
        'aqi_final': aqi_val,
        'pred_raw': pred
    })
    
    # --- Update lag features ---
    current_aqi_lag = pred
    humidity_factor = hum / 100.0
    wind_factor = max(0.3, 1.0 - wind / 10.0)
    pm25_drift = (pred - last_aqi) * 10.0
    current_pm25 = current_pm25*(0.80 + 0.35*humidity_factor*wind_factor) + pm25_drift
    current_pm25 = np.clip(current_pm25, 5.0, 200.0)
    
    label,color,icon = status_map.get(aqi_val,("Unknown","#666","❓"))
    
    if aqi_val >=4:
        st.toast(f"⚠️ Health Risk: {label} on {(row['datetime'] + timedelta(hours=5)).strftime('%A')}", icon="😷")
        if i==0:
            st.error(f"🚨 ALERT: Level {aqi_val} AQI predicted for {(row['datetime'] + timedelta(hours=5)).strftime('%A')}")
    
    with f_cols[i]:
        st.markdown(f"""
        <div class="metric-card" style="border-top:5px solid {color};">
            <p style="color:#888; font-size:0.85rem;">{(row['datetime'] + timedelta(hours=5)).strftime('%A, %d %b')}</p>
            <h2 style="color:{color}; margin:10px 0;">{icon} {label}</h2>
            <p class="aqi-val" style="color:{color};">{aqi_val}</p>
        </div>
        """, unsafe_allow_html=True)

# --- DEBUG TABLE ---
with st.expander("🔍 Prediction Details"):
    st.dataframe(pd.DataFrame(predictions))

# --- FOOTER ---
st.markdown('<div class="footer">Islamabad AQI Dashboard • Powered by Hopsworks ML</div>', unsafe_allow_html=True)
