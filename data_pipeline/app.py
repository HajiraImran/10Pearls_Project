import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import altair as alt
from utils import fetch_weather_forecast, fetch_historical_aqi_data

st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- UI STYLING ---
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

# --- LOAD MODEL + FEATURE STORE ---
@st.cache_resource(ttl=3600)
def load_assets():
    project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_KEY"))
    mr = project.get_model_registry()
    fs = project.get_feature_store()

    model_meta = max(mr.get_models("best_islamabad_aqi_model"), key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))

    return model, model_meta, fs

model, best_meta, fs = load_assets()

# --- HEADER ---
st.title("🌬️ Islamabad Air Quality Index")
st.markdown(f"🛰️ **Live Monitoring:** {datetime.now().strftime('%A, %d %b %Y | %I:%M %p')}")
st.markdown("---")

# --- LATEST AQI DATA ---
fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
latest_df = fg.read(read_options={"use_hive": True}).sort_values("datetime", ascending=False).head(1)

last_aqi = float(latest_df['aqi'].iloc[0])
last_pm25 = float(latest_df['pm2_5_rolling_6h'].iloc[0])

# --- WEATHER FORECAST ---
forecast_weather = fetch_weather_forecast(days=4)

if forecast_weather.empty or len(forecast_weather) < 4:
    st.error("Weather forecast data insufficient.")
    st.stop()

# --- METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Current AQI", int(last_aqi))
m2.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
m3.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
m4.metric("PM2.5", f"{last_pm25:.1f} µg/m³")

# --- HISTORICAL CHART ---
st.subheader("📈 Historical AQI Trends")
hist_df = fetch_historical_aqi_data(fs, num_days=7)
if not hist_df.empty:
    chart = alt.Chart(hist_df).mark_line(point=True).encode(
        x='Date:T', y='Average AQI:Q'
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

# --- FORECAST PREDICTIONS ---
st.markdown("---")
st.subheader("📅 3-Day AQI Forecast")

FEATURE_ORDER = [
    'temperature','humidity','wind_speed',
    'hour','weekday','month',
    'aqi_lag_1','pm2_5_rolling_6h','wind_stagnant'
]

future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
f_cols = st.columns(3)

current_aqi_lag = last_aqi
current_pm25 = last_pm25

status_map = {
    1: ("Good", "#00cc96", "🌿"),
    2: ("Fair", "#fec032", "🌳"),
    3: ("Moderate", "#ffa15a", "😷"),
    4: ("Poor", "#ef553b", "⚠️"),
    5: ("Hazardous", "#ab63fa", "🚨")
}

predictions = []

for i, row in future_df.iterrows():

    forecast_datetime = row['datetime']

    feat = pd.DataFrame([{
        'temperature': float(row['temperature']),
        'humidity': float(row['humidity']),
        'wind_speed': float(row['wind_speed']),
        'hour': 12.0,
        'weekday': float(forecast_datetime.weekday()),
        'month': float(forecast_datetime.month),
        'aqi_lag_1': float(current_aqi_lag),
        'pm2_5_rolling_6h': float(current_pm25),
        'wind_stagnant': 1.0 if row['wind_speed'] < 2.0 else 0.0
    }])

    # SAFETY CLIPPING
    feat['aqi_lag_1'] = np.clip(feat['aqi_lag_1'], 1, 5)
    feat['pm2_5_rolling_6h'] = np.clip(feat['pm2_5_rolling_6h'], 15, 250)
    feat = feat[FEATURE_ORDER]

    pred = float(model.predict(feat)[0])
    pred_clipped = float(np.clip(pred, 1, 5))
    aqi_val = int(round(pred_clipped))

    predictions.append({
        'Day': i+1,
        'Raw Prediction': pred,
        'Clipped Prediction': pred_clipped,
        'Final AQI': aqi_val
    })

    # 🔁 FEEDBACK FIX
    current_aqi_lag = pred_clipped

    humidity_factor = row['humidity'] / 100.0
    wind_factor = max(0.4, 1.0 - (row['wind_speed'] / 12.0))
    pm25_drift = (pred_clipped - last_aqi) * 5.0
    current_pm25 = current_pm25 * (0.85 + 0.25 * humidity_factor * wind_factor) + pm25_drift
    current_pm25 = float(np.clip(current_pm25, 15.0, 200.0))

    label, color, icon = status_map[aqi_val]

    with f_cols[i]:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 5px solid {color};">
            <p>{forecast_datetime.strftime('%A, %d %b')}</p>
            <h2 style="color:{color};">{icon} {label}</h2>
            <p class="aqi-val" style="color:{color};">{aqi_val}</p>
        </div>
        """, unsafe_allow_html=True)

# --- DEBUG TABLE ---
with st.expander("🔍 Prediction Debug"):
    st.dataframe(pd.DataFrame(predictions), use_container_width=True)
