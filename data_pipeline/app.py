import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import altair as alt
from utils import fetch_weather_forecast, fetch_historical_aqi_data

# --- CONFIG ---
st.set_page_config(
    page_title="Islamabad Air Quality Insight",
    layout="wide",
    page_icon="🌬️"
)

# --- CSS STYLING ---
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

# --- LOAD HOPSWORKS ASSETS ---
@st.cache_resource(ttl=3600)
def load_assets():
    try:
        HOPSWORKS_KEY = os.environ["HOPSWORKS_KEY"]  # Set this in Streamlit Secrets
        project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
        mr = project.get_model_registry()
        fs = project.get_feature_store()
        
        # Fetch latest model version
        models_list = mr.get_models("best_islamabad_aqi_model")
        if not models_list:
            st.error("❌ Model 'best_islamabad_aqi_model' not found in Registry!")
            st.stop()
        best_model_meta = max(models_list, key=lambda m: m.version)
        model_dir = best_model_meta.download()
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        
        # Fetch benchmark models (optional)
        benchmark_names = ["islamabad_aqi_randomforest", "islamabad_aqi_xgboost", "islamabad_aqi_gradientboosting"]
        all_models = []
        for name in benchmark_names:
            versions = mr.get_models(name)
            if versions:
                all_models.append(max(versions, key=lambda v: v.version))
        
        return model, best_model_meta, all_models, fs
    except Exception as e:
        st.error(f"Failed to load Hopsworks assets: {e}")
        st.stop()

model, best_meta, all_models, fs = load_assets()

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
try:
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
    if latest_df.empty:
        st.error("❌ No data found in feature store!")
        st.stop()
except Exception as e:
    st.error(f"Failed to fetch feature store data: {e}")
    st.stop()

last_aqi = float(latest_df['aqi'].iloc[0])
last_pm25 = float(latest_df['pm2_5_rolling_6h'].iloc[0])

# --- WEATHER FORECAST ---
forecast_weather = fetch_weather_forecast(days=4)
if forecast_weather.empty or len(forecast_weather) < 4:
    st.error("❌ Weather forecast data is insufficient for predictions.")
    st.stop()

# --- KEY METRICS ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Current AQI", int(last_aqi))
with m2:
    st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
with m3:
    st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
with m4:
    st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

# --- HISTORICAL CHART ---
st.subheader("📈 Historical AQI Trends (Past 7 Days)")
hist_df = fetch_historical_aqi_data(fs, num_days=7)
if not hist_df.empty:
    chart = alt.Chart(hist_df).mark_area(
        line={'color':'#00cc96'},
        color=alt.Gradient(
            gradient='linear', 
            stops=[alt.GradientStop(color='#00cc96', offset=0), alt.GradientStop(color='transparent', offset=1)]
        )
    ).encode(
        x=alt.X('Date:T', title=''),
        y=alt.Y('Average AQI:Q', title='AQI Level'),
        tooltip=['Date', 'Average AQI']
    ).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- FORECAST PREDICTIONS ---
st.subheader("📅 Smart Forecast & Health Alerts")

FEATURE_ORDER = [
    'temperature', 'humidity', 'wind_speed', 
    'hour', 'weekday', 'month',
    'aqi_lag_1', 'pm2_5_rolling_6h', 'wind_stagnant'
]

future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
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
f_cols = st.columns(3)

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
    feat = feat[FEATURE_ORDER]
    
    pred = model.predict(feat)[0]
    aqi_val = int(np.clip(round(pred), 1, 5))
    
    predictions.append({
        'day': i+1,
        'date': forecast_datetime.strftime('%A, %d %b'),
        'pred_raw': float(pred),
        'aqi_final': aqi_val
    })
    
    # Update lag features
    current_aqi_lag = pred
    humidity_factor = row['humidity'] / 100.0
    wind_factor = max(0.3, 1.0 - (row['wind_speed'] / 10.0))
    pm25_drift = (pred - last_aqi) * 10.0
    current_pm25 = current_pm25 * (0.80 + 0.35 * humidity_factor * wind_factor) + pm25_drift
    current_pm25 = np.clip(current_pm25, 5.0, 200.0)
    
    label, color, icon = status_map.get(aqi_val, ("Unknown", "#666", "❓"))
    
    # Alerts
    if aqi_val >= 4:
        st.toast(f"⚠️ Health Risk: {label} air quality on {forecast_datetime.strftime('%A')}", icon="😷")
        if i == 0: 
            st.error(f"🚨 ALERT: Level {aqi_val} AQI predicted for {forecast_datetime.strftime('%A')}")
    
    # Display card
    with f_cols[i]:
        st.markdown(f"""
            <div class="metric-card" style="border-top: 5px solid {color};">
                <p style="color: #888; font-size: 0.85rem;">{forecast_datetime.strftime('%A, %d %b')}</p>
                <h2 style="color: {color}; margin: 10px 0;">{icon} {label}</h2>
                <p class="aqi-val" style="color: {color};">{aqi_val}</p>
                <hr style="opacity: 0.1;">
                <p style="font-size: 0.8rem; opacity: 0.8;">🌡️ {row['temperature']:.1f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
            </div>
        """, unsafe_allow_html=True)

# --- DEBUG / Prediction Table ---
with st.expander("🔍 Prediction Details & Evolution", expanded=False):
    st.dataframe(pd.DataFrame(predictions))

# --- SIDEBAR INFO ---
with st.sidebar:
    st.title("🔬 Analytics")
    st.success(f"📌 Active Model: v{best_meta.version}")
    if hasattr(model, 'feature_importances_'):
        feats = ['Temp', 'Humid', 'Wind', 'Hour', 'Weekday', 'Month', 'Lag AQI', 'PM2.5', 'Stagnant']
        imp_series = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(5)
        st.bar_chart(imp_series)

st.markdown('<div class="footer">Islamabad AQI Dashboard • Powered by Hopsworks ML • Real-time Predictions</div>', unsafe_allow_html=True)
