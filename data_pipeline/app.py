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
pk_tz = pytz.timezone('Asia/Karachi') # Standard for Pakistan
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
    api_key = st.secrets["HOPSWORKS_KEY"] if "HOPSWORKS_KEY" in st.secrets else os.getenv("HOPSWORKS_KEY")
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    best_models_list = mr.get_models("best_islamabad_aqi_model")
    model_meta = max(best_models_list, key=lambda m: m.version)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    model_names = ["islamabad_aqi_randomforest", "islamabad_aqi_xgboost", "islamabad_aqi_gradientboosting"]
    all_models_meta = []
    for name in model_names:
        try:
            versions = mr.get_models(name)
            if versions:
                all_models_meta.append(max(versions, key=lambda v: v.version))
        except: pass
            
    return model, model_meta, all_models_meta, fs

try:
    model, best_meta, all_models, fs = load_assets()
    
    # 1. Header Section
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.title("🌬️ Islamabad Air Quality Index")
        st.markdown(f"🛰️ **Live Monitoring (PKT):** {now_pk.strftime('%A, %d %b %Y | %I:%M %p')}")
    with col_t2:
        if os.path.exists("Islamabad.jpg"):
            st.image("Islamabad.jpg", width=250)

    st.markdown("---")

    # 2. Key Metrics Row
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    if not latest_df.empty:
        last_aqi = float(latest_df['aqi'].values[0])
        last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0])
    else:
        last_aqi, last_pm25 = 1.0, 10.0

    forecast_weather = fetch_weather_forecast(days=4)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current AQI", int(last_aqi))
    with m2:
        st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    with m3:
        st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    with m4:
        st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

    # 3. Graph Section
    st.subheader("📈 Historical AQI Trends (Islamabad)")
    hist_df = fetch_historical_aqi_data(fs, num_days=7)
    if not hist_df.empty:
        chart = alt.Chart(hist_df).mark_area(
            line={'color':'#00cc96'},
            color=alt.Gradient(
                gradient='linear', 
                stops=[alt.GradientStop(color='#00cc96', offset=0), alt.GradientStop(color='transparent', offset=1)], 
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Average AQI:Q', scale=alt.Scale(domain=[1, 5]), title='AQI Level'),
            tooltip=['Date', 'Average AQI']
        ).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

    # 4. Forecast Section (CRITICAL FIXES HERE)
    st.markdown("---")
    st.subheader("📅 Smart Forecast & Health Alerts")
    f_cols = st.columns(3)
    
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    current_aqi_lag = float(last_aqi) # Explicit float
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Hazardous", "#ab63fa", "🚨")
    }

    for i, row in future_df.iterrows():
        # Ensure model receives correct column order and data types
        feat = pd.DataFrame([{
            'temperature': float(row['temperature']), 
            'humidity': float(row['humidity']), 
            'wind_speed': float(row['wind_speed']),
            'hour': float(row['datetime'].hour), 
            'weekday': float(row['datetime'].weekday()), 
            'month': float(row['datetime'].month),
            'aqi_lag_1': float(current_aqi_lag), 
            'pm2_5_rolling_6h': float(last_pm25), 
            'wind_stagnant': 1.0 if float(row['wind_speed']) < 2.0 else 0.0
        }])
        
        # Explicitly align columns if model was trained on specific order
        # feat = feat[['temperature', 'humidity', 'wind_speed', 'hour', 'weekday', 'month', 'aqi_lag_1', 'pm2_5_rolling_6h', 'wind_stagnant']]
        
        pred = model.predict(feat)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Update lag for the next iteration
        current_aqi_lag = pred 
        
        label, color, icon = status_map.get(aqi_val, ("Unknown", "#666", "❓"))

        with f_cols[i]:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <p style="color: #888; font-size: 0.85rem;">{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color: {color}; margin: 10px 0;">{icon} {label}</h2>
                    <p class="aqi-val" style="color: {color};">{aqi_val}</p>
                    <hr style="opacity: 0.1;">
                    <p style="font-size: 0.8rem; opacity: 0.8;">🌡️ {row['temperature']:.1f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
                </div>
            """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("🔬 Analytics")
        st.success(f"📌 Model: **Version {best_meta.version}**")
        st.metric("Training R² Accuracy", f"{best_meta.training_metrics.get('r2', 0):.4f}")
        
        if hasattr(model, 'feature_importances_'):
            st.write("---")
            st.subheader("💡 Prediction Drivers")
            # Feature names matching the training set
            feats = ['Temp', 'Humid', 'Wind', 'Hour', 'Weekday', 'Month', 'Lag AQI', 'PM2.5', 'Stagnant']
            imp_series = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(5)
            st.bar_chart(imp_series)

except Exception as e:
    st.error(f"System Error: {e}")

st.markdown('<div class="footer">Islamabad AQI Dashboard • MLOps System • Hopsworks 4.2</div>', unsafe_allow_html=True)