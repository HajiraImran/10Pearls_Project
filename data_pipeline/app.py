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

# 1. Environment Variables Load karein (.env file se)
load_dotenv()

# Timezone Setup
pk_tz = pytz.timezone('Asia/Karachi')
now_pk = datetime.now(pk_tz)

# Streamlit Page Config
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
    .aqi-val { font-size: 3.5rem; font-weight: 800; margin: 0; }
    .footer { text-align: center; color: #666; padding: 20px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource(ttl=3600) 
def load_assets():
    # Pehle .env se check karein, phir Streamlit Cloud Secrets se
    api_key = os.getenv("HOPSWORKS_KEY")
    if not api_key and "HOPSWORKS_KEY" in st.secrets:
        api_key = st.secrets["HOPSWORKS_KEY"]
    
    if not api_key:
        st.error("❌ HOPSWORKS_KEY nahi mili! Apni .env file check karein.")
        st.stop()
        
    try:
        project = hopsworks.login(api_key_value=api_key)
        mr = project.get_model_registry()
        fs = project.get_feature_store()
        
        # Model fetching
        best_models_list = mr.get_models("best_islamabad_aqi_model")
        if not best_models_list:
            st.error("Model registry mein model nahi mila!")
            st.stop()
            
        model_meta = max(best_models_list, key=lambda m: m.version)
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        
        return model, model_meta, fs
    except Exception as e:
        st.error(f"Hopsworks Connection Error: {e}")
        st.stop()

try:
    model, best_meta, fs = load_assets()
    
    # 2. Header Section
    st.title("🌬️ Islamabad Air Quality Index")
    st.markdown(f"🛰️ **Live Monitoring (PKT):** {now_pk.strftime('%A, %d %b %Y | %I:%M %p')}")
    st.markdown("---")

    # 3. Key Metrics Row (Latest Data from Hopsworks)
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read(read_options={"use_trend": False}).sort_values("datetime", ascending=False).head(1)
    
    # Defaults agar data na mile
    last_aqi = float(latest_df['aqi'].values[0]) if not latest_df.empty else 2.0
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0]) if not latest_df.empty else 15.0
    
    forecast_weather = fetch_weather_forecast(days=4)

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Current AQI", int(last_aqi))
    with m2: st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    with m3: st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    with m4: st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

    # 4. Graph Section
    st.subheader("📈 Historical AQI Trends (Past 7 Days)")
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
        ).properties(height=250).interactive()
        st.altair_chart(chart, use_container_width=True)

    # 5. Forecast Section
    st.markdown("---")
    st.subheader("📅 Smart 3-Day Forecast")
    f_cols = st.columns(3)
    
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    current_aqi_lag = last_aqi
    current_pm25 = last_pm25
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Hazardous", "#ab63fa", "🚨")
    }

    

    for i, row in future_df.iterrows():
        # ✅ NumPy Array approach for strict feature order
        # Order: aqi_lag_1, humidity, hour, month, pm2_5_rolling_6h, temperature, weekday, wind_speed, wind_stagnant
        input_values = [
            max(float(current_aqi_lag), 1.0),   # Lag AQI (Minimum 1)
            float(row['humidity']),            # Humidity
            12.0,                              # Hour (Static Noon for Daily Forecast)
            float(row['datetime'].month),      # Month
            max(float(current_pm25), 5.0),     # PM2.5 (Minimum 5)
            float(row['temperature']),         # Temperature
            float(row['datetime'].weekday()),  # Weekday
            float(row['wind_speed']),          # Wind Speed
            1.0 if float(row['wind_speed']) < 2.5 else 0.0 # Stagnancy
        ]
        
        # Predict using Array to avoid feature name mismatch
        pred = model.predict(np.array([input_values]))[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Recursive State Update
        current_aqi_lag = pred
        current_pm25 = current_pm25 * 0.9 + (pred * 5.0) * 0.1
        
        label, color, icon = status_map.get(aqi_val, ("Good", "#00cc96", "🌿"))

        with f_cols[i]:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <p style="color: #888; font-size: 0.9rem;">{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color: {color}; margin: 10px 0;">{icon} {label}</h2>
                    <p class="aqi-val" style="color: {color};">{aqi_val}</p>
                    <p style="font-size: 0.8rem; opacity: 0.8;">🌡️ {row['temperature']:.1f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
                </div>
            """, unsafe_allow_html=True)

    # Sidebar: Analytics
    with st.sidebar:
        st.header("🔬 Model Details")
        st.success(f"📌 Version: {best_meta.version}")
        st.info(f"R² Score: {best_meta.training_metrics.get('r2', 0):.4f}")
        
        if hasattr(model, 'feature_importances_'):
            st.write("---")
            st.subheader("💡 Key Drivers")
            feats = ['Lag AQI', 'Humid', 'Hour', 'Month', 'PM2.5', 'Temp', 'W-Day', 'Wind', 'Stag']
            imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(5)
            st.bar_chart(imp)

except Exception as e:
    st.error(f"Application Error: {e}")

st.markdown('<div class="footer">Islamabad AQI Dashboard • Powered by Hopsworks & Streamlit</div>', unsafe_allow_html=True)