import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import altair as alt
from utils import fetch_weather_forecast, fetch_historical_aqi_data
from dotenv import load_dotenv

# Environment & Config
load_dotenv()
st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- ADVANCED CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0b0e14; color: #ffffff; }
    
    /* Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    /* AQI Card Colors */
    .aqi-val { font-size: 3rem; font-weight: 800; margin: 0; }
    .aqi-label { font-size: 1.2rem; font-weight: 500; margin-bottom: 10px; }
    
    /* Sidebar Styling */
    .css-1d391kg { background-color: #161b22; }
    
    /* Footer */
    .footer { text-align: center; color: #666; padding: 20px; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    model_meta = mr.get_model("best_islamabad_aqi_model", version=1)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    
    model_names = ["islamabad_aqi_randomforest", "islamabad_aqi_xgboost", "islamabad_aqi_gradientboosting"]
    all_models_meta = []
    for name in model_names:
        try:
            m = mr.get_model(name, version=1)
            if m: all_models_meta.append(m)
        except: pass
    return model, model_meta, all_models_meta, fs

# --- MAIN DASHBOARD LOGIC ---
try:
    model, best_meta, all_models, fs = load_assets()
    
    # 1. Header Section
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.title("🌬️ Islamabad Air Quality Index")
        st.markdown(f"🛰️ **Live Monitoring:** {datetime.now().strftime('%A, %d %b %Y | %I:%M %p')}")
    
    with col_t2:
        # Small centered image
        st.image("Islamabad.jpg", width=250)

    st.markdown("---")

    # 2. Key Metrics Row
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
    
    last_aqi = float(latest_df['aqi'].values[0])
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0])
    forecast_weather = fetch_weather_forecast(days=4)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current AQI", int(last_aqi), delta_color="inverse")
    with m2:
        curr_temp = forecast_weather.iloc[0]['temperature']
        st.metric("Temperature", f"{curr_temp:.1f}°C")
    with m3:
        curr_hum = forecast_weather.iloc[0]['humidity']
        st.metric("Humidity", f"{curr_hum:.1f}%")
    with m4:
        st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

    # 3. Graph Section
    st.subheader("📈 Historical AQI Trends (Past 7 Days)")
    hist_df = fetch_historical_aqi_data(fs, num_days=7)
    if not hist_df.empty:
        chart = alt.Chart(hist_df).mark_area(
            line={'color':'#00cc96'},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='#00cc96', offset=0),
                       alt.GradientStop(color='transparent', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Average AQI:Q', scale=alt.Scale(domain=[1, 5]), title='AQI Level'),
            tooltip=['Date', 'Average AQI']
        ).properties(height=350).interactive()
        st.altair_chart(chart, use_container_width=True)

    # 4. Weekend Forecast Section
    st.markdown("---")
    st.subheader("📅 Smart Forecast (Next 3 Days)")
    f_cols = st.columns(3)
    
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    current_aqi_lag = last_aqi
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Very Poor", "#ab63fa", "☠️")
    }

    for i, row in future_df.iterrows():
        # Prediction logic
        feat = pd.DataFrame([{
            'temperature': row['temperature'], 'humidity': row['humidity'], 'wind_speed': row['wind_speed'],
            'hour': 12.0, 'weekday': float(row['datetime'].weekday()), 'month': float(row['datetime'].month),
            'aqi_lag_1': current_aqi_lag, 'pm2_5_rolling_6h': last_pm25, 'wind_stagnant': 1.0 if row['wind_speed'] < 2.0 else 0.0
        }])
        pred = model.predict(feat)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        current_aqi_lag = pred
        label, color, icon = status_map.get(aqi_val)

        with f_cols[i]:
            st.markdown(f"""
                <div class="metric-card" style="border-top: 5px solid {color};">
                    <p style="color: #666; margin-bottom: 5px;">{row['datetime'].strftime('%A, %d %b')}</p>
                    <h2 style="color: {color}; margin: 0;">{icon} {label}</h2>
                    <p class="aqi-val" style="color: {color};">{aqi_val}</p>
                    <hr style="opacity: 0.1;">
                    <p style="font-size: 0.9rem;">🌡️ {row['temperature']:.1f}°C | 💨 {row['wind_speed']:.1f} km/h</p>
                </div>
            """, unsafe_allow_html=True)

    # Sidebar: Analytics
    with st.sidebar:
        st.title("🔬 Analytics")
        st.info(f"Using Champion Model: **{best_meta.name} v{best_meta.version}**")
        st.metric("Model R² Accuracy", f"{best_meta.training_metrics.get('r2', 0):.4f}")
        
        if all_models:
            st.write("---")
            st.write("📊 **Algorithm Benchmark**")
            comp_df = pd.DataFrame([{"Model": m.name.split('_')[-1].upper(), "R2": m.training_metrics.get('r2', 0)} for m in all_models])
            st.dataframe(comp_df, hide_index=True)

except Exception as e:
    st.error(f"System Error: {e}")

st.markdown('<div class="footer">Data Pipeline: GitHub Actions → Hopsworks → Streamlit Cloud</div>', unsafe_allow_html=True)