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

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Islamabad Air Quality Insight", layout="wide", page_icon="🌬️")

# --- STYLING ---
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

# --- ASSET LOADING (Optimized for Hopsworks 4.2) ---
@st.cache_resource(ttl=3600) 
def load_assets():
    try:
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
        mr = project.get_model_registry()
        fs = project.get_feature_store()
        
        # Latest Model Loading
        best_models_list = mr.get_models("best_islamabad_aqi_model")
        if not best_models_list:
            st.error("Model 'best_islamabad_aqi_model' not found!")
            st.stop()
        
        model_meta = max(best_models_list, key=lambda m: m.version)
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        
        # Benchmark Models
        model_names = ["islamabad_aqi_randomforest", "islamabad_aqi_xgboost", "islamabad_aqi_gradientboosting"]
        all_models_meta = []
        for name in model_names:
            try:
                versions = mr.get_models(name)
                if versions:
                    all_models_meta.append(max(versions, key=lambda v: v.version))
            except: pass
                
        return model, model_meta, all_models_meta, fs
    except Exception as e:
        st.error(f"Failed to connect to Hopsworks: {e}")
        st.stop()

# --- MAIN APP LOGIC ---
try:
    model, best_meta, all_models, fs = load_assets()
    
    # 1. Header
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.title("🌬️ Islamabad Air Quality Index")
        st.markdown(f"🛰️ **Live Monitoring:** {datetime.now().strftime('%A, %d %b %Y | %I:%M %p')}")
    with col_t2:
        if os.path.exists("Islamabad.jpg"):
            st.image("Islamabad.jpg", width=250)

    st.markdown("---")

    # 2. Key Metrics & Data Fetching (With Out-of-bounds Fix)
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    
    # Use hive mode to avoid DuckDB binder errors in older backends
    try:
        latest_df = fg.read(read_options={"use_hive": True}).sort_values("datetime", ascending=False).head(1)
    except:
        latest_df = fg.show(1) # Fallback to preview mode

    # Safety Check for iloc[0]
    if not latest_df.empty:
        last_aqi = float(latest_df['aqi'].values[0])
        last_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0])
    else:
        st.warning("⚠️ Data currently unavailable in Feature Store. Using fallback values.")
        last_aqi, last_pm25 = 1.0, 15.0 # Generic defaults

    forecast_weather = fetch_weather_forecast(days=4)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current AQI", int(last_aqi), delta_color="inverse")
    
    # Weather Metrics Safety
    if forecast_weather is not None and not forecast_weather.empty:
        with m2: st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
        with m3: st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    else:
        with m2: st.metric("Temperature", "N/A")
        with m3: st.metric("Humidity", "N/A")
        
    with m4:
        st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

    # 3. Graph Section
    st.subheader("📈 Historical AQI Trends (Past 7 Days)")
    hist_df = fetch_historical_aqi_data(fs, num_days=7)
    if hist_df is not None and not hist_df.empty:
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
    else:
        st.info("Historical data is loading or temporarily unavailable.")

    # 4. Forecast Section (CRITICAL FIX FOR OUT-OF-BOUNDS)
    st.markdown("---")
    st.subheader("📅 Smart Forecast & Health Alerts")
    
    if forecast_weather is not None and len(forecast_weather) > 1:
        # Get next 3 days safely
        future_count = min(4, len(forecast_weather))
        future_df = forecast_weather.iloc[1:future_count].reset_index(drop=True)
        f_cols = st.columns(len(future_df))
        
        current_aqi_lag = last_aqi
        status_map = {
            1: ("Good", "#00cc96", "🌿"), 2: ("Fair", "#fec032", "🌳"), 
            3: ("Moderate", "#ffa15a", "😷"), 4: ("Poor", "#ef553b", "⚠️"), 5: ("Hazardous", "#ab63fa", "🚨")
        }

        for i, row in future_df.iterrows():
            # Build feature set for XGBoost
            feat = pd.DataFrame([{
                'temperature': float(row['temperature']), 
                'humidity': float(row['humidity']), 
                'wind_speed': float(row['wind_speed']),
                'hour': 12.0, 
                'weekday': float(row['datetime'].weekday()), 
                'month': float(row['datetime'].month),
                'aqi_lag_1': float(current_aqi_lag), 
                'pm2_5_rolling_6h': float(last_pm25), 
                'wind_stagnant': 1.0 if row['wind_speed'] < 2.0 else 0.0
            }])
            
            try:
                pred = model.predict(feat)[0]
                aqi_val = int(np.clip(round(pred), 1, 5))
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
            except Exception as pred_err:
                st.error(f"Forecast error: {pred_err}")
    else:
        st.info("Weather forecast data is currently insufficient for predictions.")

    # Sidebar: Analytics
    with st.sidebar:
        st.title("🔬 Analytics")
        st.success(f"📌 Model: **v{best_meta.version}**")
        if best_meta.training_metrics:
            st.metric("R² Accuracy", f"{best_meta.training_metrics.get('r2', 0):.4f}")
        
        if hasattr(model, 'feature_importances_'):
            st.write("---")
            st.subheader("💡 Drivers")
            feats = ['Temp', 'Humid', 'Wind', 'Hour', 'Weekday', 'Month', 'Lag AQI', 'PM2.5', 'Stagnant']
            imp_series = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(5)
            st.bar_chart(imp_series)

except Exception as e:
    st.error(f"Global Application Error: {e}")

st.markdown('<div class="footer">Islamabad AQI Dashboard • MLOps System • Hopsworks 4.2</div>', unsafe_allow_html=True)