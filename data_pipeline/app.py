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
    try:
        # Login to Hopsworks
        project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
        mr = project.get_model_registry()
        fs = project.get_feature_store()
        
        # Get latest model version
        best_models_list = mr.get_models("best_islamabad_aqi_model")
        if not best_models_list:
            st.error("Model 'best_islamabad_aqi_model' not found in Registry!")
            st.stop()
        
        # Sort and pick the max version
        model_meta = max(best_models_list, key=lambda m: m.version)
        
        # Download and load model
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "best_model.pkl"))
        
        # Load benchmark models
        model_names = ["islamabad_aqi_randomforest", "islamabad_aqi_xgboost", "islamabad_aqi_gradientboosting"]
        all_models_meta = []
        for name in model_names:
            try:
                versions = mr.get_models(name)
                if versions:
                    all_models_meta.append(max(versions, key=lambda v: v.version))
            except: 
                pass
                
        return model, model_meta, all_models_meta, fs
    except Exception as e:
        st.error(f"Failed to load assets: {e}")
        st.stop()

try:
    model, best_meta, all_models, fs = load_assets()
    
    # 1. Header Section
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.title("🌬️ Islamabad Air Quality Index")
        st.markdown(f"🛰️ **Live Monitoring:** {datetime.now().strftime('%A, %d %b %Y | %I:%M %p')}")
    with col_t2:
        if os.path.exists("Islamabad.jpg"):
            st.image("Islamabad.jpg", width=250)

    st.markdown("---")

    # 2. Fetch Latest Data from Feature Store
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
    latest_df = fg.read().sort_values("datetime", ascending=False).head(1)
    
    # Ensure proper data extraction
    if latest_df.empty:
        st.error("❌ No data found in feature store!")
        st.stop()
    
    last_aqi = float(latest_df['aqi'].iloc[0])
    last_pm25 = float(latest_df['pm2_5_rolling_6h'].iloc[0])
    
    # 3. Fetch Weather Forecast
    forecast_weather = fetch_weather_forecast(days=4)
    
    # 🐛 WEATHER FORECAST DEBUG
    with st.expander("🌤️ Weather Forecast Debug", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Forecast Data Quality:**")
            st.write(f"- Rows: {len(forecast_weather)}")
            st.write(f"- Columns: {list(forecast_weather.columns)}")
            if len(forecast_weather) > 0:
                st.write(f"- Temp Range: {forecast_weather['temperature'].min():.1f}°C - {forecast_weather['temperature'].max():.1f}°C")
                st.write(f"- Humidity Range: {forecast_weather['humidity'].min():.1f}% - {forecast_weather['humidity'].max():.1f}%")
                st.write(f"- Wind Range: {forecast_weather['wind_speed'].min():.1f} - {forecast_weather['wind_speed'].max():.1f} km/h")
        with col2:
            st.write("**Complete Forecast Data:**")
            st.dataframe(forecast_weather, use_container_width=True)
    
    if forecast_weather.empty or len(forecast_weather) < 4:
        st.error("❌ Weather forecast data is currently insufficient for predictions.")
        st.stop()

    # 4. Key Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Current AQI", int(last_aqi), delta_color="inverse")
    with m2:
        st.metric("Temperature", f"{forecast_weather.iloc[0]['temperature']:.1f}°C")
    with m3:
        st.metric("Humidity", f"{forecast_weather.iloc[0]['humidity']:.1f}%")
    with m4:
        st.metric("PM2.5 (Rolling)", f"{last_pm25:.1f} µg/m³")

    # 5. Graph Section
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
        ).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

    # 6. Forecast Section with ALERTS
    st.markdown("---")
    st.subheader("📅 Smart Forecast & Health Alerts")
    
    # 🐛 SYSTEM DIAGNOSTICS
    with st.expander("🔬 System Diagnostics", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Information:**")
            st.write(f"- Version: {best_meta.version}")
            st.write(f"- R² Score: {best_meta.training_metrics.get('r2', 0):.4f}")
            st.write(f"- Model Type: {type(model).__name__}")
            if hasattr(model, 'feature_names_in_'):
                st.write(f"- Features: {len(model.feature_names_in_)}")
            
        with col2:
            st.write("**Initial Conditions:**")
            st.write(f"- Current AQI: {last_aqi:.2f}")
            st.write(f"- Current PM2.5: {last_pm25:.2f}")
            st.write(f"- Latest Time: {latest_df['datetime'].iloc[0]}")
        
        st.write("**Feature Store Latest Record:**")
        st.dataframe(latest_df.head(1))
    
    # Define feature order (CRITICAL)
    FEATURE_ORDER = [
        'temperature', 'humidity', 'wind_speed', 
        'hour', 'weekday', 'month',
        'aqi_lag_1', 'pm2_5_rolling_6h', 'wind_stagnant'
    ]
    
    f_cols = st.columns(3)
    future_df = forecast_weather.iloc[1:4].reset_index(drop=True)
    
    # Initialize lagged features
    current_aqi_lag = last_aqi
    current_pm25 = last_pm25
    
    status_map = {
        1: ("Good", "#00cc96", "🌿"), 
        2: ("Fair", "#fec032", "🌳"), 
        3: ("Moderate", "#ffa15a", "😷"), 
        4: ("Poor", "#ef553b", "⚠️"), 
        5: ("Hazardous", "#ab63fa", "🚨")
    }

    predictions = []  # Debug storage

    for i, row in future_df.iterrows():
        # Calculate time features
        forecast_datetime = row['datetime']
        hour_of_day = 12.0  # Noon prediction
        weekday = float(forecast_datetime.weekday())
        month = float(forecast_datetime.month)
        
        # Build feature DataFrame
        feat = pd.DataFrame([{
            'temperature': float(row['temperature']),
            'humidity': float(row['humidity']),
            'wind_speed': float(row['wind_speed']),
            'hour': hour_of_day,
            'weekday': weekday,
            'month': month,
            'aqi_lag_1': float(current_aqi_lag),
            'pm2_5_rolling_6h': float(current_pm25),
            'wind_stagnant': 1.0 if row['wind_speed'] < 2.0 else 0.0
        }])
        
        # Enforce feature order
        feat = feat[FEATURE_ORDER]
        
        # Make prediction
        pred = model.predict(feat)[0]
        aqi_val = int(np.clip(round(pred), 1, 5))
        
        # Store debug info
        predictions.append({
            'day': i + 1,
            'date': forecast_datetime.strftime('%A, %d %b'),
            'temp': float(row['temperature']),
            'humidity': float(row['humidity']),
            'wind': float(row['wind_speed']),
            'aqi_lag_in': float(current_aqi_lag),
            'pm25_in': float(current_pm25),
            'pred_raw': float(pred),
            'aqi_final': aqi_val
        })
        
        # 🔧 UPDATE LAG FEATURES FOR NEXT ITERATION
        # Use raw prediction (not clipped integer)
        current_aqi_lag = pred
        
        # Evolve PM2.5 based on weather conditions and AQI trend
        humidity_factor = row['humidity'] / 100.0
        wind_factor = max(0.3, 1.0 - (row['wind_speed'] / 10.0))
        
        # AQI-PM2.5 correlation
        pm25_drift = (pred - last_aqi) * 10.0
        
        # Weather impact on PM2.5 accumulation
        current_pm25 = current_pm25 * (0.80 + 0.35 * humidity_factor * wind_factor) + pm25_drift
        current_pm25 = np.clip(current_pm25, 5.0, 200.0)
        
        label, color, icon = status_map.get(aqi_val, ("Unknown", "#666", "❓"))

        # 🚨 TRIGGER ALERTS
        if aqi_val >= 4:
            st.toast(f"⚠️ Health Risk: {label} air quality on {forecast_datetime.strftime('%A')}", icon="😷")
            if i == 0: 
                st.error(f"🚨 **ALERT:** Level {aqi_val} AQI predicted for {forecast_datetime.strftime('%A')}. Sensitive groups should limit outdoor activities.")

        # Display forecast card
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

    # 🐛 PREDICTION EVOLUTION DEBUG
    with st.expander("🔍 Prediction Details & Evolution", expanded=False):
        st.write("**How predictions evolve across days:**")
        debug_df = pd.DataFrame(predictions)
        st.dataframe(debug_df, use_container_width=True)
        
        st.write("**Key Observations:**")
        st.write(f"- Day 1 uses historical AQI lag: {last_aqi:.2f}")
        st.write(f"- Day 2 uses Day 1 prediction as lag: {predictions[0]['pred_raw']:.2f}")
        st.write(f"- Day 3 uses Day 2 prediction as lag: {predictions[1]['pred_raw']:.2f}")
        st.write(f"- PM2.5 evolves based on weather + AQI trend")

    # Sidebar: Analytics & Controls
    with st.sidebar:
        st.title("🔬 Analytics")
        
        # Model version info
        st.success(f"📌 Active Model: **v{best_meta.version}**")
        st.metric("R² Accuracy", f"{best_meta.training_metrics.get('r2', 0):.4f}")
        
        # Cache control
        st.write("---")
        if st.button("🔄 Refresh All Data"):
            st.cache_resource.clear()
            st.rerun()
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            st.write("---")
            st.subheader("💡 Top Prediction Drivers")
            feats = ['Temp', 'Humid', 'Wind', 'Hour', 'Weekday', 'Month', 'Lag AQI', 'PM2.5', 'Stagnant']
            imp_series = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False).head(5)
            st.bar_chart(imp_series)

        # Model Benchmark
        if all_models:
            st.write("---")
            st.write("📊 **Algorithm Comparison**")
            comp_df = pd.DataFrame([{
                "Model": m.name.split('_')[-1].upper(), 
                "Ver": m.version, 
                "R²": f"{m.training_metrics.get('r2', 0):.4f}"
            } for m in all_models])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
        
        # System Info
        st.write("---")
        st.caption("💻 System Status")
        st.caption(f"Model: {type(model).__name__}")
        st.caption(f"Features: {len(FEATURE_ORDER)}")
        st.caption(f"Environment: Streamlit Cloud")

except Exception as e:
    st.error(f"❌ Critical System Error")
    st.exception(e)
    
    # Show detailed error for debugging
    import traceback
    with st.expander("🐛 Technical Details"):
        st.code(traceback.format_exc())

st.markdown('<div class="footer">Islamabad AQI Dashboard • Powered by Hopsworks ML • Real-time Predictions</div>', unsafe_allow_html=True)