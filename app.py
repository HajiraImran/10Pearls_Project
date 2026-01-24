import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime
import joblib
import hopsworks
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def load_resources():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Dynamic Model Loading
    model_obj = mr.get_model("islamabad_aqi_randomforest", version=15) 
    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    
    return model, fs, model_obj

try:
    st.set_page_config(page_title="Islamabad AQI Predictor", page_icon="🇵🇰", layout="wide")
    st.title("🇵🇰 Islamabad Real-Time 3-Day AQI Forecast")
    st.markdown("---")
    
    model, fs, model_obj = load_resources()
    
    # 1. Fetch latest data
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=1)
    batch_data = fg.read()
    latest_record = batch_data.sort_values(by='datetime').iloc[-1]
    
    current_pm25 = float(latest_record['pm2_5'])
    current_aqi_raw = float(latest_record['aqi'])

    # 2. Forecast API
    API_KEY = os.getenv("OPENWEATHER_KEY")
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat=33.72&lon=73.04&appid={API_KEY}&units=metric"
    forecast_res = requests.get(url).json()

    # 3. Features & Predictions
    features_order = ['pm2_5', 'hour', 'weekday', 'month', 'aqi_lag_1', 'pm2_5_rolling_6h', 'aqi_change_rate']
    forecast_list = []
    daily_indices = [8, 16, 24] 

    for i in daily_indices:
        day_data = forecast_res['list'][i]
        future_dt = datetime.fromtimestamp(day_data['dt'])
        
        input_dict = {
            'pm2_5': current_pm25, 
            'hour': float(future_dt.hour),
            'weekday': float(future_dt.weekday()),
            'month': float(future_dt.month),
            'aqi_lag_1': current_aqi_raw,
            'pm2_5_rolling_6h': float(latest_record['pm2_5_rolling_6h']),
            'aqi_change_rate': float(latest_record['aqi_change_rate'])
        }

        df_input = pd.DataFrame([input_dict])[features_order]
        prediction_index = model.predict(df_input)[0] 
        forecast_list.append({"Date": future_dt.strftime('%A (%d %b)'), "AQI_Index": round(prediction_index, 2)})

    # Layout for Analysis
    col_graph, col_shap = st.columns([2, 1])

    with col_graph:
        st.subheader("📈 3-Day AQI Index Trend")
        df_forecast = pd.DataFrame(forecast_list)
        fig = px.line(df_forecast, x="Date", y="AQI_Index", markers=True, range_y=[0,6])
        st.plotly_chart(fig, use_container_width=True)

    with col_shap:
        st.subheader("🔍 Feature Analysis (SHAP)")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': features_order, 'Weight': importances}).sort_values(by='Weight')
        fig_importance = px.bar(feat_df, x='Weight', y='Feature', orientation='h', color='Weight', color_continuous_scale='Reds')
        st.plotly_chart(fig_importance, use_container_width=True)

    # Explanation Section for Mentor
    with st.expander("📝 What do these features mean?"):
        st.write("""
        * **PM2.5:** Current fine particulate matter concentration.
        * **AQI Lag 1:** The previous hour's AQI value (Predicts continuity).
        * **Rolling 6h:** Average pollution over the last 6 hours (Predicts trends).
        * **AQI Change Rate:** How fast the pollution is rising or falling.
        """)

    st.markdown("---")

    # 4. Results Cards
    st.subheader("🚀 Forecast Details")
    cols = st.columns(3)
    for idx, row in df_forecast.iterrows():
        display_val = int(round(row["AQI_Index"]))
        with cols[idx]:
            st.metric(label=row["Date"], value=f"Index: {display_val}/5")
            if display_val >= 5: st.error("🛑 **Hazardous**")
            elif display_val == 4: st.error("🔴 **Poor**")
            elif display_val == 3: st.warning("🟠 **Moderate**")
            else: st.success("🍃 **Good**")

    # Sidebar Metrics
    st.sidebar.markdown(f"### 📡 Live Sensor Status")
    st.sidebar.metric("Live PM2.5", f"{current_pm25} µg/m³")
    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 Dynamic Model Metrics")
    st.sidebar.write(f"Model: **Random Forest (v{model_obj.version})**")
    
    r2 = model_obj.training_metrics.get('r2', 0) * 100
    rmse = model_obj.training_metrics.get('rmse', 0)
    st.sidebar.write(f"Reliability (R2): **{r2:.2f}%**")
    st.sidebar.write(f"Error (RMSE): **{rmse:.4f}**")

except Exception as e:
    st.error(f"❌ Error: {e}")