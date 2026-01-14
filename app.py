import streamlit as st
import hopsworks
import joblib
import pandas as pd
import datetime
import os

st.title("🌍 Islamabad AQI Predictor (3-Day Forecast)")

# 1. Connect to Hopsworks
@st.cache_resource
def get_model():
    project = hopsworks.login(api_key_value="mnDlPOrqaHLxdkdM.wlO02bSDougG2LJpZOVNWbKXlkZvvQ60HGDjlP5AJb1Dmokdkp5BbJAfXW3wE1ov")
    mr = project.get_model_registry()
    model_meta = mr.get_model("aqi_randomforest", version=1)
    model_dir = model_meta.download()
    return joblib.load(os.path.join(model_dir, "model.pkl")), project.get_feature_store()

model, fs = get_model()

# 2. Fetch Latest Data
aqi_fg = fs.get_feature_group(name="islamabad_aqi_v10", version=1)
latest_df = aqi_fg.read().sort_values("datetime").tail(1)

# 3. Compute 3-Day Prediction
current_aqi = float(latest_df['aqi'].values[0])
current_pm25 = float(latest_df['pm2_5_rolling_6h'].values[0])
forecast = []

temp_aqi = current_aqi
for i in range(1, 73):
    future_time = datetime.datetime.now() + datetime.timedelta(hours=i)
    input_df = pd.DataFrame([{'hour': float(future_time.hour), 'weekday': float(future_time.weekday()), 
                              'aqi_lag_1': temp_aqi, 'pm2_5_rolling_6h': current_pm25}])
    pred = model.predict(input_df)[0]
    forecast.append({"Time": future_time, "AQI": round(pred, 2)})
    temp_aqi = pred

# 4. Display Dashboard
df_forecast = pd.DataFrame(forecast)
st.metric("Current AQI (Estimated)", f"{current_aqi}")
st.line_chart(df_forecast.set_index("Time"))
st.write("Next 72 Hours Forecast:", df_forecast)