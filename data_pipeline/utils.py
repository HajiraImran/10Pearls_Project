import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def fetch_weather_forecast(days=4):
    """Stable Open-Meteo API for Cloud"""
    lat, lon = 33.72, 73.04
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max&timezone=Asia/Karachi"
        res = requests.get(url, timeout=15).json()
        
        forecast_data = []
        for i in range(days):
            forecast_data.append({
                'datetime': pd.to_datetime(res['daily']['time'][i]),
                'temperature': res['daily']['temperature_2m_mean'][i],
                'humidity': res['daily']['relative_humidity_2m_mean'][i],
                'wind_speed': res['daily']['wind_speed_10m_max'][i]
            })
        return pd.DataFrame(forecast_data)
    except Exception as e:
        print(f"Weather Forecast Error: {e}")
        # Deterministic fallback if API fails
        dates = [datetime.now() + timedelta(days=i) for i in range(days)]
        return pd.DataFrame({'datetime': dates, 'temperature': [18.0]*days, 'humidity': [55.0]*days, 'wind_speed': [5.0]*days})

def fetch_historical_aqi_data(fs, num_days=7):
    """Fetch using Hive to prevent Binder/DuckDB errors on Streamlit Cloud"""
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        # CRITICAL: use_hive is mandatory for Streamlit Cloud stability
        query_df = fg.read(read_options={"use_hive": True})
        
        if query_df.empty:
            return pd.DataFrame()

        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        if query_df['datetime'].dt.tz is None:
            query_df['datetime'] = query_df['datetime'].dt.tz_localize('UTC')
        
        start_date = datetime.now(timezone.utc) - timedelta(days=num_days)
        historical_df = query_df[query_df['datetime'] >= start_date].copy()
        
        historical_df['date_only'] = historical_df['datetime'].dt.date
        daily_avg = historical_df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={'date_only': 'Date', 'aqi': 'Average AQI'}, inplace=True)
        return daily_avg.sort_values('Date')
    except Exception as e:
        print(f"Historical Data Error: {e}")
        return pd.DataFrame()