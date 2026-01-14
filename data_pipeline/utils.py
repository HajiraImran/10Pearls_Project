import os
import requests
import pandas as pd
import datetime
from datetime import timezone
from dotenv import load_dotenv

load_dotenv()

def fetch_raw_data(days=120):
    """External API raw weather aur pollutant data."""
    API_KEY = os.getenv("OPENWEATHER_KEY")
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    end = int(datetime.datetime.now(timezone.utc).timestamp())
    start = end - (days * 86400)
    
    params = {"lat": 33.72, "lon": 73.04, "start": start, "end": end, "appid": API_KEY}
    res = requests.get(url, params=params).json()
    
    data_list = []
    for entry in res['list']:
        dt = datetime.datetime.fromtimestamp(entry['dt'], tz=timezone.utc)
        data_list.append({
            "datetime": dt,
            "city": "Islamabad",
            "aqi": float(entry['main']['aqi']),
            "pm2_5": float(entry['components']['pm2_5'])
        })
    return pd.DataFrame(data_list)

def apply_feature_engineering(df):
    """ from Raw data features compute."""
    df = df.sort_values("datetime").drop_duplicates(subset=['datetime'])
    
    # 1. Time-based Features
    df['hour'] = df['datetime'].dt.hour.astype(float)
    df['weekday'] = df['datetime'].dt.weekday.astype(float)
    df['month'] = df['datetime'].dt.month.astype(float)
    
    # 2. Derived/Lag Features (Derived from raw data)
    # Previous hour's AQI (Trend analysis)
    df['aqi_lag_1'] = df['aqi'].shift(1).fillna(df['aqi'])
    
    # 3. Rolling Features (Derived for smoothing)
    # 6-hour rolling average for PM2.5
    df['pm2_5_rolling_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
    
    # 4. AQI Change Rate
    df['aqi_change_rate'] = df['aqi'].diff().fillna(0)
    
    return df