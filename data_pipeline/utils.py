import os
import requests
import pandas as pd
import datetime
from datetime import timezone
from dotenv import load_dotenv

load_dotenv()

def fetch_raw_data(days=120):
    API_KEY = os.getenv("OPENWEATHER_KEY")
    # Historical data API use kar rahe hain
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    end = int(datetime.datetime.now(timezone.utc).timestamp())
    start = end - (days * 86400)
    
    params = {"lat": 33.72, "lon": 73.04, "start": start, "end": end, "appid": API_KEY}
    response = requests.get(url, params=params)
    res = response.json()
    
    # Check if 'list' exists in response to avoid KeyError
    if 'list' not in res:
        print(f"⚠️ API Error: {res}")
        return pd.DataFrame()

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
    if df.empty: return df
    
    df = df.sort_values("datetime").drop_duplicates(subset=['datetime'])
    
    # Time Features
    df['hour'] = df['datetime'].dt.hour.astype(float)
    df['weekday'] = df['datetime'].dt.weekday.astype(float)
    df['month'] = df['datetime'].dt.month.astype(float)
    
    # Lag & Rolling Features
    df['aqi_lag_1'] = df['aqi'].shift(1).fillna(df['aqi'])
    df['pm2_5_rolling_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
    df['aqi_change_rate'] = df['aqi'].diff().fillna(0)
    
    return df