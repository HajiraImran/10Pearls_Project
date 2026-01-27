import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly, Daily # Daily added for forecast
from dotenv import load_dotenv

load_dotenv()

def fetch_weather_history(days=120):
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    now_time = datetime.now() 
    start_time = now_time - timedelta(days=days)
    
    data = Hourly(location, start_time, now_time)
    df = data.fetch()
    
    if df.empty: return pd.DataFrame()
    
    available_cols = [c for c in ['temp', 'rhum', 'wspd'] if c in df.columns]
    df = df[available_cols]
    df.reset_index(inplace=True)
    df.rename(columns={'time': 'datetime', 'temp': 'temperature', 'rhum': 'humidity', 'wspd': 'wind_speed'}, inplace=True)
    
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    return df

# --- NEXT 3 DAYS FORECAST FUNCTION ---
def fetch_weather_forecast(days=3):
    """Meteostat se aglay 3 din ka predicted weather lata hai."""
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    
    # Forecast starts from today
    start = datetime.now()
    end = start + timedelta(days=days)
    
    # Hourly forecast (Meteostat models predict ahead)
    data = Hourly(location, start, end)
    df = data.fetch()
    
    if df.empty:
        return pd.DataFrame()
    
    df.reset_index(inplace=True)
    # Humein rozana ki aik average reading chahiye prediction ke liye
    df['datetime'] = pd.to_datetime(df['time'])
    
    # Rozana ki averages nikalna
    forecast_daily = df.resample('D', on='datetime').agg({
        'temp': 'mean',
        'rhum': 'mean',
        'wspd': 'mean'
    }).reset_index()
    
    forecast_daily.rename(columns={
        'temp': 'temperature', 
        'rhum': 'humidity', 
        'wspd': 'wind_speed'
    }, inplace=True)
    
    return forecast_daily.head(days)

def fetch_raw_pollution(days=120):
    API_KEY = os.getenv("OPENWEATHER_KEY")
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    now_utc = datetime.now(timezone.utc)
    end_ts = int(now_utc.timestamp())
    start_ts = end_ts - (days * 86400)
    
    params = {"lat": 33.72, "lon": 73.04, "start": start_ts, "end": end_ts, "appid": API_KEY}
    res = requests.get(url, params=params).json()
    
    if 'list' not in res: return pd.DataFrame()

    data_list = []
    for entry in res['list']:
        data_list.append({
            "datetime": datetime.fromtimestamp(entry['dt'], tz=timezone.utc),
            "city": "Islamabad",
            "aqi": float(entry['main']['aqi']),
            "pm2_5": float(entry['components']['pm2_5']),
            "no2": float(entry['components']['no2']),
            "so2": float(entry['components']['so2'])
        })
    return pd.DataFrame(data_list)

def clean_and_merge(pol_df, wea_df):
    if pol_df.empty or wea_df.empty: return pd.DataFrame()
    pol_df = pol_df.sort_values('datetime')
    wea_df = wea_df.sort_values('datetime')
    combined = pd.merge_asof(pol_df, wea_df, on='datetime', direction='nearest')
    return combined

def apply_feature_engineering(df):
    if df.empty: return df
    df = df.sort_values("datetime").drop_duplicates(subset=['datetime'])
    
    df['hour'] = df['datetime'].dt.hour.astype(float)
    df['weekday'] = df['datetime'].dt.weekday.astype(float)
    df['month'] = df['datetime'].dt.month.astype(float)
    
    if 'aqi' in df.columns:
        df['aqi_lag_1'] = df['aqi'].shift(1)
    if 'pm2_5' in df.columns:
        df['pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(window=6, min_periods=1).mean()
    
    if 'wind_speed' in df.columns:
        df['wind_stagnant'] = (df['wind_speed'] < 2.0).astype(float)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df.fillna(0)