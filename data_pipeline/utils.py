import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly
import numpy as np

# Try Streamlit secrets if available (cloud), fallback to .env locally
try:
    import streamlit as st
    HOPSWORKS_KEY = st.secrets.get("HOPSWORKS_KEY")
    OPENWEATHER_KEY = st.secrets.get("OPENWEATHER_KEY")
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    HOPSWORKS_KEY = os.getenv("HOPSWORKS_KEY")
    OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")


def fetch_weather_history(days=120):
    """Fetch historical weather from Meteostat"""
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    now_time = datetime.now()
    start_time = now_time - timedelta(days=days)

    data = Hourly(location, start_time, now_time)
    df = data.fetch()

    if df.empty:
        return pd.DataFrame()

    available_cols = [c for c in ['temp', 'rhum', 'wspd'] if c in df.columns]
    df = df[available_cols]
    df.reset_index(inplace=True)
    df.rename(columns={
        'time': 'datetime',
        'temp': 'temperature',
        'rhum': 'humidity',
        'wspd': 'wind_speed'
    }, inplace=True)

    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    return df


def fetch_weather_forecast(days=3):
    """Fetch 3-day weather forecast from Meteostat (daily avg)"""
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)

    start = datetime.now()
    end = start + timedelta(days=days)

    data = Hourly(location, start, end)
    df = data.fetch()
    if df.empty:
        # fallback deterministic forecast
        dates = [start + timedelta(days=i) for i in range(days)]
        return pd.DataFrame({
            'datetime': dates,
            'temperature': [20 + i*0.5 for i in range(days)],
            'humidity': [50 - i for i in range(days)],
            'wind_speed': [5 + (-1)**i for i in range(days)]
        })

    df.reset_index(inplace=True)
    df['datetime'] = pd.to_datetime(df['time'])

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


def fetch_raw_pollution(days=120, api_key=None):
    """Fetch historical pollution from OpenWeather"""
    if api_key is None:
        api_key = OPENWEATHER_KEY
    if not api_key:
        print("⚠️ No OpenWeather API key provided")
        return pd.DataFrame()

    lat, lon = 33.72, 73.04
    now_utc = datetime.now(timezone.utc)
    end_ts = int(now_utc.timestamp())
    start_ts = end_ts - (days * 86400)

    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {"lat": lat, "lon": lon, "start": start_ts, "end": end_ts, "appid": api_key}

    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"⚠️ Pollution API error: {e}")
        return pd.DataFrame()

    if 'list' not in data or not data['list']:
        return pd.DataFrame()

    records = []
    for entry in data['list']:
        records.append({
            "datetime": datetime.fromtimestamp(entry['dt'], tz=timezone.utc),
            "city": "Islamabad",
            "aqi": float(entry['main']['aqi']),
            "pm2_5": float(entry['components']['pm2_5']),
            "no2": float(entry['components']['no2']),
            "so2": float(entry['components']['so2'])
        })

    return pd.DataFrame(records)


def clean_and_merge(pol_df, wea_df):
    """Merge pollution and weather data on datetime"""
    if pol_df.empty or wea_df.empty:
        # If no pollution data, return only weather forecast (for cloud fallback)
        return wea_df.copy()
    pol_df = pol_df.sort_values('datetime')
    wea_df = wea_df.sort_values('datetime')
    return pd.merge_asof(pol_df, wea_df, on='datetime', direction='nearest')


def apply_feature_engineering(df, is_forecast=False):
    """Compute features for ML model"""
    if df.empty:
        return df

    df = df.sort_values("datetime").drop_duplicates(subset=['datetime'])
    df['hour'] = df['datetime'].dt.hour.astype(float)
    df['weekday'] = df['datetime'].dt.weekday.astype(float)
    df['month'] = df['datetime'].dt.month.astype(float)

    # Lag features only if not forecast
    if not is_forecast:
        if 'aqi' in df.columns:
            df['aqi_lag_1'] = df['aqi'].shift(1)
        if 'pm2_5' in df.columns:
            df['pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(window=6, min_periods=1).mean()
    else:
        # Forecast mode: fill lags with zeros
        df['aqi_lag_1'] = 0
        if 'pm2_5' in df.columns:
            df['pm2_5_rolling_6h'] = df['pm2_5'].fillna(0)

    if 'wind_speed' in df.columns:
        df['wind_stagnant'] = (df['wind_speed'] < 2.0).astype(float)

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df.fillna(0)
