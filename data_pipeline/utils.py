import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly
from dotenv import load_dotenv

load_dotenv()

# --- WEATHER HISTORY ---
def fetch_weather_history(days=120):
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    now_time = datetime.now(timezone.utc)
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

# --- NEXT 3 DAYS FORECAST ---
def fetch_weather_forecast(days=3):
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days)
    
    data = Hourly(location, start, end)
    df = data.fetch()
    if df.empty: return pd.DataFrame()
    
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

# --- HISTORICAL AQI ---
def fetch_historical_aqi_data(fs, num_days=7):
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        query_df = fg.read()
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=num_days)
        
        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        if query_df['datetime'].dt.tz is None:
            query_df['datetime'] = query_df['datetime'].dt.tz_localize('UTC')
        
        historical_df = query_df[query_df['datetime'] >= start_date]
        historical_df['date_only'] = historical_df['datetime'].dt.date
        daily_avg = historical_df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={'date_only': 'Date', 'aqi': 'Average AQI'}, inplace=True)
        return daily_avg.sort_values('Date')
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()
