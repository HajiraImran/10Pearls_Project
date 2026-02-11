import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly
from dotenv import load_dotenv

load_dotenv()

def fetch_weather_forecast(days=3):
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    start = datetime.now()
    end = start + timedelta(days=days)
    
    try:
        data = Hourly(location, start, end)
        df = data.fetch()
        
        if df.empty:
            # Fallback data taake app crash na kare
            dates = [start + timedelta(days=i) for i in range(days + 1)]
            return pd.DataFrame({
                'datetime': dates,
                'temperature': [18.0]*len(dates),
                'humidity': [50.0]*len(dates),
                'wind_speed': [5.0]*len(dates)
            })

        df = df.reset_index().copy()
        df.rename(columns={'time': 'datetime', 'temp': 'temperature', 'rhum': 'humidity', 'wspd': 'wind_speed'}, inplace=True)
        
        # Aggregation logic
        forecast_daily = df.resample('D', on='datetime').agg({
            'temperature': 'mean', 'humidity': 'mean', 'wind_speed': 'mean'
        }).reset_index()
        
        return forecast_daily.head(days + 1)
    except:
        return pd.DataFrame()

def fetch_historical_aqi_data(fs, num_days=7):
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        query_df = fg.read().copy()
        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        
        if query_df['datetime'].dt.tz is None:
            query_df['datetime'] = query_df['datetime'].dt.tz_localize('UTC')
        
        start_date = datetime.now(timezone.utc) - timedelta(days=num_days)
        historical_df = query_df[query_df['datetime'] >= start_date].copy()
        historical_df['date_only'] = historical_df['datetime'].dt.date
        
        daily_avg = historical_df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={'date_only': 'Date', 'aqi': 'Average AQI'}, inplace=True)
        return daily_avg.sort_values('Date')
    except:
        return pd.DataFrame()