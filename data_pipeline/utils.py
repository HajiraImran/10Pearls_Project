import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly
from dotenv import load_dotenv

load_dotenv()

def fetch_weather_forecast(days=3):
    """Meteostat se aglay 3 din ka predicted weather lata hai."""
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    
    start = datetime.now()
    end = start + timedelta(days=days)
    
    try:
        data = Hourly(location, start, end)
        df = data.fetch()
        
        # Agar Meteostat data na de (API Down/Empty), toh fallback use karein
        if df.empty:
            dates = [start + timedelta(days=i) for i in range(days + 1)]
            return pd.DataFrame({
                'datetime': dates,
                'temperature': [20.0] * len(dates), 
                'humidity': [50.0] * len(dates),
                'wind_speed': [5.0] * len(dates)
            })

        # ✅ Fix: Create a proper copy to avoid SettingWithCopyWarning
        df = df.reset_index().copy()
        df.rename(columns={'time': 'datetime', 'temp': 'temperature', 'rhum': 'humidity', 'wspd': 'wind_speed'}, inplace=True)

        # Rozana ki averages nikalna
        forecast_daily = df.resample('D', on='datetime').agg({
            'temperature': 'max',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        
        return forecast_daily.head(days + 1)
    except Exception as e:
        print(f"Forecast Error: {e}")
        return pd.DataFrame()

def fetch_historical_aqi_data(fs, num_days=7):
    """Hopsworks se pichle 7 din ka data nikal kar daily average deta hai."""
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        # ✅ Fix: Pure dataframe ki copy banayein
        query_df = fg.read().copy()
        
        # ✅ Fix: .loc use karein column update karne ke liye
        query_df.loc[:, 'datetime'] = pd.to_datetime(query_df['datetime'])
        
        if query_df['datetime'].dt.tz is None:
            query_df.loc[:, 'datetime'] = query_df['datetime'].dt.tz_localize('UTC')
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=num_days)
        
        # Filter data
        historical_df = query_df[query_df['datetime'] >= start_date].copy()
        historical_df.loc[:, 'date_only'] = historical_df['datetime'].dt.date
        
        # Grouping
        daily_avg = historical_df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={'date_only': 'Date', 'aqi': 'Average AQI'}, inplace=True)
        
        return daily_avg.sort_values('Date')
        
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def apply_feature_engineering(df):
    """Training aur Pipepline dono ke liye features tyar karna."""
    if df.empty: return df
    df = df.sort_values("datetime").drop_duplicates(subset=['datetime']).copy()
    
    df.loc[:, 'hour'] = df['datetime'].dt.hour.astype(float)
    df.loc[:, 'weekday'] = df['datetime'].dt.weekday.astype(float)
    df.loc[:, 'month'] = df['datetime'].dt.month.astype(float)
    
    if 'aqi' in df.columns:
        df.loc[:, 'aqi_lag_1'] = df['aqi'].shift(1)
    if 'pm2_5' in df.columns:
        df.loc[:, 'pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(window=6, min_periods=1).mean()
    
    if 'wind_speed' in df.columns:
        df.loc[:, 'wind_stagnant'] = (df['wind_speed'] < 2.5).astype(float)
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = df[numeric_cols].ffill().bfill()
    return df.fillna(0)