import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly
from dotenv import load_dotenv

load_dotenv()

# --- FORECAST FIX (The reason for '1') ---
def fetch_weather_forecast(days=3):
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    
    # Current time se lekar 3 din aagay tak
    start = datetime.now()
    end = start + timedelta(days=days)
    
    try:
        data = Hourly(location, start, end)
        df = data.fetch()
        
        if df.empty:
            # Fallback: Agar Meteostat data na de toh Islamabad ka average weather use karein
            # Taake model 0 values dekh kar AQI 1 na predict kare
            print("⚠️ Meteostat empty, using seasonal fallbacks")
            dates = [start + timedelta(days=i) for i in range(days + 1)]
            return pd.DataFrame({
                'datetime': dates,
                'temperature': [18.0] * len(dates), 
                'humidity': [50.0] * len(dates),
                'wind_speed': [5.0] * len(dates)
            })

        df.reset_index(inplace=True)
        df.rename(columns={'time': 'datetime', 'temp': 'temperature', 'rhum': 'humidity', 'wspd': 'wind_speed'}, inplace=True)

        # Rozana ki Max/Mean ka mix taake model sensitive rahe
        forecast_daily = df.resample('D', on='datetime').agg({
            'temperature': 'max', # Din ka garam waqt AQI ko affect karta hai
            'humidity': 'mean',
            'wind_speed': 'min'  # Sabse kam hawa (stagnation) AQI kharab karti hai
        }).reset_index()
        
        return forecast_daily.head(days + 1)
    except Exception as e:
        print(f"Forecast Error: {e}")
        return pd.DataFrame()

# --- FEATURE ENGINEERING SYNC ---
def apply_feature_engineering(df):
    if df.empty: return df
    df = df.sort_values("datetime").drop_duplicates(subset=['datetime'])
    
    df['hour'] = df['datetime'].dt.hour.astype(float)
    df['weekday'] = df['datetime'].dt.weekday.astype(float)
    df['month'] = df['datetime'].dt.month.astype(float)
    
    # Lags logic
    if 'aqi' in df.columns:
        df['aqi_lag_1'] = df['aqi'].shift(1)
    
    if 'pm2_5' in df.columns:
        # Rolling mean for 6 hours
        df['pm2_5_rolling_6h'] = df['pm2_5'].rolling(window=6, min_periods=1).mean()
    
    # Wind Stagnation: Islamabad context mein 2.5-3.0 km/h se kam hawa zeher hai
    if 'wind_speed' in df.columns:
        df['wind_stagnant'] = (df['wind_speed'] < 2.5).astype(float)
    
    # Fill gaps taake model ko 0 na miley (0 = AQI 1 error)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df