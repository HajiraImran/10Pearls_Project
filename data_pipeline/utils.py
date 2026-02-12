import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly  # For fallback historical weather
from dotenv import load_dotenv

load_dotenv()

# --- LOCATION ---
LAT, LON = 33.72, 73.04

# --- FETCH WEATHER FORECAST (3 DAYS) ---
def fetch_weather_forecast(days=3):
    """Fetch 3-day weather forecast from OpenWeatherMap."""
    API_KEY = os.getenv("OPENWEATHER_KEY")
    if not API_KEY:
        return pd.DataFrame()
    
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units":"metric"}
    
    try:
        res = requests.get(url, params=params).json()
        if "list" not in res:
            return pd.DataFrame()
        
        # Extract forecast every 24h (roughly midday)
        forecast_list = res['list']
        df_list = []
        seen_dates = set()
        for f in forecast_list:
            dt = datetime.utcfromtimestamp(f['dt'])
            date_only = dt.date()
            if date_only not in seen_dates:
                df_list.append({
                    "datetime": dt,
                    "temperature": f['main']['temp'],
                    "humidity": f['main']['humidity'],
                    "wind_speed": f['wind']['speed']
                })
                seen_dates.add(date_only)
            if len(df_list) >= days:
                break
        
        return pd.DataFrame(df_list)
    
    except Exception as e:
        print(f"Forecast fetch error: {e}")
        return pd.DataFrame()


# --- HISTORICAL WEATHER (Fallback) ---
def fetch_weather_history(days=3):
    """Fetch past N days of weather using Meteostat (fallback)."""
    location = Point(LAT, LON)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days)
    
    try:
        data = Hourly(location, start_time, end_time)
        df = data.fetch()
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.rename(columns={
            "time":"datetime",
            "temp":"temperature",
            "rhum":"humidity",
            "wspd":"wind_speed"
        }, inplace=True)
        
        # Only keep last N days
        df = df.tail(days).reset_index(drop=True)
        return df[['datetime','temperature','humidity','wind_speed']]
    
    except Exception as e:
        print(f"Historical weather fetch error: {e}")
        return pd.DataFrame()


# --- FETCH HISTORICAL AQI DATA FROM HOPSWORKS ---
def fetch_historical_aqi_data(fs, num_days=7):
    """Get past 7 days AQI from Hopsworks Feature Store for chart."""
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        query_df = fg.read()
        if query_df.empty:
            return pd.DataFrame()
        
        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        if query_df['datetime'].dt.tz is None:
            query_df['datetime'] = query_df['datetime'].dt.tz_localize('UTC')
        
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=num_days)
        df = query_df[query_df['datetime'] >= start_date].copy()
        df['date_only'] = df['datetime'].dt.date
        daily_avg = df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={'date_only':'Date','aqi':'Average AQI'}, inplace=True)
        return daily_avg.sort_values('Date')
    
    except Exception as e:
        print(f"Error fetching historical AQI: {e}")
        return pd.DataFrame()
