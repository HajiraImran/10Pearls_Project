import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly, Stations
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def fetch_weather_forecast(days=4):
    lat, lon = 33.72, 73.04
    weather_key = os.getenv("WEATHER_API_KEY")
    
    # ATTEMPT 1: WeatherAPI.com
    if weather_key:
        try:
            url = f"http://api.weatherapi.com/v1/forecast.json?key={weather_key}&q={lat},{lon}&days={days}&aqi=no"
            response = requests.get(url, timeout=10)
            data = response.json()
            forecast_data = []
            for day in data['forecast']['forecastday']:
                forecast_data.append({
                    'datetime': pd.to_datetime(day['date']),
                    'temperature': float(day['day']['avgtemp_c']),
                    'humidity': float(day['day']['avghumidity']),
                    'wind_speed': float(day['day']['maxwind_kph'])
                })
            return pd.DataFrame(forecast_data)
        except: pass

    # ATTEMPT 2: Open-Meteo
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&forecast_days={days}&timezone=Asia/Karachi"
        res = requests.get(url).json()
        df = pd.DataFrame({'datetime': pd.to_datetime(res['hourly']['time']), 'temperature': res['hourly']['temperature_2m'], 'humidity': res['hourly']['relative_humidity_2m'], 'wind_speed': res['hourly']['wind_speed_10m']})
        return df.resample('D', on='datetime').mean().reset_index()
    except:
        return _generate_fallback_forecast(days)

def _generate_fallback_forecast(days=4):
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    return pd.DataFrame({'datetime': dates, 'temperature': [18.0]*days, 'humidity': [55.0]*days, 'wind_speed': [5.0]*days})

def fetch_historical_aqi_data(fs, num_days=7):
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        # CRITICAL FIX: Added use_hive for Hopsworks 4.2
        query_df = fg.read(read_options={"use_hive": True})
        
        if query_df.empty: return pd.DataFrame()

        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        start_date = datetime.now(timezone.utc) - timedelta(days=num_days)
        
        # Ensure timezone consistency
        if query_df['datetime'].dt.tz is None:
            query_df['datetime'] = query_df['datetime'].dt.tz_localize('UTC')
            
        hist_df = query_df[query_df['datetime'] >= start_date].copy()
        hist_df['Date'] = hist_df['datetime'].dt.date
        return hist_df.groupby('Date')['aqi'].mean().reset_index().rename(columns={'aqi': 'Average AQI'})
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()