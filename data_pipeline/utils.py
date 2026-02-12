import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from meteostat import Point, Hourly
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def fetch_weather_history(days=120):
    """Fetch historical weather data from Meteostat"""
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


def fetch_weather_forecast(days=4):
    """
    Fetch weather forecast with multiple API fallbacks
    Priority: WeatherAPI.com → Open-Meteo → Deterministic Fallback
    """
    lat, lon = 33.72, 73.04
    
    # ATTEMPT 1: WeatherAPI.com (most reliable)
    weather_key = os.getenv("WEATHER_API_KEY")
    if weather_key:
        try:
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": weather_key,
                "q": f"{lat},{lon}",
                "days": days,
                "aqi": "no"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast_data = []
            for day in data['forecast']['forecastday']:
                forecast_data.append({
                    'datetime': pd.to_datetime(day['date']),
                    'temperature': float(day['day']['avgtemp_c']),
                    'humidity': float(day['day']['avghumidity']),
                    'wind_speed': float(day['day']['maxwind_kph'])
                })
            
            df = pd.DataFrame(forecast_data)
            print(f"✅ WeatherAPI.com: {len(df)} days fetched")
            return df.head(days)
            
        except Exception as e:
            print(f"⚠️ WeatherAPI.com failed: {e}")
    
    # ATTEMPT 2: Open-Meteo (free, no key needed)
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
            "forecast_days": days,
            "timezone": "Asia/Karachi"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data['hourly']
        df = pd.DataFrame({
            'datetime': pd.to_datetime(hourly['time']),
            'temperature': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'wind_speed': hourly['wind_speed_10m']
        })
        
        # Aggregate to daily
        df['date'] = df['datetime'].dt.date
        daily_forecast = df.groupby('date').agg({
            'datetime': 'first',
            'temperature': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index(drop=True)
        
        daily_forecast['datetime'] = pd.to_datetime(daily_forecast['datetime'])
        print(f"✅ Open-Meteo: {len(daily_forecast)} days fetched")
        return daily_forecast.head(days)
        
    except Exception as e:
        print(f"⚠️ Open-Meteo failed: {e}")
    
    # ATTEMPT 3: Deterministic fallback
    print(f"⚠️ All APIs failed - using deterministic fallback")
    return _generate_fallback_forecast(days)


def _generate_fallback_forecast(days=4):
    """
    Generate deterministic (non-random) weather forecast
    Based on Islamabad seasonal patterns
    """
    current_month = datetime.now().month
    current_day = datetime.now().day
    
    # Islamabad February typical weather
    # Mid-winter transitioning to spring
    base_temp = 15.0  # °C
    base_humidity = 55.0  # %
    base_wind = 5.5  # km/h
    
    # Apply seasonal adjustments
    if current_month == 2:  # February
        base_temp = 14.0 + (current_day / 28.0) * 5.0  # Warming through month
        base_humidity = 60.0 - (current_day / 28.0) * 10.0  # Drying
    elif current_month == 3:  # March
        base_temp = 20.0
        base_humidity = 50.0
    
    dates = [datetime.now() + timedelta(days=i) for i in range(days)]
    
    forecast_data = []
    for i, date in enumerate(dates):
        # Progressive changes (not random)
        temp_trend = i * 0.4  # Slight daily warming
        humidity_trend = -i * 1.5  # Decreasing humidity
        wind_variation = 0.7 * ((-1) ** i)  # Alternating pattern
        
        forecast_data.append({
            'datetime': date,
            'temperature': base_temp + temp_trend,
            'humidity': max(35.0, min(80.0, base_humidity + humidity_trend)),
            'wind_speed': max(2.0, base_wind + wind_variation)
        })
    
    return pd.DataFrame(forecast_data)


def fetch_raw_pollution(days=120):
    """Fetch historical pollution data from OpenWeather API"""
    API_KEY = os.getenv("OPENWEATHER_KEY")
    if not API_KEY:
        print("⚠️ OPENWEATHER_KEY not found")
        return pd.DataFrame()
    
    url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
    now_utc = datetime.now(timezone.utc)
    end_ts = int(now_utc.timestamp())
    start_ts = end_ts - (days * 86400)
    
    params = {
        "lat": 33.72, 
        "lon": 73.04, 
        "start": start_ts, 
        "end": end_ts, 
        "appid": API_KEY
    }
    
    try:
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(f"⚠️ Pollution API error: {e}")
        return pd.DataFrame()
    
    if 'list' not in data: 
        return pd.DataFrame()

    data_list = []
    for entry in data['list']:
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
    """Merge pollution and weather data on datetime"""
    if pol_df.empty or wea_df.empty: 
        return pd.DataFrame()
    
    pol_df = pol_df.sort_values('datetime')
    wea_df = wea_df.sort_values('datetime')
    combined = pd.merge_asof(pol_df, wea_df, on='datetime', direction='nearest')
    
    return combined


def apply_feature_engineering(df):
    """Apply time-based features and rolling statistics"""
    if df.empty: 
        return df
    
    df = df.sort_values("datetime").drop_duplicates(subset=['datetime'])
    
    # Time features
    df['hour'] = df['datetime'].dt.hour.astype(float)
    df['weekday'] = df['datetime'].dt.weekday.astype(float)
    df['month'] = df['datetime'].dt.month.astype(float)
    
    # Lag features
    if 'aqi' in df.columns:
        df['aqi_lag_1'] = df['aqi'].shift(1)
    
    if 'pm2_5' in df.columns:
        df['pm2_5_rolling_6h'] = df['pm2_5'].shift(1).rolling(
            window=6, min_periods=1
        ).mean()
    
    # Wind stagnation
    if 'wind_speed' in df.columns:
        df['wind_stagnant'] = (df['wind_speed'] < 2.0).astype(float)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    return df.fillna(0)


def fetch_historical_aqi_data(fs, num_days=7):
    """
    Fetch historical AQI data from Hopsworks Feature Store
    Returns daily averages for visualization
    """
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        query_df = fg.read()
        
        if query_df.empty:
            print("⚠️ Feature group is empty")
            return pd.DataFrame(columns=['Date', 'Average AQI'])
        
        # Filter for last N days
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=num_days)
        
        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        if query_df['datetime'].dt.tz is None:
            query_df['datetime'] = query_df['datetime'].dt.tz_localize('UTC')
        
        historical_df = query_df[query_df['datetime'] >= start_date].copy()
        
        if historical_df.empty:
            print(f"⚠️ No data in last {num_days} days")
            return pd.DataFrame(columns=['Date', 'Average AQI'])
        
        # Calculate daily averages
        historical_df['date_only'] = historical_df['datetime'].dt.date
        daily_avg = historical_df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={
            'date_only': 'Date', 
            'aqi': 'Average AQI'
        }, inplace=True)
        
        daily_avg['Date'] = pd.to_datetime(daily_avg['Date'])
        
        print(f"✅ Historical data: {len(daily_avg)} days")
        return daily_avg.sort_values('Date')
        
    except Exception as e:
        print(f"❌ Error fetching historical data: {e}")
        return pd.DataFrame(columns=['Date', 'Average AQI'])