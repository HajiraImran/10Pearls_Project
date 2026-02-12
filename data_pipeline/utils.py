import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# --- HISTORICAL WEATHER (still Meteostat) ---
from meteostat import Point, Hourly

def fetch_weather_history(days=120):
    lat, lon = 33.72, 73.04
    location = Point(lat, lon)
    now_time = datetime.utcnow()
    start_time = now_time - timedelta(days=days)
    
    data = Hourly(location, start_time, now_time)
    df = data.fetch()
    if df.empty: return pd.DataFrame()
    
    available_cols = [c for c in ['temp', 'rhum', 'wspd'] if c in df.columns]
    df = df[available_cols]
    df.reset_index(inplace=True)
    df.rename(columns={'time':'datetime','temp':'temperature','rhum':'humidity','wspd':'wind_speed'}, inplace=True)
    
    return df

# --- OPENWEATHERMAP 3-DAY FORECAST ---
def fetch_weather_forecast(days=3):
    """
    Fetch next `days` forecast from OpenWeatherMap API
    """
    API_KEY = os.getenv("OPENWEATHER_KEY")
    if not API_KEY:
        return pd.DataFrame()
    
    lat, lon = 33.72, 73.04
    url = f"https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if "list" not in res: 
            return pd.DataFrame()
        
        df_list = []
        for item in res["list"]:
            dt = datetime.utcfromtimestamp(item["dt"])
            temp = item["main"]["temp"]
            humidity = item["main"]["humidity"]
            wind_speed = item["wind"]["speed"]
            df_list.append({"datetime": dt, "temperature": temp, "humidity": humidity, "wind_speed": wind_speed})
        
        df = pd.DataFrame(df_list)
        df.set_index("datetime", inplace=True)
        # Take daily average
        daily_df = df.resample("D").mean().reset_index()
        return daily_df.head(days)
    
    except Exception as e:
        print(f"Weather forecast fetch error: {e}")
        return pd.DataFrame()


# --- HISTORICAL AQI (Hopsworks) ---
def fetch_historical_aqi_data(fs, num_days=7):
    try:
        fg = fs.get_feature_group(name="islamabad_aqi_v12", version=5)
        query_df = fg.read()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=num_days)
        
        query_df['datetime'] = pd.to_datetime(query_df['datetime'])
        historical_df = query_df[query_df['datetime'] >= start_date]
        historical_df['date_only'] = historical_df['datetime'].dt.date
        daily_avg = historical_df.groupby('date_only')['aqi'].mean().reset_index()
        daily_avg.rename(columns={'date_only':'Date','aqi':'Average AQI'}, inplace=True)
        return daily_avg.sort_values('Date')
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()
