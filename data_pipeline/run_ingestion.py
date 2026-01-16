import hopsworks
import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def ingest_hourly_data():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="islamabad_aqi_v12", version=1)

    # 1. Pichla data uthayein taake Lag aur Rolling calculate ho sakay
    history_df = fg.read().sort_values(by='datetime')
    last_aqi = history_df.iloc[-1]['aqi']
    last_6h_pm25 = history_df.tail(6)['pm2_5']

    # 2. Naya data fetch karein OpenWeather se
    API_KEY = os.getenv("OPENWEATHER_KEY")
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=33.72&lon=73.04&appid={API_KEY}"
    res = requests.get(url).json()
    current_pm25 = float(res['list'][0]['components']['pm2_5'])
    current_aqi = float(res['list'][0]['main']['aqi'])
    now = datetime.now()

    # 3. FEATURE ENGINEERING (Matching your Model Schema)
    new_data = {
        'city': ['Islamabad'],
        'datetime': [pd.to_datetime(now)],
        'pm2_5': [current_pm25],
        'aqi': [current_aqi],
        'hour': [float(now.hour)],
        'weekday': [float(now.weekday())],
        'month': [float(now.month)],
        'aqi_lag_1': [float(last_aqi)], # Pichle ghante ka AQI
        'pm2_5_rolling_6h': [float(last_6h_pm25.mean())], # 6-hour average
        'aqi_change_rate': [float(current_aqi - last_aqi)] # Change rate
    }
    
    df = pd.DataFrame(new_data)
    
    # 4. Insert into Feature Group
    fg.insert(df)
    print(f"✅ Full Feature Set inserted at {now}")

if __name__ == "__main__":
    ingest_hourly_data()