import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_KEY")
LAT = 33.72
LON = 73.04

# OpenWeather Air Pollution Endpoint
url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"

response = requests.get(url).json()

# Data extraction
if "list" in response:
    components = response['list'][0]['components']
    aqi_index = response['list'][0]['main']['aqi'] # 1=Good, 5=Very Poor
    pm25 = components['pm2_5']
    
    print(f"🌍 OpenWeather Islamabad Data:")
    print(f"📍 PM2.5: {pm25} µg/m³")
    print(f"📊 AQI Index (1-5 Scale): {aqi_index}")
else:
    print("Error fetching data:", response)