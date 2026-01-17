import hopsworks
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from utils import fetch_raw_data, apply_feature_engineering
from dotenv import load_dotenv

load_dotenv()

def run_backfill():
    # Login
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()

    # Get or Create Feature Group
    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_v12",
        version=1,
        primary_key=['city', 'datetime'],
        event_time='datetime',
        online_enabled=False,
        description="Fresh Clean AQI data after cleanup"
    )

    # --- UPDATED LOGIC FOR ONLY NEW ROWS ---
    fetch_days = 1 # Default: Sirf naya data mangain
    
    try:
        # Check karein ke kya FG mein pehle se data hai
        # Hum sirf count check kar rahe hain taake query tez ho
        # Statistics tab (2887 records) confirm karta hai ke data already hai
        fg_meta = aqi_fg.get_statistics()
        
        if fg_meta:
            fetch_days = 1 # Agar statistics maujood hain, matlab data hai
            print("✅ Data already exists. Switching to Incremental Mode (1 day).")
        else:
            fetch_days = 120
            print("⏳ Feature Group is empty. Starting Initial Backfill (120 days).")
            
    except Exception as e:
        print(f"⚠️ Could not check stats, defaulting to 1 day to save API quota. Error: {e}")
        fetch_days = 1

    # Fetch and Engineering
    print(f"🚀 Fetching data for the last {fetch_days} day(s)...")
    raw_df = fetch_raw_data(days=fetch_days)
    engineered_df = apply_feature_engineering(raw_df)

    # Final Insert
    if not engineered_df.empty:
        # Duplicate check Hopsworks primary key khud kar lega
        aqi_fg.insert(engineered_df)
        print(f"⭐ Successfully updated V12 with {len(engineered_df)} records.")
    else:
        print("❌ No new data fetched. Check OpenWeather API connection.")

if __name__ == "__main__":
    run_backfill()