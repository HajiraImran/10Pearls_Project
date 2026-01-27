import hopsworks
import os
import pandas as pd
from utils import fetch_raw_pollution, fetch_weather_history, apply_feature_engineering, clean_and_merge
from dotenv import load_dotenv

load_dotenv()

def run_backfill():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()

    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_v12",
        version=5,
        primary_key=['city', 'datetime'],
        event_time='datetime',
        description="Smart Incremental AQI + Weather data"
    )

    # Smart Logic to decide fetch_days
    try:
        # Check if feature group has data
        if len(aqi_fg.read(read_options={"use_hive": True})) > 0:
            fetch_days = 2
            print("✅ Data exists. Mode: Incremental (2 days).")
        else:
            fetch_days = 120
            print("⏳ FG is empty. Mode: Initial Backfill (120 days).")
    except:
        fetch_days = 120 # First time running

    print(f"🚀 Fetching data for {fetch_days} days...")
    df_pol = fetch_raw_pollution(days=fetch_days)
    df_wet = fetch_weather_history(days=fetch_days)

    # Use the now-imported clean_and_merge
    combined_df = clean_and_merge(df_pol, df_wet)
    engineered_df = apply_feature_engineering(combined_df)

    if not engineered_df.empty:
        aqi_fg.insert(engineered_df)
        print(f"⭐ Successfully inserted {len(engineered_df)} records.")
    else:
        print("❌ Nothing to insert.")

if __name__ == "__main__":
    run_backfill()