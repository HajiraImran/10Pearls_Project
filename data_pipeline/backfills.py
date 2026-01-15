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

    # Create Fresh Feature Group (V12)
    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_v12",
        version=1,
        primary_key=['city', 'datetime'],
        event_time='datetime',
        online_enabled=False,  # <--- Isay False kar dein
        description="Fresh Clean AQI data after cleanup"
    )

    # Check for existing data
    try:
        query = aqi_fg.select(["datetime"]).order_by("datetime", descending=True).limit(1)
        latest_data = query.read()
        if not latest_data.empty:
            last_date = pd.to_datetime(latest_data.iloc[0]['datetime']).replace(tzinfo=timezone.utc)
            fetch_days = 2
        else:
            fetch_days = 120
    except:
        fetch_days = 120

    # Fetch and Engineering
    raw_df = fetch_raw_data(days=fetch_days)
    engineered_df = apply_feature_engineering(raw_df)

    # Final Insert
    if not engineered_df.empty:
        # datetime columns ko string me convert karein takay Hopsworks asani se handle kare
        aqi_fg.insert(engineered_df)
        print(f"⭐ Successfully created V12 with {len(engineered_df)} records.")
    else:
        print("❌ Failed to fetch data. Check API Key.")

if __name__ == "__main__":
    run_backfill()