import hopsworks
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from utils import fetch_raw_data, apply_feature_engineering
from dotenv import load_dotenv

load_dotenv()

def run_backfill():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()

    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_final_v10",
        version=1,
        primary_key=['city', 'datetime'],
        event_time='datetime',
        online_enabled=False,
        description="Final cleaned AQI data with incremental loading"
    )

    # --- DUPLICATE CHECK LOGIC ---
    try:
        # Check karein ke last date kya thi
        query = aqi_fg.select(["datetime"]).order_by("datetime", descending=True).limit(1)
        latest_data = query.read()
        
        if not latest_data.empty:
            last_date = pd.to_datetime(latest_data.iloc[0]['datetime']).replace(tzinfo=timezone.utc)
            # Sirf tab fetch karein agar data 24h purana ho
            if datetime.now(timezone.utc) - last_date < timedelta(hours=23):
                print("✅ Data is already up to date. No need to fetch.")
                return
            fetch_days = 2 # Sirf delta mangwayein
        else:
            fetch_days = 120 # First time setup
    except:
        fetch_days = 120

    # 3. Fetch & Filter
    raw_df = fetch_raw_data(days=fetch_days)
    engineered_df = apply_feature_engineering(raw_df)

    # Extra Filter: Sirf last_date se agay ki rows rakhein
    if fetch_days < 120:
        last_date_naive = last_date.replace(tzinfo=None)
        engineered_df = engineered_df[pd.to_datetime(engineered_df['datetime']) > last_date_naive]

    # 4. Final Insert
    if not engineered_df.empty:
        aqi_fg.insert(engineered_df, write_options={"start_offline_materialization": True})
        print(f"⭐ Successfully added {len(engineered_df)} new records.")
    else:
        print("No new unique records found.")

if __name__ == "__main__":
    run_backfill()