import hopsworks
import os
from utils import fetch_raw_data, apply_feature_engineering
from dotenv import load_dotenv

# .env keys load
load_dotenv()

def run_backfill():
    # 1. Hopsworks Login
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()

    # 2. Data Fetching (Using utils.py)
    print("Fetching 120 days of historical data...")
    raw_df = fetch_raw_data(days=120)

    # 3. Feature Engineering (Using utils.py)
    print("Applying feature engineering...")
    engineered_df = apply_feature_engineering(raw_df)

    # 4. Naya Feature Group Create Karein
    # Hum iska naam "islamabad_aqi_final" rakhte hain
    aqi_fg = fs.get_or_create_feature_group(
        name="islamabad_aqi_final_v10",
        version=1,
        primary_key=['city', 'datetime'],
        event_time='datetime',
        online_enabled=False,
        description="Final cleaned AQI data for Islamabad with lag and rolling features"
    )

    # 5. Data Insert 
    print("Uploading data to Hopsworks...")
    aqi_fg.insert(engineered_df, write_options={"start_offline_materialization": True})
    
    print("⭐ Success! New Feature Group 'islamabad_aqi_final' created and loaded.")

if __name__ == "__main__":
    run_backfill()