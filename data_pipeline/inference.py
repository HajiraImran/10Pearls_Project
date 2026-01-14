import hopsworks
import joblib
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

def run_prediction():
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 1. Best Model download karein
    model_meta = mr.get_model("islamabad_aqi_model", version=1)
    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "aqi_best_model.pkl"))

    # 2. Latest data lein prediction shuru karne ke liye
    fg = fs.get_feature_group(name="islamabad_aqi_final_v10", version=1)
    df = fg.select_all().read().sort_values("datetime")
    
    # Aakhri record ko starting point banayein
    latest_record = df.iloc[-1].copy()
    current_time = pd.to_datetime(latest_record['datetime'])

    predictions = []
    
    print("Generating 72-hour forecast...")
    for i in range(72):
        # Features: ['hour', 'weekday', 'month', 'aqi_lag_1', 'pm2_5_rolling_6h', 'aqi_change_rate']
        next_time = current_time + timedelta(hours=1)
        
        # Simple Recursive Logic
        input_data = pd.DataFrame([{
            'hour': float(next_time.hour),
            'weekday': float(next_time.weekday()),
            'month': float(next_time.month),
            'aqi_lag_1': float(latest_record['aqi']),
            'pm2_5_rolling_6h': float(latest_record['pm2_5_rolling_6h']),
            'aqi_change_rate': float(latest_record['aqi_change_rate'])
        }])
        
        pred_aqi = model.predict(input_data)[0]
        
        predictions.append({
            "Time": next_time.strftime('%Y-%m-%d %H:%M'),
            "Predicted_AQI": round(pred_aqi, 2)
        })
        
        # Update latest_record for next iteration
        latest_record['aqi'] = pred_aqi
        current_time = next_time

    # Results display karein
    forecast_df = pd.DataFrame(predictions)
    print("\n--- 3-Day Forecast (Islamabad) ---")
    print(forecast_df.head(10)) # Pehle 10 ghante
    
    # Save results
    os.makedirs('data', exist_ok=True)
    forecast_df.to_csv("data/forecast.csv", index=False)
    print("\n⭐ Forecast saved to data/forecast.csv")

if __name__ == "__main__":
    run_prediction()