import hopsworks
import pandas as pd
import joblib
import os
import numpy as np 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # MSE add kiya
from dotenv import load_dotenv

load_dotenv()

def train_and_select_best():
    # 1. Hopsworks Connection
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()

    # 2. Get Feature View
    try:
        fv = fs.get_feature_view(name="islamabad_aqi_viewss", version=1)
        print("✅ Feature View retrieved!")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)

    # 4. Feature Cleaning
    cols_to_drop = ['aqi', 'pm2_5', 'city', 'datetime', 'timestamp', 'no2', 'so2']
    def clean_features(df):
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return df.select_dtypes(include=['number']).fillna(0)

    X_train = clean_features(X_train)
    X_test = clean_features(X_test)
    
    # 5. Models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    best_model, best_r2, best_name, best_metrics = None, -1, "", {}
    mr = project.get_model_registry()

    print("\n--- Training with RMSE Tracking ---")

    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        # Calculate Metrics
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds)) # RMSE Calculation
        
        print(f"📊 {name}: R2={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

        # Individual Registration
        model_file = f"{name.lower()}_aqi.pkl"
        joblib.dump(m, model_file)
        
        metrics = {"r2": r2, "mae": mae, "rmse": rmse}
        hw_model = mr.python.create_model(
            name=f"islamabad_aqi_{name.lower()}",
            metrics=metrics,
            description=f"{name} with RMSE tracking"
        )
        hw_model.save(model_file)
        
        if r2 > best_r2:
            best_r2, best_model, best_name, best_metrics = r2, m, name, metrics

    # 7. Final Best Model Registration
    joblib.dump(best_model, "best_model.pkl")
    final_model = mr.python.create_model(
        name="best_islamabad_aqi_model",
        metrics=best_metrics,
        description=f"Winner: {best_name}"
    )
    final_model.save("best_model.pkl")
    print(f"\n🏆 Saved {best_name} as 'best_islamabad_aqi_model' with RMSE: {best_metrics['rmse']:.2f}")

if __name__ == "__main__":
    train_and_select_best()