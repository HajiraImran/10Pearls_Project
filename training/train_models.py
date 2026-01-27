import hopsworks
import pandas as pd
import joblib
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

def train_and_select_best():
    # 1. Hopsworks Connection
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()

    # 2. Get the Feature View you created manually
    try:
        fv = fs.get_feature_view(name="islamabad_aqi_viewss", version=1)
        print("✅ Feature View 'islamabad_aqi_viewss' retrieved successfully!")
    except Exception as e:
        print(f"❌ Error: Could not find the FV. Check name in Hopsworks UI: {e}")
        return

    # 3. Train-Test Split
    # Is point par Hopsworks background mein query run karke data fetch karega
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)

    # 4. Feature Selection & Cleaning (Protecting against leakage)
    # Humein sirf 'aqi' ko target rakhna hai, baqi pollution indicators drop karne hain
    cols_to_drop = ['aqi', 'pm2_5', 'city', 'datetime', 'timestamp', 'no2', 'so2']
    
    def clean_features(df):
        # Drop metadata and target-related columns
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        # Sirf numeric columns rakhein (hour, temp, humidity, lags, etc.)
        return df.select_dtypes(include=['number']).fillna(0)

    X_train = clean_features(X_train)
    X_test = clean_features(X_test)
    
    print(f"🚀 Features being used: {list(X_train.columns)}")

    # 5. Define 3 Competitive Models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    best_model = None
    best_r2 = -1
    best_name = ""
    best_metrics = {}

    mr = project.get_model_registry()
    print("\n--- Model Competition Starting ---")

    # 6. Training & Evaluation Loop
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        print(f"📊 {name}: R2 = {r2:.4f}, MAE = {mae:.2f}")

        # Individual model registration (Version tracking ke liye)
        model_filename = f"{name.lower()}_aqi.pkl"
        joblib.dump(m, model_filename)
        
        hw_model = mr.python.create_model(
            name=f"islamabad_aqi_{name.lower()}",
            metrics={"r2": r2, "mae": mae},
            description=f"{name} trained on weather-augmented Islamabad data"
        )
        hw_model.save(model_filename)
        
        # Keep track of the winner
        if r2 > best_r2:
            best_r2 = r2
            best_model = m
            best_name = name
            best_metrics = {"r2": r2, "mae": mae}

    print(f"\n🏆 WINNER: {best_name} (R2: {best_r2:.4f})")
    
    # 7. Register the "BEST" model separately for the App to use
    joblib.dump(best_model, "best_model.pkl")
    final_model = mr.python.create_model(
        name="best_islamabad_aqi_model",
        metrics=best_metrics,
        description=f"Best performing model ({best_name}) for production app"
    )
    final_model.save("best_model.pkl")
    print("✅ Best model registered as 'best_islamabad_aqi_model'!")

if __name__ == "__main__":
    train_and_select_best()