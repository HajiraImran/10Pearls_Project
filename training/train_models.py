import hopsworks
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from dotenv import load_dotenv

load_dotenv()

def run_training():
    # 1. Login
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 2. Fetch Data from Feature View
    fv = fs.get_feature_view(name="aqi_model_view_v10_final", version=1)
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    
    # --- MENTOR'S CONDITION: ENOUGH DATA CHECK ---
    total_records = len(X_train) + len(X_test)
    print(f"Total records found in Feature Store: {total_records}")

    if total_records < 3000:
        print(f"⚠️ Skipping Training: Need at least 3000 records to ensure model quality. Current: {total_records}")
        return 
    # ----------------------------------------------

    # Feature Selection (Drop Leakage)
    cols_to_drop = ['city', 'datetime', 'pm2_5']
    X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "LinearRegression": LinearRegression()
    }

    print("\n--- Training Models & Storing in Registry ---")
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"{name} -> R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

        # Model Save & Registry
        m_dir = f"models/{name.lower()}_model"
        os.makedirs(m_dir, exist_ok=True)
        joblib.dump(model, f"{m_dir}/model.pkl")

        h_model = mr.python.create_model(
            name=f"islamabad_aqi_{name.lower()}", 
            metrics={"r2": r2, "rmse": rmse},
            description=f"Automated daily update. Trained on {total_records} records."
        )
        h_model.save(m_dir)
        
    print("⭐ Success! All models registered with latest versions.")

if __name__ == "__main__":
    run_training()