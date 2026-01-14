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
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Fetch Data
    fv = fs.get_feature_view(name="aqi_model_view_v10_final", version=1)
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    
    # Feature Selection
    cols_to_drop = ['city', 'datetime', 'pm2_5']
    X_train = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "LinearRegression": LinearRegression()
    }

    results = {}
    best_r2 = -float('inf')
    best_model_name = ""

    print("\n--- Training Models (RMSE Fix Applied) ---")
    for name, model in models.items():
        model.fit(X_train, y_train.values.ravel())
        preds = model.predict(X_test)
        
        # Metrics Calculation (Compatible with all sklearn versions)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse) 
        
        results[name] = {"r2": r2, "rmse": rmse}
        print(f"{name} -> R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

        # Model Save
        m_dir = f"models/{name.lower()}_model"
        os.makedirs(m_dir, exist_ok=True)
        joblib.dump(model, f"{m_dir}/model.pkl")

        # Register
        h_model = mr.python.create_model(
            name=f"islamabad_aqi_{name.lower()}", 
            metrics={"r2": r2, "rmse": rmse},
            description="Realistic model without pm2_5 leakage"
        )
        h_model.save(m_dir)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print(f"\n⭐ FINAL WINNER: {best_model_name}")

if __name__ == "__main__":
    run_training()