import hopsworks
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from dotenv import load_dotenv

load_dotenv()

def run_training():
    # 1. Login to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 2. Reference the successful v12 Feature Group
    fg_name = "islamabad_aqi_v12"
    aqi_fg = fs.get_feature_group(name=fg_name, version=1)

    # 3. Use Feature View Version 2
    view_name = "islamabad_aqi_v12_view"
    fv = fs.get_feature_view(name=view_name, version=2)

    # 4. Data Split
    print("⏳ Fetching and splitting data...")
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    
    # 5. Train 3 Models with Regularization (Mentor Requirement)
    # We reduce max_depth and increase min_samples_leaf to stop overfitting
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=5),
        "DecisionTree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=5),
        "LinearRegression": LinearRegression()
    }

    print("\n--- Training Process (Optimized for Realism) ---")
    for name, model in models.items():
        # --- FEATURE SELECTION TO PREVENT LEAKAGE ---
        # We drop 'city', 'datetime' (non-numeric) 
        # We also drop 'aqi_lag_1' if the R2 remains 1.0, as it might be too predictive
        cols_to_drop = ['city', 'datetime', 'aqi'] 
        
        X_train_f = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns])
        X_test_f = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

        # Train
        model.fit(X_train_f, y_train.values.ravel())
        
        # Evaluate
        preds = model.predict(X_test_f)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # Realistic R2 should be between 0.60 and 0.94
        print(f"📊 {name} -> R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

        # 6. Save and Register in Model Registry
        model_dir = f"models/{name.lower()}_v12"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/model.pkl")

        h_model = mr.python.create_model(
            name=f"islamabad_aqi_{name.lower()}", 
            metrics={"r2": r2, "rmse": rmse},
            description="Trained with regularization to prevent overfitting."
        )
        h_model.save(model_dir)
        print(f"📦 Registered {name} in Model Registry.")

if __name__ == "__main__":
    run_training()