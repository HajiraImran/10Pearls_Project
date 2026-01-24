import hopsworks
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from dotenv import load_dotenv

load_dotenv()

def run_training():
    # 1. Login to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_KEY"))
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # 2. Get Feature View (Ensure we use latest data)
    fv = fs.get_feature_view(name="islamabad_aqi_v12_view", version=2)

    # 3. Data Split
    print("⏳ Fetching data...")
    X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    
    # --- ANTI-OVERFITTING MODEL CONFIGURATION ---
    
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=7,            
            min_samples_leaf=5,      
            random_state=42
        ),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=5,            
            min_samples_leaf=10
        ),
        "LinearRegression": LinearRegression() # Baseline model
    }

    print("\n--- Training Process (Optimized) ---")
    for name, model in models.items():
        
        cols_to_drop = ['city', 'datetime', 'aqi'] 
        X_train_f = X_train.drop(columns=[c for c in cols_to_drop if c in X_train.columns], errors='ignore')
        X_test_f = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns], errors='ignore')

        # Train
        model.fit(X_train_f, y_train.values.ravel())
        
        # Evaluate
        preds = model.predict(X_test_f)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        
        print(f"📊 {name} -> R2 Score: {r2:.4f}, RMSE: {rmse:.4f}")

        # 4. Schema Registration
        model_schema = ModelSchema(Schema(X_train_f), Schema(y_train))

        # 5. Save and Register
        model_dir = f"models/{name.lower()}"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, f"{model_dir}/model.pkl")

        h_model = mr.python.create_model(
            name=f"islamabad_aqi_{name.lower()}", 
            metrics={"r2": r2, "rmse": rmse},
            model_schema=model_schema,
            description=f"Anti-overfitting version of {name}."
        )
        h_model.save(model_dir)
        print(f"📦 Registered {name} (Version {h_model.version})")

if __name__ == "__main__":
    run_training()