🌬️ Islamabad Air Quality Forecasting System (MLOps)
An end-to-end Machine Learning operations (MLOps) project that predicts the Air Quality Index (AQI) for Islamabad, Pakistan. The system features automated data pipelines, a centralized feature store, and an interactive dashboard with model explainability.

🚀 Project Overview
This project goes beyond simple model training. It implements a complete life cycle:

Automated Data Ingestion: Fetches real-time weather and pollutant data.

Feature Store: Uses Hopsworks to manage features and model versions.

CI/CD: Automated pipelines via GitHub Actions.

Explainable AI (XAI): Uses SHAP to interpret model decisions.

🛠️ System Architecture

The project is divided into three main components:
1. Feature Pipeline (data_pipeline/)Source: Fetches raw data from OpenWeather and AQICN APIs.Processing: Computes time-based features (hour, weekday) and engineered features like aqi_lag_1 and pm2_5_rolling_6h.Automation: Triggered every hour via GitHub Actions.Storage: Processed data is pushed to the Hopsworks Feature Store.
2. Training PipelineExperimentation: Models tested include Random Forest, XGBoost, and Ridge Regression.Registry: The "Champion" model is saved in the Hopsworks Model Registry.Evaluation: Models are evaluated based on RMSE, MAE, and $R^2$ metrics.
3. Web Dashboard (app.py)Framework: Built with Streamlit.Real-time Inference: Pulls the latest features and model from Hopsworks to predict AQI for the next 3 days.Hazardous Alerts: Integrated health alerts that trigger warnings for AQI levels 4 (Poor) and 5 (Hazardous).

🔬 Advanced Analytics & Model Explainability
To ensure transparency, we utilized SHAP (SHapley Additive exPlanations).

Global Importance: Analysis shows that aqi_lag_1 (previous day's air quality) is the strongest predictor, followed by pm2_5_rolling_6h.

Weather Impact: High wind_speed shows a negative correlation with AQI, meaning stronger winds significantly improve air quality by dispersing pollutants.

📋 Requirements Fulfillment
Requirement	Status	Implementation
Feature Pipeline	✅	Automated hourly runs via GitHub Actions.
Feature Store	✅	Centralized storage in Hopsworks.
Model Registry	✅	Versioning and storage of best-performing models.
Dashboard	✅	Interactive Streamlit UI with 3-day forecasting.
Hazardous Alerts	✅	Real-time notifications for dangerous pollution levels.
SHAP/LIME	✅	Deep-dive feature importance analysis.

⚙️ Installation & Setup
Clone the Repository:


git clone https://github.com/your-username/10pearls_project.git
cd 10pearls_project

Install Dependencies:
pip install -r requirements.txt

Environment Variables: Create a .env file and add your keys:
HOPSWORKS_KEY=your_api_key_here

Run the Dashboard:
streamlit run data_pipeline/app.py

https://10pearlsproject-tadmaj4bxhkwpoemecozcg.streamlit.app/

👥 Contributor
Hajira Imran