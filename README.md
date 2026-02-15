🌬️ Islamabad Air Quality Forecasting System (MLOps)

An end-to-end Machine Learning Operations (MLOps) project that predicts the Air Quality Index (AQI) for Islamabad, Pakistan, up to 3 days ahead.
The system demonstrates automated data pipelines, a centralized feature store, model versioning, and an interactive dashboard with explainability.

🚀 Project Overview

This project implements the complete ML lifecycle, moving beyond standalone model training:

Automated Data Ingestion: Fetches historical and near–real-time weather and pollution data

Feature Store Integration: Centralized feature management using Hopsworks

Model Training & Registry: Multiple models trained, evaluated, and versioned

CI/CD Automation: Feature and training pipelines automated with GitHub Actions

Interactive Dashboard: Real-time AQI visualization, forecasting, and health alerts

🛠️ System Architecture

The system is composed of three major components:

1️⃣ Feature Pipeline (data_pipeline/)

Data Sources

Weather data from Meteostat API

Pollution data from OpenWeather Air Pollution API

Feature Engineering

Time-based features: hour, weekday, month

Lag features: aqi_lag_1

Rolling statistics: pm2_5_rolling_6h

Derived indicators: wind stagnation flag

Automation

Runs hourly using GitHub Actions

Storage

Features stored in Hopsworks Feature Store with primary keys and event time

2️⃣ Training Pipeline (training/)

Model Experimentation

Random Forest

XGBoost

Gradient Boosting

Evaluation Metrics

RMSE

MAE

R²

Model Registry

All trained models are versioned

Best-performing model is promoted as the production (champion) model in Hopsworks

Automation

Runs daily via GitHub Actions

3️⃣ Web Dashboard (app.py)

Framework

Built using Streamlit

Real-Time Inference

Loads latest features and the champion model directly from Hopsworks

Predicts AQI for the next 3 days

Visualizations

Historical AQI trends

Forecasted AQI levels

Health Alerts

Automatic alerts for:

AQI Level 4 (Poor)

AQI Level 5 (Hazardous)

🔬 Advanced Analytics & Model Explainability

To enhance interpretability, feature importance analysis is integrated:

Global Feature Importance

aqi_lag_1 is the strongest predictor

pm2_5_rolling_6h significantly influences AQI

Higher wind_speed correlates with improved air quality

Dashboard Explainability

Tree-based feature importance visualized directly in the UI

Helps explain prediction drivers to non-technical users

📋 Requirements Fulfillment
Requirement	Status	Implementation
Feature Pipeline	✅	Hourly automated runs via GitHub Actions
Feature Store	✅	Centralized storage using Hopsworks
Model Registry	✅	Versioned models with champion selection
Training Automation	✅	Daily retraining pipeline
Dashboard	✅	Streamlit-based interactive UI
3-Day Forecasting	✅	Weather-driven AQI prediction
Hazardous Alerts	✅	Real-time health warnings
Explainability	✅	Feature importance analysis
⚙️ Installation & Setup
Clone the Repository
git clone https://github.com/HajiraImran/10pearls_project.git
cd 10pearls_project

Install Dependencies
pip install -r requirements.txt

Environment Variables

Create a .env file:

HOPSWORKS_KEY=your_api_key_here
OPENWEATHER_KEY=your_api_key_here

Run the Dashboard
streamlit run data_pipeline/app.py

🌐 Live Demo

🔗 https://huggingface.co/spaces/Hajiraaa/myProject

👥 Contributor

Hajira Imran