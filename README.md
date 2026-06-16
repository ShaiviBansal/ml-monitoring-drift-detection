# Production ML Monitoring and Drift Detection Service

A monitoring system for a fraud detection model that logs every prediction to PostgreSQL, computes Population Stability Index (PSI) on input feature distributions, tracks SHAP feature importance drift over time, and visualizes everything through a Streamlit dashboard. Alerts are persisted to a dedicated database table when drift crosses defined thresholds.

## Architecture

- **Model**: XGBoost classifier trained on the Kaggle Credit Card Fraud Detection dataset
- **API**: FastAPI service exposing a `/predict` endpoint, logging every prediction to PostgreSQL
- **Drift detection**: PSI computed per feature against a saved training-data reference; SHAP importance compared against a saved baseline
- **Alerting**: Drift events above WARNING/ALERT thresholds are persisted to a `drift_alerts` table
- **Dashboard**: Streamlit app visualizing prediction volume, PSI status per feature, alert history, and SHAP importance comparison
- **Infrastructure**: Full stack containerized with Docker Compose (PostgreSQL, API, dashboard)

## Setup

1. Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project root.

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Train the model (this also generates `reference_data.csv`, the PSI baseline):
   ```
   python train_model.py
   ```

4. Generate the SHAP baseline:
   ```
   python shap_baseline.py
   ```

5. Start the full stack:
   ```
   docker compose up -d --build
   ```

6. API available at `http://localhost:8000`, dashboard at `http://localhost:8501`.

## Testing drift detection

`simulate_traffic.py` sends normal sampled transactions through the API. `simulate_drift.py` sends transactions with artificially shifted `Amount` and `V14` values to demonstrate the alerting system catching real drift. Run `monitor_drift.py` and `monitor_shap_drift.py` after either to compute and log drift metrics.
