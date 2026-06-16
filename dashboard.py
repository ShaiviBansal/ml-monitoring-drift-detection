import streamlit as st
import pandas as pd
import psycopg2
import pickle
import shap
import numpy as np
from psi_utils import calculate_psi
import os

st.set_page_config(page_title="Fraud Detection Monitoring", layout="wide")

st.title("Production ML Monitoring Dashboard")
st.caption("Fraud detection model — drift and prediction monitoring")


# def get_connection():
#     return psycopg2.connect(
#         host="localhost",
#         port=5432,
#         dbname="ml_monitoring",
#         user="mluser",
#         password="mlpassword",
#     )

def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=5432,
        dbname="ml_monitoring",
        user="mluser",
        password="mlpassword",
    )


feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

conn = get_connection()

total_predictions = pd.read_sql("SELECT COUNT(*) as count FROM predictions;", conn).iloc[0]["count"]
fraud_flagged = pd.read_sql("SELECT COUNT(*) as count FROM predictions WHERE predicted_class = 1;", conn).iloc[0]["count"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions Logged", total_predictions)
col2.metric("Flagged as Fraud", fraud_flagged)
col3.metric("Flagged Rate", f"{(fraud_flagged / total_predictions * 100):.2f}%" if total_predictions > 0 else "N/A")

st.divider()

st.subheader("Feature Drift (PSI)")

reference = pd.read_csv("reference_data.csv")
columns_sql = ", ".join([f'"{c}"' for c in feature_cols])
current = pd.read_sql(f"SELECT {columns_sql} FROM predictions ORDER BY id DESC LIMIT 500;", conn)

psi_results = []
for col in feature_cols:
    psi_value = calculate_psi(reference[col], current[col])
    if psi_value >= 0.25:
        status = "ALERT"
    elif psi_value >= 0.1:
        status = "WARNING"
    else:
        status = "OK"
    psi_results.append({"feature": col, "psi": round(psi_value, 4), "status": status})

psi_df = pd.DataFrame(psi_results).sort_values("psi", ascending=False)


def highlight_status(row):
    if row["status"] == "ALERT":
        return ["background-color: #ffcccc"] * len(row)
    elif row["status"] == "WARNING":
        return ["background-color: #fff3cd"] * len(row)
    return [""] * len(row)


# st.dataframe(psi_df.style.apply(highlight_status, axis=1), use_container_width=True, height=300)
st.dataframe(psi_df.style.apply(highlight_status, axis=1), width="stretch", height=300)

st.divider()

st.subheader("Recent Drift Alerts")
alerts = pd.read_sql(
    "SELECT timestamp, metric_type, feature_name, metric_value, severity FROM drift_alerts ORDER BY timestamp DESC LIMIT 20;",
    conn,
)
# st.dataframe(alerts, use_container_width=True)
st.dataframe(alerts, width="stretch")

st.divider()

st.subheader("SHAP Feature Importance: Baseline vs Current")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

baseline_shap = pd.read_csv("baseline_shap_importance.csv").set_index("feature")["mean_abs_shap"]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(current)
current_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=current.columns)

shap_comparison = pd.DataFrame({"baseline": baseline_shap, "current": current_shap})
shap_comparison["pct_change"] = (
    (shap_comparison["current"] - shap_comparison["baseline"]) / shap_comparison["baseline"] * 100
)
shap_comparison = shap_comparison.sort_values("pct_change", ascending=False)

st.bar_chart(shap_comparison[["baseline", "current"]])

conn.close()