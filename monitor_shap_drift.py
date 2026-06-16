import pandas as pd
import psycopg2
import shap
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

baseline = pd.read_csv("baseline_shap_importance.csv").set_index("feature")["mean_abs_shap"]

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_monitoring",
    user="mluser",
    password="mlpassword",
)

feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
columns_sql = ", ".join([f'"{c}"' for c in feature_cols])
query = f"SELECT {columns_sql} FROM predictions ORDER BY id DESC LIMIT 500;"
current = pd.read_sql(query, conn)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(current)

current_importance = pd.Series(
    np.abs(shap_values).mean(axis=0), index=current.columns
)

comparison = pd.DataFrame({
    "baseline": baseline,
    "current": current_importance,
})
comparison["pct_change"] = (
    (comparison["current"] - comparison["baseline"]) / comparison["baseline"] * 100
)
comparison = comparison.sort_values("pct_change", ascending=False)

SHAP_WARNING_THRESHOLD = 15
SHAP_ALERT_THRESHOLD = 30

cur = conn.cursor()
alerts_logged = 0

print("SHAP importance: baseline vs current (sorted by % change)\n")
print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

for feature, row in comparison.iterrows():
    pct_change = row["pct_change"]
    abs_pct_change = abs(pct_change)

    if abs_pct_change >= SHAP_ALERT_THRESHOLD:
        severity = "ALERT"
    elif abs_pct_change >= SHAP_WARNING_THRESHOLD:
        severity = "WARNING"
    else:
        continue

    cur.execute(
        """INSERT INTO drift_alerts (metric_type, feature_name, metric_value, severity)
           VALUES (%s, %s, %s, %s)""",
        ("SHAP", feature, float(pct_change), severity),
    )
    alerts_logged += 1

conn.commit()
cur.close()
conn.close()

print(f"\n{alerts_logged} SHAP alert(s) logged to drift_alerts table.")