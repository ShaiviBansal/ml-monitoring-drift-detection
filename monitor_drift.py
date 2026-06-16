import pandas as pd
import psycopg2
from psi_utils import calculate_psi

reference = pd.read_csv("reference_data.csv")

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

print(f"Comparing {len(current)} recent predictions against {len(reference)} reference rows.\n")

PSI_WARNING_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.25

results = []
for col in feature_cols:
    psi_value = calculate_psi(reference[col], current[col])
    results.append((col, psi_value))

results.sort(key=lambda x: x[1], reverse=True)

cur = conn.cursor()
alerts_logged = 0

print(f"{'Feature':<10} {'PSI':>8}  Status")
print("-" * 35)
for col, psi_value in results:
    if psi_value >= PSI_ALERT_THRESHOLD:
        status = "ALERT"
    elif psi_value >= PSI_WARNING_THRESHOLD:
        status = "WARNING"
    else:
        status = "OK"
    print(f"{col:<10} {psi_value:>8.4f}  {status}")

    if status in ("WARNING", "ALERT"):
        cur.execute(
            """INSERT INTO drift_alerts (metric_type, feature_name, metric_value, severity)
               VALUES (%s, %s, %s, %s)""",
            ("PSI", col, float(psi_value), status),
        )
        alerts_logged += 1

conn.commit()
cur.close()
conn.close()

print(f"\n{alerts_logged} alert(s) logged to drift_alerts table.")