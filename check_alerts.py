import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_monitoring",
    user="mluser",
    password="mlpassword",
)
cur = conn.cursor()

cur.execute("""
    SELECT timestamp, metric_type, feature_name, metric_value, severity
    FROM drift_alerts
    ORDER BY timestamp DESC;
""")
rows = cur.fetchall()

print(f"Total alerts in table: {len(rows)}\n")
for row in rows:
    print(row)

cur.close()
conn.close()