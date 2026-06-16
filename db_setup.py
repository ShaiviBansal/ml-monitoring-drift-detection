import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_monitoring",
    user="mluser",
    password="mlpassword",
)
cur = conn.cursor()

feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
feature_columns_sql = ", ".join([f'"{col}" FLOAT' for col in feature_cols])

create_predictions_sql = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    {feature_columns_sql},
    predicted_class INTEGER,
    fraud_probability FLOAT
);
"""

create_alerts_sql = """
CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    metric_type VARCHAR(10),
    feature_name VARCHAR(20),
    metric_value FLOAT,
    severity VARCHAR(10)
);
"""

cur.execute(create_predictions_sql)
cur.execute(create_alerts_sql)
conn.commit()

print("Tables 'predictions' and 'drift_alerts' created successfully.")

cur.close()
conn.close()