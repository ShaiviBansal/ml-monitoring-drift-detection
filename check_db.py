import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_monitoring",
    user="mluser",
    password="mlpassword",
)
cur = conn.cursor()

cur.execute("SELECT id, timestamp, predicted_class, fraud_probability FROM predictions ORDER BY id DESC LIMIT 5;")
rows = cur.fetchall()

print("Most recent predictions logged:")
for row in rows:
    print(row)

cur.close()
conn.close()