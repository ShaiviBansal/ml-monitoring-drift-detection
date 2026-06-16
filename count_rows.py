import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    dbname="ml_monitoring",
    user="mluser",
    password="mlpassword",
)
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM predictions;")
total = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM predictions WHERE predicted_class = 1;")
fraud_flagged = cur.fetchone()[0]

print(f"Total predictions logged: {total}")
print(f"Flagged as fraud: {fraud_flagged}")

cur.close()
conn.close()