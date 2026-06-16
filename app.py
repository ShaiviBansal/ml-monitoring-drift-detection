from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import psycopg2

app = FastAPI(title="Fraud Detection Monitoring Service")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


# def get_connection():
#     return psycopg2.connect(
#         host="localhost",
#         port=5432,
#         dbname="ml_monitoring",
#         user="mluser",
#         password="mlpassword",
#     )

import os

def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        port=5432,
        dbname="ml_monitoring",
        user="mluser",
        password="mlpassword",
    )


@app.get("/")
def root():
    return {"status": "Fraud detection service is running"}


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.dict()
    input_df = pd.DataFrame([data])[feature_cols]

    predicted_class = int(model.predict(input_df)[0])
    fraud_probability = float(model.predict_proba(input_df)[0][1])

    conn = get_connection()
    cur = conn.cursor()

    columns = feature_cols + ["predicted_class", "fraud_probability"]
    values = [data[c] for c in feature_cols] + [predicted_class, fraud_probability]
    placeholders = ", ".join(["%s"] * len(values))
    columns_sql = ", ".join([f'"{c}"' for c in columns])

    insert_sql = f"INSERT INTO predictions ({columns_sql}) VALUES ({placeholders})"
    cur.execute(insert_sql, values)
    conn.commit()
    cur.close()
    conn.close()

    return {
        "predicted_class": predicted_class,
        "fraud_probability": round(fraud_probability, 4),
    }