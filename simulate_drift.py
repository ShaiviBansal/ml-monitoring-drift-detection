import pandas as pd
import requests

df = pd.read_csv("creditcard.csv")
sample = df.sample(n=300, random_state=2).drop(columns=["Class"]).copy()

# Artificially shift the Amount and V14 columns to simulate drift
sample["Amount"] = sample["Amount"] * 15 + 500
sample["V14"] = sample["V14"] + 8

print(f"Sending {len(sample)} artificially drifted transactions to the API...")

success_count = 0
for i, row in sample.iterrows():
    payload = row.to_dict()
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if response.status_code == 200:
        success_count += 1

print(f"Done. {success_count}/{len(sample)} requests succeeded.")