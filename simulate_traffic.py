import pandas as pd
import requests
import time

df = pd.read_csv("creditcard.csv")

sample = df.sample(n=500, random_state=1).drop(columns=["Class"])

print(f"Sending {len(sample)} simulated transactions to the API...")

success_count = 0
for i, row in sample.iterrows():
    payload = row.to_dict()
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    if response.status_code == 200:
        success_count += 1
    if (success_count) % 100 == 0 and success_count > 0:
        print(f"  {success_count} sent...")

print(f"Done. {success_count}/{len(sample)} requests succeeded.")