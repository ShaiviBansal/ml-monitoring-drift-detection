import requests
import pandas as pd

df = pd.read_csv("creditcard.csv")
row = df.iloc[0].drop("Class").to_dict()

response = requests.post("http://127.0.0.1:8000/predict", json=row)

print("Status code:", response.status_code)
print("Response:", response.json())