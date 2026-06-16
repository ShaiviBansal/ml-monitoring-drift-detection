import pandas as pd

df = pd.read_csv("creditcard.csv")

print("Shape (rows, columns):", df.shape)
print()
print("Column names:", list(df.columns))
print()
print("Fraud vs non-fraud counts:")
print(df["Class"].value_counts())
print()
print("Fraud percentage: {:.4f}%".format(df["Class"].mean() * 100))
print()
print("First 5 rows:")
print(df.head())