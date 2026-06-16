import pandas as pd
import numpy as np
import shap
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

reference = pd.read_csv("reference_data.csv")
sample = reference.sample(n=5000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": sample.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv("baseline_shap_importance.csv", index=False)

print("Baseline SHAP importance (top 10 features):")
print(importance_df.head(10).to_string(index=False))