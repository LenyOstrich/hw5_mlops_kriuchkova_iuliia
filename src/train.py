import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LogisticRegression

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

max_iter = params["model"]["max_iter"]

df = pd.read_csv("data/processed/iris_clean.csv")

X = df.drop(columns=["variety"])
y = df["variety"]

model = LogisticRegression(max_iter=max_iter)
model.fit(X, y)

joblib.dump(model, "model.pkl")