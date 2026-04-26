import pandas as pd

df = pd.read_csv("data/raw/iris.csv")

df.dropna(inplace=True)

df.to_csv("data/processed/iris_clean.csv", index=False)