import pandas as pd

# Update this path if needed
df = pd.read_excel("data/data.xlsx", engine="openpyxl")
df.to_csv("data/data.csv", index=False)
