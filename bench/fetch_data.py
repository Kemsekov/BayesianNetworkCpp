#!/usr/bin/env python3
"""Download the UCI Congressional Voting Records dataset and write a clean
integer-encoded CSV (0/1) with rows containing missing values dropped."""
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
COLUMNS = [
    "class",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missile",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa",
]

df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
print(f"raw rows: {len(df)}, columns: {len(df.columns)}")
print("missing per column (rows with any '?'):")
print(df.isna().sum().to_string())

# Drop rows with any missing vote, map yes/no -> 1/0.
df = df.dropna()
for col in COLUMNS:
    df[col] = df[col].map({"y": 1, "n": 0})

# Drop columns that became constant after cleaning (no variability -> degenerate CPD).
constant = [c for c in COLUMNS if df[c].nunique() < 2]
if constant:
    print(f"dropping constant columns: {constant}")
    df = df.drop(columns=constant)

df = df.astype(int)
df.to_csv("voting.csv", index=False)
print(f"\ncleaned rows: {len(df)}")
print(f"remaining columns ({len(df.columns)}): {list(df.columns)}")
print(df.describe().T[["min", "max", "mean"]].to_string())
