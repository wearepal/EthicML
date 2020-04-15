"""Transparently show how the Hearitage Health dataset was modified from the raw download."""
# The Heritage Health dataset. It needs some (mild) preprocessing before we can plug and play.

import pandas as pd

df = pd.read_csv("raw/health.csv")

# Drop columns which contains NaNs
print(f"Dropping columns: {df.columns[df.isna().any()].tolist()}")
df = df.dropna(axis=1)

# Add binary class column
df["Charlson>0"] = df["CharlsonIndexI_max"] > 0
df["Charlson>0"] = df["Charlson>0"].astype(int)

# Drop rows with missing age and sex
df = df[(df["age_MISS"] == 0) & (df["sexMISS"] == 0)]

# Shuffle the data
df = df.sample(frac=1.0, random_state=888).reset_index(drop=True)

# Save the CSV
df.to_csv("./health.csv", index=False)
