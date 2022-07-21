"""Transparently show how the UCI Nursey dataset was modified from the raw download."""
# The Heritage Health dataset. It needs some (mild) preprocessing before we can plug and play.

from typing import Hashable, List

import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv(
        "raw/nursery_raw.csv",
        names=[
            "parents",
            "has_nurs",
            "form",
            "children",
            "housing",
            "finance",
            "social",
            "health",
            "class",
        ],
    )

    # Drop rows
    # indexNames = df[df['class'] == 'recommend'].index
    # df.drop(indexNames, inplace=True)

    # Add binary class column
    features = ["children"]
    for column in features:
        df[column] = df[column].astype("category").cat.codes

    features1: List[Hashable] = [
        'form',
        'health',
        "finance",
        "class",
        "parents",
        "has_nurs",
        "housing",
        "social",
    ]
    x1 = df.drop(features1, axis=1)

    x2 = df.drop(features, axis=1)
    x2 = pd.get_dummies(x2)

    df = pd.concat([x1, x2], axis=1)

    # Shuffle the data
    df = df.sample(frac=1.0, random_state=888).reset_index(drop=True)

    # Save the CSV
    compression_opts = dict(method='zip', archive_name='nursery.csv')
    df.to_csv("./nursery.csv.zip", index=False, compression=compression_opts)
