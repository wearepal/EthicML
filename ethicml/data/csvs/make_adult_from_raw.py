"""Transparently show how the UCI Adult dataset was generated from the raw download."""

import numpy as np
import pandas as pd


def run_generate_adult() -> None:
    """Generate the UCI Adult dataset from scratch."""
    # Load the data
    train = pd.read_csv("raw/adult.data", header=None)  # type: ignore[call-overload]
    test = pd.read_csv("raw/adult.test", skiprows=[0], header=None)  # type: ignore[call-overload]

    # Give data column names
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]

    train.columns = pd.Index(columns)
    test.columns = pd.Index(columns)

    # Concat the data
    all_data = pd.concat([train, test], axis=0)

    for col in all_data.columns:
        if all_data[col].dtype == np.object:  # type: ignore[attr-defined]
            all_data[col] = all_data[col].str.strip()

    # Replace full stop in the label of the test set
    all_data = all_data.replace("<=50K.", np.str_("<=50K"))
    all_data = all_data.replace(">50K.", np.str_(">50K"))

    # Drop NaNs
    all_data = all_data.replace(r"^\s*\?+\s*$", np.nan, regex=True).dropna()

    # OHE
    all_data = pd.get_dummies(all_data)

    # Shuffle the data
    all_data = all_data.sample(frac=1.0, random_state=888).reset_index(drop=True)

    # Save the CSV
    all_data.to_csv("./adult.csv", index=False)


if __name__ == "__main__":
    run_generate_adult()
