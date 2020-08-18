"""Set of scripts for splitting the train and test datasets based on conditions."""

import re
from typing import Hashable, Optional, Tuple

import pandas as pd

from ethicml.utility import DataTuple

__all__ = ["dataset_from_cond", "domain_split", "query_dt"]


def dataset_from_cond(dataset: pd.DataFrame, cond: str) -> pd.DataFrame:
    """Return the dataframe that meets some condition."""
    original_column_names = dataset.columns
    # make column names query-friendly
    dataset = dataset.rename(axis="columns", mapper=make_valid_variable_name)
    subset = dataset.query(cond).reset_index(drop=True)
    subset.columns = original_column_names
    return subset


def query_dt(datatup: DataTuple, query_str: str) -> DataTuple:
    """Query a datatuple."""
    assert isinstance(query_str, str)
    assert isinstance(datatup, DataTuple)

    def _query_func(joined_data_frame: pd.DataFrame) -> pd.DataFrame:
        return dataset_from_cond(joined_data_frame, cond=query_str)

    return datatup.apply_to_joined_df(_query_func)


def domain_split(datatup: DataTuple, tr_cond: str, te_cond: str) -> Tuple[DataTuple, DataTuple]:
    """Splits a datatuple based on a condition.

    Args:
        datatup: DataTuple
        tr_cond: condition for the training set
        te_cond: condition for the test set

    Returns:
        Tuple of DataTuple split into train and test. The test is all those that meet
        the test condition plus the same percentage again of the train set.
    """
    dataset = datatup.x

    train_dataset = dataset_from_cond(dataset, tr_cond)
    test_dataset = dataset_from_cond(dataset, te_cond)

    assert train_dataset.shape[0] + test_dataset.shape[0] == dataset.shape[0]

    test_pct = test_dataset.shape[0] / dataset.shape[0]
    train_pct = 1 - test_pct

    train_train_pcnt = (1 - (test_pct * 2)) / train_pct

    train_train = train_dataset.sample(frac=train_train_pcnt, random_state=888)
    test_train = train_dataset.drop(train_train.index, axis="index")

    test = pd.concat([test_train, test_dataset], axis="index")

    train_x = datatup.x.iloc[train_train.index].reset_index(drop=True)
    train_s = datatup.s.iloc[train_train.index].reset_index(drop=True)
    train_y = datatup.y.iloc[train_train.index].reset_index(drop=True)

    train_datatup = DataTuple(x=train_x, s=train_s, y=train_y, name=datatup.name)

    test_x = datatup.x.iloc[test.index].reset_index(drop=True)
    test_s = datatup.s.iloc[test.index].reset_index(drop=True)
    test_y = datatup.y.iloc[test.index].reset_index(drop=True)

    test_datatup = DataTuple(x=test_x, s=test_s, y=test_y, name=datatup.name)

    return train_datatup, test_datatup


def make_valid_variable_name(name: Optional[Hashable]) -> str:
    """Convert a string into a valid Python variable name."""
    # Ensure that it's a string
    name = str(name)

    # Python variable names may only contain digits, letters and underscores
    explicit_substitutions = {
        "-": "_",  # Replace hyphens with underscores
        "+": "_plus_",
        "=": "_eq_",
    }
    for original, replacement in explicit_substitutions.items():
        name = name.replace(original, replacement)

    # Remove every other disallowed character
    name = re.sub("[^0-9a-zA-Z_]", "", name)

    # Python variables must start with a letter or an underscore
    # Thus, we add an underscore, if the string starts with a digit
    if name[0] in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"):
        name = "_" + name
    return name
