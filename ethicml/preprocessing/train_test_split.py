"""
Split into train and test data
"""
from typing import Tuple
import numpy as np
from pandas.testing import assert_index_equal
import pandas as pd

from ethicml.utility.data_structures import DataTuple


def call_numpy_to_split(
    dataframe: pd.DataFrame, train_percentage, random_seed
) -> (Tuple[pd.DataFrame, pd.DataFrame]):
    """Split a DataFrame along the rows"""
    # permute
    dataframe = dataframe.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # split
    train_len = int(train_percentage * len(dataframe))
    train = dataframe.iloc[:train_len]
    test = dataframe.iloc[train_len:]

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def train_test_split(
    data: DataTuple, train_percentage: float = 0.8, random_seed: int = 0
) -> (Tuple[DataTuple, DataTuple]):
    """Split a data tuple into two datatuple along the rows of the DataFrames"""
    x_columns: pd.Index = data.x.columns
    s_columns: pd.Index = data.s.columns
    y_columns: pd.Index = data.y.columns

    all_data: pd.DataFrame = pd.concat([data.x, data.s, data.y], axis="columns")

    all_data = all_data.sample(frac=1, random_state=1).reset_index(drop=True)

    all_data_train_test: Tuple[pd.DataFrame, pd.DataFrame] = call_numpy_to_split(
        all_data, train_percentage, random_seed=random_seed
    )

    all_data_train, all_data_test = all_data_train_test

    train: DataTuple = DataTuple(
        x=all_data_train[x_columns],
        s=all_data_train[s_columns],
        y=all_data_train[y_columns],
        name=f"{data.name} - Train",
    )

    test: DataTuple = DataTuple(
        x=all_data_test[x_columns],
        s=all_data_test[s_columns],
        y=all_data_test[y_columns],
        name=f"{data.name} - Test",
    )

    assert isinstance(train.x, pd.DataFrame)
    assert isinstance(test.x, pd.DataFrame)
    assert_index_equal(train.x.columns, x_columns)
    assert_index_equal(test.x.columns, x_columns)

    assert isinstance(train.s, pd.DataFrame)
    assert isinstance(test.s, pd.DataFrame)
    assert_index_equal(train.s.columns, s_columns)
    assert_index_equal(test.s.columns, s_columns)

    assert isinstance(train.y, pd.DataFrame)
    assert isinstance(test.y, pd.DataFrame)
    assert_index_equal(train.y.columns, y_columns)
    assert_index_equal(test.y.columns, y_columns)

    return train, test


def fold_data(data: DataTuple, folds: int):
    """
    So much love to sklearn for making their source code open
    """

    indices: np.ndarray[int] = np.arange(data.x.shape[0])

    fold_sizes: np.ndarray[int] = np.full(folds, data.x.shape[0] // folds, dtype=int)
    fold_sizes[: data.x.shape[0] % folds] += 1

    current = 0
    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, current + fold_size
        val_inds: np.ndarray[int] = indices[start:stop]
        train_inds = [i for i in indices if i not in val_inds]  # Pretty sure this is inefficient

        train_x = data.x.iloc[train_inds].reset_index(drop=True)
        train_s = data.s.iloc[train_inds].reset_index(drop=True)
        train_y = data.y.iloc[train_inds].reset_index(drop=True)

        assert train_x.shape == (len(train_inds), data.x.shape[1])
        assert train_s.shape == (len(train_inds), data.s.shape[1])
        assert train_y.shape == (len(train_inds), data.y.shape[1])

        val_x = data.x.iloc[val_inds].reset_index(drop=True)
        val_s = data.s.iloc[val_inds].reset_index(drop=True)
        val_y = data.y.iloc[val_inds].reset_index(drop=True)

        assert val_x.shape == (len(val_inds), data.x.shape[1])
        assert val_s.shape == (len(val_inds), data.s.shape[1])
        assert val_y.shape == (len(val_inds), data.y.shape[1])

        yield DataTuple(
            x=train_x, s=train_s, y=train_y, name=f"{data.name} - train fold {i}"
        ), DataTuple(x=val_x, s=val_s, y=val_y, name=f"{data.name} - test fold {i}")

        current = stop
