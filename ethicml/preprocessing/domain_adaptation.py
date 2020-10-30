"""Set of scripts for splitting the train and test datasets based on conditions."""

from typing import Tuple

import pandas as pd

from ethicml.utility import DataTuple

__all__ = ["dataset_from_cond", "domain_split", "query_dt"]


def dataset_from_cond(dataset: pd.DataFrame, cond: str) -> pd.DataFrame:
    """Return the dataframe that meets some condition."""
    return dataset.query(cond).reset_index(drop=True)


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
