"""Split into train and test data."""
import itertools
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple
from typing_extensions import Literal

import numpy as np
import pandas as pd
from numpy.random import RandomState
from ranzen import implements

from ethicml.utility import DataTuple
from ethicml.utility.data_helpers import shuffle_df

__all__ = [
    "BalancedTestSplit",
    "DataSplitter",
    "ProportionalSplit",
    "RandomSplit",
    "SequentialSplit",
    "train_test_split",
]


class DataSplitter(ABC):
    """Base class for classes that split data."""

    @abstractmethod
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        """Split the given data.

        :param data: the data to split
        :param split_id: ID that makes this split unique; this affects for example the random seed
        :returns: A tuple consisting of two datatuples that are the result of the split and a
            dictionary that contains information about the split that is useful for logging.
        """


class SequentialSplit(DataSplitter):
    """Take the first N samples for train set and the rest for test set; no shuffle."""

    def __init__(self, train_percentage: float):
        self.train_percentage = train_percentage

    @implements(DataSplitter)
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        del split_id
        train_len = round(self.train_percentage * len(data))

        train = data.apply_to_joined_df(lambda df: df.iloc[:train_len].reset_index(drop=True))
        train = train.replace(name=f"{data.name} - Train")

        test = data.apply_to_joined_df(lambda df: df.iloc[train_len:].reset_index(drop=True))
        test = test.replace(name=f"{data.name} - Test")

        assert len(train) + len(test) == len(data)
        return train, test, {}


def train_test_split(
    data: DataTuple,
    train_percentage: Optional[float] = 0.8,
    random_seed: int = 0,
    num_test_samples: Optional[int] = None,
) -> Tuple[DataTuple, DataTuple]:
    """Split a data tuple into two datatuple along the rows of the DataFrames.

    :param data: data tuple to split
    :param train_percentage: percentage for train split. Must be `None` if `num_test_samples` given. (Default: 0.8)
    :param random_seed: seed to make splitting reproducible (Default: 0)
    :param num_test_samples: Number of samples to make the test set. Must be `None` if `train_percentage` given. (Default: None)
    :returns: train split and test split
    """
    if train_percentage is None:
        assert num_test_samples is not None
    if num_test_samples is None:
        assert train_percentage is not None

    # ======================== concatenate the datatuple to one dataframe =========================
    # save the column names for later
    x_columns: pd.Index = data.x.columns
    s_column = data.s.name
    y_column = data.y.name
    assert isinstance(s_column, str) and isinstance(y_column, str)

    all_data: pd.DataFrame = pd.concat([data.x, data.s, data.y], axis="columns")

    all_data = shuffle_df(all_data, random_state=1)

    # ============================== split the concatenated dataframe =============================
    # permute
    all_data = shuffle_df(all_data, random_state=random_seed)

    # split
    if train_percentage is not None:
        train_len = int(train_percentage * len(all_data))
    else:
        train_len = len(all_data) - num_test_samples
    all_data_train = all_data.iloc[:train_len]
    all_data_test = all_data.iloc[train_len:]

    assert isinstance(all_data_train, pd.DataFrame)
    assert isinstance(all_data_test, pd.DataFrame)

    all_data_train = all_data_train.reset_index(drop=True)
    all_data_test = all_data_test.reset_index(drop=True)

    # ================================== assemble train and test ==================================
    train: DataTuple = DataTuple(
        x=all_data_train[x_columns],
        s=all_data_train[s_column],
        y=all_data_train[y_column],
        name=f"{data.name} - Train",
    )

    test: DataTuple = DataTuple(
        x=all_data_test[x_columns],
        s=all_data_test[s_column],
        y=all_data_test[y_column],
        name=f"{data.name} - Test",
    )

    assert isinstance(train.x, pd.DataFrame)
    assert isinstance(test.x, pd.DataFrame)
    pd.testing.assert_index_equal(train.x.columns, x_columns)
    pd.testing.assert_index_equal(test.x.columns, x_columns)

    assert isinstance(train.s, pd.Series)
    assert isinstance(test.s, pd.Series)
    assert train.s.name == s_column
    assert test.s.name == s_column

    assert isinstance(train.y, pd.Series)
    assert isinstance(test.y, pd.Series)
    assert train.y.name == y_column
    assert test.y.name == y_column

    return train, test


class RandomSplit(DataSplitter):
    """Standard train test split.

    :param train_percentage: how much of the data to use for the train split
    :param start_seed: random seed for the first split
    """

    def __init__(self, train_percentage: float = 0.8, start_seed: Optional[int] = 0):
        super().__init__()
        self.start_seed = start_seed
        self.train_percentage = train_percentage

    def _get_seed(self, split_id: int) -> int:
        return self.start_seed + 2410 * split_id if self.start_seed is not None else 0

    @implements(DataSplitter)
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        random_seed = self._get_seed(split_id)
        split_info: Dict[str, float] = {"seed": random_seed}
        return train_test_split(data, self.train_percentage, random_seed) + (split_info,)


def generate_proportional_split_indexes(
    data: DataTuple, train_percentage: float, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the indexes of the train and test splits using a proportional sampling scheme.

    :param data:
    :param train_percentage:
    :param random_seed:  (Default: 42)
    """
    # local random state that won't affect the global state
    random = RandomState(seed=random_seed)

    s_vals: List[int] = list(map(int, data.s.unique()))
    y_vals: List[int] = list(map(int, data.y.unique()))

    train_indexes: List[np.ndarray] = []
    test_indexes: List[np.ndarray] = []

    # iterate over all combinations of s and y
    for s, y in itertools.product(s_vals, y_vals):
        # find all indices for this group
        idx = ((data.s == s) & (data.y == y)).to_numpy().nonzero()[0]

        # shuffle and take subsets
        random.shuffle(idx)
        split_indexes: int = round(len(idx) * train_percentage)
        # append index subsets to the list of train indices
        train_indexes.append(idx[:split_indexes])
        test_indexes.append(idx[split_indexes:])

    train_indexes_ = np.concatenate(train_indexes, axis=0)
    test_indexes_ = np.concatenate(test_indexes, axis=0)
    del train_indexes
    del test_indexes

    num_groups = len(s_vals) * len(y_vals)
    expected_train_len = round(len(data) * train_percentage)
    # assert that we (at least approximately) achieved the specified `train_percentage`
    # the maximum error occurs when all the group splits favor train or all favor test
    assert expected_train_len - num_groups <= len(train_indexes_) <= expected_train_len + num_groups

    return train_indexes_, test_indexes_


class ProportionalSplit(RandomSplit):
    """Split into train and test while preserving the proportion of s and y."""

    @implements(DataSplitter)
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        random_seed = self._get_seed(split_id)
        train_indexes, test_indexes = generate_proportional_split_indexes(
            data, train_percentage=self.train_percentage, random_seed=random_seed
        )

        train: DataTuple = DataTuple(
            x=data.x.iloc[train_indexes].reset_index(drop=True),
            s=data.s.iloc[train_indexes].reset_index(drop=True),  # type: ignore[call-overload]
            y=data.y.iloc[train_indexes].reset_index(drop=True),  # type: ignore[call-overload]
            name=f"{data.name} - Train",
        )

        test: DataTuple = DataTuple(
            x=data.x.iloc[test_indexes].reset_index(drop=True),
            s=data.s.iloc[test_indexes].reset_index(drop=True),  # type: ignore[call-overload]
            y=data.y.iloc[test_indexes].reset_index(drop=True),  # type: ignore[call-overload]
            name=f"{data.name} - Test",
        )

        # assert that no data points got lost anywhere
        assert len(data) == len(train) + len(test)

        split_info: Dict[str, float] = {"seed": random_seed}

        return train, test, split_info


class BalancedTestSplit(RandomSplit):
    """Split data such that the test set is balanced and the training set is proportional.

    :param balance_type: how to do the balancing
    :param train_percentage: how much of the data to use for the train split
    :param start_seed: random seed for the first split
    """

    def __init__(
        self,
        balance_type: Literal["P(s|y)=0.5", "P(y|s)=0.5", "P(s,y)=0.25"] = "P(s|y)=0.5",
        train_percentage: float = 0.8,
        start_seed: int = 0,
    ):
        super().__init__(start_seed=start_seed, train_percentage=train_percentage)
        self.balance_type = balance_type

    @implements(DataSplitter)
    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> Tuple[DataTuple, DataTuple, Dict[str, float]]:
        random_seed = self._get_seed(split_id)
        random = RandomState(seed=random_seed)

        s_vals: List[int] = list(map(int, data.s.unique()))
        y_vals: List[int] = list(map(int, data.y.unique()))

        train_indexes: List[np.ndarray] = []
        test_indexes: List[np.ndarray] = []

        num_test: Dict[Tuple[int, int], int] = {}
        # find out how many samples are available for the test set
        for s, y in itertools.product(s_vals, y_vals):
            # find all indices for this group
            idx = ((data.s == s) & (data.y == y)).to_numpy().nonzero()[0]
            # how many elements are in this "quadrant"
            quadrant_size = len(idx)
            # compute how many elements would be available for the test set
            num_test[(s, y)] = round(quadrant_size * (1 - self.train_percentage))

        # compute how much we should take for the test set to make it balanced
        if self.balance_type == "P(s|y)=0.5":
            minimize_over_s = {y: min(num_test[(s, y)] for s in s_vals) for y in y_vals}
            num_test_balanced = {(s, y): minimize_over_s[y] for s in s_vals for y in y_vals}
        elif self.balance_type == "P(y|s)=0.5":
            minimize_over_y = {s: min(num_test[(s, y)] for y in y_vals) for s in s_vals}
            num_test_balanced = {(s, y): minimize_over_y[s] for s in s_vals for y in y_vals}
        elif self.balance_type == "P(s,y)=0.25":
            smallest_quadrant = min(num_test[(s, y)] for s in s_vals for y in y_vals)
            num_test_balanced = {(s, y): smallest_quadrant for s in s_vals for y in y_vals}
        else:
            raise ValueError("Unknown balance_type")

        num_dropped = 0
        # iterate over all combinations of s and y
        for s, y in itertools.product(s_vals, y_vals):
            # find all indices for this group
            idx = ((data.s == s) & (data.y == y)).to_numpy().nonzero()[0]

            # shuffle and take subsets
            random.shuffle(idx)
            split_indexes: int = round(len(idx) * self.train_percentage)
            # append index subsets to the list of train indices
            train_indexes.append(idx[:split_indexes])
            test_indexes.append(idx[split_indexes : (split_indexes + num_test_balanced[(s, y)])])
            num_dropped += num_test[(s, y)] - num_test_balanced[(s, y)]

        train_idx = np.concatenate(train_indexes, axis=0)
        test_idx = np.concatenate(test_indexes, axis=0)

        train: DataTuple = DataTuple(
            x=data.x.iloc[train_idx].reset_index(drop=True),
            s=data.s.iloc[train_idx].reset_index(drop=True),
            y=data.y.iloc[train_idx].reset_index(drop=True),
            name=f"{data.name} - Train",
        )

        test: DataTuple = DataTuple(
            x=data.x.iloc[test_idx].reset_index(drop=True),
            s=data.s.iloc[test_idx].reset_index(drop=True),
            y=data.y.iloc[test_idx].reset_index(drop=True),
            name=f"{data.name} - Test",
        )

        unbalanced_test_len = round(len(data) * (1 - self.train_percentage))
        split_info = {
            "seed": random_seed,
            "percent_dropped": num_dropped / unbalanced_test_len,
            self.balance_type: 1,
        }

        return train, test, split_info


def fold_data(data: DataTuple, folds: int) -> Iterator[Tuple[DataTuple, DataTuple]]:
    """So much love to sklearn for making their source code open.

    :param data:
    :param folds:
    """
    indices: np.ndarray = np.arange(data.x.shape[0])

    fold_sizes: np.ndarray = np.full(folds, data.x.shape[0] // folds, dtype=np.int32)
    fold_sizes[: data.x.shape[0] % folds] += np.int32(1)

    current = 0
    for i, fold_size in enumerate(fold_sizes):
        start, stop = current, int(current + fold_size)
        val_inds: np.ndarray = indices[start:stop]
        train_inds = np.array([i for i in indices if i not in val_inds])  # probably inefficient

        train_x = data.x.iloc[train_inds].reset_index(drop=True)
        train_s = data.s.iloc[train_inds].reset_index(drop=True)  # type: ignore[call-overload]
        train_y = data.y.iloc[train_inds].reset_index(drop=True)  # type: ignore[call-overload]

        assert train_x.shape == (len(train_inds), data.x.shape[1])
        assert train_s.shape == (len(train_inds),)
        assert train_y.shape == (len(train_inds),)

        val_x = data.x.iloc[val_inds].reset_index(drop=True)
        val_s = data.s.iloc[val_inds].reset_index(drop=True)  # type: ignore[call-overload]
        val_y = data.y.iloc[val_inds].reset_index(drop=True)  # type: ignore[call-overload]

        assert val_x.shape == (len(val_inds), data.x.shape[1])
        assert val_s.shape == (len(val_inds),)
        assert val_y.shape == (len(val_inds),)

        yield DataTuple(
            x=train_x, s=train_s, y=train_y, name=f"{data.name} - train fold {i}"
        ), DataTuple(x=val_x, s=val_s, y=val_y, name=f"{data.name} - test fold {i}")

        current = stop