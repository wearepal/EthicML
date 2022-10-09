"""Simple upsampler that makes subgroups the same size as the majority group."""
from __future__ import annotations
from dataclasses import dataclass
from enum import auto
import itertools
from typing import Optional

import pandas as pd
from ranzen import StrEnum, implements

from ethicml.utility import DataTuple, SoftPrediction

from ..inprocess.logistic_regression import LR
from .pre_algorithm import PreAlgorithm, T

__all__ = ["Upsampler", "UpsampleStrategy"]


class UpsampleStrategy(StrEnum):
    """Strategy for upsampling."""

    uniform = auto()
    preferential = auto()
    naive = auto()


@dataclass
class Upsampler(PreAlgorithm):
    """Upsampler algorithm.

    Given a datatuple, create a larger datatuple such that the subgroups have a balanced number
    of samples.
    """

    strategy: UpsampleStrategy = UpsampleStrategy.uniform

    def __post_init__(self) -> None:
        self._out_size: Optional[int] = None

    @property
    @implements(PreAlgorithm)
    def name(self) -> str:
        return f"Upsample {self.strategy}"

    @property
    @implements(PreAlgorithm)
    def out_size(self) -> int:
        assert self._out_size is not None
        return self._out_size

    @implements(PreAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> tuple[Upsampler, DataTuple]:
        self._out_size = train.x.shape[1]
        new_train, _ = upsample(train, train, self.strategy, seed, name=self.name)
        return self, new_train

    @implements(PreAlgorithm)
    def transform(self, data: T) -> T:
        return data.rename(f"{self.name}: {data.name}")

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: T, seed: int = 888) -> tuple[DataTuple, T]:
        self._out_size = train.x.shape[1]
        return upsample(train, test, self.strategy, seed, name=self.name)


def concat_datatuples(first_dt: DataTuple, second_dt: DataTuple) -> DataTuple:
    """Given 2 datatuples, concatenate them and shuffle."""
    assert (first_dt.x.columns == second_dt.x.columns).all()
    assert first_dt.s_column == second_dt.s_column
    assert first_dt.y_column == second_dt.y_column

    a_combined: pd.DataFrame = first_dt.data
    b_combined: pd.DataFrame = second_dt.data

    combined = pd.concat([a_combined, b_combined], axis="index")
    combined: pd.DataFrame = combined.sample(frac=1.0, random_state=1).reset_index(drop=True)

    return first_dt.replace_data(combined)


def upsample(
    dataset: DataTuple,
    test: T,
    strategy: UpsampleStrategy,
    seed: int,
    name: str,
) -> tuple[DataTuple, T]:
    """Upsample a datatuple.

    :param dataset: Dataset that is used to determine the imbalance.
    :param test: Test set that is upsampled along with ``dataset``.
    :param strategy: Strategy to use for upsampling.
    :param seed: Seed for the upsampling.
    :param name: Name of the upsampling strategy.
    :returns: The upsampled train and test data.
    """
    s_vals: list[int] = list(map(int, dataset.s.unique()))
    y_vals: list[int] = list(map(int, dataset.y.unique()))

    groups = itertools.product(s_vals, y_vals)

    data: dict[tuple[int, int], DataTuple] = {}
    for s, y in groups:
        s_y_mask = (dataset.s == s) & (dataset.y == y)
        data[(s, y)] = dataset.replace_data(dataset.data.loc[s_y_mask].reset_index(drop=True))

    percentages: dict[tuple[int, int], float] = {}

    vals: list[int] = []
    for key, val in data.items():
        vals.append(val.x.shape[0])

    for key, val in data.items():
        if strategy is UpsampleStrategy.naive:
            percentages[key] = max(vals) / val.x.shape[0]
        else:
            s_val: int = key[0]
            y_val: int = key[1]

            y_eq_y = dataset.y.loc[dataset.y == y_val].count()
            s_eq_s = dataset.s.loc[dataset.s == s_val].count()

            num_samples = dataset.y.count()
            num_batch = val.y.count()

            percentages[key] = round((y_eq_y * s_eq_s / (num_batch * num_samples)), 8)

    upsampled: dict[tuple[int, int], DataTuple] = {}
    all_data: pd.DataFrame
    for key, val in data.items():
        all_data = val.data.sample(
            frac=percentages[key], random_state=seed, replace=True
        ).reset_index(drop=True)
        upsampled[key] = val.replace_data(all_data)

    upsampled_datatuple: DataTuple | None = None
    for key, val in upsampled.items():
        if upsampled_datatuple is None:
            upsampled_datatuple = val
        else:
            upsampled_datatuple = concat_datatuples(upsampled_datatuple, val)
            upsampled_datatuple = upsampled_datatuple.rename(f"{name}: {dataset.name}")

    if strategy is UpsampleStrategy.preferential:
        ranker = LR()
        rank: SoftPrediction = ranker.run(dataset, dataset)

        selected: list[pd.DataFrame] = []

        all_data = dataset.data
        all_data = pd.concat(
            [all_data, pd.DataFrame(rank.soft[:, 1], columns=["preds"])], axis="columns"
        )

        for key in data:

            s_val = key[0]
            y_val = key[1]
            s_y_mask = (dataset.s == s_val) & (dataset.y == y_val)

            ascending = s_val <= 0

            if percentages[key] > 1.0:
                selected.append(all_data.loc[s_y_mask])
                percentages[key] -= 1.0

            weight = all_data.loc[s_y_mask][dataset.y_column].count()
            selected.append(
                all_data.loc[s_y_mask]
                .sort_values(by=["preds"], ascending=ascending)
                .iloc[: int(percentages[key] * weight)]
            )

        upsampled_dataframes: pd.DataFrame
        for i, df in enumerate(selected):
            if i == 0:
                upsampled_dataframes = df.drop(["preds"], axis="columns")
            else:
                upsampled_dataframes = pd.concat(
                    [upsampled_dataframes, df.drop(["preds"], axis="columns")], axis="index"
                ).reset_index(drop=True)
        upsampled_datatuple = dataset.replace_data(
            upsampled_dataframes, name=f"{name}: {dataset.name}"
        )

    assert upsampled_datatuple is not None
    return upsampled_datatuple, test.rename(name=f"{name}: {test.name}")
