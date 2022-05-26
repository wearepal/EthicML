"""Simple upsampler that makes subgroups the same size as the majority group."""
import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Hashable, List, Optional, Tuple

import polars as pd
from ranzen import enum_name_str, implements

from ethicml.utility import DataTuple, SoftPrediction, TestTuple

from ..inprocess.logistic_regression import LR
from .pre_algorithm import PreAlgorithm, T

__all__ = ["Upsampler", "UpsampleStrategy"]


@enum_name_str
class UpsampleStrategy(Enum):
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

    @implements(PreAlgorithm)
    def get_name(self) -> str:
        return f"Upsample {self.strategy}"

    @implements(PreAlgorithm)
    def get_out_size(self) -> int:
        assert self._out_size is not None
        return self._out_size

    @implements(PreAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> Tuple["Upsampler", DataTuple]:
        self._out_size = train.x.shape[1]
        new_train, _ = upsample(train, train, self.strategy, seed, name=self.name)
        return self, new_train

    @implements(PreAlgorithm)
    def transform(self, data: T) -> T:
        return data.replace(name=f"{self.name}: {data.name}")

    @implements(PreAlgorithm)
    def run(
        self, train: DataTuple, test: TestTuple, seed: int = 888
    ) -> Tuple[DataTuple, TestTuple]:
        self._out_size = train.x.shape[1]
        return upsample(train, test, self.strategy, seed, name=self.name)


def concat_datatuples(first_dt: DataTuple, second_dt: DataTuple) -> DataTuple:
    """Given 2 datatuples, concatenate them and shuffle."""
    assert (first_dt.x.columns == second_dt.x.columns).all()  # type: ignore[attr-defined]
    assert first_dt.s.name == second_dt.s.name
    assert first_dt.y.name == second_dt.y.name

    x_columns: pd.Index = first_dt.x.columns
    s_column: Optional[Hashable] = first_dt.s.name
    y_column: Optional[Hashable] = first_dt.y.name
    assert isinstance(s_column, str) and isinstance(y_column, str)

    a_combined: pd.DataFrame = pd.concat([first_dt.x, first_dt.s, first_dt.y], axis="columns")
    b_combined: pd.DataFrame = pd.concat([second_dt.x, second_dt.s, second_dt.y], axis="columns")

    combined = pd.concat([a_combined, b_combined], axis="index")
    combined: pd.DataFrame = combined.sample(frac=1.0, random_state=1).reset_index(drop=True)  # type: ignore[assignment]

    return DataTuple(
        x=combined[x_columns], s=combined[s_column], y=combined[y_column], name=first_dt.name
    )


def upsample(
    dataset: DataTuple,
    test: TestTuple,
    strategy: UpsampleStrategy,
    seed: int,
    name: str,
) -> Tuple[DataTuple, TestTuple]:
    """Upsample a datatuple.

    :param dataset: Dataset that is used to determine the imbalance.
    :param test: Test set that is upsampled along with ``dataset``.
    :param strategy: Strategy to use for upsampling.
    :param seed: Seed for the upsampling.
    :param name: Name of the upsampling strategy.
    """
    s_vals: List[int] = list(map(int, dataset.s.unique()))
    y_vals: List[int] = list(map(int, dataset.y.unique()))

    groups = itertools.product(s_vals, y_vals)

    data: Dict[Tuple[int, int], DataTuple] = {}
    for s, y in groups:
        s_y_mask = (dataset.s == s) & (dataset.y == y)
        data[(s, y)] = DataTuple(
            x=dataset.x.loc[s_y_mask].reset_index(drop=True),
            s=dataset.s.loc[s_y_mask].reset_index(drop=True),
            y=dataset.y.loc[s_y_mask].reset_index(drop=True),
            name=dataset.name,
        )

    percentages: Dict[Tuple[int, int], float] = {}

    vals: List[int] = []
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

    x_columns: pd.Index = dataset.x.columns
    s_column: Optional[Hashable] = dataset.s.name
    y_column: Optional[Hashable] = dataset.y.name
    assert isinstance(s_column, str) and isinstance(y_column, str)

    upsampled: Dict[Tuple[int, int], DataTuple] = {}
    all_data: pd.DataFrame
    for key, val in data.items():
        all_data = pd.concat([val.x, val.s, val.y], axis="columns")
        all_data = all_data.sample(  # type: ignore[assignment]
            frac=percentages[key], random_state=seed, replace=True
        ).reset_index(drop=True)
        upsampled[key] = DataTuple(
            x=all_data[x_columns], s=all_data[s_column], y=all_data[y_column], name=dataset.name
        )

    upsampled_datatuple: Optional[DataTuple] = None
    for key, val in upsampled.items():
        if upsampled_datatuple is None:
            upsampled_datatuple = val
        else:
            upsampled_datatuple = concat_datatuples(upsampled_datatuple, val)
            upsampled_datatuple = upsampled_datatuple.replace(name=f"{name}: {dataset.name}")

    if strategy is UpsampleStrategy.preferential:
        ranker = LR()
        rank: SoftPrediction = ranker.run(dataset, dataset)

        selected: List[pd.DataFrame] = []

        all_data = pd.concat([dataset.x, dataset.s, dataset.y], axis="columns")
        all_data = pd.concat(
            [all_data, pd.DataFrame(rank.soft[:, 1], columns=["preds"])], axis="columns"
        )

        for key, val in data.items():

            s_val = key[0]
            y_val = key[1]
            s_y_mask = (dataset.s == s_val) & (dataset.y == y_val)

            ascending = s_val <= 0

            if percentages[key] > 1.0:
                selected.append(all_data.loc[s_y_mask])
                percentages[key] -= 1.0

            weight = all_data.loc[s_y_mask][y_column].count()
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
        upsampled_datatuple = DataTuple(
            x=upsampled_dataframes[x_columns],
            s=upsampled_dataframes[s_column],
            y=upsampled_dataframes[y_column],
            name=f"{name}: {dataset.name}",
        )

    assert upsampled_datatuple is not None
    return upsampled_datatuple, TestTuple(x=test.x, s=test.s, name=f"{name}: {test.name}")
