"""Kamiran&Calders 2012, massaging."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from ranzen import implements

from ethicml.utility import DataTuple, SoftPrediction, concat

from ..inprocess.logistic_regression import LR
from .pre_algorithm import PreAlgorithm, T

__all__ = ["Calders"]


@dataclass
class Calders(PreAlgorithm):
    """Massaging algorithm from Kamiran&Calders 2012."""

    preferable_class: int = 1
    disadvantaged_group: int = 0

    def __post_init__(self) -> None:
        self._out_size: Optional[int] = None

    @property
    @implements(PreAlgorithm)
    def out_size(self) -> int:
        assert self._out_size is not None
        return self._out_size

    @property
    @implements(PreAlgorithm)
    def name(self) -> str:
        return "Calders"

    @implements(PreAlgorithm)
    def fit(self, train: DataTuple, seed: int = 888) -> tuple[Calders, DataTuple]:
        self._out_size = train.x.shape[1]
        new_train, _ = _calders_algorithm(
            train, train, self.preferable_class, self.disadvantaged_group, seed
        )
        return self, new_train.rename(f"{self.name}: {train.name}")

    @implements(PreAlgorithm)
    def transform(self, data: T) -> T:
        return data.rename(f"{self.name}: {data.name}")

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: T, seed: int = 888) -> tuple[DataTuple, T]:
        self._out_size = train.x.shape[1]
        new_train, new_test = _calders_algorithm(
            train, test, self.preferable_class, self.disadvantaged_group, seed
        )
        return new_train.rename(name=f"{self.name}: {train.name}"), new_test.rename(
            name=f"{self.name}: {test.name}"
        )


def _calders_algorithm(
    dataset: DataTuple, test: T, good_class: int, disadvantaged_group: int, seed: int
) -> tuple[DataTuple, T]:
    s_vals: list[int] = list(map(int, dataset.s.unique()))
    y_vals: list[int] = list(map(int, dataset.y.unique()))

    assert len(s_vals) == 2
    assert len(y_vals) == 2
    s_0, s_1 = s_vals
    y_0, y_1 = y_vals

    bad_class = y_0 if good_class == y_1 else y_1
    advantaged_group = s_0 if disadvantaged_group == s_1 else s_1

    groups = ((s_0, y_0), (s_0, y_1), (s_1, y_0), (s_1, y_1))
    data: dict[tuple[int, int], DataTuple] = {}
    for s, y in groups:
        s_y_mask = (dataset.s == s) & (dataset.y == y)
        data[(s, y)] = dataset.replace_data(data=dataset.data.loc[s_y_mask].reset_index(drop=True))

    dis_group = (disadvantaged_group, bad_class)
    adv_group = (advantaged_group, good_class)

    massaging_candidates = concat([data[dis_group], data[adv_group]])

    ranker = LR()
    rank: SoftPrediction = ranker.run(dataset, massaging_candidates, seed)

    dis_group_len = len(data[dis_group])
    adv_group_len = len(data[adv_group])

    dis_group_rank = pd.Series(rank.soft[:dis_group_len][:, 1])
    adv_group_rank = pd.Series(rank.soft[dis_group_len:][:, 1]).reset_index(drop=True)
    assert len(adv_group_rank) == adv_group_len

    # sort the ranking
    dis_group_rank.sort_values(ascending=False, inplace=True)
    adv_group_rank.sort_values(inplace=True)

    # use the rank to sort the data
    for group, ranking in [(dis_group, dis_group_rank), (adv_group, adv_group_rank)]:
        unsorted_data = data[group]
        data[group] = unsorted_data.replace_data(
            data=unsorted_data.data.reindex(index=ranking.index).reset_index(drop=True)
        )

    all_disadvantaged = len(data[(disadvantaged_group, good_class)]) + dis_group_len
    all_advantaged = adv_group_len + len(data[(advantaged_group, bad_class)])
    dis_group_good_len = all_disadvantaged - dis_group_len

    # ensure that the ratio of good_class to bad_class is the same in both groups.
    # for this, we have to swap some labels
    num_to_swap = round(
        (adv_group_len * all_disadvantaged - dis_group_good_len * all_advantaged) / len(dataset)
    )
    data[dis_group].y.iloc[:num_to_swap] = good_class
    data[adv_group].y.iloc[:num_to_swap] = bad_class

    return concat(list(data.values())), test
