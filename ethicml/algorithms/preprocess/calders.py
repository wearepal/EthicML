"""Kamiran&Calders 2012, massaging."""
from typing import Dict, List, Tuple

from kit import implements

from ethicml.utility import DataTuple, SoftPrediction, TestTuple, concat_dt

from ..inprocess.logistic_regression import LRProb
from .pre_algorithm import PreAlgorithm

__all__ = ["Calders"]


class Calders(PreAlgorithm):
    """Massaging algorithm from Kamiran&Calders 2012."""

    def __init__(self, preferable_class: int, disadvantaged_group: int):
        super().__init__(name="Calders")
        self.preferable_class = preferable_class
        self.disadvantaged_group = disadvantaged_group

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        return _calders_algorithm(train, test, self.preferable_class, self.disadvantaged_group)


def _calders_algorithm(
    dataset: DataTuple, test: TestTuple, good_class: int, disadvantaged_group: int
) -> Tuple[DataTuple, TestTuple]:
    s_col = dataset.s.columns[0]
    y_col = dataset.y.columns[0]

    s_vals: List[int] = list(map(int, dataset.s[s_col].unique()))
    y_vals: List[int] = list(map(int, dataset.y[y_col].unique()))

    assert len(s_vals) == 2
    assert len(y_vals) == 2
    s_0, s_1 = s_vals
    y_0, y_1 = y_vals

    bad_class = y_0 if good_class == y_1 else y_1
    advantaged_group = s_0 if disadvantaged_group == s_1 else s_1

    groups = ((s_0, y_0), (s_0, y_1), (s_1, y_0), (s_1, y_1))
    data: Dict[Tuple[int, int], DataTuple] = {}
    for s, y in groups:
        s_y_mask = (dataset.s[s_col] == s) & (dataset.y[y_col] == y)
        data[(s, y)] = DataTuple(
            x=dataset.x.loc[s_y_mask].reset_index(drop=True),
            s=dataset.s.loc[s_y_mask].reset_index(drop=True),
            y=dataset.y.loc[s_y_mask].reset_index(drop=True),
            name=dataset.name,
        )

    dis_group = (disadvantaged_group, bad_class)
    adv_group = (advantaged_group, good_class)

    massaging_candidates = concat_dt([data[dis_group], data[adv_group]])

    ranker = LRProb()
    rank: SoftPrediction = ranker.run(dataset, massaging_candidates)

    dis_group_len = len(data[dis_group])
    adv_group_len = len(data[adv_group])

    dis_group_rank = rank.soft.iloc[:dis_group_len]
    adv_group_rank = rank.soft.iloc[dis_group_len:].reset_index(drop=True)
    assert len(adv_group_rank) == adv_group_len

    # sort the ranking
    dis_group_rank.sort_values(ascending=False, inplace=True)
    adv_group_rank.sort_values(inplace=True)

    # use the rank to sort the data
    for group, ranking in [(dis_group, dis_group_rank), (adv_group, adv_group_rank)]:
        unsorted_data = data[group]
        data[group] = DataTuple(
            x=unsorted_data.x.reindex(index=ranking.index).reset_index(drop=True),
            s=unsorted_data.s.reindex(index=ranking.index).reset_index(drop=True),
            y=unsorted_data.y.reindex(index=ranking.index).reset_index(drop=True),
            name=unsorted_data.name,
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

    return concat_dt(list(data.values())), test
