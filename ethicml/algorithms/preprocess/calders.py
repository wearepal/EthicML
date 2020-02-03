"""Kamiran&Calders 2012, massaging."""
from typing import List, Tuple, Dict

from ethicml.common import implements
from ethicml.algorithms.inprocess import LRProb
from ethicml.utility import TestTuple, DataTuple, SoftPrediction, concat_dt
from .pre_algorithm import PreAlgorithm


class Calders(PreAlgorithm):
    """Upsampler algorithm.

    Given a datatuple, create a larger datatuple such that the subgroups have a balanced number
    of samples.
    """

    @implements(PreAlgorithm)
    def run(self, train: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
        return calders_algorithm(train, test)

    @property
    def name(self) -> str:
        """Getter for algorithm name."""
        return "Calders"


def calders_algorithm(dataset: DataTuple, test: TestTuple) -> Tuple[DataTuple, TestTuple]:
    """Massaging."""
    s_col = dataset.s.columns[0]
    y_col = dataset.y.columns[0]

    s_vals: List[int] = list(map(int, dataset.s[s_col].unique()))
    y_vals: List[int] = list(map(int, dataset.y[y_col].unique()))

    assert len(s_vals) == 2
    assert len(y_vals) == 2
    s_0, s_1 = s_vals
    y_0, y_1 = y_vals

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

    # find the smallest quadrant (i.e. the minimum in `groups` according to the length of the data)
    min_group_id = min(range(len(groups)), key=lambda group_id: len(data[groups[group_id]]))
    # mapping that maps each group to its opposite (that is, s0->s1 and y0->y1 and so on)
    opposite_groups = {0: 3, 1: 2, 2: 1, 3: 0}
    opp_group_id = opposite_groups[min_group_id]

    min_group = groups[min_group_id]
    opp_group = groups[opp_group_id]

    massaging_candidates = concat_dt([data[min_group], data[opp_group]])

    ranker = LRProb()
    rank: SoftPrediction = ranker.run(dataset, massaging_candidates)

    min_group_len = len(data[min_group])
    opp_group_len = len(data[opp_group])

    min_group_rank = rank.soft.iloc[:min_group_len]
    opp_group_rank = rank.soft.iloc[min_group_len:].reset_index(drop=True)
    assert len(opp_group_rank) == opp_group_len

    # sort the ranking
    min_group_rank.sort_values(ascending=False, inplace=True)
    opp_group_rank.sort_values(inplace=True)

    # use the rank to sort the data
    for group, ranking in [(min_group, min_group_rank), (opp_group, opp_group_rank)]:
        unsorted_data = data[group]
        data[group] = DataTuple(
            x=unsorted_data.x.reindex(index=ranking.index).reset_index(drop=True),
            s=unsorted_data.s.reindex(index=ranking.index).reset_index(drop=True),
            y=unsorted_data.y.reindex(index=ranking.index).reset_index(drop=True),
            name=unsorted_data.name,
        )

    all_for_s_min = sum(len(quadrant) for (s, y), quadrant in data.items() if s == min_group[0])
    all_for_s_opp = sum(len(quadrant) for (s, y), quadrant in data.items() if s == opp_group[0])

    num_to_swap = round(
        (opp_group_len * all_for_s_min - (all_for_s_min - min_group_len) * all_for_s_opp) / len(dataset)
    )

    temp = data[min_group].y.iloc[:num_to_swap]
    data[min_group].y.iloc[:num_to_swap] = data[opp_group].y.iloc[:num_to_swap]
    data[opp_group].y.iloc[:num_to_swap] = temp

    return concat_dt(list(data.values())), test
