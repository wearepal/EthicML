"""Generate biased subsets."""
from __future__ import annotations
from typing import Sequence

import numpy as np
import pandas as pd
import teext as tx

from ethicml.utility.data_structures import DataTuple, concat

from .domain_adaptation import query_dt
from .splits import DataSplitter, ProportionalSplit

__all__ = [
    "BiasedDebiasedSubsets",
    "BiasedSubset",
    "get_biased_and_debiased_subsets",
    "get_biased_subset",
]


class BiasedSubset(DataSplitter):
    """Split the given data into a biased subset and a normal subset.

    :param mixing_factors: List of mixing factors; they are chosen based on the split ID
    :param unbiased_pcnt: how much of the data should be reserved for the unbiased subset
    :param seed: random seed for the splitting
    :param data_efficient: if True, try to keep as many data points as possible
    """

    def __init__(
        self,
        unbiased_pcnt: float,
        mixing_factors: Sequence[float] = (0,),
        seed: int = 42,
        data_efficient: bool = True,
    ):
        super().__init__()
        self.unbiased_pcnt = unbiased_pcnt
        self.mixing_factors = mixing_factors
        self.seed = seed
        self.data_efficient = data_efficient

    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> tuple[DataTuple, DataTuple, dict[str, float]]:
        """Do BiasedSplit."""
        mixing_factor = self.mixing_factors[split_id]
        biased, unbiased = get_biased_subset(
            data, mixing_factor, self.unbiased_pcnt, self.seed, self.data_efficient
        )
        return biased, unbiased, {"mix_fact": mixing_factor}


def get_biased_subset(
    data: DataTuple,
    mixing_factor: float,
    unbiased_pcnt: float,
    seed: int = 42,
    data_efficient: bool = True,
) -> tuple[DataTuple, DataTuple]:
    """Split the given data into a biased subset and a normal subset.

    The two subsets don't generally sum up to the whole set.

    Example behavior:

    * mixing_factor=0.0: in biased, s=y everywhere; unbiased is just a subset of `data`
    * mixing_factor=0.5: biased and unbiased are both just subsets of `data`
    * mixing_factor=1.0: in biased, s!=y everywhere; unbiased is just a subset of `data`

    :param data: data in form of a DataTuple
    :param mixing_factor: How much of the debiased data should be mixed into the biased subset? If this
                       factor is 0, the biased subset is maximally biased.
    :param unbiased_pcnt: how much of the data should be reserved for the unbiased subset
    :param seed: random seed for the splitting (Default: 42)
    :param data_efficient: if True, try to keep as many data points as possible (Default: True)
    :returns: biased and unbiased dataset
    """
    assert tx.is_percentage(mixing_factor), f"mixing_factor: {mixing_factor}"
    assert tx.is_percentage(unbiased_pcnt), f"unbiased_pcnt: {unbiased_pcnt}"
    s_name = data.s.name
    y_name = data.y.name
    assert isinstance(s_name, str) and isinstance(y_name, str)

    normal_subset, for_biased_subset = _random_split(data, first_pcnt=unbiased_pcnt, seed=seed)

    sy_equal, sy_opposite = _get_sy_equal_and_opp(for_biased_subset, s_name, y_name)

    mix_fact = mixing_factor  # how much of sy_opp should be mixed into the biased subset

    if data_efficient:
        if mix_fact < 0.5:
            sy_equal_fraction = 1.0
            sy_opp_fraction = 2 * mix_fact
        else:
            sy_equal_fraction = 2 * (1 - mix_fact)
            sy_opp_fraction = 1.0
    else:
        sy_equal_fraction = 1 - mix_fact
        sy_opp_fraction = mix_fact

    sy_equal_for_biased_ss, _ = _random_split(sy_equal, first_pcnt=sy_equal_fraction, seed=seed)
    sy_opp_for_biased_ss, _ = _random_split(sy_opposite, first_pcnt=sy_opp_fraction, seed=seed)

    biased_subset = concat([sy_equal_for_biased_ss, sy_opp_for_biased_ss], ignore_index=True)

    if mix_fact == 0.0:
        # s and y should be very correlated in the biased subset
        assert all(biased_subset.s == biased_subset.y)

    biased_subset = biased_subset.rename(f"{data.name} - Biased (tm={mixing_factor})")
    normal_subset = normal_subset.rename(f"{data.name} - Subset (tm={mixing_factor})")
    return biased_subset, normal_subset


class BiasedDebiasedSubsets(DataSplitter):
    """Split the given data into a biased subset and a debiased subset.

    :param mixing_factors: List of mixing factors; they are chosen based on the split ID
    :param unbiased_pcnt: how much of the data should be reserved for the unbiased subset
    :param seed: random seed for the splitting
    :param fixed_unbiased: if True, then the unbiased dataset is independent from the mixing factor
    """

    def __init__(
        self,
        unbiased_pcnt: float,
        mixing_factors: Sequence[float] = (0,),
        seed: int = 42,
        fixed_unbiased: bool = True,
    ):
        super().__init__()
        self.unbiased_pcnt = unbiased_pcnt
        self.mixing_factors = mixing_factors
        self.seed = seed
        self.fixed_unbiased = fixed_unbiased

    def __call__(
        self, data: DataTuple, split_id: int = 0
    ) -> tuple[DataTuple, DataTuple, dict[str, float]]:
        """Do Biased and Debiased Split."""
        mixing_factor = self.mixing_factors[split_id]
        biased, unbiased = get_biased_and_debiased_subsets(
            data, mixing_factor, self.unbiased_pcnt, self.seed, self.fixed_unbiased
        )
        return biased, unbiased, {"mix_fact": mixing_factor}


def get_biased_and_debiased_subsets(
    data: DataTuple,
    mixing_factor: float,
    unbiased_pcnt: float,
    seed: int = 42,
    fixed_unbiased: bool = True,
) -> tuple[DataTuple, DataTuple]:
    """Split the given data into a biased subset and a debiased subset.

    In contrast to :func:`get_biased_subset()`, this function makes the unbiased subset *really*
    unbiased.

    The two subsets don't generally sum up to the whole set.

    Example behavior:

    * mixing_factor=0.0: in biased, s=y everywhere; in debiased, 50% s=y and 50% s!=y
    * mixing_factor=0.5: biased is just a subset of `data`; in debiased, 50% s=y and 50% s!=y
    * mixing_factor=1.0: in biased, s!=y everywhere; in debiased, 50% s=y and 50% s!=y

    :param data: data in form of a DataTuple
    :param mixing_factor: How much of the debiased data should be mixed into the biased subset? If this
                       factor is 0, the biased subset is maximally biased.
    :param unbiased_pcnt: how much of the data should be reserved for the unbiased subset
    :param seed: random seed for the splitting (Default: 42)
    :param fixed_unbiased: if True, then the unbiased dataset is independent from the mixing factor (Default: True)
    :returns: biased and unbiased dataset
    """
    assert tx.is_percentage(mixing_factor), f"mixing_factor: {mixing_factor}"
    assert tx.is_percentage(unbiased_pcnt), f"unbiased_pcnt: {unbiased_pcnt}"
    s_name = data.s.name
    y_name = data.y.name
    assert isinstance(s_name, str) and isinstance(y_name, str)
    sy_equal, sy_opposite = _get_sy_equal_and_opp(data, s_name, y_name)

    # how much of sy_equal should be reserved for the biased subset:

    if fixed_unbiased:
        sy_equal_for_debiased_ss, sy_equal_remaining = _random_split(
            sy_equal, first_pcnt=unbiased_pcnt, seed=seed
        )
        sy_opp_for_debiased_ss, sy_opp_remaining = _random_split(
            sy_opposite, first_pcnt=unbiased_pcnt, seed=seed
        )
        if mixing_factor < 0.5:
            sy_equal_fraction = 1.0
            sy_opp_fraction = 2 * mixing_factor
        else:
            sy_equal_fraction = 2 * (1 - mixing_factor)
            sy_opp_fraction = 1.0

        sy_equal_for_biased_ss, _ = _random_split(
            sy_equal_remaining, first_pcnt=sy_equal_fraction, seed=seed
        )
        sy_opp_for_biased_ss, _ = _random_split(
            sy_opp_remaining, first_pcnt=sy_opp_fraction, seed=seed
        )
    else:
        biased_pcnt = 1 - unbiased_pcnt
        sy_equal_for_biased_ss, sy_equal_for_debiased_ss = _random_split(
            sy_equal, first_pcnt=biased_pcnt * (1 - mixing_factor), seed=seed
        )
        sy_opp_for_biased_ss, sy_opp_for_debiased_ss = _random_split(
            sy_opposite, first_pcnt=biased_pcnt * mixing_factor, seed=seed
        )

    biased_subset = concat([sy_equal_for_biased_ss, sy_opp_for_biased_ss], ignore_index=True)

    # the debiased set is constructed from two sets of the same size
    min_size = min(len(sy_equal_for_debiased_ss), len(sy_opp_for_debiased_ss))

    def _get_equal_sized_subset(df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(n=min_size, random_state=seed)

    debiased_subset_part1 = sy_equal_for_debiased_ss.apply_to_joined_df(_get_equal_sized_subset)
    debiased_subset_part2 = sy_opp_for_debiased_ss.apply_to_joined_df(_get_equal_sized_subset)
    debiased_subset = concat([debiased_subset_part1, debiased_subset_part2], ignore_index=True)

    # s and y should not be correlated in the debiased subset
    assert abs(debiased_subset.s.corr(debiased_subset.y)) < 0.5

    if mixing_factor == 0.0:
        # s and y should be very correlated in the biased subset
        assert biased_subset.s.corr(biased_subset.y) > 0.99

    biased_subset = biased_subset.rename(f"{data.name} - Biased (tm={mixing_factor})")
    debiased_subset = debiased_subset.rename(f"{data.name} - Debiased (tm={mixing_factor})")
    return biased_subset, debiased_subset


def _random_split(data: DataTuple, first_pcnt: float, seed: int) -> tuple[DataTuple, DataTuple]:
    if len(data) == 0:
        return data, data
    splitter = ProportionalSplit(train_percentage=first_pcnt, start_seed=seed)
    return splitter(data)[:2]


def _get_sy_equal_and_opp(data: DataTuple, s_name: str, y_name: str) -> tuple[DataTuple, DataTuple]:
    """Get the subset where s and y are equal and the subset where they are opposite."""
    s_values = np.unique(data.s.to_numpy())
    y_values = np.unique(data.y.to_numpy())
    assert len(s_values) == 2, "function only works with binary sensitive attribute"
    assert len(y_values) == 2, "function only works with binary labels"
    # we implicitly assume that the minimum value of s and y are equal; same for the maximum value
    s0, s1 = s_values
    y0, y1 = y_values
    # datapoints where s and y are the same
    s_and_y_equal = query_dt(
        data,
        f"(`{s_name}` == {s0} & `{y_name}` == {y0}) | (`{s_name}` == {s1} & `{y_name}` == {y1})",
    )
    # datapoints where they are not the same
    s_and_y_opposite = query_dt(
        data,
        f"(`{s_name}` == {s0} & `{y_name}` == {y1}) | (`{s_name}` == {s1} & `{y_name}` == {y0})",
    )
    return s_and_y_equal, s_and_y_opposite
