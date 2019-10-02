"""Generate biased subsets"""
from typing import Tuple

import pandas as pd

from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.preprocessing.domain_adaptation import query_dt, make_valid_variable_name
from ethicml.utility.data_structures import concat_dt, DataTuple


def _random_split(data: DataTuple, first_pcnt: float, seed: int) -> Tuple[DataTuple, DataTuple]:
    return train_test_split(data, train_percentage=first_pcnt, random_seed=seed)


def _get_sy_equal_and_opp(data: DataTuple, s_name: str, y_name: str) -> Tuple[DataTuple, DataTuple]:
    """Get the subset where s and y are equal and the subset where they are opposite"""
    s_name = make_valid_variable_name(s_name)
    y_name = make_valid_variable_name(y_name)
    # datapoints where s and y are the same
    s_and_y_equal = query_dt(
        data, f"({s_name} == 0 & {y_name} == 0) | ({s_name} == 1 & {y_name} == 1)"
    )
    # datapoints where they are not the same
    s_and_y_opposite = query_dt(
        data, f"({s_name} == 1 & {y_name} == 0) | ({s_name} == 0 & {y_name} == 1)"
    )
    return s_and_y_equal, s_and_y_opposite


def get_biased_subset(
    data: DataTuple,
    mixing_factor: float,
    unbiased_pcnt: float,
    seed: int = 42,
    data_efficient: bool = True,
) -> Tuple[DataTuple, DataTuple]:
    """Split the given data into a biased subset and a normal subset

    The two subsets don't generally sum up to the whole set.

    Example behavior:

    * mixing_factor=0.0: in biased, s=y everywhere; unbiased is just a subset of `data`
    * mixing_factor=0.5: biased and unbiased are both just subsets of `data`
    * mixing_factor=1.0: in biased, s!=y everywhere; unbiased is just a subset of `data`

    Args:
        data: data in form of a DataTuple
        mixing_factor: How much of the debiased data should be mixed into the biased subset? If this
                       factor is 0, the biased subset is maximally biased.
        unbiased_pcnt: how much of the data should be reserved for the unbiased subset
        seed: random seed for the splitting
        data_efficient: if True, try to keep as many data points as possible

    Returns:
        biased and unbiased dataset
    """
    s_name = data.s.columns[0]
    y_name = data.y.columns[0]

    # task and meta are just normal subsets
    for_biased_subset, normal_subset = _random_split(data, first_pcnt=unbiased_pcnt, seed=seed)

    sy_equal, sy_opposite = _get_sy_equal_and_opp(for_biased_subset, s_name, y_name)

    mix_fact = mixing_factor  # how much of sy_opp should be mixed into task train set

    if data_efficient:
        sy_equal_fraction = min(1.0, 2 * (1 - mix_fact))
        sy_opp_fraction = min(1.0, 2 * mix_fact)
    else:
        sy_equal_fraction = 1 - mix_fact
        sy_opp_fraction = mix_fact

    sy_equal_task_train, _ = _random_split(sy_equal, first_pcnt=sy_equal_fraction, seed=seed)
    sy_opp_task_train, _ = _random_split(sy_opposite, first_pcnt=sy_opp_fraction, seed=seed)

    biased_subset = concat_dt(
        [sy_equal_task_train, sy_opp_task_train], axis='index', ignore_index=True
    )

    if mix_fact == 0:
        # s and y should be very correlated in the biased subset
        assert biased_subset.s[s_name].corr(biased_subset.y[y_name]) > 0.99

    return biased_subset, normal_subset


def get_biased_and_debiased_subsets(
    data: DataTuple, mixing_factor: float, unbiased_pcnt: float, seed: int = 42
) -> Tuple[DataTuple, DataTuple]:
    """Split the given data into a biased subset and a debiased subset

    In contrast to :func:`get_biased_subset()`, this function makes the unbiased subset *really*
    unbiased.

    The two subsets don't generally sum up to the whole set.

    Example behavior:

    * mixing_factor=0.0: in biased, s=y everywhere; in debiased, 50% s=y and 50% s!=y
    * mixing_factor=0.5: biased is just a subset of `data`; in debiased, 50% s=y and 50% s!=y
    * mixing_factor=1.0: in biased, s!=y everywhere; in debiased, 50% s=y and 50% s!=y

    Args:
        data: data in form of a DataTuple
        mixing_factor: How much of the debiased data should be mixed into the biased subset? If this
                       factor is 0, the biased subset is maximally biased.
        unbiased_pcnt: how much of the data should be reserved for the unbiased subset
        seed: random seed for the splitting

    Returns:
        biased and unbiased dataset
    """
    s_name = data.s.columns[0]
    y_name = data.y.columns[0]
    sy_equal, sy_opposite = _get_sy_equal_and_opp(data, s_name, y_name)

    assert 0 <= unbiased_pcnt <= 1
    # how much of sy_equal should be reserved for the biased subset:
    biased_pcnt = 1 - unbiased_pcnt

    sy_equal_for_biased_ss, sy_equal_for_debiased_ss = _random_split(
        sy_equal, first_pcnt=biased_pcnt * (1 - mixing_factor), seed=seed
    )
    sy_opp_for_biased_ss, sy_opp_for_debiased_ss = _random_split(
        sy_opposite, first_pcnt=biased_pcnt * mixing_factor, seed=seed
    )

    biased_subset = concat_dt(
        [sy_equal_for_biased_ss, sy_opp_for_biased_ss], axis='index', ignore_index=True
    )

    # the debiased set is constructed from two sets of the same size
    min_size = min(sy_equal_for_debiased_ss.x.shape[0], sy_opp_for_debiased_ss.x.shape[0])

    def _get_equal_sized_subset(df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(n=min_size, random_state=seed)

    debiased_subset_part1 = sy_equal_for_debiased_ss.apply_to_joined_df(_get_equal_sized_subset)
    debiased_subset_part2 = sy_opp_for_debiased_ss.apply_to_joined_df(_get_equal_sized_subset)
    debiased_subset = concat_dt(
        [debiased_subset_part1, debiased_subset_part2], axis='index', ignore_index=True
    )

    # s and y should not be correlated in the debiased subset
    assert abs(debiased_subset.s[s_name].corr(debiased_subset.y[y_name])) < 0.1

    if mixing_factor == 0:
        # s and y should be very correlated in the biased subset
        assert biased_subset.s[s_name].corr(biased_subset.y[y_name]) > 0.99

    return biased_subset, debiased_subset
