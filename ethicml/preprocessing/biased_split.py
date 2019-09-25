"""Generate biased subsets"""
from typing import Tuple, Callable

from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.preprocessing.domain_adaptation import query_dt, make_valid_variable_name
from ethicml.utility.data_structures import concat_dt, DataTuple


def _random_split(data: DataTuple, first_pcnt: float, seed: int) -> Tuple[DataTuple, DataTuple]:
    return train_test_split(data, train_percentage=first_pcnt, random_seed=seed)


def _get_sy_equal_and_opp(data: DataTuple, s_name: str, y_name: str) -> Tuple[DataTuple, DataTuple]:
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


def get_biased_subset(data: DataTuple, s_name: str, y_name: str, mixing_factor: float, biased_pcnt: float, seed: int = 42,
                      data_efficient: bool = True) -> Tuple[DataTuple, DataTuple]:
    """Split the given data such that the task train set is very biased

    Meta train and task are just subsets of the Adult dataset.
    """

    # task and meta are just normal subsets
    normal_subset, for_biased_subset = _random_split(data, first_pcnt=biased_pcnt, seed=seed)

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
        # s & y should be very correlated in the task train set
        assert biased_subset.s[s_name].corr(biased_subset.y[y_name]) > 0.99

    return biased_subset, normal_subset


def get_biased_and_unbiased_subset(
    data: DataTuple, s_name: str, y_name: str, mixing_factor: float, biased_pcnt: float, seed: int = 42
) -> Tuple[DataTuple, DataTuple]:
    """Split the given data such that the task train set is very biased

    In contrast to the (new) generate_splits() function, this function makes the meta train and the
    task set *unbiased*.

    Args:
        data: data in form of a DataTuple
        s_name: name of the sensitive attribute
        y_name: name of the class label
        mixing_factor: How much of the debiased data should be mixed into the biased subset? If this
                       factor is 0, the biased subset is maximally biased.
    """
    sy_equal, sy_opposite = _get_sy_equal_and_opp(data, s_name, y_name)

    # how much of sy_equal should be reserved for the task train set:
    mix_fact = mixing_factor  # how much of sy_opp should be mixed into task train set

    sy_equal_task_train, sy_equal_meta_train = _random_split(
        sy_equal, first_pcnt=biased_pcnt * (1 - mix_fact), seed=seed
    )
    sy_opp_task_train, sy_opp_meta_train = _random_split(
        sy_opposite, first_pcnt=biased_pcnt * mix_fact, seed=seed
    )

    very_biased_subset = concat_dt(
        [sy_equal_task_train, sy_opp_task_train], axis='index', ignore_index=True
    )
    debiased_subset = concat_dt(
        [sy_equal_meta_train, sy_opp_meta_train], axis='index', ignore_index=True
    )

    if mix_fact == 0:
        # s and y should not be overly correlated in the meta train set
        assert abs(debiased_subset.s[s_name].corr(debiased_subset.y[y_name])) < 0.1
        # but they should be very correlated in the task train set
        assert very_biased_subset.s[s_name].corr(very_biased_subset.y[y_name]) > 0.99

    return debiased_subset, very_biased_subset
