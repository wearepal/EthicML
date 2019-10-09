"""
Test preprocessing capabilities
"""
import math
from typing import Tuple

from pytest import approx
import pandas as pd

from ethicml.utility import DataTuple
from ethicml.data import load_data, Toy, Adult
from ethicml.preprocessing import (
    ProportionalTrainTestSplit,
    train_test_split,
    bin_cont_feats,
    get_biased_subset,
    get_biased_and_debiased_subsets,
)

from .run_algorithm_test import count_true


def test_train_test_split():
    """test train test split"""
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train, test = train_test
    assert train is not None
    assert test is not None
    assert train.x.shape[0] > test.x.shape[0]
    assert train.x['a1'].values[0] == 2.7839204628851526
    assert train.x['a2'].values[0] == 6.083679468491366

    assert train.x.shape[0] == train.s.shape[0]
    assert train.s.shape[0] == train.y.shape[0]

    len_default = math.floor((2000 / 100) * 80)
    assert train.s.shape[0] == len_default
    assert test.s.shape[0] == 2000 - len_default

    len_0_9 = math.floor((2000 / 100) * 90)
    train, test = train_test_split(data, train_percentage=0.9)
    assert train.s.shape[0] == len_0_9
    assert test.s.shape[0] == 2000 - len_0_9

    len_0_7 = math.floor((2000 / 100) * 70)
    train, test = train_test_split(data, train_percentage=0.7)
    assert train.s.shape[0] == len_0_7
    assert test.s.shape[0] == 2000 - len_0_7

    len_0_5 = math.floor((2000 / 100) * 50)
    train, test = train_test_split(data, train_percentage=0.5)
    assert train.s.shape[0] == len_0_5
    assert test.s.shape[0] == 2000 - len_0_5

    len_0_3 = math.floor((2000 / 100) * 30)
    train, test = train_test_split(data, train_percentage=0.3)
    assert train.s.shape[0] == len_0_3
    assert test.s.shape[0] == 2000 - len_0_3

    len_0_1 = math.floor((2000 / 100) * 10)
    train, test = train_test_split(data, train_percentage=0.1)
    assert train.s.shape[0] == len_0_1
    assert test.s.shape[0] == 2000 - len_0_1

    len_0_0 = math.floor((2000 / 100) * 0)
    train, test = train_test_split(data, train_percentage=0.0)
    assert train.s.shape[0] == len_0_0
    assert train.name == "Toy - Train"
    assert test.s.shape[0] == 2000 - len_0_0
    assert test.name == "Toy - Test"


def test_prop_train_test_split():
    """test prop train test split"""
    data: DataTuple = load_data(Toy())
    train: DataTuple
    test: DataTuple
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.8)(data, split_id=0)
    assert train is not None
    assert test is not None
    assert train.x.shape[0] > test.x.shape[0]
    assert train.x['a1'].values[0] == 2.9624320112634743
    assert train.x['a2'].values[0] == 1.996089217523236

    assert train.x.shape[0] == train.s.shape[0]
    assert train.s.shape[0] == train.y.shape[0]

    len_default = math.floor((2000 / 100) * 80)
    assert train.s.shape[0] == len_default
    assert test.s.shape[0] == 2000 - len_default

    # assert that the proportion of s=0 to s=1 has remained the same (and also test for y=0/y=1)
    assert count_true(train.s.to_numpy() == 0) == round(0.8 * count_true(data.s.to_numpy() == 0))
    assert count_true(train.y.to_numpy() == 0) == round(0.8 * count_true(data.y.to_numpy() == 0))

    len_0_9 = math.floor((2000 / 100) * 90)
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.9)(data, split_id=0)
    assert train.s.shape[0] == len_0_9
    assert test.s.shape[0] == 2000 - len_0_9
    assert count_true(train.s.to_numpy() == 0) == round(0.9 * count_true(data.s.to_numpy() == 0))
    assert count_true(train.y.to_numpy() == 0) == round(0.9 * count_true(data.y.to_numpy() == 0))

    len_0_7 = math.floor((2000 / 100) * 70)
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.7)(data, split_id=0)
    assert train.s.shape[0] == len_0_7
    assert test.s.shape[0] == 2000 - len_0_7
    assert count_true(train.s.to_numpy() == 0) == round(0.7 * count_true(data.s.to_numpy() == 0))
    assert count_true(train.y.to_numpy() == 0) == round(0.7 * count_true(data.y.to_numpy() == 0))

    len_0_5 = math.floor((2000 / 100) * 50)
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.5)(data, split_id=0)
    assert train.s.shape[0] == len_0_5
    assert test.s.shape[0] == 2000 - len_0_5

    len_0_3 = math.floor((2000 / 100) * 30)
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.3)(data, split_id=0)
    assert train.s.shape[0] == len_0_3
    assert test.s.shape[0] == 2000 - len_0_3

    len_0_1 = math.floor((2000 / 100) * 10)
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.1)(data, split_id=0)
    assert train.s.shape[0] == len_0_1
    assert test.s.shape[0] == 2000 - len_0_1

    len_0_0 = math.floor((2000 / 100) * 0)
    train, test, _ = ProportionalTrainTestSplit(train_percentage=0.0)(data, split_id=0)
    assert train.s.shape[0] == len_0_0
    assert train.name == "Toy - Train"
    assert test.s.shape[0] == 2000 - len_0_0
    assert test.name == "Toy - Test"


def test_random_seed():
    """test random seed"""
    data: DataTuple = load_data(Toy())
    train_test_0: Tuple[DataTuple, DataTuple] = train_test_split(data)
    train_0, test_0 = train_test_0
    assert train_0 is not None
    assert test_0 is not None
    assert train_0.x.shape[0] > test_0.x.shape[0]
    assert train_0.x['a1'].values[0] == 2.7839204628851526
    assert train_0.x['a2'].values[0] == 6.083679468491366

    assert train_0.x.shape[0] == train_0.s.shape[0]
    assert train_0.s.shape[0] == train_0.y.shape[0]

    train_test_1: Tuple[DataTuple, DataTuple] = train_test_split(data, random_seed=1)
    train_1, test_1 = train_test_1
    assert train_1 is not None
    assert test_1 is not None
    assert train_1.x.shape[0] > test_1.x.shape[0]
    assert train_1.x['a1'].values[0] == 2.6158100167119773
    assert train_1.x['a2'].values[0] == -2.026281606912103

    assert train_1.x.shape[0] == train_1.s.shape[0]
    assert train_1.s.shape[0] == train_1.y.shape[0]

    train_test_2: Tuple[DataTuple, DataTuple] = train_test_split(data, random_seed=2)
    train_2, test_2 = train_test_2
    assert train_2 is not None
    assert test_2 is not None
    assert train_2.x.shape[0] > test_2.x.shape[0]
    assert train_2.x['a1'].values[0] == 0.6431486026206228
    assert train_2.x['a2'].values[0] == -0.09879806963941018

    assert train_2.x.shape[0] == train_2.s.shape[0]
    assert train_2.s.shape[0] == train_2.y.shape[0]

    train_test_3: Tuple[DataTuple, DataTuple] = train_test_split(data, random_seed=3)
    train_3, test_3 = train_test_3
    assert train_3 is not None
    assert test_3 is not None
    assert train_3.x.shape[0] > test_3.x.shape[0]
    assert train_3.x['a1'].values[0] == 0.8165458710908045
    assert train_3.x['a2'].values[0] == -5.456548268244318

    assert train_3.x.shape[0] == train_3.s.shape[0]
    assert train_3.s.shape[0] == train_3.y.shape[0]


def test_binning():
    """test binning"""
    data: DataTuple = load_data(Adult())

    binned: DataTuple = bin_cont_feats(data)

    assert len([col for col in binned.x.columns if col not in data.x.columns]) == 25
    assert 'age' not in binned.x.columns


def test_biased_split():
    """test biased split"""
    data = DataTuple(
        x=pd.DataFrame([0] * 1000, columns=['feat1']),
        s=pd.DataFrame([1] * 750 + [0] * 250, columns=["sens"]),
        y=pd.DataFrame([1] * 500 + [0] * 250 + [1] * 125 + [0] * 125, columns=["label"]),
    )

    # ================================== mixing factor = 0 ========================================
    biased1, subset = get_biased_subset(data, mixing_factor=0.0, unbiased_pcnt=0.5)
    # expected behavior: in biased1, s=y everywhere; `subset` is just a subset of `data`

    assert biased1.s.shape == (310, 1)
    assert biased1.y.shape == (310, 1)
    assert (biased1.s.to_numpy() == biased1.y.to_numpy()).all()

    # size is exactly 0.5 of the total
    assert subset.s.shape == (500, 1)
    assert subset.y.shape == (500, 1)

    # the fraction of `data`points where y=s should be the same in the data and the subset
    subset_mean = (subset.s.to_numpy() == subset.y.to_numpy()).mean()
    data_mean = (data.s.to_numpy() == data.y.to_numpy()).mean()
    assert subset_mean == approx(data_mean, abs=0.01)

    biased2, debiased = get_biased_and_debiased_subsets(
        data, mixing_factor=0.0, unbiased_pcnt=0.5, fixed_unbiased=False
    )
    # expected behavior: in biased2, s=y everywhere; in debiased, 50% s=y and 50% s!=y

    assert biased2.s.shape == (312, 1)
    assert biased2.y.shape == (312, 1)
    assert (biased2.s.to_numpy() == biased2.y.to_numpy()).all()

    assert debiased.s.shape == (626, 1)
    assert debiased.y.shape == (626, 1)
    # for the debiased subset, s=y half of the time
    count = count_true(debiased.s.to_numpy() == debiased.y.to_numpy())
    assert count == 626 // 2

    # ================================= mixing factor = 0.5 =======================================
    biased1, subset = get_biased_subset(data, mixing_factor=0.5, unbiased_pcnt=0.5)
    # expected behavior: biased1 and `subset` are both just subsets of `data`

    assert biased1.s.shape == (500, 1)
    assert biased1.y.shape == (500, 1)
    data_mean = (data.s.to_numpy() == data.y.to_numpy()).mean()
    biased1_mean = (biased1.s.to_numpy() == biased1.y.to_numpy()).mean()
    assert biased1_mean == approx(data_mean, abs=0.01)

    assert subset.s.shape == (500, 1)
    assert subset.y.shape == (500, 1)

    # the fraction of `data`points where y=s should be the same in the data and the subset
    subset_mean = (subset.s.to_numpy() == subset.y.to_numpy()).mean()
    assert subset_mean == approx(data_mean, abs=0.01)

    biased2, debiased = get_biased_and_debiased_subsets(
        data, mixing_factor=0.5, unbiased_pcnt=0.5, fixed_unbiased=False
    )
    # expected behavior: biased2 is just a subset of `data`; in debiased, 50% s=y and 50% s!=y

    assert biased2.s.shape == (249, 1)
    assert biased2.y.shape == (249, 1)
    data_mean = (data.s.to_numpy() == data.y.to_numpy()).mean()
    biased2_mean = (biased2.s.to_numpy() == biased2.y.to_numpy()).mean()
    assert biased2_mean == approx(data_mean, abs=0.01)

    assert debiased.s.shape == (564, 1)
    assert debiased.y.shape == (564, 1)
    # for the debiased subset, s=y half of the time
    count = count_true(debiased.s.to_numpy() == debiased.y.to_numpy())
    assert count == 564 // 2

    # ================================== mixing factor = 1 ========================================
    biased1, subset = get_biased_subset(data, mixing_factor=1.0, unbiased_pcnt=0.5)
    # expected behavior: in biased1, s!=y everywhere; `subset` is just a subset of `data`

    assert biased1.s.shape == (190, 1)
    assert biased1.y.shape == (190, 1)
    assert (biased1.s.to_numpy() != biased1.y.to_numpy()).all()

    assert subset.s.shape == (500, 1)
    assert subset.y.shape == (500, 1)

    # the fraction of `data`points where y=s should be the same in the data and the subset
    subset_mean = (subset.s.to_numpy() == subset.y.to_numpy()).mean()
    assert subset_mean == approx(data_mean, abs=0.01)

    biased2, debiased = get_biased_and_debiased_subsets(
        data, mixing_factor=1.0, unbiased_pcnt=0.5, fixed_unbiased=False
    )
    # expected behavior: in biased2, s!=y everywhere; in debiased, 50% s=y and 50% s!=y

    assert biased2.s.shape == (187, 1)
    assert biased2.y.shape == (187, 1)
    assert (biased2.s.to_numpy() != biased2.y.to_numpy()).all()

    assert debiased.s.shape == (376, 1)
    assert debiased.y.shape == (376, 1)
    # for the debiased subset, s=y half of the time
    count = count_true(debiased.s.to_numpy() == debiased.y.to_numpy())
    assert count == 376 // 2
