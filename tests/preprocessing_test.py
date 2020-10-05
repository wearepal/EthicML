"""Test preprocessing capabilities."""
import math
from typing import Tuple

import pandas as pd
from pytest import approx

import ethicml as em
from ethicml import DataTuple, ProportionalSplit

from .run_algorithm_test import count_true


def test_train_test_split():
    """test train test split"""
    data: DataTuple = em.load_data(em.toy())
    train_test: Tuple[DataTuple, DataTuple] = em.train_test_split(data)
    train, test = train_test
    assert train is not None
    assert test is not None
    assert train.x.shape[0] > test.x.shape[0]
    assert train.x["a1"].values[0] == 0.2365572108691669
    assert train.x["a2"].values[0] == 0.008603090240657633

    assert train.x.shape[0] == train.s.shape[0]
    assert train.s.shape[0] == train.y.shape[0]

    NUM_SAMPLES = len(data)

    len_default = math.floor((NUM_SAMPLES / 100) * 80)
    assert train.s.shape[0] == len_default
    assert test.s.shape[0] == NUM_SAMPLES - len_default

    len_0_9 = math.floor((NUM_SAMPLES / 100) * 90)
    train, test = em.train_test_split(data, train_percentage=0.9)
    assert train.s.shape[0] == len_0_9
    assert test.s.shape[0] == NUM_SAMPLES - len_0_9

    len_0_7 = math.floor((NUM_SAMPLES / 100) * 70)
    train, test = em.train_test_split(data, train_percentage=0.7)
    assert train.s.shape[0] == len_0_7
    assert test.s.shape[0] == NUM_SAMPLES - len_0_7

    len_0_5 = math.floor((NUM_SAMPLES / 100) * 50)
    train, test = em.train_test_split(data, train_percentage=0.5)
    assert train.s.shape[0] == len_0_5
    assert test.s.shape[0] == NUM_SAMPLES - len_0_5

    len_0_3 = math.floor((NUM_SAMPLES / 100) * 30)
    train, test = em.train_test_split(data, train_percentage=0.3)
    assert train.s.shape[0] == len_0_3
    assert test.s.shape[0] == NUM_SAMPLES - len_0_3

    len_0_1 = math.floor((NUM_SAMPLES / 100) * 10)
    train, test = em.train_test_split(data, train_percentage=0.1)
    assert train.s.shape[0] == len_0_1
    assert test.s.shape[0] == NUM_SAMPLES - len_0_1

    len_0_0 = math.floor((NUM_SAMPLES / 100) * 0)
    train, test = em.train_test_split(data, train_percentage=0.0)
    assert train.s.shape[0] == len_0_0
    assert train.name == "Toy - Train"
    assert test.s.shape[0] == NUM_SAMPLES - len_0_0
    assert test.name == "Toy - Test"


def test_prop_train_test_split():
    """test prop train test split"""
    data: DataTuple = em.load_data(em.toy())
    train: DataTuple
    test: DataTuple
    train, test, _ = ProportionalSplit(train_percentage=0.8)(data, split_id=0)
    assert train is not None
    assert test is not None
    assert train.x.shape[0] > test.x.shape[0]
    assert train.x["a1"].values[0] == -0.7135614558562237
    assert train.x["a2"].values[0] == 1.1211390799513148

    assert train.x.shape[0] == train.s.shape[0]
    assert train.s.shape[0] == train.y.shape[0]

    NUM_SAMPLES = len(data)
    len_default = math.floor((NUM_SAMPLES / 100) * 80)
    assert train.s.shape[0] == len_default
    assert test.s.shape[0] == NUM_SAMPLES - len_default

    # assert that the proportion of s=0 to s=1 has remained the same (and also test for y=0/y=1)
    assert count_true(train.s.to_numpy() == 0) == round(0.8 * count_true(data.s.to_numpy() == 0))
    assert count_true(train.y.to_numpy() == 0) == round(0.8 * count_true(data.y.to_numpy() == 0))

    len_0_9 = math.floor((NUM_SAMPLES / 100) * 90)
    train, test, _ = ProportionalSplit(train_percentage=0.9)(data, split_id=0)
    assert train.s.shape[0] == len_0_9
    assert test.s.shape[0] == NUM_SAMPLES - len_0_9
    assert count_true(train.s.to_numpy() == 0) == approx(
        round(0.9 * count_true(data.s.to_numpy() == 0)), abs=1
    )
    assert count_true(train.y.to_numpy() == 0) == approx(
        round(0.9 * count_true(data.y.to_numpy() == 0)), abs=1
    )

    len_0_7 = math.floor((NUM_SAMPLES / 100) * 70)
    train, test, _ = ProportionalSplit(train_percentage=0.7)(data, split_id=0)
    assert train.s.shape[0] == len_0_7
    assert test.s.shape[0] == NUM_SAMPLES - len_0_7
    assert count_true(train.s.to_numpy() == 0) == approx(
        round(0.7 * count_true(data.s.to_numpy() == 0)), abs=1
    )
    assert count_true(train.y.to_numpy() == 0) == approx(
        round(0.7 * count_true(data.y.to_numpy() == 0)), abs=1
    )

    len_0_5 = math.floor((NUM_SAMPLES / 100) * 50)
    train, test, _ = ProportionalSplit(train_percentage=0.5)(data, split_id=0)
    assert train.s.shape[0] == len_0_5
    assert test.s.shape[0] == NUM_SAMPLES - len_0_5

    len_0_3 = math.floor((NUM_SAMPLES / 100) * 30)
    train, test, _ = ProportionalSplit(train_percentage=0.3)(data, split_id=0)
    assert train.s.shape[0] == len_0_3
    assert test.s.shape[0] == NUM_SAMPLES - len_0_3

    len_0_1 = math.floor((NUM_SAMPLES / 100) * 10)
    train, test, _ = ProportionalSplit(train_percentage=0.1)(data, split_id=0)
    assert train.s.shape[0] == len_0_1
    assert test.s.shape[0] == NUM_SAMPLES - len_0_1

    len_0_0 = math.floor((NUM_SAMPLES / 100) * 0)
    train, test, _ = ProportionalSplit(train_percentage=0.0)(data, split_id=0)
    assert train.s.shape[0] == len_0_0
    assert train.name == "Toy - Train"
    assert test.s.shape[0] == NUM_SAMPLES - len_0_0
    assert test.name == "Toy - Test"


def test_random_seed():
    """test random seed"""
    data: DataTuple = em.load_data(em.toy())
    train_test_0: Tuple[DataTuple, DataTuple] = em.train_test_split(data)
    train_0, test_0 = train_test_0
    assert train_0 is not None
    assert test_0 is not None
    assert train_0.x.shape[0] > test_0.x.shape[0]
    assert train_0.x["a1"].values[0] == 0.2365572108691669
    assert train_0.x["a2"].values[0] == 0.008603090240657633

    assert train_0.x.shape[0] == train_0.s.shape[0]
    assert train_0.s.shape[0] == train_0.y.shape[0]

    train_test_1: Tuple[DataTuple, DataTuple] = em.train_test_split(data, random_seed=1)
    train_1, test_1 = train_test_1
    assert train_1 is not None
    assert test_1 is not None
    assert train_1.x.shape[0] > test_1.x.shape[0]
    assert train_1.x["a1"].values[0] == 1.3736566330173798
    assert train_1.x["a2"].values[0] == 0.21742296144957174

    assert train_1.x.shape[0] == train_1.s.shape[0]
    assert train_1.s.shape[0] == train_1.y.shape[0]

    train_test_2: Tuple[DataTuple, DataTuple] = em.train_test_split(data, random_seed=2)
    train_2, test_2 = train_test_2
    assert train_2 is not None
    assert test_2 is not None
    assert train_2.x.shape[0] > test_2.x.shape[0]
    assert train_2.x["a1"].values[0] == 1.2255705960148289
    assert train_2.x["a2"].values[0] == -1.208089015454192

    assert train_2.x.shape[0] == train_2.s.shape[0]
    assert train_2.s.shape[0] == train_2.y.shape[0]

    train_test_3: Tuple[DataTuple, DataTuple] = em.train_test_split(data, random_seed=3)
    train_3, test_3 = train_test_3
    assert train_3 is not None
    assert test_3 is not None
    assert train_3.x.shape[0] > test_3.x.shape[0]
    assert train_3.x["a1"].values[0] == 0.21165963748018515
    assert train_3.x["a2"].values[0] == -2.425137404779957

    assert train_3.x.shape[0] == train_3.s.shape[0]
    assert train_3.s.shape[0] == train_3.y.shape[0]


def test_binning():
    """test binning"""
    data: DataTuple = em.load_data(em.adult())

    binned: DataTuple = em.bin_cont_feats(data)

    assert len([col for col in binned.x.columns if col not in data.x.columns]) == 25
    assert "age" not in binned.x.columns


def test_sequential_split():
    """test sequential split"""
    data: DataTuple = em.load_data(em.toy())
    train: DataTuple
    test: DataTuple
    train, test, _ = em.SequentialSplit(train_percentage=0.8)(data)
    assert all(data.x.iloc[0] == train.x.iloc[0])
    assert all(data.x.iloc[-1] == test.x.iloc[-1])
    assert len(train) == 320
    assert len(test) == 80


def test_biased_split():
    """test biased split"""
    data = DataTuple(
        x=pd.DataFrame([0] * 1000, columns=["feat1-"]),
        s=pd.DataFrame([1] * 750 + [0] * 250, columns=["sens="]),
        y=pd.DataFrame([1] * 500 + [0] * 250 + [1] * 125 + [0] * 125, columns=["label<"]),
        name="TestData",
    )

    # =================================== mixing factor = 0 =======================================
    biased1, subset = em.get_biased_subset(data, mixing_factor=0.0, unbiased_pcnt=0.5)
    # expected behavior: in biased1, s=y everywhere; `subset` is just a subset of `data`

    assert biased1.s.shape == (313, 1)
    assert biased1.y.shape == (313, 1)
    assert (biased1.s.to_numpy() == biased1.y.to_numpy()).all()
    assert biased1.name == "TestData - Biased (tm=0.0)"

    # size is exactly 0.5 of the total
    assert subset.s.shape[0] == approx(500, abs=4)  # can be off by 4 because of proportional split
    assert subset.y.shape[0] == approx(500, abs=4)
    assert subset.name == "TestData - Subset (tm=0.0)"

    # the fraction of `data`points where y=s should be the same in the data and the subset
    subset_mean = (subset.s.to_numpy() == subset.y.to_numpy()).mean()
    data_mean = (data.s.to_numpy() == data.y.to_numpy()).mean()
    assert subset_mean == approx(data_mean, abs=0.01)

    biased2, debiased = em.get_biased_and_debiased_subsets(
        data, mixing_factor=0.0, unbiased_pcnt=0.5, fixed_unbiased=True
    )
    # expected behavior: in biased2, s=y everywhere; in debiased, 50% s=y and 50% s!=y

    assert biased2.s.shape == (313, 1)
    assert biased2.y.shape == (313, 1)
    assert (biased2.s.to_numpy() == biased2.y.to_numpy()).all()

    # assert debiased.s.shape == (626, 1)
    # assert debiased.y.shape == (626, 1)
    assert debiased.s.shape == (374, 1)
    assert debiased.y.shape == (374, 1)
    # for the debiased subset, s=y half of the time
    count = count_true(debiased.s.to_numpy() == debiased.y.to_numpy())
    # assert count == 626 // 2
    assert count == 374 // 2

    # ================================= mixing factor = 0.5 =======================================
    biased1, subset = em.get_biased_subset(data, mixing_factor=0.5, unbiased_pcnt=0.5)
    # expected behavior: biased1 and `subset` are both just subsets of `data`

    assert biased1.s.shape[0] == approx(500, abs=4)
    assert biased1.y.shape[0] == approx(500, abs=4)
    data_mean = (data.s.to_numpy() == data.y.to_numpy()).mean()
    biased1_mean = (biased1.s.to_numpy() == biased1.y.to_numpy()).mean()
    assert biased1_mean == approx(data_mean, abs=0.01)

    assert subset.s.shape[0] == approx(500, abs=4)
    assert subset.y.shape[0] == approx(500, abs=4)

    # the fraction of `data`points where y=s should be the same in the data and the subset
    subset_mean = (subset.s.to_numpy() == subset.y.to_numpy()).mean()
    assert subset_mean == approx(data_mean, abs=0.01)

    biased2, debiased = em.get_biased_and_debiased_subsets(
        data, mixing_factor=0.5, unbiased_pcnt=0.5, fixed_unbiased=True
    )
    # expected behavior: biased2 is just a subset of `data`; in debiased, 50% s=y and 50% s!=y

    assert biased2.s.shape[0] == approx(500, abs=4)
    assert biased2.y.shape[0] == approx(500, abs=4)
    data_mean = (data.s.to_numpy() == data.y.to_numpy()).mean()
    biased2_mean = (biased2.s.to_numpy() == biased2.y.to_numpy()).mean()
    assert biased2_mean == approx(data_mean, abs=0.01)

    # assert debiased.s.shape == (564, 1)
    # assert debiased.y.shape == (564, 1)
    assert debiased.s.shape == (374, 1)
    assert debiased.y.shape == (374, 1)
    # for the debiased subset, s=y half of the time
    count = count_true(debiased.s.to_numpy() == debiased.y.to_numpy())
    # assert count == 564 // 2
    assert count == 374 // 2

    # ================================== mixing factor = 1 ========================================
    biased1, subset = em.get_biased_subset(data, mixing_factor=1.0, unbiased_pcnt=0.5)
    # expected behavior: in biased1, s!=y everywhere; `subset` is just a subset of `data`

    assert biased1.s.shape == (188, 1)
    assert biased1.y.shape == (188, 1)
    assert (biased1.s.to_numpy() != biased1.y.to_numpy()).all()

    assert subset.s.shape[0] == approx(500, abs=4)
    assert subset.y.shape[0] == approx(500, abs=4)

    # the fraction of `data`points where y=s should be the same in the data and the subset
    subset_mean = (subset.s.to_numpy() == subset.y.to_numpy()).mean()
    assert subset_mean == approx(data_mean, abs=0.01)

    biased2, debiased = em.get_biased_and_debiased_subsets(
        data, mixing_factor=1.0, unbiased_pcnt=0.5, fixed_unbiased=True
    )
    # expected behavior: in biased2, s!=y everywhere; in debiased, 50% s=y and 50% s!=y

    assert biased2.s.shape == (188, 1)
    assert biased2.y.shape == (188, 1)
    assert (biased2.s.to_numpy() != biased2.y.to_numpy()).all()

    # assert debiased.s.shape == (376, 1)
    # assert debiased.y.shape == (376, 1)
    assert debiased.s.shape == (374, 1)
    assert debiased.y.shape == (374, 1)
    # for the debiased subset, s=y half of the time
    count = count_true(debiased.s.to_numpy() == debiased.y.to_numpy())
    # assert count == 376 // 2
    assert count == 374 // 2


def test_biased_split_sizes():
    """test biased split sizes"""
    data = DataTuple(
        x=pd.DataFrame([0] * 1000, columns=["feat1-"]),
        s=pd.DataFrame([1] * 750 + [0] * 250, columns=["sens="]),
        y=pd.DataFrame([1] * 500 + [0] * 250 + [1] * 125 + [0] * 125, columns=["label<"]),
        name="TestData",
    )

    biased1, subset = em.get_biased_subset(data, mixing_factor=0.0, unbiased_pcnt=0.8)
    assert len(biased1) < len(subset)
    biased1, subset = em.get_biased_subset(data, mixing_factor=0.0, unbiased_pcnt=0.2)
    assert len(biased1) > len(subset)

    biased2, debiased = em.get_biased_and_debiased_subsets(
        data, mixing_factor=0.0, unbiased_pcnt=0.8, fixed_unbiased=True
    )
    assert len(biased2) < len(debiased)
    biased2, debiased = em.get_biased_and_debiased_subsets(
        data, mixing_factor=0.0, unbiased_pcnt=0.2, fixed_unbiased=True
    )
    assert len(biased2) > len(debiased)


def test_biased_split_nonbinary():
    """test biased split nonbinary"""
    # generate data that uses -1 and 1 instead of 0 and 1 for s and y
    data = DataTuple(
        x=pd.DataFrame([0] * 1000, columns=["feat1-"]),
        s=pd.DataFrame([1] * 750 + [-1] * 250, columns=["sens="]),
        y=pd.DataFrame([1] * 500 + [-1] * 250 + [1] * 125 + [-1] * 125, columns=["label<"]),
        name="TestData",
    )

    biased1, subset = em.get_biased_subset(data, mixing_factor=0.5, unbiased_pcnt=0.5)
    assert len(biased1) == approx(len(subset), abs=4)


def test_balanced_test_split(simple_data: DataTuple):
    """test biased split sizes"""
    train_percentage = 0.75
    train, test, split_info = em.BalancedTestSplit(train_percentage=train_percentage)(simple_data)

    # check that the split for training set was proportional
    assert count_true(train.s.to_numpy() == 0) == approx(round(train_percentage * 250), abs=1)
    assert count_true(train.y.to_numpy() == 0) == approx(round(train_percentage * 400), abs=1)

    # check that the test set is balanced
    assert len(em.query_dt(test, "s == 0 & y == 0")) == round((1 - train_percentage) * 150)
    assert len(em.query_dt(test, "s == 1 & y == 0")) == round((1 - train_percentage) * 150)
    assert len(em.query_dt(test, "s == 0 & y == 1")) == round((1 - train_percentage) * 100)
    assert len(em.query_dt(test, "s == 1 & y == 1")) == round((1 - train_percentage) * 100)

    # check how many samples were droppped
    assert split_info["percent_dropped"] == 0.496


def test_balanced_test_split_by_s(simple_data: DataTuple):
    """test biased split sizes"""
    train_percentage = 0.75
    train, test, split_info = em.BalancedTestSplit(
        balance_type="P(y|s)=0.5", train_percentage=train_percentage
    )(simple_data)

    # check that the split for training set was proportional
    assert count_true(train.s.to_numpy() == 0) == approx(round(train_percentage * 250), abs=1)
    assert count_true(train.y.to_numpy() == 0) == approx(round(train_percentage * 400), abs=1)

    # check that the test set is balanced
    assert len(em.query_dt(test, "s == 0 & y == 0")) == round((1 - train_percentage) * 100)
    assert len(em.query_dt(test, "s == 0 & y == 1")) == round((1 - train_percentage) * 100)
    assert len(em.query_dt(test, "s == 1 & y == 0")) == round((1 - train_percentage) * 250)
    assert len(em.query_dt(test, "s == 1 & y == 1")) == round((1 - train_percentage) * 250)

    # check how many samples were droppped
    assert split_info["percent_dropped"] == 0.304


def test_balanced_test_split_by_s_and_y(simple_data: DataTuple):
    """test biased split sizes"""
    train_percentage = 0.75
    train, test, split_info = em.BalancedTestSplit(
        balance_type="P(s,y)=0.25", train_percentage=train_percentage
    )(simple_data)

    # check that the split for training set was proportional
    assert count_true(train.s.to_numpy() == 0) == approx(round(train_percentage * 250), abs=1)
    assert count_true(train.y.to_numpy() == 0) == approx(round(train_percentage * 400), abs=1)

    # check that the test set is balanced
    assert len(em.query_dt(test, "s == 0 & y == 0")) == round((1 - train_percentage) * 100)
    assert len(em.query_dt(test, "s == 0 & y == 1")) == round((1 - train_percentage) * 100)
    assert len(em.query_dt(test, "s == 1 & y == 0")) == round((1 - train_percentage) * 100)
    assert len(em.query_dt(test, "s == 1 & y == 1")) == round((1 - train_percentage) * 100)

    # check how many samples were droppped
    assert split_info["percent_dropped"] == 0.6
