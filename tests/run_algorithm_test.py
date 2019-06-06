"""
Test that an algorithm can run against some data
"""

from typing import Tuple, List
import pytest
import numpy as np

from ethicml.algorithms.inprocess import InAlgorithm, LR, SVM, Majority
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess import Beutel, PreAlgorithm
from ethicml.algorithms.utils import DataTuple
from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data
from ethicml.data import Adult, Compas, German, Sqf, Toy
from ethicml.evaluators import run_in_parallel
from ethicml.evaluators.evaluate_models import evaluate_models
from ethicml.evaluators.per_sensitive_attribute import MetricNotApplicable
from ethicml.metrics import Accuracy, CV, TPR, Metric
from ethicml.preprocessing.train_test_split import train_test_split


def get_train_test() -> Tuple[DataTuple, DataTuple]:
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test


def count_true(mask: 'np.ndarray[bool]') -> int:
    """Count the number of elements that are True"""
    return mask.nonzero()[0].shape[0]


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_run_parallel():
    data0 = get_train_test()
    data1 = get_train_test()
    result = run_in_parallel([LR(), SVM(), Majority()], [data0, data1], max_parallel=2)
    # LR
    assert count_true(result[0][0].values == 1) == 211
    assert count_true(result[0][0].values == -1) == 189
    assert count_true(result[0][1].values == 1) == 211
    assert count_true(result[0][1].values == -1) == 189
    # SVM
    assert count_true(result[1][0].values == 1) == 201
    assert count_true(result[1][0].values == -1) == 199
    assert count_true(result[1][1].values == 1) == 201
    assert count_true(result[1][1].values == -1) == 199
    # Majority
    assert count_true(result[2][0].values == 1) == 0
    assert count_true(result[2][0].values == -1) == 400
    assert count_true(result[2][1].values == 1) == 0
    assert count_true(result[2][1].values == -1) == 400


# def test_run_alg_suite():
#     dataset = Adult("Race")
#     dataset.sens_attrs = ["race_White"]
#     datasets: List[Dataset] = [dataset, Toy()]
#     preprocess_models: List[PreAlgorithm] = [Beutel()]
#     inprocess_models: List[InAlgorithm] = [LR()]
#     postprocess_models: List[PostAlgorithm] = []
#     metrics: List[Metric] = [Accuracy(), CV()]
#     per_sens_metrics: List[Metric] = [Accuracy(), TPR()]
#     evaluate_models(datasets, preprocess_models, inprocess_models,
#                     postprocess_models, metrics, per_sens_metrics, repeats=1, test_mode=True)
#
#
# def test_run_alg_suite_wrong_metrics():
#     datasets: List[Dataset] = [Toy(), Adult()]
#     preprocess_models: List[PreAlgorithm] = [Beutel()]
#     inprocess_models: List[InAlgorithm] = [SVM(), LR()]
#     postprocess_models: List[PostAlgorithm] = []
#     metrics: List[Metric] = [Accuracy(), CV()]
#     per_sens_metrics: List[Metric] = [Accuracy(), TPR(), CV()]
#     with pytest.raises(MetricNotApplicable):
#         evaluate_models(datasets, preprocess_models, inprocess_models,
#                         postprocess_models, metrics, per_sens_metrics, repeats=1, test_mode=True)
