"""
Test that an algorithm can run against some data
"""

from typing import Tuple, List
import pytest

from ethicml.algorithms.inprocess import InAlgorithm, LR, SVM
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess import Beutel, PreAlgorithm
from ethicml.algorithms.utils import DataTuple
from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data
from ethicml.data import Adult, Compas, German, Sqf, Toy
from ethicml.evaluators.evaluate_models import evaluate_models
from ethicml.evaluators.per_sensitive_attribute import MetricNotApplicable
from ethicml.metrics import Accuracy, CV, TPR, Metric
from ethicml.preprocessing.train_test_split import train_test_split


def get_train_test():
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


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
