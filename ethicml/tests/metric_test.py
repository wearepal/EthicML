"""
Test that we can get some metrics on predictions
"""

import numpy as np

from ethicml.algorithms.algorithm import Algorithm
from ethicml.algorithms.svm import SVM
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.metric import Metric
from ethicml.evaluators.per_sensitive_attribute import metric_per_sensitive_attribute
from ethicml.tests.run_algorithm_test import get_train_test


def test_get_acc_of_predictions():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: np.array = model.run(train, test)
    acc: Metric = Accuracy()
    assert acc.get_name() == "Accuracy"
    score = acc.score(predictions, test)
    assert score == 0.88


def test_accuracy_per_sens_attr():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: np.array = model.run(train, test)
    acc_per_sens = metric_per_sensitive_attribute(predictions, test, Accuracy())
    assert acc_per_sens == {0: 0.8780487804878049, 1: 0.882051282051282}
