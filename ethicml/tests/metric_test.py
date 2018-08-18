"""
Test that we can get some metrics on predictions
"""

import numpy as np

from ethicml.algorithms.algorithm import Algorithm
from ethicml.algorithms.svm import SVM
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.metric import Metric
from ethicml.evaluators.per_sensitive_attribute import metric_per_sensitive_attribute
from ethicml.evaluators.per_sensitive_attribute import diff_per_sensitive_attribute
from ethicml.metrics.tpr import TPR
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


def test_tpr_diff():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: np.array = model.run(train, test)
    tprs = metric_per_sensitive_attribute(predictions, test, TPR())
    assert TPR().get_name() == "TPR"
    assert tprs == {0: 0.7704918032786885, 1: 0.9312977099236641}
    tpr_diff = diff_per_sensitive_attribute(tprs)
    assert tpr_diff["0-1"] == 0.16080590664497563
