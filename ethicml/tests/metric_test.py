"""
Test that we can get some metrics on predictions
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np

from ethicml.algorithms.algorithm import Algorithm
from ethicml.algorithms.svm import SVM
from ethicml.data.test import Test
from ethicml.data.load import load_data
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.metric import Metric
from ethicml.preprocessing.train_test_split import train_test_split


def get_train_test():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    train_test: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(data)
    return train_test


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_get_acc_of_predictions():
    train, test = get_train_test()
    model: Algorithm = SVM()
    predictions: np.array = model.run(train, test)
    acc: Metric = Accuracy()
    assert acc.get_name() == "Accuracy"
    score = acc.score(predictions, test['y'])
    assert score == 0.88
