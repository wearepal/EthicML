"""
Test that an algorithm can run against some data
"""

from typing import Tuple, Dict
import pandas as pd
import numpy as np

from ethicml.algorithms.algorithm import Algorithm
from ethicml.algorithms.logistic_regression import LR
from ethicml.algorithms.svm import SVM
from ethicml.data.test import Test
from ethicml.data.load import load_data
from ethicml.preprocessing.train_test_split import train_test_split


def get_train_test():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    train_test: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(data)
    return train_test


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_can_make_predictions():
    train, test = get_train_test()

    model: Algorithm = SVM()
    assert model is not None
    assert model.get_name() == "SVM"

    predictions: np.array = model.run(train, test)
    assert sum([1 for x in predictions if x == 1]) == 194
    assert sum([1 for x in predictions if x == -1]) == 206


def test_lr():
    train, test = get_train_test()

    model: Algorithm = LR()
    assert model is not None
    assert model.get_name() == "Logistic Regression"

    predictions: np.array = model.run(train, test)
    assert sum([1 for x in predictions if x == 1]) == 199
    assert sum([1 for x in predictions if x == -1]) == 201
