from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import pytest

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.installed_model import InstalledModel
from ethicml.algorithms.inprocess.kamishima import Kamishima
from ethicml.algorithms.inprocess.logistic_regression_cross_validated import LRCV
from ethicml.algorithms.inprocess.logistic_regression_probability import LRProb
from ethicml.algorithms.inprocess.logistic_regression import LR
from ethicml.algorithms.inprocess.svm import SVM
from ethicml.evaluators.cross_validator import CrossValidator
from ethicml.metrics import Accuracy
from ethicml.utility.heaviside import Heaviside
from tests.run_algorithm_test import get_train_test


def test_svm():
    train, test = get_train_test()

    model: InAlgorithm = SVM()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 201
    assert predictions[predictions.values == -1].count().values[0] == 199


def test_cv_svm():
    train, test = get_train_test()

    hyperparams: Dict[str, List[Any]] = {'C': [1, 10, 100], 'kernel': ['rbf', 'linear']}

    svm_cv = CrossValidator(SVM, hyperparams, folds=3)

    assert svm_cv is not None
    assert isinstance(svm_cv.model(), InAlgorithm)

    svm_cv.run(train)

    best_model = svm_cv.best(Accuracy())

    predictions: pd.DataFrame = best_model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 201
    assert predictions[predictions.values == -1].count().values[0] == 199


def test_cv_lr():
    train, test = get_train_test()

    hyperparams: Dict[str, List[int]] = {'C': [1, 10, 100]}

    lr_cv = CrossValidator(LR, hyperparams, folds=3)

    assert lr_cv is not None
    assert isinstance(lr_cv.model(), InAlgorithm)

    lr_cv.run(train)

    best_model = lr_cv.best(Accuracy())

    predictions: pd.DataFrame = best_model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 211
    assert predictions[predictions.values == -1].count().values[0] == 189


@pytest.fixture(scope="module")
def kamishima():
    kamishima_algo = Kamishima()
    yield kamishima_algo
    print("teardown Kamishima")
    kamishima_algo.remove()


def test_kamishima(kamishima):
    train, test = get_train_test()

    model: InAlgorithm = kamishima

    assert model is not None
    assert model.name == "Kamishima"

    predictions: pd.DataFrame = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 208
    assert predictions[predictions.values == -1].count().values[0] == 192


def test_threaded_svm():
    train, test = get_train_test()

    model: InAlgorithm = SVM()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run(train, test, sub_process=True)
    assert predictions[predictions.values == 1].count().values[0] == 201
    assert predictions[predictions.values == -1].count().values[0] == 199

    predictions_non_threaded: pd.DataFrame = model.run(train, test)
    assert predictions_non_threaded[predictions_non_threaded.values == 1].count().values[0] == 201
    assert predictions_non_threaded[predictions_non_threaded.values == -1].count().values[0] == 199


def test_lr():
    train, test = get_train_test()

    model: InAlgorithm = LR()
    assert model is not None
    assert model.name == "Logistic Regression"

    predictions: np.array = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 211
    assert predictions[predictions.values == -1].count().values[0] == 189


def test_threaded_lr():
    train, test = get_train_test()

    model: InAlgorithm = LR()
    assert model.name == "Logistic Regression"

    predictions: pd.DataFrame = model.run(train, test, sub_process=True)
    assert predictions[predictions.values == 1].count().values[0] == 211
    assert predictions[predictions.values == -1].count().values[0] == 189

    predictions_non_threaded: pd.DataFrame = model.run(train, test)
    assert predictions_non_threaded[predictions_non_threaded.values == 1].count().values[0] == 211
    assert predictions_non_threaded[predictions_non_threaded.values == -1].count().values[0] == 189


def test_threaded_lr_cross_validated():
    train, test = get_train_test()

    model: InAlgorithm = LRCV()
    assert model.name == "LRCV"

    predictions: pd.DataFrame = model.run(train, test, sub_process=True)
    assert predictions[predictions.values == 1].count().values[0] == 211
    assert predictions[predictions.values == -1].count().values[0] == 189

    predictions_non_threaded: pd.DataFrame = model.run(train, test)
    assert predictions_non_threaded[predictions_non_threaded.values == 1].count().values[0] == 211
    assert predictions_non_threaded[predictions_non_threaded.values == -1].count().values[0] == 189


def test_threaded_lr_prob():
    train, test = get_train_test()

    model: InAlgorithm = LRProb()
    assert model.name == "Logistic Regression Prob"

    heavi = Heaviside()

    predictions_non_threaded: pd.DataFrame = model.run(train, test)
    predictions_non_threaded = predictions_non_threaded.apply(heavi.apply)
    predictions_non_threaded = predictions_non_threaded.replace(0, -1)

    assert predictions_non_threaded[predictions_non_threaded.values == 1].count().values[0] == 211
    assert predictions_non_threaded[predictions_non_threaded.values == -1].count().values[0] == 189

    predictions: pd.DataFrame = model.run(train, test, sub_process=True)
    predictions = predictions.apply(heavi.apply)
    predictions = predictions.replace(0, -1)
    assert predictions[predictions.values == 1].count().values[0] == 211
    assert predictions[predictions.values == -1].count().values[0] == 189
