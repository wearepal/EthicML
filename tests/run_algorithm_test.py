"""
Test that an algorithm can run against some data
"""
from typing import Tuple
import pandas as pd
import numpy as np
import pytest

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.threaded.threaded_in_algorithm import ThreadedInAlgorithm
from ethicml.algorithms.inprocess.logistic_regression import LR
from ethicml.algorithms.inprocess.threaded.logistic_regression import ThreadedLR
from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.preprocess.beutel import Beutel
from ethicml.algorithms.utils import make_data_tuple, DataTuple
from ethicml.data.adult import Adult
from ethicml.data.compas import Compas
from ethicml.data.german import German
from ethicml.data.load import load_data
from ethicml.data.sqf import Sqf
from ethicml.data.test import Test
from ethicml.evaluators.evaluate_models import evaluate_models, call_on_saved_data
from ethicml.evaluators.per_sensitive_attribute import MetricNotApplicable
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.cv import CV
from ethicml.metrics.tpr import TPR
from ethicml.preprocessing.train_test_split import train_test_split


def get_train_test():
    data: DataTuple = load_data(Test())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_can_make_predictions():
    train, test = get_train_test()

    model: InAlgorithm = SVM()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 201
    assert predictions[predictions.values == -1].count().values[0] == 199


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

    model: ThreadedInAlgorithm = ThreadedLR()

    predictions: pd.DataFrame = call_on_saved_data(model, train, test)
    assert predictions[predictions.values == 1].count().values[0] == 211
    assert predictions[predictions.values == -1].count().values[0] == 189


def test_beutel():
    train, test = get_train_test()

    model: InAlgorithm = Beutel()
    assert model is not None
    assert model.name == "Beutel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = make_data_tuple(new_xtrain, train.s, train.y)
    new_test = make_data_tuple(new_xtest, test.s, test.y)

    model: InAlgorithm = SVM()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run_test(new_train, new_test)
    assert predictions[predictions.values == 1].count().values[0] == 208
    assert predictions[predictions.values == -1].count().values[0] == 192


def test_run_alg_suite():
    datasets = [Test(), Adult(), Compas(), Sqf(), German()]
    preprocess_models = [Beutel()]
    inprocess_models = [LR()]
    postprocess_models = []
    metrics = [Accuracy(), CV()]
    per_sens_metrics = [Accuracy(), TPR()]
    evaluate_models(datasets, preprocess_models, inprocess_models,
                    postprocess_models, metrics, per_sens_metrics, test_mode=True)


def test_run_alg_suite_wrong_metrics():
    datasets = [Test(), Adult()]
    preprocess_models = [Beutel()]
    inprocess_models = [SVM(), LR()]
    postprocess_models = []
    metrics = [Accuracy(), CV()]
    per_sens_metrics = [Accuracy(), TPR(), CV()]
    with pytest.raises(MetricNotApplicable):
        evaluate_models(datasets, preprocess_models, inprocess_models,
                        postprocess_models, metrics, per_sens_metrics, test_mode=True)
