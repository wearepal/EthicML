"""
Test that an algorithm can run against some data
"""
import os
from typing import Tuple, Dict
import pandas as pd
import numpy as np

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.logistic_regression import LR
from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.preprocess.beutel import Beutel
from ethicml.algorithms.utils import make_dict
from ethicml.data.load import load_data
from ethicml.data.test import Test
from ethicml.evaluators.evaluate_models import evaluate_models
from ethicml.metrics.accuracy import Accuracy
from ethicml.metrics.tpr import TPR
from ethicml.preprocessing.train_test_split import train_test_split


def get_train_test():
    data: Dict[str, pd.DataFrame] = load_data(Test())
    train_test: Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]] = train_test_split(data)
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


def test_beutel():
    train, test = get_train_test()

    model: InAlgorithm = Beutel()
    assert model is not None
    assert model.name == "Beutel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train['x'].shape[0]
    assert new_xtest.shape[0] == test['x'].shape[0]

    new_train = make_dict(new_xtrain, train['s'], train['y'])
    new_test = make_dict(new_xtest, test['s'], test['y'])

    model: InAlgorithm = SVM()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run_test(new_train, new_test)
    assert predictions[predictions.values == 1].count().values[0] == 208
    assert predictions[predictions.values == -1].count().values[0] == 192


def test_run_alg_suite():
    datasets = [Test()]
    models = [SVM(), LR()]
    metrics = [Accuracy()]
    per_sens_metrics = [Accuracy(), TPR()]
    result = evaluate_models(datasets, models, metrics, per_sens_metrics)

    outdir = '../results'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    result.to_csv("../results/res.csv", index=False)
