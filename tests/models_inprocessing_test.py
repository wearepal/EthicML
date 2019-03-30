from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pytest

from ethicml.algorithms.inprocess import (Agarwal, GPyT, GPyTDemPar, GPyTEqOdds, InAlgorithm,
                                          Kamishima, LRCV, LRProb, LR, SVM)
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


@pytest.fixture(scope="module")
def gpyt_models():
    # the gpyt models are created together because they share a git repository
    gpyt = GPyT(epochs=2, length_scale=2.4)
    gpyt_dem_par = GPyTDemPar(epochs=2, length_scale=.05)
    gpyt_eq_odds = GPyTEqOdds(epochs=2, tpr=1.0, length_scale=.05)
    yield (gpyt, gpyt_dem_par, gpyt_eq_odds)
    gpyt.remove()  # only has to be called from one of the models because they share a directory


def test_gpyt(gpyt_models):
    train, test = get_train_test()

    baseline: InAlgorithm = gpyt_models[0]
    dem_par: InAlgorithm = gpyt_models[1]
    eq_odds: InAlgorithm = gpyt_models[2]

    assert baseline.name == "GPyT_in_True"
    predictions: pd.DataFrame = baseline.run(train, test)
    np.testing.assert_allclose(predictions[predictions.values == 1].count().values[0], 210, 0.02)
    np.testing.assert_allclose(predictions[predictions.values == -1].count().values[0], 190, 0.02)

    assert dem_par.name == "GPyT_dem_par_in_True"
    predictions: pd.DataFrame = dem_par.run(train, test)
    np.testing.assert_allclose(predictions[predictions.values == 1].count().values[0], 182, 0.02)
    np.testing.assert_allclose(predictions[predictions.values == -1].count().values[0], 218, 0.02)

    assert eq_odds.name == "GPyT_eq_odds_in_True_tpr_1.0"
    predictions: pd.DataFrame = eq_odds.run(train, test)
    np.testing.assert_allclose(predictions[predictions.values == 1].count().values[0], 179, 0.02)
    np.testing.assert_allclose(predictions[predictions.values == -1].count().values[0], 221, 0.02)


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


def test_agarwal():
    train, test = get_train_test()
    model: InAlgorithm = Agarwal()
    assert model is not None
    assert model.name == "Agarwal LR"

    predictions: np.array = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 145
    assert predictions[predictions.values == -1].count().values[0] == 255

    model = Agarwal(fairness="EqOd")
    predictions = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 162
    assert predictions[predictions.values == -1].count().values[0] == 238

    model = Agarwal(classifier="SVM")
    predictions = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 141
    assert predictions[predictions.values == -1].count().values[0] == 259

    model = Agarwal(classifier="SVM", fairness="EqOd")
    predictions = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 159
    assert predictions[predictions.values == -1].count().values[0] == 241

    model = Agarwal(classifier="SVM", fairness="EqOd")
    predictions = model.run(train, test, sub_process=True)
    assert predictions[predictions.values == 1].count().values[0] == 159
    assert predictions[predictions.values == -1].count().values[0] == 241


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
