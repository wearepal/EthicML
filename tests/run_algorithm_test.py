"""
Test that an algorithm can run against some data
"""
from typing import Tuple, List
import pandas as pd
import numpy as np
import pytest

from exsvm.SVM import SVMEXAMPLE

from ethicml.algorithms.inprocess.in_algorithm import InAlgorithm
from ethicml.algorithms.inprocess.logistic_regression_cross_validated import LRCV
from ethicml.algorithms.inprocess.logistic_regression_probability import LRProb
from ethicml.algorithms.inprocess.logistic_regression import LR
from ethicml.algorithms.inprocess.svm import SVM
from ethicml.algorithms.postprocess.post_algorithm import PostAlgorithm
from ethicml.algorithms.preprocess.beutel import Beutel
from ethicml.algorithms.preprocess.pre_algorithm import PreAlgorithm
from ethicml.algorithms.utils import DataTuple
from ethicml.data.dataset import Dataset
from ethicml.data.load import load_data
from ethicml.data import Adult, Compas, German, Sqf, Toy
from ethicml.evaluators.evaluate_models import evaluate_models
from ethicml.evaluators.per_sensitive_attribute import MetricNotApplicable
from ethicml.metrics import Accuracy, CV, TPR, Metric
from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.utility.heaviside import Heaviside


def get_train_test():
    data: DataTuple = load_data(Toy())
    train_test: Tuple[DataTuple, DataTuple] = train_test_split(data)
    return train_test


def test_can_load_test_data():
    train, test = get_train_test()
    assert train is not None
    assert test is not None


def test_svm():
    train, test = get_train_test()

    model: InAlgorithm = SVM()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run(train, test)
    assert predictions[predictions.values == 1].count().values[0] == 201
    assert predictions[predictions.values == -1].count().values[0] == 199


def test_svm_import():
    train, test = get_train_test()

    model: InAlgorithm = SVMEXAMPLE()
    assert model is not None
    assert model.name == "SVM"

    predictions: pd.DataFrame = model.run(train, test, sub_process=True)
    assert predictions[predictions.values == 1].count().values[0] == 201
    assert predictions[predictions.values == -1].count().values[0] == 199

    predictions_non_threaded: pd.DataFrame = model.run(train, test)
    assert predictions_non_threaded[predictions_non_threaded.values == 1].count().values[0] == 201
    assert predictions_non_threaded[predictions_non_threaded.values == -1].count().values[0] == 199


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


def test_beutel():
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = beut_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: pd.DataFrame = svm_model.run_test(new_train, new_test)
    assert predictions[predictions.values == 1].count().values[0] == 208
    assert predictions[predictions.values == -1].count().values[0] == 192


def test_threaded_beutel():
    train, test = get_train_test()

    model: PreAlgorithm = Beutel()
    assert model is not None
    assert model.name == "Beutel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = model.run(train, test, sub_process=True)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert predictions[predictions.values == 1].count().values[0] == 208
    assert predictions[predictions.values == -1].count().values[0] == 192

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_xtrain_xtest = beut_model.run(train, test)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions[predictions.values == 1].count().values[0] == 208
    assert predictions[predictions.values == -1].count().values[0] == 192


def test_threaded_custom_beutel():
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel(epochs=5, fairness="Eq. Opp")
    assert beut_model is not None
    assert beut_model.name == "Beutel"

    new_xtrain_xtest_non_thread: Tuple[pd.DataFrame, pd.DataFrame] = beut_model.run(train, test)
    new_xtrain_nt, new_xtest_nt = new_xtrain_xtest_non_thread

    assert new_xtrain_nt.shape[0] == train.x.shape[0]
    assert new_xtest_nt.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain_nt, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest_nt, s=test.s, y=test.y)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions[predictions.values == 1].count().values[0] == 202
    assert predictions[predictions.values == -1].count().values[0] == 198

    model: PreAlgorithm = Beutel(epochs=5, fairness="Eq. Opp")
    assert model is not None
    assert model.name == "Beutel"

    new_xtrain_xtest: Tuple[pd.DataFrame, pd.DataFrame] = model.run(train, test, sub_process=True)
    new_xtrain, new_xtest = new_xtrain_xtest

    assert new_xtrain.shape[0] == train.x.shape[0]
    assert new_xtest.shape[0] == test.x.shape[0]

    new_train = DataTuple(x=new_xtrain, s=train.s, y=train.y)
    new_test = DataTuple(x=new_xtest, s=test.s, y=test.y)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    treaded_predictions: pd.DataFrame = classifier.run_test(new_train, new_test)
    assert treaded_predictions[treaded_predictions.values == 1].count().values[0] == 202
    assert treaded_predictions[treaded_predictions.values == -1].count().values[0] == 198


def test_run_alg_suite():
    datasets: List[Dataset] = [Toy(), Adult(), Compas(), Sqf(), German()]
    preprocess_models: List[PreAlgorithm] = [Beutel()]
    inprocess_models: List[InAlgorithm] = [LR()]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR()]
    evaluate_models(datasets, preprocess_models, inprocess_models,
                    postprocess_models, metrics, per_sens_metrics, test_mode=True)


def test_run_alg_suite_wrong_metrics():
    datasets: List[Dataset] = [Toy(), Adult()]
    preprocess_models: List[PreAlgorithm] = [Beutel()]
    inprocess_models: List[InAlgorithm] = [SVM(), LR()]
    postprocess_models: List[PostAlgorithm] = []
    metrics: List[Metric] = [Accuracy(), CV()]
    per_sens_metrics: List[Metric] = [Accuracy(), TPR(), CV()]
    with pytest.raises(MetricNotApplicable):
        evaluate_models(datasets, preprocess_models, inprocess_models,
                        postprocess_models, metrics, per_sens_metrics, test_mode=True)
