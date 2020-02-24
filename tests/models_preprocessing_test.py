"""Test preprocessing models"""
from typing import Tuple

import numpy as np
import pandas as pd
from pytest import approx

from ethicml.algorithms import run_blocking
from ethicml.algorithms.inprocess import LR, SVM, InAlgorithm
from ethicml.algorithms.preprocess import (
    VFAE,
    Beutel,
    Calders,
    PreAlgorithm,
    PreAlgorithmAsync,
    Upsampler,
    Zemel,
)
from ethicml.preprocessing import query_dt, train_test_split
from ethicml.utility import DataTuple, Prediction, TestTuple
from tests.run_algorithm_test import get_train_test


def test_vfae():
    """test vfae"""
    train, test = get_train_test()

    vfae_model: PreAlgorithm = VFAE(dataset="Toy", epochs=10, batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_train_test: Tuple[DataTuple, TestTuple] = vfae_model.run(train, test)
    new_train, new_test = new_train_test

    assert len(new_train) == len(train)
    assert new_test.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions: Prediction = svm_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 201
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 199

    vfae_model = VFAE(dataset="Toy", supervised=True, epochs=10, fairness="Eq. Opp", batch_size=100)
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_train_test = vfae_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "VFAE: " + str(test.name)
    assert new_train.name == "VFAE: " + str(train.name)

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 201
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 199

    vfae_model = VFAE(
        dataset="Toy", supervised=False, epochs=10, fairness="Eq. Opp", batch_size=100
    )
    assert vfae_model is not None
    assert vfae_model.name == "VFAE"

    new_train_test = vfae_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 207
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 193


def test_threaded_zemel():
    """test threaded zemel"""
    train, test = get_train_test()

    model: PreAlgorithmAsync = Zemel()
    assert model is not None
    assert model.name == "Zemel"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(model.run_async(train, test))
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: Prediction = classifier.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 182
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 218

    beut_model: PreAlgorithm = Zemel()
    assert beut_model is not None
    assert beut_model.name == "Zemel"

    new_train_test = beut_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "Zemel: " + str(test.name)
    assert new_train.name == "Zemel: " + str(train.name)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 182
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 218


def test_threaded_beutel():
    """test threaded beutel"""
    train, test = get_train_test()

    model: PreAlgorithmAsync = Beutel()
    assert model is not None
    assert model.name == "Beutel DP"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(model.run_async(train, test))
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "Beutel DP: " + str(test.name)
    assert new_train.name == "Beutel DP: " + str(train.name)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: Prediction = classifier.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 201
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 199

    beut_model: PreAlgorithm = Beutel()
    assert beut_model is not None
    assert beut_model.name == "Beutel DP"

    new_train_test = beut_model.run(train, test)
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "Beutel DP: " + str(test.name)
    assert new_train.name == "Beutel DP: " + str(train.name)

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 201
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 199


def test_threaded_custom_beutel():
    """test threaded custom beutel"""
    train, test = get_train_test()

    beut_model: PreAlgorithm = Beutel(epochs=5, fairness="EqOp")
    assert beut_model is not None
    assert beut_model.name == "Beutel EqOp"

    new_train_test_non_thread: Tuple[DataTuple, TestTuple] = beut_model.run(train, test)
    new_train_nt, new_test_nt = new_train_test_non_thread

    assert new_train_nt.x.shape[0] == train.x.shape[0]
    assert new_test_nt.x.shape[0] == test.x.shape[0]

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM"

    predictions = svm_model.run_test(new_train_nt, new_test_nt)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 202
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 198

    model: PreAlgorithmAsync = Beutel(epochs=5, fairness="EqOp")
    assert model is not None
    assert model.name == "Beutel EqOp"

    new_train_test: Tuple[DataTuple, TestTuple] = run_blocking(model.run_async(train, test))
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "Beutel EqOp: " + str(test.name)
    assert new_train.name == "Beutel EqOp: " + str(train.name)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    treaded_predictions: Prediction = classifier.run_test(new_train, new_test)
    assert treaded_predictions.hard.values[treaded_predictions.hard.values == 1].shape[0] == 202
    assert treaded_predictions.hard.values[treaded_predictions.hard.values == -1].shape[0] == 198


def test_upsampler():
    """test upsampler"""
    train, test = get_train_test()

    upsampler: PreAlgorithm = Upsampler(strategy="naive")
    assert upsampler is not None
    assert upsampler.name == "Upsample naive"

    new_train: DataTuple
    new_test: TestTuple
    new_train, new_test = upsampler.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == test.name
    assert new_train.name == train.name

    lr_model: InAlgorithm = LR()
    assert lr_model is not None
    assert lr_model.name == "Logistic Regression, C=1.0"

    predictions = lr_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 209
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 191

    upsampler = Upsampler(strategy="uniform")
    new_train, new_test = upsampler.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == test.name
    assert new_train.name == train.name

    predictions = lr_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 215
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 185

    upsampler = Upsampler(strategy="preferential")
    new_train, new_test = upsampler.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == test.name
    assert new_train.name == train.name

    predictions = lr_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 148
    assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 252


def test_calders():
    """test calders"""
    data = DataTuple(
        x=pd.DataFrame(np.linspace(0, 1, 100), columns=["x"]),
        s=pd.DataFrame([1] * 75 + [0] * 25, columns=["s"]),
        y=pd.DataFrame([1] * 50 + [0] * 25 + [1] * 10 + [0] * 15, columns=["y"]),
        name="TestData",
    )
    data, _ = train_test_split(data, train_percentage=1.0)
    assert len(query_dt(data, "s == 0 & y == 0")) == 15
    assert len(query_dt(data, "s == 0 & y == 1")) == 10
    assert len(query_dt(data, "s == 1 & y == 0")) == 25
    assert len(query_dt(data, "s == 1 & y == 1")) == 50
    assert query_dt(data, "s == 1 & y == 0").x.min().min() == approx(0.50, abs=0.01)

    calders: PreAlgorithm = Calders(preferable_class=1, disadvantaged_group=0)
    new_train, new_test = calders.run(data, data.remove_y())

    pd.testing.assert_frame_equal(new_test.x, data.x)
    pd.testing.assert_frame_equal(new_test.s, data.s)

    assert len(query_dt(new_train, "s == 0 & y == 0")) == 10
    assert len(query_dt(new_train, "s == 0 & y == 1")) == 15
    assert len(query_dt(new_train, "s == 1 & y == 0")) == 30
    assert len(query_dt(new_train, "s == 1 & y == 1")) == 45

    assert len(data) == len(new_train)
    assert query_dt(new_train, "s == 1 & y == 1").x.min().min() == 0
    assert query_dt(new_train, "s == 1 & y == 0").x.min().min() == approx(0.45, abs=0.01)
