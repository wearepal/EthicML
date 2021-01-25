"""Test preprocessing models"""
from typing import Tuple

import numpy as np
import pandas as pd
from pytest import approx

import ethicml as em
from ethicml import (
    LR,
    SVM,
    VFAE,
    Beutel,
    Calders,
    DataTuple,
    InAlgorithm,
    PreAlgorithm,
    PreAlgorithmAsync,
    Prediction,
    TestTuple,
    TrainTestPair,
    Upsampler,
    Zemel,
)


def test_vfae(toy_train_test: TrainTestPair):
    """test vfae"""
    train, test = toy_train_test

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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 65
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 15

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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 65
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 15

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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 44
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 36


def test_threaded_zemel(toy_train_test: TrainTestPair):
    """test threaded zemel"""
    train, test = toy_train_test

    model: PreAlgorithmAsync = Zemel()
    assert model is not None
    assert model.name == "Zemel"

    new_train_test: Tuple[DataTuple, TestTuple] = em.run_blocking(model.run_async(train, test))
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: Prediction = classifier.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 51
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 29

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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 51
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 29


def test_threaded_beutel(toy_train_test: TrainTestPair):
    """test threaded beutel"""
    train, test = toy_train_test

    model: PreAlgorithmAsync = Beutel()
    assert model is not None
    assert model.name == "Beutel DP"

    new_train_test: Tuple[DataTuple, TestTuple] = em.run_blocking(model.run_async(train, test))
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "Beutel DP: " + str(test.name)
    assert new_train.name == "Beutel DP: " + str(train.name)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    predictions: Prediction = classifier.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 49
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 31

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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 49
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 31


def test_threaded_custom_beutel(toy_train_test: TrainTestPair):
    """test threaded custom beutel"""
    train, test = toy_train_test

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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 56
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 24

    model: PreAlgorithmAsync = Beutel(epochs=5, fairness="EqOp")
    assert model is not None
    assert model.name == "Beutel EqOp"

    new_train_test: Tuple[DataTuple, TestTuple] = em.run_blocking(model.run_async(train, test))
    new_train, new_test = new_train_test

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == "Beutel EqOp: " + str(test.name)
    assert new_train.name == "Beutel EqOp: " + str(train.name)

    classifier: InAlgorithm = SVM()
    assert classifier is not None
    assert classifier.name == "SVM"

    treaded_predictions: Prediction = classifier.run_test(new_train, new_test)
    assert treaded_predictions.hard.values[treaded_predictions.hard.values == 1].shape[0] == 56
    assert treaded_predictions.hard.values[treaded_predictions.hard.values == 0].shape[0] == 24


def test_upsampler(toy_train_test: TrainTestPair):
    """test upsampler"""
    train, test = toy_train_test

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
    assert lr_model.name == "Logistic Regression (C=1.0)"

    predictions = lr_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 39
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 41

    upsampler = Upsampler(strategy="uniform")
    new_train, new_test = upsampler.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == test.name
    assert new_train.name == train.name

    predictions = lr_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 43
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 37

    upsampler = Upsampler(strategy="preferential")
    new_train, new_test = upsampler.run(train, test)

    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == test.name
    assert new_train.name == train.name

    predictions = lr_model.run_test(new_train, new_test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 44
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 36


def test_calders():
    """test calders"""
    data = DataTuple(
        x=pd.DataFrame(np.linspace(0, 1, 100), columns=["x"]),
        s=pd.DataFrame([1] * 75 + [0] * 25, columns=["s"]),
        y=pd.DataFrame([1] * 50 + [0] * 25 + [1] * 10 + [0] * 15, columns=["y"]),
        name="TestData",
    )
    data, _ = em.train_test_split(data, train_percentage=1.0)
    assert len(em.query_dt(data, "s == 0 & y == 0")) == 15
    assert len(em.query_dt(data, "s == 0 & y == 1")) == 10
    assert len(em.query_dt(data, "s == 1 & y == 0")) == 25
    assert len(em.query_dt(data, "s == 1 & y == 1")) == 50
    assert em.query_dt(data, "s == 1 & y == 0").x.min().min() == approx(0.50, abs=0.01)

    calders: PreAlgorithm = Calders(preferable_class=1, disadvantaged_group=0)
    new_train, new_test = calders.run(data, data.remove_y())

    pd.testing.assert_frame_equal(new_test.x, data.x)
    pd.testing.assert_frame_equal(new_test.s, data.s)

    assert len(em.query_dt(new_train, "s == 0 & y == 0")) == 10
    assert len(em.query_dt(new_train, "s == 0 & y == 1")) == 15
    assert len(em.query_dt(new_train, "s == 1 & y == 0")) == 30
    assert len(em.query_dt(new_train, "s == 1 & y == 1")) == 45

    assert len(data) == len(new_train)
    assert em.query_dt(new_train, "s == 1 & y == 1").x.min().min() == 0
    assert em.query_dt(new_train, "s == 1 & y == 0").x.min().min() == approx(0.45, abs=0.01)
