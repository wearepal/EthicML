"""Test preprocessing models."""
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from pytest import approx

import ethicml as em
from ethicml import (
    LR,
    SVM,
    VFAE,
    Beutel,
    Calders,
    DataTuple,
    FairnessType,
    InAlgorithm,
    PreAlgorithm,
    PreAlgorithmAsync,
    Prediction,
    TestTuple,
    TrainTestPair,
    Upsampler,
    UpsampleStrategy,
    Zemel,
)


class PreprocessTest(NamedTuple):
    """Define a test for a preprocess model."""

    model: PreAlgorithm
    name: str
    num_pos: int


METHOD_LIST = [
    PreprocessTest(
        model=VFAE(
            dir='/tmp',
            dataset="Toy",
            supervised=True,
            epochs=10,
            fairness="Eq. Opp",
            batch_size=100,
        ),
        name="VFAE",
        num_pos=56,
    ),
    PreprocessTest(
        model=VFAE(
            dir='/tmp',
            dataset="Toy",
            supervised=False,
            epochs=10,
            fairness="Eq. Opp",
            batch_size=100,
        ),
        name="VFAE",
        num_pos=47,
    ),
    PreprocessTest(model=Zemel(dir='/tmp'), name="Zemel", num_pos=51),
    PreprocessTest(model=Beutel(dir='/tmp'), name="Beutel dp", num_pos=49),
    PreprocessTest(
        model=Beutel(dir='/tmp', epochs=5, fairness=FairnessType.eqod),
        name="Beutel EqOp",
        num_pos=56,
    ),
    PreprocessTest(
        model=Upsampler(strategy=UpsampleStrategy.naive), name="Upsample naive", num_pos=43
    ),
    PreprocessTest(
        model=Upsampler(strategy=UpsampleStrategy.uniform), name="Upsample uniform", num_pos=44
    ),
    PreprocessTest(
        model=Upsampler(strategy=UpsampleStrategy.preferential),
        name="Upsample preferential",
        num_pos=45,
    ),
    PreprocessTest(
        model=Calders(preferable_class=1, disadvantaged_group=0), name="Calders", num_pos=43
    ),
]


@pytest.mark.parametrize("model,name,num_pos", METHOD_LIST)
def test_pre(toy_train_test: TrainTestPair, model: PreAlgorithm, name: str, num_pos: int):
    """Test preprocessing."""
    train, test = toy_train_test

    svm_model: InAlgorithm = SVM()

    assert model.name == name
    new_train, new_test = model.run(train, test)

    if not isinstance(model, Upsampler):
        assert new_train.x.shape[0] == train.x.shape[0]
        assert new_test.x.shape[0] == test.x.shape[0]

    assert new_train.x.shape[1] == model.out_size
    assert new_test.x.shape[1] == model.out_size
    assert new_test.name == f"{name}: " + str(test.name)
    assert new_train.name == f"{name}: " + str(train.name)

    preds = svm_model.run_test(new_train, new_test)
    assert preds.hard.values[preds.hard.values == 1].shape[0] == num_pos
    assert preds.hard.values[preds.hard.values == 0].shape[0] == len(preds) - num_pos


@pytest.mark.parametrize("model,name,num_pos", METHOD_LIST)
def test_pre_sep_fit_transform(
    toy_train_test: TrainTestPair, model: PreAlgorithm, name: str, num_pos: int
):
    """Test preprocessing."""
    train, test = toy_train_test

    svm_model: InAlgorithm = SVM()

    assert model.name == name
    model, new_train = model.fit(train)
    new_test = model.transform(test)

    if not isinstance(model, Upsampler):
        assert new_train.x.shape[0] == train.x.shape[0]
        assert new_test.x.shape[0] == test.x.shape[0]

    assert new_train.x.shape[1] == model.out_size
    assert new_test.x.shape[1] == model.out_size
    assert new_test.name == f"{name}: " + str(test.name)
    assert new_train.name == f"{name}: " + str(train.name)

    preds = svm_model.run_test(new_train, new_test)
    assert preds.hard.values[preds.hard.values == 1].shape[0] == num_pos
    assert preds.hard.values[preds.hard.values == 0].shape[0] == len(preds) - num_pos


@pytest.mark.parametrize(
    "model,name,num_pos",
    [
        PreprocessTest(
            model=VFAE(
                dir='/tmp',
                dataset="Toy",
                supervised=True,
                epochs=10,
                fairness="Eq. Opp",
                batch_size=100,
            ),
            name="VFAE",
            num_pos=56,
        ),
        PreprocessTest(
            model=VFAE(
                dir='/tmp',
                dataset="Toy",
                supervised=False,
                epochs=10,
                fairness="Eq. Opp",
                batch_size=100,
            ),
            name="VFAE",
            num_pos=47,
        ),
        PreprocessTest(model=Zemel(dir='/tmp'), name="Zemel", num_pos=51),
        PreprocessTest(model=Beutel(dir='/tmp'), name="Beutel dp", num_pos=49),
        PreprocessTest(
            model=Beutel(dir='/tmp', epochs=5, fairness=FairnessType.eqod),
            name="Beutel EqOp",
            num_pos=56,
        ),
    ],
)
def test_threaded_pre(toy_train_test: TrainTestPair, model: PreAlgorithm, name: str, num_pos: int):
    """Test vfae."""
    train, test = toy_train_test

    svm_model: InAlgorithm = SVM()
    assert svm_model is not None
    assert svm_model.name == "SVM (rbf)"

    assert model.name == name
    new_train_test = model.run(train, test)
    new_train, new_test = new_train_test

    assert len(new_train) == len(train)
    assert new_test.x.shape[0] == test.x.shape[0]

    assert new_train.x.shape[0] == train.x.shape[0]
    assert new_test.x.shape[0] == test.x.shape[0]
    assert new_test.name == f"{name}: " + str(test.name)
    assert new_train.name == f"{name}: " + str(train.name)

    preds = svm_model.run_test(new_train, new_test)
    assert preds.hard.values[preds.hard.values == 1].shape[0] == num_pos
    assert preds.hard.values[preds.hard.values == 0].shape[0] == len(preds) - num_pos


def test_calders():
    """Test calders."""
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
