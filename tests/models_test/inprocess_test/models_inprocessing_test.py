"""EthicML Tests."""
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, List, Mapping, NamedTuple

import numpy as np
import pytest
from pytest import approx
from ranzen import implements

from ethicml import (
    DRO,
    LR,
    LRCV,
    MLP,
    SVM,
    AbsCV,
    Accuracy,
    Agarwal,
    BaseMetric,
    Blind,
    ClassifierType,
    Corels,
    CrossValidator,
    DataTuple,
    DPOracle,
    FairnessType,
    InAlgoArgs,
    InAlgorithm,
    InAlgorithmSubprocess,
    Kamiran,
    Kamishima,
    KernelType,
    LRProb,
    Majority,
    Oracle,
    Prediction,
    TrainTestPair,
    compas,
    evaluate_models,
    load_data,
    toy,
    train_test_split,
)
from ethicml.algorithms.inprocess.in_algorithm import HyperParamType


class InprocessTest(NamedTuple):
    """Define a test for an inprocess model."""

    name: str
    model: InAlgorithm
    num_pos: int


INPROCESS_TESTS = [
    InprocessTest(name="Agarwal, lr, dp", model=Agarwal(dir='/tmp'), num_pos=45),
    InprocessTest(
        name="Agarwal, lr, eq_odds",
        model=Agarwal(dir='/tmp', fairness=FairnessType.eq_odds),
        num_pos=44,
    ),
    InprocessTest(
        name="Agarwal, svm, dp",
        model=Agarwal(dir='/tmp', classifier=ClassifierType.svm),
        num_pos=45,
    ),
    InprocessTest(
        name="Agarwal, svm, dp",
        model=Agarwal(dir='/tmp', classifier=ClassifierType.svm, kernel=KernelType.linear),
        num_pos=42,
    ),
    InprocessTest(
        name="Agarwal, svm, eq_odds",
        model=Agarwal(dir='/tmp', classifier=ClassifierType.svm, fairness=FairnessType.eq_odds),
        num_pos=45,
    ),
    InprocessTest(
        name="Agarwal, svm, eq_odds",
        model=Agarwal(
            dir='/tmp',
            classifier=ClassifierType.svm,
            fairness=FairnessType.eq_odds,
            kernel=KernelType.linear,
        ),
        num_pos=42,
    ),
    InprocessTest(name="Blind", model=Blind(), num_pos=48),
    InprocessTest(name="DemPar. Oracle", model=DPOracle(), num_pos=53),
    InprocessTest(name="Dist Robust Optim", model=DRO(eta=0.5, dir="/tmp"), num_pos=45),
    InprocessTest(name="Dist Robust Optim", model=DRO(eta=5.0, dir="/tmp"), num_pos=59),
    InprocessTest(name="Kamiran & Calders lr C=1.0", model=Kamiran(), num_pos=44),
    InprocessTest(name="Logistic Regression (C=1.0)", model=LR(), num_pos=44),
    InprocessTest(name="Logistic Regression Prob (C=1.0)", model=LRProb(), num_pos=44),
    InprocessTest(name="LRCV", model=LRCV(), num_pos=40),
    InprocessTest(name="Majority", model=Majority(), num_pos=80),
    InprocessTest(name="MLP", model=MLP(), num_pos=43),
    InprocessTest(name="Oracle", model=Oracle(), num_pos=41),
    InprocessTest(name="SVM (rbf)", model=SVM(), num_pos=45),
    InprocessTest(name="SVM (linear)", model=SVM(kernel=KernelType.linear), num_pos=41),
]


@pytest.mark.parametrize("name,model,num_pos", INPROCESS_TESTS)
def test_inprocess(toy_train_test: TrainTestPair, name: str, model: InAlgorithm, num_pos: int):
    """Test an inprocess model."""
    train, test = toy_train_test

    assert isinstance(model, InAlgorithm)
    assert model is not None
    assert model.name == name

    predictions: Prediction = model.run(train, test)
    assert np.count_nonzero(predictions.hard.values == 1) == num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - num_pos


def test_kamiran_weights(toy_train_test: TrainTestPair):
    """Test the weights of the Kamiran model are accessible."""
    train, test = toy_train_test

    model = Kamiran()

    _ = model.run(train, test)
    assert model.group_weights is not None
    assert model.group_weights == {
        0: {'weight': 0.8451576576576577, 'count': 111.0, 'sensitive-attr': 1.0, 'decision': 1.0},
        1: {'weight': 0.7929216867469879, 'count': 83.0, 'sensitive-attr': 0.0, 'decision': 0.0},
        2: {'weight': 1.2175632911392404, 'count': 79.0, 'sensitive-attr': 0.0, 'decision': 1.0},
        3: {'weight': 1.365691489361702, 'count': 47.0, 'sensitive-attr': 1.0, 'decision': 0.0},
    }


@pytest.mark.parametrize("name,model,num_pos", INPROCESS_TESTS)
def test_inprocess_sep_train_pred(
    toy_train_test: TrainTestPair, name: str, model: InAlgorithm, num_pos: int
):
    """Test an inprocess model with distinct train and predict steps."""
    train, test = toy_train_test

    assert isinstance(model, InAlgorithm)
    assert model is not None
    assert model.name == name

    model = model.fit(train)
    predictions: Prediction = model.predict(test)
    assert np.count_nonzero(predictions.hard.values == 1) == num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - num_pos


def test_corels(toy_train_test: TrainTestPair) -> None:
    """Test corels."""
    model: InAlgorithm = Corels()
    assert model is not None
    assert model.name == "CORELS"

    train_toy, test_toy = toy_train_test
    with pytest.raises(RuntimeError):
        model.run(train_toy, test_toy)

    data: DataTuple = load_data(compas())
    train, test = train_test_split(data)

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 428
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos


def test_fair_cv_lr(toy_train_test: TrainTestPair) -> None:
    """Test fair cv lr."""
    train, _ = toy_train_test

    hyperparams: Dict[str, List[float]] = {"C": [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}

    lr_cv = CrossValidator(LR, hyperparams, folds=3)

    assert lr_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results = lr_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)
    print(best_result)

    assert best_result.params["C"] == 1e-4
    assert best_result.scores["Accuracy"] == approx(0.888, abs=0.001)
    assert best_result.scores["CV absolute"] == approx(0.832, abs=0.001)


@pytest.fixture(scope="session")
def kamishima_gen() -> Generator[Kamishima, None, None]:
    """This fixtures tears down Kamishima after all tests have finished.

    This has to be done with a fixture because otherwise it will not happen when the test fails.
    """
    yield Kamishima()
    print("teardown Kamishima")
    Kamishima().remove()  # delete the downloaded code


@pytest.mark.slow
def test_kamishima(toy_train_test: TrainTestPair, kamishima_gen: Kamishima) -> None:
    """Test Kamishima."""
    train, test = toy_train_test

    model = kamishima_gen  # this will download the code from github and install pipenv
    assert model.name == "Kamishima"

    assert model is not None

    predictions: Prediction = model.run(train, test)
    assert np.count_nonzero(predictions.hard.values == 1) == 42
    assert np.count_nonzero(predictions.hard.values == 0) == 38

    another_model = Kamishima()
    new_predictions: Prediction = another_model.fit(train).predict(test)
    assert np.count_nonzero(new_predictions.hard.values == 1) == 42
    assert np.count_nonzero(new_predictions.hard.values == 0) == 38


def test_local_installed_lr(toy_train_test: TrainTestPair):
    """Test local installed lr."""
    train, test = toy_train_test

    class _LocalInstalledLR(InAlgorithmSubprocess):
        is_fairness_algo: ClassVar[bool] = False

        def get_name(self) -> str:
            return "local installed LR"

        def _get_path_to_script(self) -> List[str]:
            return [str((Path(__file__).parent.parent.parent / "local_installed_lr.py").resolve())]

        def _get_flags(self) -> Mapping[str, Any]:
            return {}

        def get_hyperparameters(self) -> HyperParamType:
            return {}

    model: InAlgorithm = _LocalInstalledLR()
    assert model is not None
    assert model.name == "local installed LR"

    predictions: Prediction = model.run(train, test, seed=0)
    expected_num_pos = 44
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos


@pytest.mark.slow
def test_threaded_agarwal():
    """Test threaded agarwal."""
    models: List[InAlgorithmSubprocess] = [
        Agarwal(dir='/tmp', classifier=ClassifierType.svm, fairness=FairnessType.eq_odds)
    ]

    class AssertResult(BaseMetric):
        apply_per_sensitive: ClassVar[bool] = True
        _name: ClassVar[str] = "assert_result"

        def score(self, prediction, actual) -> float:
            return (
                np.count_nonzero(prediction.hard.values == 1) == 45
                and np.count_nonzero(prediction.hard.values == 0) == 35
            )

    results = evaluate_models(
        datasets=[toy()], inprocess_models=models, metrics=[AssertResult()], delete_prev=True
    )
    assert results["assert_result"].iloc[0]
