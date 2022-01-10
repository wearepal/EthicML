"""EthicML Tests."""
from pathlib import Path
from typing import Dict, Generator, List, NamedTuple

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
    Blind,
    Corels,
    CrossValidator,
    DataTuple,
    DPOracle,
    Heaviside,
    InAlgorithm,
    InAlgorithmAsync,
    Kamiran,
    Kamishima,
    LRProb,
    Majority,
    Metric,
    Oracle,
    Prediction,
    SoftPrediction,
    TrainTestPair,
    ZafarAccuracy,
    ZafarBaseline,
    ZafarFairness,
    compas,
    evaluate_models_async,
    load_data,
    query_dt,
    toy,
    train_test_split,
)
from ethicml.algorithms.inprocess.shared import flag_interface
from tests.run_algorithm_test import count_true


class InprocessTest(NamedTuple):
    """Define a test for an inprocess model."""

    name: str
    model: InAlgorithm
    num_pos: int


INPROCESS_TESTS = [
    InprocessTest(name="Agarwal, LR, DP", model=Agarwal(dir='/tmp'), num_pos=45),
    InprocessTest(name="Agarwal, LR, EqOd", model=Agarwal(dir='/tmp', fairness="EqOd"), num_pos=44),
    InprocessTest(name="Agarwal, SVM, DP", model=Agarwal(dir='/tmp', classifier="SVM"), num_pos=45),
    InprocessTest(
        name="Agarwal, SVM, DP",
        model=Agarwal(dir='/tmp', classifier="SVM", kernel="linear"),
        num_pos=42,
    ),
    InprocessTest(
        name="Agarwal, SVM, EqOd",
        model=Agarwal(dir='/tmp', classifier="SVM", fairness="EqOd"),
        num_pos=45,
    ),
    InprocessTest(
        name="Agarwal, SVM, EqOd",
        model=Agarwal(dir='/tmp', classifier="SVM", fairness="EqOd", kernel="linear"),
        num_pos=42,
    ),
    InprocessTest(name="Blind", model=Blind(), num_pos=48),
    InprocessTest(name="DemPar. Oracle", model=DPOracle(), num_pos=53),
    InprocessTest(name="Dist Robust Optim", model=DRO(eta=0.5, dir="/tmp"), num_pos=45),
    InprocessTest(name="Dist Robust Optim", model=DRO(eta=5.0, dir="/tmp"), num_pos=59),
    InprocessTest(name="Kamiran & Calders LR", model=Kamiran(), num_pos=44),
    InprocessTest(name="Logistic Regression (C=1.0)", model=LR(), num_pos=44),
    InprocessTest(name="Logistic Regression Prob (C=1.0)", model=LRProb(), num_pos=44),
    InprocessTest(name="LRCV", model=LRCV(), num_pos=40),
    InprocessTest(name="Majority", model=Majority(), num_pos=80),
    InprocessTest(name="MLP", model=MLP(), num_pos=43),
    InprocessTest(name="Oracle", model=Oracle(), num_pos=41),
    InprocessTest(name="SVM", model=SVM(), num_pos=45),
    InprocessTest(name="SVM (linear)", model=SVM(kernel="linear"), num_pos=41),
    # InprocessTest(name="Zafar", model=ZafarAccuracy(), num_pos=41),
    # InprocessTest(name="Zafar", model=ZafarBaseline(), num_pos=41),
    # InprocessTest(name="Zafar", model=ZafarFairness(), num_pos=41),
]


@pytest.mark.parametrize("name,model,num_pos", INPROCESS_TESTS)
def test_inprocess(toy_train_test: TrainTestPair, name: str, model: InAlgorithm, num_pos: int):
    """Test an inprocess model."""
    train, test = toy_train_test

    assert isinstance(model, InAlgorithm)
    assert model is not None
    assert model.name == name

    predictions: Prediction = model.run(train, test)
    assert count_true(predictions.hard.values == 1) == num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - num_pos


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
    assert count_true(predictions.hard.values == 1) == num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - num_pos


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
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos


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


@pytest.fixture(scope="module")
def kamishima() -> Generator[Kamishima, None, None]:
    kamishima_algo = Kamishima()
    yield kamishima_algo
    print("teardown Kamishima")
    kamishima_algo.remove()


def test_kamishima(toy_train_test: TrainTestPair, kamishima: Kamishima):
    train, test = toy_train_test

    model: InAlgorithm = kamishima
    assert model.name == "Kamishima"

    assert model is not None

    predictions: Prediction = model.run(train, test)
    assert count_true(predictions.hard.values == 1) == 42
    assert count_true(predictions.hard.values == 0) == 38

    another_model = Kamishima()
    new_predictions: Prediction = another_model.fit(train).predict(test)
    assert count_true(new_predictions.hard.values == 1) == 42
    assert count_true(new_predictions.hard.values == 0) == 38


def test_local_installed_lr(toy_train_test: TrainTestPair):
    """Test local installed lr."""
    train, test = toy_train_test

    class _LocalInstalledLR(InAlgorithmAsync):
        def __init__(self):
            super().__init__(name="local installed LR", seed=0, is_fairness_algo=False)

        @implements(InAlgorithmAsync)
        def _run_script_command(
            self, train_path: Path, test_path: Path, pred_path: Path
        ) -> List[str]:
            script = str((Path(__file__).parent.parent.parent / "local_installed_lr.py").resolve())
            return [script, str(train_path), str(test_path), str(pred_path)]

        @implements(InAlgorithmAsync)
        def _fit_script_command(self, train_path: Path, model_path: Path) -> List[str]:
            raise NotImplementedError("this model doesn't support the fit/predict split yet")

        @implements(InAlgorithmAsync)
        def _predict_script_command(
            self, model_path: Path, test_path: Path, pred_path: Path
        ) -> List[str]:
            raise NotImplementedError("this model doesn't support the fit/predict split yet")

    model: InAlgorithm = _LocalInstalledLR()
    assert model is not None
    assert model.name == "local installed LR"

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 44
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos


def test_threaded_agarwal():
    """Test threaded agarwal."""
    models: List[InAlgorithmAsync] = [Agarwal(dir='/tmp', classifier="SVM", fairness="EqOd")]

    class AssertResult(Metric):
        _name = "assert_result"

        def score(self, prediction, actual) -> float:
            x = count_true(prediction.hard.values == 1)
            y = count_true(prediction.hard.values == 0)
            assert x == 41 and y == 39, str(x) + " " + str(y)
            return (
                count_true(prediction.hard.values == 1) == 41
                and count_true(prediction.hard.values == 0) == 39
            )

    results = evaluate_models_async(
        datasets=[toy()], inprocess_models=models, metrics=[AssertResult()], delete_prev=True
    )
    assert results["assert_result"].iloc[0] == 1.0
