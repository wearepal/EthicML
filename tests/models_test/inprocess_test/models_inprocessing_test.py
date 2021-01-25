"""EthicML Tests."""
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import pandas as pd
import pytest
from pytest import approx

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
    Heaviside,
    InAlgorithm,
    InAlgorithmAsync,
    InstalledModel,
    Kamiran,
    LRProb,
    Majority,
    Metric,
    Prediction,
    ProbPos,
    SoftPrediction,
    TrainTestPair,
    compas,
    diff_per_sensitive_attribute,
    evaluate_models_async,
    load_data,
    metric_per_sensitive_attribute,
    query_dt,
    run_blocking,
    run_in_parallel,
    toy,
    train_test_split,
)
from ethicml.algorithms.inprocess.oracle import DPOracle, Oracle
from tests.run_algorithm_test import count_true


class InprocessTest(NamedTuple):
    """Define a test for an inprocess model."""

    name: str
    model: InAlgorithm
    num_pos: int


INPROCESS_TESTS = [
    InprocessTest(name="SVM", model=SVM(), num_pos=45),
    InprocessTest(name="SVM (linear)", model=SVM(kernel="linear"), num_pos=41),
    InprocessTest(name="Majority", model=Majority(), num_pos=80),
    InprocessTest(name="Blind", model=Blind(), num_pos=48),
    InprocessTest(name="MLP", model=MLP(), num_pos=43),
    InprocessTest(name="Logistic Regression (C=1.0)", model=LR(), num_pos=44),
    InprocessTest(name="LRCV", model=LRCV(), num_pos=40),
    InprocessTest(name="Dist Robust Optim", model=DRO(eta=0.5), num_pos=45),
    InprocessTest(name="Dist Robust Optim", model=DRO(eta=5.0), num_pos=59),
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


def test_oracle():
    """Test an inprocess model."""
    data: DataTuple = toy().load()
    train, test = train_test_split(data)

    model = Oracle()
    name = "Oracle"
    num_pos = 41

    assert isinstance(model, InAlgorithm)
    assert model is not None
    assert model.name == name

    predictions: Prediction = model.run(train, test)
    assert count_true(predictions.hard.values == 1) == num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - num_pos

    test = test.remove_y()
    with pytest.raises(AssertionError):
        _ = model.run(train, test)


def test_dp_oracle():
    """Test an inprocess model."""
    data: DataTuple = toy().load()
    train, test = train_test_split(data)

    model = DPOracle()
    name = "DemPar. Oracle"
    num_pos = 53

    assert isinstance(model, InAlgorithm)
    assert model is not None
    assert model.name == name

    predictions: Prediction = model.run(train, test)
    assert count_true(predictions.hard.values == 1) == num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - num_pos

    diffs = diff_per_sensitive_attribute(
        metric_per_sensitive_attribute(predictions, test, ProbPos())
    )
    for name, diff in diffs.items():
        assert 0 == pytest.approx(diff, abs=1e-2)

    test = test.remove_y()
    with pytest.raises(AssertionError):
        _ = model.run(train, test)


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


# @pytest.fixture(scope="module")
# def kamishima():
#     kamishima_algo = Kamishima()
#     yield kamishima_algo
#     print("teardown Kamishima")
#     kamishima_algo.remove()


# def test_kamishima(kamishima):
#     train, test = get_train_test()

#     model: InAlgorithmAsync = kamishima
#     assert model.name == "Kamishima"

#     assert model is not None

#     predictions: Prediction = run_blocking(model.run_async(train, test))
#     assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 208
#     assert predictions.hard.values[predictions.hard.values == -1].shape[0] == 192


def test_local_installed_lr(toy_train_test: TrainTestPair):
    """Test local installed lr."""
    train, test = toy_train_test

    class _LocalInstalledLR(InstalledModel):
        def __init__(self):
            super().__init__(
                name="local installed LR", dir_name="../..", top_dir="", executable=sys.executable
            )

        def _script_command(
            self, train_path: Path, test_path: Path, pred_path: Path
        ) -> (List[str]):
            script = "./tests/local_installed_lr.py"
            return [
                script,
                str(train_path),
                str(test_path),
                str(pred_path),
            ]

    model: InAlgorithm = _LocalInstalledLR()
    assert model is not None
    assert model.name == "local installed LR"

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 44
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos


def test_agarwal(toy_train_test: TrainTestPair):
    """Test agarwal."""
    train, test = toy_train_test

    agarwal_variants: List[InAlgorithmAsync] = []
    model_names: List[str] = []
    expected_results: List[Tuple[int, int]] = []

    agarwal_variants.append(Agarwal())
    model_names.append("Agarwal, LR, DP")
    expected_results.append((45, 35))

    agarwal_variants.append(Agarwal(fairness="EqOd"))
    model_names.append("Agarwal, LR, EqOd")
    expected_results.append((44, 36))

    agarwal_variants.append(Agarwal(classifier="SVM"))
    model_names.append("Agarwal, SVM, DP")
    expected_results.append((45, 35))

    agarwal_variants.append(Agarwal(classifier="SVM", kernel="linear"))
    model_names.append("Agarwal, SVM, DP")
    expected_results.append((42, 38))

    agarwal_variants.append(Agarwal(classifier="SVM", fairness="EqOd"))
    model_names.append("Agarwal, SVM, EqOd")
    expected_results.append((45, 35))

    agarwal_variants.append(Agarwal(classifier="SVM", fairness="EqOd", kernel="linear"))
    model_names.append("Agarwal, SVM, EqOd")
    expected_results.append((42, 38))

    results = run_blocking(
        run_in_parallel(agarwal_variants, [TrainTestPair(train, test)], max_parallel=1)
    )

    for model, results_for_model, model_name, (pred_true, pred_false) in zip(
        agarwal_variants, results, model_names, expected_results
    ):
        assert model.name == model_name
        assert count_true(results_for_model[0].hard.to_numpy() == 1) == pred_true, model_name
        assert count_true(results_for_model[0].hard.to_numpy() == 0) == pred_false, model_name


def test_threaded_agarwal():
    """Test threaded agarwal."""
    models: List[InAlgorithmAsync] = [Agarwal(classifier="SVM", fairness="EqOd")]

    class AssertResult(Metric):
        _name = "assert_result"

        def score(self, prediction, actual) -> float:
            return (
                count_true(prediction.hard.values == 1) == 241
                and count_true(prediction.hard.values == 0) == 159
            )

    results = run_blocking(
        evaluate_models_async(
            datasets=[toy()], inprocess_models=models, metrics=[AssertResult()], delete_prev=True
        )
    )
    assert results["assert_result"].iloc[0] == 0.0


def test_lr_prob(toy_train_test: TrainTestPair):
    """Test lr prob."""
    train, test = toy_train_test

    model: LRProb = LRProb()
    assert model.name == "Logistic Regression Prob (C=1.0)"

    heavi = Heaviside()

    predictions: SoftPrediction = model.run(train, test)
    hard_predictions = pd.Series(heavi.apply(predictions.soft.to_numpy()))
    pd.testing.assert_series_equal(hard_predictions, predictions.hard)
    assert hard_predictions.values[hard_predictions.values == 1].shape[0] == 44
    assert hard_predictions.values[hard_predictions.values == 0].shape[0] == 36


def test_kamiran(toy_train_test: TrainTestPair):
    """Test kamiran."""
    train, test = toy_train_test

    kamiran_model: InAlgorithm = Kamiran()
    assert kamiran_model is not None
    assert kamiran_model.name == "Kamiran & Calders LR"

    predictions: Prediction = kamiran_model.run(train, test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 44
    assert predictions.hard.values[predictions.hard.values == 0].shape[0] == 36

    # remove all samples with s=0 & y=1 from the data
    train_no_s0y1 = query_dt(train, "`sensitive-attr` != 0 | decision != 1")
    predictions = kamiran_model.run(train_no_s0y1, test)
