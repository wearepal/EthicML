from typing import Dict, List

import pytest
from pytest import approx

from ethicml.algorithms import run_blocking
from ethicml.algorithms.inprocess import (
    InAlgorithm,
    InAlgorithmAsync,
    ZafarAccuracy,
    ZafarBaseline,
    ZafarEqOdds,
    ZafarEqOpp,
    ZafarFairness,
)
from ethicml.evaluators import CrossValidator, CVResults
from ethicml.metrics import AbsCV, Accuracy
from ethicml.utility import Prediction, TrainTestPair


@pytest.fixture(scope="module")
def zafar_models():
    """Create and destroy Zafar models

    Returns:
        Tuple of three Zafar model classes
    """
    zafarAcc_algo = ZafarAccuracy
    zafarBas_algo = ZafarBaseline
    zafarFai_algo = ZafarFairness
    yield (zafarAcc_algo, zafarBas_algo, zafarFai_algo)
    print("teardown Zafar Accuracy")
    zafarAcc_algo().remove()


def test_zafar(zafar_models, toy_train_test: TrainTestPair) -> None:
    """

    Args:
        zafarModels:
        toy_train_test:
    """
    train, test = toy_train_test

    model: InAlgorithmAsync = zafar_models[0]()
    assert model.name == "ZafarAccuracy, γ=0.5"

    assert model is not None

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 42
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos

    predictions = run_blocking(model.run_async(train, test))
    expected_num_pos = 42
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos

    hyperparams: Dict[str, List[float]] = {"gamma": [1, 1e-1, 1e-2]}

    model = zafar_models[0]
    zafar_cv = CrossValidator(model, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["gamma"] == 0.01
    assert best_result.scores["Accuracy"] == approx(0.995, abs=0.001)
    assert best_result.scores["CV absolute"] == approx(0.851, abs=0.001)

    model = zafar_models[1]()
    assert model.name == "ZafarBaseline"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 241
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos

    predictions = run_blocking(model.run_async(train, test))
    expected_num_pos = 241
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos

    model = zafar_models[2]()
    assert model.name == "ZafarFairness, c=0.001"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 42
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos

    predictions = run_blocking(model.run_async(train, test))
    expected_num_pos = 42
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == expected_num_pos
    num_neg = predictions.hard.values[predictions.hard.values == 0].shape[0]
    assert num_neg == len(predictions) - expected_num_pos

    hyperparams = {"c": [1, 1e-1, 1e-2]}

    model = zafar_models[2]
    zafar_cv = CrossValidator(model, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["c"] == 0.01
    assert best_result.scores["Accuracy"] == approx(0.830, abs=0.001)
    assert best_result.scores["CV absolute"] == approx(0.970, rel=0.001)

    # ==================== Zafar Equality of Opportunity ========================
    zafar_eq_opp: InAlgorithm = ZafarEqOpp()
    assert zafar_eq_opp.name == "ZafarEqOpp, τ=5.0, μ=1.2"

    predictions = zafar_eq_opp.run(train, test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 241

    # ==================== Zafar Equalised Odds ========================
    zafar_eq_odds: InAlgorithm = ZafarEqOdds()
    assert zafar_eq_odds.name == "ZafarEqOdds, τ=5.0, μ=1.2"

    predictions = zafar_eq_odds.run(train, test)
    assert predictions.hard.values[predictions.hard.values == 1].shape[0] == 241
