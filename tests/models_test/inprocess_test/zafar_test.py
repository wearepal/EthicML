from typing import Dict, List, Type

import pytest
from pytest import approx

from ethicml import (
    AbsCV,
    Accuracy,
    CrossValidator,
    CVResults,
    InAlgorithm,
    Prediction,
    TrainTestPair,
    ZafarAccuracy,
    ZafarBaseline,
    ZafarEqOdds,
    ZafarEqOpp,
    ZafarFairness,
)
from tests.run_algorithm_test import count_true


@pytest.mark.slow
def test_zafar(toy_train_test: TrainTestPair) -> None:
    """

    Args:
        toy_train_test:
    """
    train, test = toy_train_test

    model: InAlgorithm = ZafarAccuracy()
    assert model.name == "ZafarAccuracy, γ=0.5"

    assert model is not None

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 42
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 42
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    hyperparams: Dict[str, List[float]] = {"gamma": [1, 1e-1, 1e-2]}

    model_class: Type[InAlgorithm] = ZafarAccuracy
    zafar_cv = CrossValidator(model_class, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["gamma"] == 1
    assert best_result.scores["Accuracy"] == approx(0.956, abs=1e-3)
    assert best_result.scores["CV absolute"] == approx(0.834, abs=1e-3)

    model = ZafarBaseline()
    assert model.name == "ZafarBaseline"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 40
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 40
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    model = ZafarFairness()
    assert model.name == "ZafarFairness, c=0.001"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 51
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 51
    assert count_true(predictions.hard.values == 1) == expected_num_pos
    assert count_true(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    hyperparams = {"c": [1, 1e-1, 1e-2]}

    model_class = ZafarFairness
    zafar_cv = CrossValidator(model_class, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["c"] == 0.01
    assert best_result.scores["Accuracy"] == approx(0.703, abs=1e-3)
    assert best_result.scores["CV absolute"] == approx(0.855, rel=1e-3)

    # ==================== Zafar Equality of Opportunity ========================
    zafar_eq_opp: InAlgorithm = ZafarEqOpp()
    assert zafar_eq_opp.name == "ZafarEqOpp, τ=5.0, μ=1.2"

    predictions = zafar_eq_opp.run(train, test)
    assert count_true(predictions.hard.values == 1) == 40

    # ==================== Zafar Equalised Odds ========================
    zafar_eq_odds: InAlgorithm = ZafarEqOdds()
    assert zafar_eq_odds.name == "ZafarEqOdds, τ=5.0, μ=1.2"

    predictions = zafar_eq_odds.run(train, test)
    assert count_true(predictions.hard.values == 1) == 40

    print("teardown Zafar")
    ZafarBaseline().remove()
