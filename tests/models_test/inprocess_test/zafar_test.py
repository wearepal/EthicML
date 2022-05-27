from typing import Dict, Generator, List, Type

import numpy as np
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


@pytest.fixture(scope="session")
def zafar_teardown() -> Generator[None, None, None]:
    """This fixtures tears down Zafar after all tests have finished.

    This has to be done with a fixture because otherwise it will not happen when the test fails.
    """
    yield
    print("teardown Zafar")
    ZafarBaseline().remove()


@pytest.mark.slow
def test_zafar(toy_train_test: TrainTestPair, zafar_teardown: None) -> None:
    """Test zafar."""
    train, test = toy_train_test

    model: InAlgorithm = ZafarAccuracy()
    assert model.name == "ZafarAccuracy, γ=0.5"

    assert model is not None

    predictions: Prediction = model.run(train, test)
    expected_num_pos = 41
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.fit(train).predict(test)
    expected_num_pos = 41
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.run(train, test)
    expected_num_pos = 41
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

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
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.fit(train).predict(test)
    expected_num_pos = 40
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    model = ZafarFairness()
    assert model.name == "ZafarFairness, C=0.001"

    assert model is not None

    predictions = model.run(train, test)
    expected_num_pos = 51
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    predictions = model.fit(train).predict(test)
    expected_num_pos = 51
    assert np.count_nonzero(predictions.hard.values == 1) == expected_num_pos
    assert np.count_nonzero(predictions.hard.values == 0) == len(predictions) - expected_num_pos

    hyperparams = {"C": [1, 1e-1, 1e-2]}

    model_class = ZafarFairness
    zafar_cv = CrossValidator(model_class, hyperparams, folds=3)

    assert zafar_cv is not None

    primary = Accuracy()
    fair_measure = AbsCV()
    cv_results: CVResults = zafar_cv.run(train, measures=[primary, fair_measure])

    best_result = cv_results.get_best_in_top_k(primary, fair_measure, top_k=3)

    assert best_result.params["C"] == 0.01
    assert best_result.scores["Accuracy"] == approx(0.703, abs=1e-3)
    assert best_result.scores["CV absolute"] == approx(0.855, rel=1e-3)

    # ==================== Zafar Equality of Opportunity ========================
    zafar_eq_opp: InAlgorithm = ZafarEqOpp()
    assert zafar_eq_opp.name == "ZafarEqOpp, τ=5.0, μ=1.2 ε=0.0001"

    predictions = zafar_eq_opp.run(train, test)
    assert np.count_nonzero(predictions.hard.values == 1) == 40

    predictions = zafar_eq_opp.fit(train).predict(test)
    assert np.count_nonzero(predictions.hard.values == 1) == 40

    # ==================== Zafar Equalised Odds ========================
    zafar_eq_odds: InAlgorithm = ZafarEqOdds()
    assert zafar_eq_odds.name == "ZafarEqOdds, τ=5.0, μ=1.2 ε=0.0001"

    predictions = zafar_eq_odds.run(train, test)
    assert np.count_nonzero(predictions.hard.values == 1) == 40

    predictions = zafar_eq_odds.fit(train).predict(test)
    assert np.count_nonzero(predictions.hard.values == 1) == 40
