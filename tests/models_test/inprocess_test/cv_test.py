"""Tests for cross validation."""
from typing import Dict, List, NamedTuple, Sequence, Type, Union

import pytest

import ethicml as em
from ethicml import LR, SVM, Accuracy, InAlgorithm, Prediction, TrainTestPair


class CvParam(NamedTuple):
    """Specification of a unit test for cross validation."""

    model: Type[InAlgorithm]
    hyperparams: Dict[str, Union[Sequence[float], List[str]]]
    num_pos: int


CV_PARAMS = [
    CvParam(model=SVM, hyperparams={"C": [1, 10, 100], "kernel": ["rbf", "linear"]}, num_pos=43),
    CvParam(model=LR, hyperparams={"C": [0.01, 0.1, 1.0]}, num_pos=44),
]


@pytest.mark.parametrize("model,hyperparams,num_pos", CV_PARAMS)
def test_cv(
    toy_train_test: TrainTestPair,
    model: Type[InAlgorithm],
    hyperparams: Dict[str, Union[Sequence[float], List[str]]],
    num_pos: int,
):
    """test cv svm"""
    train, test = toy_train_test
    cross_validator = em.CrossValidator(model, hyperparams)
    assert cross_validator is not None

    cv_results = cross_validator.run(train)
    best_model = cv_results.best(Accuracy())
    assert isinstance(best_model, InAlgorithm)

    preds: Prediction = best_model.run(train, test)
    assert preds.hard.values[preds.hard.values == 1].shape[0] == num_pos
    assert preds.hard.values[preds.hard.values == 0].shape[0] == len(preds) - num_pos


@pytest.mark.parametrize("model,hyperparams,num_pos", CV_PARAMS)
def test_parallel_cv(
    toy_train_test: TrainTestPair,
    model: Type[InAlgorithm],
    hyperparams: Dict[str, Union[Sequence[float], List[str]]],
    num_pos: int,
) -> None:
    """test parallel cv."""
    train, test = toy_train_test
    measure = Accuracy()

    cross_validator = em.CrossValidator(model, hyperparams, max_parallel=1)
    cv_results: em.CVResults = em.run_blocking(cross_validator.run_async(train, measures=[measure]))
    best_model = cv_results.best(measure)
    assert isinstance(best_model, InAlgorithm)

    preds: Prediction = best_model.run(train, test)
    assert preds.hard.values[preds.hard.values == 1].shape[0] == num_pos
    assert preds.hard.values[preds.hard.values == 0].shape[0] == len(preds) - num_pos
