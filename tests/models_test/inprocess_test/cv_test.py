"""Tests for cross validation."""
from typing import Dict, List, NamedTuple, Sequence, Type, Union

import numpy as np
import pytest

from ethicml import KernelType, Prediction, TrainTestPair
from ethicml.metrics import Accuracy
from ethicml.models import InAlgorithm, LR, SVM
from ethicml.run import CrossValidator, CVResults


class CvParam(NamedTuple):
    """Specification of a unit test for cross validation."""

    model: Type[InAlgorithm]
    hyperparams: Dict[str, Union[Sequence[float], List[str], Sequence[KernelType]]]
    num_pos: int


CV_PARAMS = [
    CvParam(
        model=SVM,
        hyperparams={"C": [1, 10, 100], "kernel": [KernelType.rbf, KernelType.linear]},
        num_pos=43,
    ),
    CvParam(model=LR, hyperparams={"C": [0.01, 0.1, 1.0]}, num_pos=44),
]


@pytest.mark.parametrize(("model", "hyperparams", "num_pos"), CV_PARAMS)
def test_cv(
    toy_train_test: TrainTestPair,
    model: Type[InAlgorithm],
    hyperparams: Dict[str, Union[Sequence[float], List[str]]],
    num_pos: int,
) -> None:
    """Test cv svm."""
    train, test = toy_train_test
    cross_validator = CrossValidator(model, hyperparams)
    assert cross_validator is not None

    cv_results = cross_validator.run(train)
    best_model = cv_results.best(Accuracy())
    assert isinstance(best_model, InAlgorithm)

    preds: Prediction = best_model.run(train, test)
    assert np.count_nonzero(preds.hard.to_numpy() == 1) == num_pos
    assert np.count_nonzero(preds.hard.to_numpy() == 0) == len(preds) - num_pos


@pytest.mark.parametrize(("model", "hyperparams", "num_pos"), CV_PARAMS)
def test_parallel_cv(
    toy_train_test: TrainTestPair,
    model: Type[InAlgorithm],
    hyperparams: Dict[str, Union[Sequence[float], List[str]]],
    num_pos: int,
) -> None:
    """Test parallel cv."""
    train, test = toy_train_test
    measure = Accuracy()

    cross_validator = CrossValidator(model, hyperparams, max_parallel=1)
    cv_results: CVResults = cross_validator.run_async(train, measures=[measure])
    best_model = cv_results.best(measure)
    assert isinstance(best_model, InAlgorithm)

    preds: Prediction = best_model.run(train, test)
    assert np.count_nonzero(preds.hard.to_numpy() == 1) == num_pos
    assert np.count_nonzero(preds.hard.to_numpy() == 0) == len(preds) - num_pos
