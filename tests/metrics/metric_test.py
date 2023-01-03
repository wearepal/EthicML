"""Metric Tests."""
from typing import NamedTuple

import numpy as np
import pytest
from pytest import approx

from ethicml import LabelTuple, Prediction, TrainValPair
from ethicml.metrics import (
    Accuracy,
    AS,
    AverageOddsDiff,
    BalancedAccuracy,
    CV,
    DependencyTarget,
    F1,
    Hsic,
    Metric,
    NMI,
    RobustAccuracy,
    Theil,
)
from ethicml.models import InAlgorithm, LR, Reweighting, SVM
from tests.conftest import get_id


class MetricTest(NamedTuple):
    """Define a test for a metric."""

    model: InAlgorithm
    metric: Metric
    name: str
    expected: float


METRIC_TESTS = [
    MetricTest(model=SVM(), metric=Accuracy(), name="Accuracy", expected=0.925),
    MetricTest(model=SVM(), metric=RobustAccuracy(), name="Robust Accuracy", expected=0.921),
    MetricTest(model=SVM(), metric=F1(), name="F1", expected=0.930),
    MetricTest(model=SVM(), metric=BalancedAccuracy(), name="Balanced Accuracy", expected=0.923),
    MetricTest(
        model=SVM(), metric=NMI(base=DependencyTarget.s), name="NMI preds and s", expected=0.102
    ),
    MetricTest(model=SVM(), metric=CV(), name="CV", expected=0.630),
    MetricTest(model=SVM(), metric=AverageOddsDiff(), name="AverageOddsDiff", expected=0.105),
    MetricTest(model=SVM(), metric=Theil(), name="Theil_Index", expected=0.033),
    MetricTest(model=LR(), metric=Theil(), name="Theil_Index", expected=0.029),
    MetricTest(model=Reweighting(), metric=Theil(), name="Theil_Index", expected=0.030),
    MetricTest(model=SVM(), metric=Hsic(), name="HSIC", expected=0.020),
    MetricTest(model=SVM(), metric=AS(), name="anti_spurious", expected=0.852),
    MetricTest(
        model=SVM(), metric=NMI(base=DependencyTarget.y), name="NMI preds and y", expected=0.638
    ),
]


@pytest.mark.parametrize(("model", "metric", "name", "expected"), METRIC_TESTS, ids=get_id)
def test_get_score_of_predictions(
    toy_train_val: TrainValPair, model: InAlgorithm, name: str, metric: Metric, expected: float
) -> None:
    """Test get acc of predictions."""
    train, test = toy_train_val
    classifier: InAlgorithm = model
    predictions: Prediction = classifier.run(train, test)
    assert metric.name == name
    score = metric.score(predictions, test)
    assert score == approx(expected, abs=0.001)


class SimpleTest(NamedTuple):
    """Define a test for a metric."""

    metric: Metric
    expected: float


METRICS = [
    SimpleTest(Accuracy(), 0.5),
    SimpleTest(RobustAccuracy(), 0.33333),
    SimpleTest(F1(), 0.5),
    SimpleTest(BalancedAccuracy(), 0.5),
    SimpleTest(CV(), 0.33333),
    SimpleTest(Theil(), 0.346573),
    SimpleTest(Hsic(), 0.06144),
    SimpleTest(AS(), 0.0),
]


@pytest.mark.parametrize(("metric", "expected"), METRICS)
def test_on_label_tuple(metric: Metric, expected: float) -> None:
    preds = Prediction.from_np(np.array([0, 0, 1, 1]))
    targets = LabelTuple.from_np(s=np.array([0, 1, 1, 1]), y=np.array([0, 1, 0, 1]))
    score = metric.score(preds, targets)
    assert score == approx(expected, abs=0.001)
