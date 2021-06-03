"""Accuracy and related metrics."""
from typing import Callable, Optional

import pandas as pd
from kit import implements
from sklearn.metrics import accuracy_score, f1_score

from ethicml.utility import DataTuple, Prediction

from .metric import Metric

__all__ = ["Accuracy", "F1", "SklearnMetric"]


class SklearnMetric(Metric):
    """Wrapper around an sklearn metric."""

    def __init__(
        self,
        sklearn_metric: Callable[[pd.DataFrame, pd.Series], float],
        name: str,
        pos_class: Optional[int] = None,
    ):
        if pos_class is not None:
            super().__init__(pos_class=pos_class)
        else:
            super().__init__()
        self._metric = sklearn_metric
        self._name = name

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        return self._metric(actual.y, prediction.hard)


class Accuracy(SklearnMetric):
    """Classification accuracy."""

    def __init__(self, pos_class: Optional[int] = None):
        super().__init__(accuracy_score, "Accuracy", pos_class=pos_class)


class F1(SklearnMetric):
    """F1 score: harmonic mean of precision and recall."""

    def __init__(self, pos_class: Optional[int] = None):
        super().__init__(f1_score, "F1", pos_class=pos_class)
