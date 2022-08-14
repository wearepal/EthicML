"""For assessing NPV."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["NPV"]


@dataclass
class NPV(CfmMetric):
    """Negative predictive value."""

    _name: ClassVar[str] = "NPV"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        t_neg, _, f_neg, _ = self._confusion_matrix(prediction=prediction, actual=actual)
        return t_neg / (t_neg + f_neg)
