"""For assessing FNR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["FNR"]


@dataclass
class FNR(CfmMetric):
    """False negative rate."""

    _name: ClassVar[str] = "FNR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, _, f_neg, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return f_neg / (f_neg + t_pos)
