"""For assessing TPR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["TPR"]


@dataclass
class TPR(CfmMetric):
    """True positive rate."""

    _name: ClassVar[str] = "TPR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, _, f_neg, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return t_pos / (t_pos + f_neg)
