"""For assessing ProbNeg."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["ProbNeg"]


@dataclass
class ProbNeg(CfmMetric):
    """Probability of negative prediction."""

    _name: ClassVar[str] = "prob_neg"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        t_neg, _, f_neg, _ = self._confusion_matrix(prediction=prediction, actual=actual)
        return (t_neg + f_neg) / prediction.hard.size
