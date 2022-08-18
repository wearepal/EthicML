"""For assessing ProbPos."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["ProbPos"]


@dataclass
class ProbPos(CfmMetric):
    """Probability of positive prediction."""

    _name: ClassVar[str] = "prob_pos"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, f_pos, _, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return (t_pos + f_pos) / prediction.hard.size
