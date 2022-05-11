"""For assessing ProbPos."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["ProbPos"]


@dataclass
class ProbPos(CfmMetric):
    """Probability of positive prediction."""

    _name: ClassVar[str] = "prob_pos"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = self.confusion_matrix(prediction=prediction, actual=actual)
        return (t_pos + f_pos) / prediction.hard.size
