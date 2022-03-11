"""For assessing ProbPos."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


@dataclass
class ProbPos(CfmMetric):
    """Probability of positive prediction."""

    _name: ClassVar[str] = "prob_pos"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.pos_class, labels=self.labels
        )

        return (t_pos + f_pos) / prediction.hard.size
