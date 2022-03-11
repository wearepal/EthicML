"""For assessing ProbNeg."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


@dataclass
class ProbNeg(CfmMetric):
    """Probability of negative prediction."""

    _name: ClassVar[str] = "prob_neg"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.pos_class, labels=self.labels
        )

        return (t_neg + f_neg) / prediction.hard.size
