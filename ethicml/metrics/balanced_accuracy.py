"""Accuracy that is balanced with respect to the class labels."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


@dataclass
class BalancedAccuracy(CfmMetric):
    """Accuracy that is balanced with respect to the class labels."""

    _name: ClassVar[str] = "Balanced Accuracy"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, f_pos, f_neg, t_pos = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.pos_class, labels=self.labels
        )
        tpr = t_pos / (t_pos + f_neg)
        tnr = t_neg / (t_neg + f_pos)
        return 0.5 * (tpr + tnr)
