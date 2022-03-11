"""For assessing TPR."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


@dataclass
class TPR(CfmMetric):
    """True positive rate."""

    _name: ClassVar[str] = "TPR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, _, f_neg, t_pos = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.pos_class, labels=self.labels
        )

        return t_pos / (t_pos + f_neg)
