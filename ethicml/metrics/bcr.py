"""For assessing Balanced Classification Rate (BCR)."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .metric import CfmMetric, Metric
from .tnr import TNR
from .tpr import TPR


@dataclass
class BCR(CfmMetric):
    """Balanced Classification Rate."""

    _name: ClassVar[str] = "BCR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        tpr = TPR(pos_class=self.pos_class, labels=self.labels).score(prediction, actual)
        tnr = TNR(pos_class=self.pos_class, labels=self.labels).score(prediction, actual)

        return (tpr + tnr) / 2
