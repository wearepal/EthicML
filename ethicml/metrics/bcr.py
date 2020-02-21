"""For assessing Balanced Classification Rate (BCR)."""

from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction

from .metric import Metric
from .tnr import TNR
from .tpr import TPR


class BCR(Metric):
    """Balanced Classification Rate."""

    _name: str = "BCR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        tpr_metric = TPR()
        tpr = tpr_metric.score(prediction, actual)

        tnr_metric = TNR()
        tnr = tnr_metric.score(prediction, actual)

        return (tpr + tnr) / 2
