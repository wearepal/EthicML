"""For assessing Balanced Classification Rate (BCR)."""

from ethicml.common import implements
from ethicml.metrics.tnr import TNR
from ethicml.metrics.tpr import TPR
from ethicml.utility.data_structures import DataTuple, Prediction

from .metric import Metric


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
