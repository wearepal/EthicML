"""
For assessing Balanced Classification Rate (BCR)
"""

from ethicml.metrics.tnr import TNR
from ethicml.metrics.tpr import TPR
from ethicml.utility.data_structures import DataTuple, Predictions
from .metric import Metric


class BCR(Metric):
    """Balanced Classification Rate"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        tpr_metric = TPR()
        tpr = tpr_metric.score(prediction, actual)

        tnr_metric = TNR()
        tnr = tnr_metric.score(prediction, actual)

        return (tpr + tnr) / 2

    @property
    def name(self) -> str:
        return "BCR"
