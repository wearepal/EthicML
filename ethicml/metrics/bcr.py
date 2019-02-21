"""
For assessing Balnced Classification Rate (BCR)
"""

import numpy as np

from ethicml.metrics.tnr import TNR
from ethicml.metrics.tpr import TPR
from .metric import Metric
from ..algorithms.utils import DataTuple


class BCR(Metric):
    """Balanced Classification Rate"""
    def score(self, prediction: np.array, actual: DataTuple) -> float:
        tpr_metric = TPR()
        tpr = tpr_metric.score(prediction, actual)

        tnr_metric = TNR()
        tnr = tnr_metric.score(prediction, actual)

        return (tpr + tnr) / 2

    @property
    def name(self) -> str:
        return "BCR"
