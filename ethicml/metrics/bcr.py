"""For assessing Balanced Classification Rate (BCR)."""

import pandas as pd

from ethicml.metrics.tnr import TNR
from ethicml.metrics.tpr import TPR
from ethicml.utility.data_structures import DataTuple
from .metric import Metric


class BCR(Metric):
    """Balanced Classification Rate."""

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        """Get the score for this metric."""
        tpr_metric = TPR()
        tpr = tpr_metric.score(prediction, actual)

        tnr_metric = TNR()
        tnr = tnr_metric.score(prediction, actual)

        return (tpr + tnr) / 2

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "BCR"
