"""
For assessing Balnced Classification Rate (BCR)
"""

from typing import Dict
import pandas as pd
import numpy as np

from ethicml.metrics.tnr import TNR
from ethicml.metrics.tpr import TPR
from .metric import Metric


class BCR(Metric):
    """Balanced Classification Rate"""
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        tpr_metric = TPR()
        tpr = tpr_metric.score(prediction, actual)

        tnr_metric = TNR()
        tnr = tnr_metric.score(prediction, actual)

        return (tpr + tnr) / 2

    @property
    def name(self) -> str:
        return "BCR"
