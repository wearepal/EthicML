"""
For assessing accuracy
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from .metric import Metric
from ..algorithms.utils import DataTuple


class Accuracy(Metric):
    """Accruacy"""
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        actual_y = actual.y.values
        return accuracy_score(actual_y, prediction)

    @property
    def name(self) -> str:
        return "Accuracy"
