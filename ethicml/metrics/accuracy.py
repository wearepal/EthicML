"""
For assessing accuracy
"""

from typing import Dict
from sklearn.metrics import accuracy_score
import pandas as pd

from .metric import Metric


class Accuracy(Metric):
    """Accruacy"""
    def score(self, prediction: pd.DataFrame, actual: Dict[str, pd.DataFrame]) -> float:
        actual_y = actual['y'].values.ravel()
        return accuracy_score(actual_y, prediction)

    @property
    def name(self) -> str:
        return "Accuracy"
