"""
For assessing accuracy
"""
from sklearn.metrics import accuracy_score

from ethicml.utility.data_structures import DataTuple, Predictions
from .metric import Metric


class Accuracy(Metric):
    """Accruacy"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        actual_y = actual.y.values
        return accuracy_score(actual_y, prediction.hard)

    @property
    def name(self) -> str:
        return "Accuracy"
