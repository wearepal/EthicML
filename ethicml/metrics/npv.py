"""
For assessing NPV
"""

from ethicml.utility.data_structures import DataTuple, Predictions
from .confusion_matrix import confusion_matrix
from .metric import Metric


class NPV(Metric):
    """Negative predictive value"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual, self.positive_class)

        return t_neg / (t_neg + f_neg)

    @property
    def name(self) -> str:
        return "NPV"
