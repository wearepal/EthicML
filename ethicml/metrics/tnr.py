"""
For assessing TNR
"""

from ethicml.utility.data_structures import DataTuple, Predictions
from .confusion_matrix import confusion_matrix
from .metric import Metric


class TNR(Metric):
    """True negative rate"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        t_neg, f_pos, _, _ = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return t_neg / (t_neg + f_pos)

    @property
    def name(self) -> str:
        return "TNR"
