"""For assessing FNR."""

from kit import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class FNR(Metric):
    """False negative rate."""

    _name: str = "FNR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, _, f_neg, t_pos = confusion_matrix(prediction, actual, self.positive_class)

        return f_neg / (f_neg + t_pos)
