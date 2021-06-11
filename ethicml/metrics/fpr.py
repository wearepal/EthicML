"""For assessing FPR."""

from kit import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class FPR(Metric):
    """False positive rate."""

    _name: str = "FPR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, f_pos, _, _ = confusion_matrix(prediction, actual, self.positive_class)

        return f_pos / (f_pos + t_neg)
