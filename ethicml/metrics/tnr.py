"""For assessing TNR."""

from kit import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class TNR(Metric):
    """True negative rate."""

    _name: str = "TNR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, f_pos, _, _ = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return t_neg / (t_neg + f_pos)
