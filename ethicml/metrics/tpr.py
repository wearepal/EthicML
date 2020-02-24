"""For assessing TPR."""

from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class TPR(Metric):
    """True positive rate."""

    _name: str = "TPR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, _, f_neg, t_pos = confusion_matrix(prediction, actual, self.positive_class)

        return t_pos / (t_pos + f_neg)
