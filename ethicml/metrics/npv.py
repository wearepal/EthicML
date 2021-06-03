"""For assessing NPV."""

from kit import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class NPV(Metric):
    """Negative predictive value."""

    _name: str = "NPV"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual, self.positive_class)

        return t_neg / (t_neg + f_neg)
