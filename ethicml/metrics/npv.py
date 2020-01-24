"""For assessing NPV."""

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, Prediction
from .confusion_matrix import confusion_matrix
from .metric import Metric


class NPV(Metric):
    """Negative predictive value."""

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual, self.positive_class)

        return t_neg / (t_neg + f_neg)

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "NPV"
