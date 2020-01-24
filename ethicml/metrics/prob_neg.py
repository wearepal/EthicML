"""For assessing ProbNeg."""

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, Prediction
from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbNeg(Metric):
    """Probability of negative prediction."""

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return (t_neg + f_neg) / prediction.hard.size

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "prob_neg"
