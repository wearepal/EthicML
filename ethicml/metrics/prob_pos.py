"""For assessing ProbPos."""

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, Prediction
from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbPos(Metric):
    """Probability of positive prediction."""

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return (t_pos + f_pos) / prediction.hard.size

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "prob_pos"
