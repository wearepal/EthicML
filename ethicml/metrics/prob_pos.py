"""For assessing ProbPos."""

from kit import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbPos(Metric):
    """Probability of positive prediction."""

    _name: str = "prob_pos"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return (t_pos + f_pos) / prediction.hard.size
