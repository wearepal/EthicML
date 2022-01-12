"""For assessing ProbNeg."""

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


class ProbNeg(CfmMetric):
    """Probability of negative prediction."""

    _name: str = "prob_neg"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.positive_class, labels=self.labels
        )

        return (t_neg + f_neg) / prediction.hard.size
