"""For assessing FPR."""

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


class FPR(CfmMetric):
    """False positive rate."""

    _name: str = "FPR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        t_neg, f_pos, _, _ = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.positive_class, labels=self.labels
        )

        return f_pos / (f_pos + t_neg)
