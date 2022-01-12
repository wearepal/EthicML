"""For assessing FNR."""

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .confusion_matrix import confusion_matrix
from .metric import CfmMetric, Metric


class FNR(CfmMetric):
    """False negative rate."""

    _name: str = "FNR"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        _, _, f_neg, t_pos = confusion_matrix(
            prediction=prediction, actual=actual, pos_cls=self.positive_class, labels=self.labels
        )

        return f_neg / (f_neg + t_pos)
