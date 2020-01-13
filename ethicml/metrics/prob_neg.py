"""For assessing ProbNeg."""

import pandas as pd

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple
from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbNeg(Metric):
    """Probability of negative prediction."""

    @implements(Metric)
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return (t_neg + f_neg) / prediction.size

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "prob_neg"
