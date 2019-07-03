"""
For assessing ProbNeg
"""
import pandas as pd

from ethicml.utility.data_structures import DataTuple, Predictions
from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbNeg(Metric):
    """Probability of negative prediction"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        assert isinstance(prediction.hard, pd.DataFrame)
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return (t_neg + f_neg) / prediction.hard.size

    @property
    def name(self) -> str:
        return "prob_neg"
