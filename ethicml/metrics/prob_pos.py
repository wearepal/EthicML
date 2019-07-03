"""
For assessing ProbPos
"""
import pandas as pd

from ethicml.utility.data_structures import DataTuple, Predictions
from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbPos(Metric):
    """Probability of positive prediction"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        assert isinstance(prediction.hard, pd.DataFrame)
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual, pos_cls=self.positive_class)

        return (t_pos + f_pos) / prediction.hard.size

    @property
    def name(self) -> str:
        return "prob_pos"
