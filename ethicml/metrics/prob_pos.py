"""
For assessing ProbPos
"""

import pandas as pd

from .confusion_matrix import confusion_matrix
from .metric import Metric
from ..algorithms.utils import DataTuple


class ProbPos(Metric):
    """Probability of positive prediction"""
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual)

        return (t_pos + f_pos) / prediction.size

    @property
    def name(self) -> str:
        return "prob_pos"
