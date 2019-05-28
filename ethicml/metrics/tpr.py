"""
For assessing TPR
"""

import pandas as pd

from .confusion_matrix import confusion_matrix
from .metric import Metric
from ..algorithms.utils import DataTuple


class TPR(Metric):
    """True positive rate"""

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        _, _, f_neg, t_pos = confusion_matrix(prediction, actual, self.positive_class)

        return t_pos / (t_pos + f_neg)

    @property
    def name(self) -> str:
        return "TPR"
