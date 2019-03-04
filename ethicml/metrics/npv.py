"""
For assessing NPV
"""

import pandas as pd

from .confusion_matrix import confusion_matrix
from .metric import Metric
from ..algorithms.utils import DataTuple


class NPV(Metric):
    """Negative predictive value"""
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual)

        return t_neg / (t_neg + f_neg)

    @property
    def name(self) -> str:
        return "NPV"
