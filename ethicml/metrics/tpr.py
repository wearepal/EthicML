"""
For assessing TPR
"""

import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric
from ..algorithms.utils import DataTuple


class TPR(Metric):
    """True positive rate"""
    def score(self, prediction: np.array, actual: DataTuple) -> float:
        _, _, f_neg, t_pos = confusion_matrix(prediction, actual)

        return t_pos / (t_pos + f_neg)

    @property
    def name(self) -> str:
        return "TPR"
