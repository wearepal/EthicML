"""
For assessing TPR
"""

from typing import Dict
import pandas as pd
import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric


class TPR(Metric):
    """True positive rate"""
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        _, _, f_neg, t_pos = confusion_matrix(prediction, actual)

        return t_pos / (t_pos + f_neg)

    @property
    def name(self) -> str:
        return "TPR"
