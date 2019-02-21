"""
For assessing TNR
"""

from typing import Dict
import pandas as pd
import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric


class TNR(Metric):
    """True negative rate"""
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        t_neg, f_pos, _, _ = confusion_matrix(prediction, actual)

        return t_neg / (t_neg + f_pos)

    @property
    def name(self) -> str:
        return "TNR"
