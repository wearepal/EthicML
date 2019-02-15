"""
For assessing NPV
"""

from typing import Dict
import pandas as pd
import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric


class NPV(Metric):
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual)

        return t_neg/(t_neg+f_neg)

    @property
    def name(self) -> str:
        return "NPV"
