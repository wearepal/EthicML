"""
For assessing ProbNeg
"""

from typing import Dict
import pandas as pd
import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbNeg(Metric):
    """Probability of negative prediction"""
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        t_neg, _, f_neg, _ = confusion_matrix(prediction, actual)

        return (t_neg + f_neg) / prediction.size

    @property
    def name(self) -> str:
        return "prob_neg"
