"""
For assessing mean of logits
"""

from typing import Dict
import pandas as pd
import numpy as np

from .confusion_matrix import confusion_matrix
from .metric import Metric


class ProbOutcome(Metric):
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        return prediction.sum()/prediction.size

    @property
    def name(self) -> str:
        return "prob_outcome"
