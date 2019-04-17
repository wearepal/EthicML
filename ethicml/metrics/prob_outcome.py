"""
For assessing mean of logits
"""

import pandas as pd

from .metric import Metric
from ..algorithms.utils import DataTuple


class ProbOutcome(Metric):
    """Mean of logits"""
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        return prediction.values.sum() / prediction.size

    @property
    def name(self) -> str:
        return "prob_outcome"
