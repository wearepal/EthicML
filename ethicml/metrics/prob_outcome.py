"""
For assessing mean of logits
"""

import pandas as pd

from ethicml.utility.data_structures import DataTuple
from .metric import Metric


class ProbOutcome(Metric):
    """Mean of logits"""

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        return prediction.values.sum() / prediction.size

    @property
    def name(self) -> str:
        return "prob_outcome"
