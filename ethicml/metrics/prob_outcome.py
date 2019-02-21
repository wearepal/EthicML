"""
For assessing mean of logits
"""

import numpy as np

from .metric import Metric
from ..algorithms.utils import DataTuple


class ProbOutcome(Metric):
    """Mean of logits"""
    def score(self, prediction: np.array, actual: DataTuple) -> float:
        return prediction.sum() / prediction.size

    @property
    def name(self) -> str:
        return "prob_outcome"
