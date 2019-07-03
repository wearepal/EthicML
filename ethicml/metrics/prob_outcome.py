"""
For assessing mean of logits
"""

from ethicml.utility.data_structures import DataTuple, Predictions
from .metric import Metric


class ProbOutcome(Metric):
    """Mean of logits"""

    def score(self, prediction: Predictions, actual: DataTuple) -> float:
        return prediction.soft.values.sum() / prediction.soft.size

    @property
    def name(self) -> str:
        return "prob_outcome"
