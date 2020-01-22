"""For assessing mean of logits."""

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, Prediction
from .metric import Metric


class ProbOutcome(Metric):
    """Mean of logits."""

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        return prediction.hard.to_numpy().sum() / prediction.hard.size

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "prob_outcome"
