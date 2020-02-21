"""For assessing mean of logits."""

from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction, SoftPrediction

from .metric import Metric


class ProbOutcome(Metric):
    """Mean of logits."""

    _name: str = "prob_outcome"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        if not isinstance(prediction, SoftPrediction):
            return float("nan")  # this metric only makes sense with probs
        return prediction.soft.to_numpy().sum() / prediction.hard.size
