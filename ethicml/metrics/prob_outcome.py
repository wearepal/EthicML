"""For assessing mean of logits."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction, SoftPrediction

from .metric import Metric, MetricStaticName

__all__ = ["ProbOutcome"]


@dataclass
class ProbOutcome(MetricStaticName):
    """Mean of logits."""

    _name: ClassVar[str] = "prob_outcome"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        if not isinstance(prediction, SoftPrediction):
            return float("nan")  # this metric only makes sense with probs
        return (prediction.soft.to_numpy().sum() / prediction.hard.size).item()
