"""For assessing mean of logits."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction, SoftPrediction

from .metric import MetricStaticName

__all__ = ["ProbOutcome"]


@dataclass
class ProbOutcome(MetricStaticName):
    """Mean of logits."""

    _name: ClassVar[str] = "prob_outcome"
    pos_class: int = 1

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        return (
            (prediction.soft.sum(axis=0)[self.pos_class] / prediction.hard.size).item()
            if isinstance(prediction, SoftPrediction)
            else float("nan")
        )
