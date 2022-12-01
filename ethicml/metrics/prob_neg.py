"""For assessing ProbNeg."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["ProbNeg"]


@dataclass
class ProbNeg(CfmMetric):
    """Probability of negative prediction."""

    _name: ClassVar[str] = "prob_neg"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        t_neg, _, f_neg, _ = self._confusion_matrix(prediction=prediction, actual=actual)
        return (t_neg + f_neg) / prediction.hard.size
