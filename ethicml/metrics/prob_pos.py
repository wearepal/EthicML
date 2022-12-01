"""For assessing ProbPos."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["ProbPos"]


@dataclass
class ProbPos(CfmMetric):
    """Probability of positive prediction."""

    _name: ClassVar[str] = "prob_pos"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, f_pos, _, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return (t_pos + f_pos) / prediction.hard.size
