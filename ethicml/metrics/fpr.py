"""For assessing FPR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["FPR"]


@dataclass
class FPR(CfmMetric):
    """False positive rate."""

    _name: ClassVar[str] = "FPR"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        t_neg, f_pos, _, _ = self._confusion_matrix(prediction=prediction, actual=actual)
        return f_pos / (f_pos + t_neg)
