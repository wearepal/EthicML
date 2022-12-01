"""For assessing FNR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["FNR"]


@dataclass
class FNR(CfmMetric):
    """False negative rate."""

    _name: ClassVar[str] = "FNR"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, _, f_neg, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return f_neg / (f_neg + t_pos)
