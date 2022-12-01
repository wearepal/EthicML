"""For assessing TNR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["TNR"]


@dataclass
class TNR(CfmMetric):
    """True negative rate."""

    _name: ClassVar[str] = "TNR"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        t_neg, f_pos, _, _ = self._confusion_matrix(prediction=prediction, actual=actual)
        return t_neg / (t_neg + f_pos)
