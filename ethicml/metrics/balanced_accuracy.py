"""Accuracy that is balanced with respect to the class labels."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["BalancedAccuracy"]


@dataclass
class BalancedAccuracy(CfmMetric):
    """Accuracy that is balanced with respect to the class labels."""

    _name: ClassVar[str] = "Balanced Accuracy"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        t_neg, f_pos, f_neg, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        tpr = t_pos / (t_pos + f_neg)
        tnr = t_neg / (t_neg + f_pos)
        return 0.5 * (tpr + tnr)
