"""For assessing TPR."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["TPR"]


@dataclass
class TPR(CfmMetric):
    """True positive rate."""

    _name: ClassVar[str] = "TPR"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, _, f_neg, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return t_pos / (t_pos + f_neg)
