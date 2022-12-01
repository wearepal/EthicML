"""For assessing Balanced Classification Rate (BCR)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .tnr import TNR
from .tpr import TPR

__all__ = ["BCR"]


@dataclass
class BCR(CfmMetric):
    """Balanced Classification Rate."""

    _name: ClassVar[str] = "BCR"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        tpr = TPR(pos_class=self.pos_class, labels=self.labels).score(prediction, actual)
        tnr = TNR(pos_class=self.pos_class, labels=self.labels).score(prediction, actual)

        return (tpr + tnr) / 2
