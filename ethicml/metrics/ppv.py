"""For assessing PPV."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
from typing_extensions import override

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric

__all__ = ["PPV"]


@dataclass
class PPV(CfmMetric):
    """Positive predictive value."""

    _name: ClassVar[str] = "PPV"

    @override
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, f_pos, _, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return t_pos / (t_pos + f_pos)
