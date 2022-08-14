"""For assessing PPV."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import Metric

__all__ = ["PPV"]


@dataclass
class PPV(CfmMetric):
    """Positive predictive value."""

    _name: ClassVar[str] = "PPV"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        _, f_pos, _, t_pos = self._confusion_matrix(prediction=prediction, actual=actual)
        return t_pos / (t_pos + f_pos)
