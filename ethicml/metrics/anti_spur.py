"""Anti-spurious."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .metric import Metric, MetricStaticName

__all__ = ["AS"]


@dataclass
class AS(MetricStaticName):
    r"""Anti-spurious metric.

    Computes :math:`P(\hat{y}=y|y\neq s)`.
    """

    _name: ClassVar[str] = "anti_spurious"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        preds = prediction.hard.to_numpy()
        sens = actual.s.to_numpy()
        labels = actual.y.to_numpy()
        s_uneq_y = sens != labels
        return (preds[s_uneq_y] == labels[s_uneq_y]).mean()
