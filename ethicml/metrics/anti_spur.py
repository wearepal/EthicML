"""Anti-spurious."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .metric import BaseMetric, Metric


@dataclass
class AS(BaseMetric):
    r"""Anti-spurious metric.

    Computes :math:`P(\hat{y}=y|y\neq s)`.
    """

    _name: ClassVar[str] = "anti_spurious"
    apply_per_sensitive: ClassVar[bool] = True

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        preds = prediction.hard.to_numpy()[:, np.newaxis]
        sens = actual.s.to_numpy()
        labels = actual.y.to_numpy()
        s_uneq_y = sens != labels
        return (preds[s_uneq_y] == labels[s_uneq_y]).mean()
