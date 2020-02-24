"""Anti-spurious."""

import numpy as np

from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction

from .metric import Metric


class AS(Metric):
    r"""Anti-spurious metric.

    Computes :math:`P(\hat{y}=y|y\neq s)`.
    """

    _name: str = "anti_spurious"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        preds = prediction.hard.to_numpy()[:, np.newaxis]
        sens = actual.s.to_numpy()
        labels = actual.y.to_numpy()
        s_uneq_y = sens != labels
        return (preds[s_uneq_y] == labels[s_uneq_y]).mean()
