"""Anti-spurious."""

import pandas as pd

from ethicml.utility.data_structures import DataTuple
from .metric import Metric


class AS(Metric):
    """Anti-spurious metric. Computes :math:`P(\\hat{y}=y|y\\neq s)`."""

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        preds = prediction.to_numpy()
        sens = actual.s.to_numpy()
        labels = actual.y.to_numpy()
        s_uneq_y = sens != labels
        return (preds[s_uneq_y] == labels[s_uneq_y]).mean()

    @property
    def name(self) -> str:
        return "anti_spurious"
