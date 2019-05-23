"""
For assessing Nomralized Mutual Information
"""

import pandas as pd

from sklearn.metrics import normalized_mutual_info_score as nmis

from .metric import Metric
from ..algorithms.utils import DataTuple


class NMI(Metric):
    """Normalized Mutual Information"""
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        return nmis(actual.y.values.flatten(),
                    prediction.values.flatten(),
                    average_method='geometric')

    @property
    def name(self) -> str:
        return "NMI"


class NMIinS(Metric):
    """Normalized Mutual Information"""
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        return nmis(actual.s.values.flatten(),
                    prediction.values.flatten(),
                    average_method='geometric')

    @property
    def name(self) -> str:
        return "NMIinS"
