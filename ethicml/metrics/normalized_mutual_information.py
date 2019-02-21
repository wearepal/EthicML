"""
For assessing Nomralized Mutual Information
"""

from typing import Dict
import pandas as pd

from sklearn.metrics import normalized_mutual_info_score as nmis

from .metric import Metric


class NMI(Metric):
    """Normalized Mutual Information"""
    def score(self, prediction: pd.DataFrame, actual: Dict[str, pd.DataFrame]) -> float:
        return nmis(actual['y'].values.flatten(), prediction.values.flatten())

    @property
    def name(self) -> str:
        return "NMI"
