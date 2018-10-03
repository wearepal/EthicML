"""
Abstract class that defines methods related to metrics.
Metrics can be applied to whole or partial datasets
"""

from typing import Dict

import pandas as pd
import numpy as np

from ethicml.metrics.metric import Metric


class MeanScore(Metric):
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        return prediction.mean()

    def get_name(self) -> str:
        return "Mean Score"
