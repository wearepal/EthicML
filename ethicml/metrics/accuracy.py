"""
For assessing accuracy
"""

from typing import Dict
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

from ethicml.metrics.metric import Metric


class Accuracy(Metric):
    def score(self,
              prediction: np.array,
              actual: Dict[str, pd.DataFrame]) -> float:

        actual_y = actual['y'].values.ravel()

        return accuracy_score(actual_y, prediction)

    def get_name(self) -> str:
        return "Accuracy"
