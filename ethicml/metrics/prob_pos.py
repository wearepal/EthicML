"""
For assessing TPR
"""

from typing import Dict
import pandas as pd
import numpy as np

from ethicml.metrics.confusion_matrix import confusion_matrix
from ethicml.metrics.metric import Metric


class ProbPos(Metric):
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        _, f_pos, _, t_pos = confusion_matrix(prediction, actual)

        return (t_pos+f_pos)/prediction.size

    @property
    def name(self) -> str:
        return "prob_pos"
