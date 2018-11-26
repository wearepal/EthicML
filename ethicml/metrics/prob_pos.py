"""
For assessing TPR
"""

from typing import Dict, Tuple

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from ethicml.metrics.metric import Metric


class ProbPos(Metric):
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        actual_y = actual['y'].values.ravel()
        labels: np.array = np.array([-1,1])#np.unique(actual_y)
        conf_matr: np.ndarray = confusion_matrix(y_true=actual_y,
                                                 y_pred=prediction,
                                                 labels=labels)
        results: Tuple[int, int, int, int] = conf_matr.ravel()
        t_neg, f_pos, f_neg, t_pos = results

        return (t_pos+f_pos)/prediction.size

    def get_name(self) -> str:
        return "prob_pos"
