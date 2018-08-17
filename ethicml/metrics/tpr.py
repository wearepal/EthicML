"""
For assessing TPR
"""

from typing import Dict, Tuple

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from ethicml.metrics.metric import Metric


class TPR(Metric):
    def score(self,
              prediction: np.array,
              actual: Dict[str, pd.DataFrame]) -> float:

        actual_y = actual['y'].values.ravel()

        labels: np.array = np.unique(actual_y)

        conf_matr: np.ndarray = confusion_matrix(y_true=actual_y,
                                                 y_pred=prediction,
                                                 labels=labels)
        results: Tuple[int, int, int, int] = conf_matr.ravel()
        _, _, f_neg, t_pos = results

        return t_pos/(t_pos+f_neg)

    def get_name(self) -> str:
        return "TPR"
