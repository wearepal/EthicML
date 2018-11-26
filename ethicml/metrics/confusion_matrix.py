"""
Applies sci-kit learn's confusion matrix
"""

from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as conf_mtx


def confusion_matrix(prediction: np.array, actual: Dict[str, pd.DataFrame]) -> Tuple[int, int, int, int]:
    actual_y = actual['y'].values.ravel()
    labels: np.array = np.unique(actual_y)
    conf_matr: np.ndarray = conf_mtx(y_true=actual_y, y_pred=prediction, labels=labels)
    results: Tuple[int, int, int, int] = conf_matr.ravel()

    return results
