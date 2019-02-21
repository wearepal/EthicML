"""
Applies sci-kit learn's confusion matrix
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import confusion_matrix as conf_mtx

from ..algorithms.utils import DataTuple


def confusion_matrix(prediction: np.array, actual: DataTuple) -> Tuple[int, int, int, int]:
    """Apply sci-kit learn's confusion matrix"""
    actual_y = actual['y'].values.ravel()
    labels: np.array = np.unique(actual_y)
    if labels.size == 1:
        labels = [0, 1]
    conf_matr: np.ndarray = conf_mtx(y_true=actual_y, y_pred=prediction, labels=labels)
    results: Tuple[int, int, int, int] = conf_matr.ravel()

    return results
