"""
Applies sci-kit learn's confusion matrix
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as conf_mtx

from ..algorithms.utils import DataTuple


class LabelOutOfBounds(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead"""


def confusion_matrix(prediction: pd.DataFrame,
                     actual: DataTuple,
                     pos_cls: int) -> Tuple[int, int, int, int]:
    """Apply sci-kit learn's confusion matrix"""
    actual_y = actual.y.values
    labels: np.ndarray = np.unique(actual_y)
    if labels.size == 1:
        labels = np.array([0, 1])
    conf_matr: np.ndarray = conf_mtx(y_true=actual_y, y_pred=prediction, labels=labels)

    if not pos_cls in labels:
        raise LabelOutOfBounds("Positive class specified must exist in the test set")

    tp_idx = np.where(labels == pos_cls)

    true_pos = int(conf_matr[tp_idx, tp_idx])
    false_pos = np.sum(conf_matr[:, tp_idx])-true_pos
    false_neg = np.sum(conf_matr[tp_idx, :].flatten())-true_pos
    true_neg = np.sum(conf_matr)-true_pos-false_pos-false_neg
    return true_neg, false_pos, false_neg, true_pos
