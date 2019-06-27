"""
Applies sci-kit learn's confusion matrix
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as conf_mtx

from ethicml.utility.data_structures import DataTuple


class LabelOutOfBounds(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead"""


def confusion_matrix(
    prediction: pd.DataFrame, actual: DataTuple, pos_cls: int
) -> Tuple[int, int, int, int]:
    """Apply sci-kit learn's confusion matrix"""
    actual_y: np.ndarray[int] = actual.y.values
    labels: np.ndarray[int] = np.unique(actual_y)
    if labels.size == 1:
        labels = np.array([0, 1])
    conf_matr: np.ndarray = conf_mtx(y_true=actual_y, y_pred=prediction, labels=labels)

    if not pos_cls in labels:
        raise LabelOutOfBounds("Positive class specified must exist in the test set")

    tp_idx: int = (labels == pos_cls).nonzero()[0].item()

    true_pos = conf_matr[tp_idx, tp_idx]
    false_pos = conf_matr[:, tp_idx].sum() - true_pos
    false_neg = conf_matr[tp_idx, :].sum() - true_pos
    true_neg = conf_matr.sum() - true_pos - false_pos - false_neg
    return true_neg, false_pos, false_neg, true_pos
