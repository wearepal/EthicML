"""Applies sci-kit learn's confusion matrix."""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix as conf_mtx

from ethicml.utility import DataTuple, Prediction

__all__ = ["LabelOutOfBounds", "confusion_matrix"]


class LabelOutOfBounds(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead."""


def confusion_matrix(
    prediction: Prediction, actual: DataTuple, pos_cls: int, labels: Optional[List[int]] = None
) -> Tuple[int, int, int, int]:
    """Apply sci-kit learn's confusion matrix."""
    actual_y: np.ndarray = actual.y.to_numpy(dtype=np.int32)  # type: ignore[arg-type]
    _labels: np.ndarray = np.unique(actual_y) if labels is None else np.array(labels)
    if _labels.size == 1:
        _labels = np.array([0, 1], dtype=np.int32)
    conf_matr: np.ndarray = conf_mtx(y_true=actual_y, y_pred=prediction.hard, labels=_labels)

    if pos_cls not in _labels:
        raise LabelOutOfBounds("Positive class specified must exist in the test set")

    tp_idx: np.int64 = (_labels == pos_cls).nonzero()[0].item()
    tp_idx = int(tp_idx)

    true_pos = conf_matr[tp_idx, tp_idx]
    false_pos = conf_matr[:, tp_idx].sum() - true_pos
    false_neg = conf_matr[tp_idx, :].sum() - true_pos
    true_neg = conf_matr.sum() - true_pos - false_pos - false_neg
    return true_neg, false_pos, false_neg, true_pos
