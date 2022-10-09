"""Applies sci-kit learn's confusion matrix."""
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.metrics import confusion_matrix as conf_mtx

from ethicml.metrics.metric import MetricStaticName
from ethicml.utility.data_structures import EvalTuple, Prediction

__all__ = ["CfmMetric", "LabelOutOfBounds"]


@dataclass
class CfmMetric(MetricStaticName, ABC):
    """Confusion Matrix based metric."""

    pos_class: int = 1
    """The class to treat as being "positive"."""
    labels: Optional[List[int]] = None
    """List of possible target values. If `None`, then this is inferred from the data when run."""

    def _confusion_matrix(
        self, prediction: Prediction, actual: EvalTuple
    ) -> tuple[int, int, int, int]:
        """Apply sci-kit learn's confusion matrix.

        :param prediction: The predictions.
        :param actual: The actual labels.
        :returns: The 4 entries of the confusion matrix as a 4-tuple.
        :raises LabelOutOfBounds: If the class labels are not as expected.
        """
        actual_y: np.ndarray = actual.y.to_numpy(dtype=np.int32)
        _labels: np.ndarray = np.unique(actual_y) if self.labels is None else np.array(self.labels)
        if _labels.size == 1:
            _labels = np.array([0, 1], dtype=np.int32)
        conf_matr: np.ndarray = conf_mtx(y_true=actual_y, y_pred=prediction.hard, labels=_labels)

        if self.pos_class not in _labels:
            raise LabelOutOfBounds("Positive class specified must exist in the test set")

        tp_idx: np.int64 = (_labels == self.pos_class).nonzero()[0].item()
        tp_idx = int(tp_idx)

        true_pos = conf_matr[tp_idx, tp_idx]
        false_pos = conf_matr[:, tp_idx].sum() - true_pos
        false_neg = conf_matr[tp_idx, :].sum() - true_pos
        true_neg = conf_matr.sum() - true_pos - false_pos - false_neg
        return true_neg, false_pos, false_neg, true_pos


class LabelOutOfBounds(Exception):
    """Metric Not Applicable per sensitive attribute, apply to whole dataset instead."""
