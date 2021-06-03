"""For assessing Theil Index metric: based on aif 360 implementation.

https://github.com/IBM/AIF360/blob/b5f4591c7c7374b9c88706827080a931209e7a24/aif360/metrics/classification_metric.py
in turn based on the paper https://arxiv.org/abs/1807.00787
"""

import numpy as np
from kit import implements

from ethicml.utility import DataTuple, Prediction

from .metric import Metric


class Theil(Metric):
    """Theil Index."""

    _name: str = "Theil_Index"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        y_true_df = actual.y
        act_col = y_true_df.columns[0]
        y_pos_label = y_true_df[act_col].max()

        y_pred = prediction.hard.to_numpy().ravel()
        y_true = y_true_df.to_numpy().ravel()
        y_pred = (y_pred == y_pos_label).astype(np.float64)
        y_true = (y_true == y_pos_label).astype(np.float64)
        var_b = 1 + y_pred - y_true

        # moving the b inside the log allows for 0 values
        return float(np.mean(np.log((var_b / np.mean(var_b)) ** var_b) / np.mean(var_b)))
