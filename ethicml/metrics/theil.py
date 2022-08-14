"""For assessing Theil Index metric: based on aif 360 implementation.

https://github.com/IBM/AIF360/blob/b5f4591c7c7374b9c88706827080a931209e7a24/aif360/metrics/classification_metric.py
in turn based on the paper https://arxiv.org/abs/1807.00787
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .metric import MetricStaticName

__all__ = ["Theil"]


@dataclass
class Theil(MetricStaticName):
    """Theil Index."""

    _name: ClassVar[str] = "Theil_Index"

    @implements(MetricStaticName)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        y_true_df = actual.y
        y_pos_label = y_true_df.max()

        y_pred = prediction.hard.to_numpy().ravel()
        y_true = y_true_df.to_numpy().ravel()
        y_pred = (y_pred == y_pos_label).astype(np.float64)
        y_true = (y_true == y_pos_label).astype(np.float64)
        var_b = 1 + y_pred - y_true

        # moving the b inside the log allows for 0 values
        return float(np.mean(np.log((var_b / np.mean(var_b)) ** var_b) / np.mean(var_b)))
