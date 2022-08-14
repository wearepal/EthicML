"""For assessing Average Odds Difference metric."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .fpr import FPR
from .metric import Metric
from .per_sensitive_attribute import diff_per_sens, metric_per_sens
from .tpr import TPR

__all__ = ["AverageOddsDiff"]


@dataclass
class AverageOddsDiff(CfmMetric):
    r"""Average Odds Difference.

    :math:`\tfrac{1}{2}\left[(FPR_{s=0} - FPR_{s=1}) + (TPR_{s=0} - TPR_{s=1}))\right]`.

    A value of 0 indicates equality of odds.
    """

    _name: ClassVar[str] = "AverageOddsDiff"
    apply_per_sensitive: ClassVar[bool] = False

    @implements(Metric)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        tpr = TPR(pos_class=self.pos_class, labels=self.labels)
        tpr_per_sens = metric_per_sens(prediction, actual, tpr)
        fpr = FPR(pos_class=self.pos_class, labels=self.labels)
        fpr_per_sens = metric_per_sens(prediction, actual, fpr)

        tpr_diff = diff_per_sens(tpr_per_sens)
        fpr_diff = diff_per_sens(fpr_per_sens)

        tpr_sum = 0.0
        fpr_sum = 0.0
        total = 0
        for ((_, tpr_v), (_, fpr_v)) in zip(tpr_diff.items(), fpr_diff.items()):
            total += 1
            tpr_sum += tpr_v
            fpr_sum += fpr_v

        return 0.5 * ((fpr_sum / total) + (tpr_sum / total))
