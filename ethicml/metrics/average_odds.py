"""For assessing Average Odds Difference metric."""


from kit import implements

from ethicml.utility import DataTuple, Prediction

from .metric import Metric


class AverageOddsDiff(Metric):
    r"""Average Odds Difference.

    :math:`\tfrac{1}{2}\left[(FPR_{s=0} - FPR_{s=1}) + (TPR_{s=0} - TPR_{s=1}))\right]`.

    A value of 0 indicates equality of odds.
    """

    _name: str = "AverageOddsDiff"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        # has to be imported on demand because otherwise we get circular imports
        from ethicml import FPR, TPR
        from ethicml.evaluators.per_sensitive_attribute import (
            diff_per_sensitive_attribute,
            metric_per_sensitive_attribute,
        )

        tpr_per_sens = metric_per_sensitive_attribute(prediction, actual, TPR())
        fpr_per_sens = metric_per_sensitive_attribute(prediction, actual, FPR())

        tpr_diff = diff_per_sensitive_attribute(tpr_per_sens)
        fpr_diff = diff_per_sensitive_attribute(fpr_per_sens)

        tpr_sum = 0
        fpr_sum = 0
        total = 0
        for ((tpr_k, tpr_v), (fpr_k, fpr_v)) in zip(tpr_diff.items(), fpr_diff.items()):
            total += 1
            tpr_sum += tpr_v
            fpr_sum += fpr_v

        return 0.5 * ((fpr_sum / total) + (tpr_sum / total))

    @property
    def apply_per_sensitive(self) -> bool:
        """Can this metric be applied per sensitive attribute group?"""
        return False
