"""For assessing Calder-Verwer metric: 1-(P(Y=1|S=1)-P(Y=1|S!=1))."""

import pandas as pd

from ethicml.metrics.prob_pos import ProbPos
from ethicml.utility.data_structures import DataTuple
from .metric import Metric


class CV(Metric):
    """Calder-Verwer."""

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        """Get the score for this metric."""
        from ethicml.evaluators.per_sensitive_attribute import (
            metric_per_sensitive_attribute,
            diff_per_sensitive_attribute,
        )

        per_sens = metric_per_sensitive_attribute(prediction, actual, ProbPos())
        diffs = diff_per_sensitive_attribute(per_sens)

        return 1 - list(diffs.values())[0]

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "CV"

    @property
    def apply_per_sensitive(self) -> bool:
        """Can this metric be applied per sensitive attribute group?"""
        return False


class AbsCV(CV):
    """Absolute value of Calder-Verwer.

    This metric is supposed to make it easier to compare results.
    """

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        """Get the score for this metric."""
        cv_score = super().score(prediction, actual)
        # the following is equivalent to 1 - abs(diff)
        if cv_score > 1:
            return 2 - cv_score
        return cv_score

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "CV absolute"
