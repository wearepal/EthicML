"""For assessing Calder-Verwer metric, :math:`1-(P(Y=1|S=1)-P(Y=1|S!=1))`."""
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

from .confusion_matrix import CfmMetric
from .metric import MetricStaticName
from .per_sensitive_attribute import diff_per_sens, metric_per_sens
from .prob_pos import ProbPos

__all__ = ["AbsCV", "CV"]


@dataclass
class CV(CfmMetric):
    """Calder-Verwer."""

    _name: ClassVar[str] = "CV"
    apply_per_sensitive: ClassVar[bool] = False

    @implements(MetricStaticName)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        prob_pos = ProbPos(pos_class=self.pos_class, labels=self.labels)
        per_sens = metric_per_sens(prediction, actual, metric=prob_pos)
        diffs = diff_per_sens(per_sens)

        return 1 - list(diffs.values())[0]


@dataclass
class AbsCV(CV):
    """Absolute value of Calder-Verwer.

    This metric is supposed to make it easier to compare results.
    """

    _name: ClassVar[str] = "CV absolute"

    @implements(MetricStaticName)
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        cv_score = super().score(prediction, actual)
        # the following is equivalent to 1 - abs(diff)
        if cv_score > 1:
            return 2 - cv_score
        return cv_score
