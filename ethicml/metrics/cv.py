"""For assessing Calder-Verwer metric, :math:`1-(P(Y=1|S=1)-P(Y=1|S!=1))`."""
from dataclasses import dataclass
from typing import ClassVar

from ranzen import implements

from ethicml.utility import DataTuple, Prediction

from .metric import Metric
from .per_sensitive_attribute import diff_per_sensitive_attribute, metric_per_sensitive_attribute
from .prob_pos import ProbPos


@dataclass
class CV(Metric):
    """Calder-Verwer."""

    _name: ClassVar[str] = "CV"
    apply_per_sensitive = False

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        per_sens = metric_per_sensitive_attribute(prediction, actual, metric=ProbPos())
        diffs = diff_per_sensitive_attribute(per_sens)

        return 1 - list(diffs.values())[0]

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._name


@dataclass
class AbsCV(CV):
    """Absolute value of Calder-Verwer.

    This metric is supposed to make it easier to compare results.
    """

    _name: ClassVar[str] = "CV absolute"

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        cv_score = super().score(prediction, actual)
        # the following is equivalent to 1 - abs(diff)
        if cv_score > 1:
            return 2 - cv_score
        return cv_score
