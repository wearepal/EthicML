"""
For assessing Calder-Verwer metric: 1-(P(Y=1|S=1)-P(Y=1|S!=1))
"""

from typing import Dict
import pandas as pd
import numpy as np

from ethicml.metrics.prob_pos import ProbPos
from .metric import Metric


class CV(Metric):
    def score(self, prediction: np.array, actual: Dict[str, pd.DataFrame]) -> float:
        from ethicml.evaluators.per_sensitive_attribute \
            import metric_per_sensitive_attribute, diff_per_sensitive_attribute
        per_sens = metric_per_sensitive_attribute(prediction, actual, ProbPos())
        diffs = diff_per_sensitive_attribute(per_sens)

        return 1 - list(diffs.values())[0]

    @property
    def name(self) -> str:
        return "CV"

    @property
    def apply_per_sensitive(self) -> bool:
        return False
