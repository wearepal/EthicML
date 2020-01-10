"""For assessing mean of logits."""

import pandas as pd

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple
from .metric import Metric


class ProbOutcome(Metric):
    """Mean of logits."""

    @implements(Metric)
    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        return prediction.to_numpy().sum() / prediction.size

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return "prob_outcome"
