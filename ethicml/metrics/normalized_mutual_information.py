"""For assessing Nomalized Mutual Information."""

import pandas as pd

from sklearn.metrics import normalized_mutual_info_score as nmis

from ethicml.utility.data_structures import DataTuple
from .metric import Metric


class NMI(Metric):
    """Normalized Mutual Information."""

    def __init__(self, pos_class: int = 1, base: str = "y"):
        """Init NMI."""
        super().__init__(pos_class=pos_class)
        if base not in ["s", "y"]:
            raise NotImplementedError("Can only calculate NMI of predictions with regard to y or s")
        self.base = base

    def score(self, prediction: pd.DataFrame, actual: DataTuple) -> float:
        """Get the score for this metric."""
        if self.base == "y":
            base_values = actual.y.to_numpy().flatten()
        else:
            base_values = actual.s.to_numpy().flatten()
        return nmis(base_values, prediction.to_numpy().flatten(), average_method="geometric")

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return f"NMI preds and {self.base}"
