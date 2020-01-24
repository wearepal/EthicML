"""For assessing Nomalized Mutual Information."""

from sklearn.metrics import normalized_mutual_info_score as nmis

from ethicml.common import implements
from ethicml.utility.data_structures import DataTuple, Prediction
from .metric import Metric


class NMI(Metric):
    """Normalized Mutual Information."""

    def __init__(self, pos_class: int = 1, base: str = "y"):
        """Init NMI."""
        super().__init__(pos_class=pos_class)
        if base not in ["s", "y"]:
            raise NotImplementedError(
                "Can only calculate NMI of prediction.hards with regard to y or s"
            )
        self.base = base

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        if self.base == "y":
            base_values = actual.y.to_numpy().flatten()
        else:
            base_values = actual.s.to_numpy().flatten()
        return nmis(base_values, prediction.hard.to_numpy().flatten(), average_method="geometric")

    @property
    def name(self) -> str:
        """Getter for the metric name."""
        return f"NMI preds and {self.base}"
