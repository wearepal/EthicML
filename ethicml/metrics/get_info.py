"""Metric that gets additional information from the prediction."""
from ethicml.common import implements
from ethicml.utility import DataTuple, Prediction

from .metric import Metric


class GetInfo(Metric):
    """Metric that gets additional information from the prediction."""

    def __init__(self, key: str):
        """Init GetInfo."""
        super().__init__()
        self._key = key

    @implements(Metric)
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        return prediction.info.get(self._key, float("nan"))

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._key

    @property
    def apply_per_sensitive(self) -> bool:
        """Whether the metric can be applied per sensitive attribute."""
        return False
