"""Abstract Base Class of all metrics in the framework."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ethicml.utility import DataTuple, Prediction


class Metric(ABC):
    """Base class for all metrics."""

    # the following instance attribute should be overwritten in the subclass
    # unfortunately this cannot be enforced with mypy yet
    # see https://github.com/python/mypy/issues/4019 for more information on this limitation
    _name: str = "<unnamed metric>"

    def __init__(self, pos_class: int = 1) -> None:
        self.positive_class = pos_class

    @abstractmethod
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        """Compute score.

        Args:
            prediction: predicted labels
            actual: DataTuple with the actual labels and the sensitive attributes

        Returns:
            the score as a single number
        """

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._name

    @property
    def apply_per_sensitive(self) -> bool:
        """Whether the metric can be applied per sensitive attribute."""
        return True


class CfmMetric(Metric):
    """Confusion Matrix based Metric."""

    def __init__(self, pos_class: int = 1, labels: Optional[List[int]] = None):
        """Confusion Matrix based Metrics.

        Args:
            pos_class: The class to treat as being "positive"
            labels: List of possible target values. If `None` is provided then this is inferred from the data when run.
        """
        super().__init__(pos_class=pos_class)
        self.labels = labels
