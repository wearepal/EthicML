"""Abstract Base Class of all metrics in the framework."""

from abc import ABC, abstractmethod

from ethicml.utility.data_structures import DataTuple, Prediction


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self, pos_class: int = 1):
        """Init Metric."""
        self.positive_class = pos_class

    @abstractmethod
    def score(self, prediction: Prediction, actual: DataTuple) -> float:
        """Compute score.

        Args:
            prediction: predicted labels
            actual: dictionary with the actual labels and the sensitive attributes

        Returns:
            the score as a single number
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        raise NotImplementedError()

    @property
    def apply_per_sensitive(self) -> bool:
        """Whether the metric can be applied per sensitive attribute."""
        return True
