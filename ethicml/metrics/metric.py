"""Abstract Base Class of all metrics in the framework."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional
from typing_extensions import Protocol

from ethicml.utility import DataTuple, Prediction

__all__ = ["CfmMetric", "BaseMetric", "Metric"]


class Metric(Protocol):
    """Base class for all metrics."""

    apply_per_sensitive: ClassVar[bool]
    """Whether the metric can be applied per sensitive attribute."""

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
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""


class BaseMetric(Metric, Protocol):
    """Metric base class for metrics whose name does not depend on instance variables."""

    _name: ClassVar[str]

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._name


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class CfmMetric(BaseMetric):
    """Confusion Matrix based metric."""

    pos_class: int = 1
    """The class to treat as being "positive"."""
    labels: Optional[List[int]] = None
    """List of possible target values. If `None`, then this is inferred from the data when run."""
    apply_per_sensitive: ClassVar[bool] = True
    _name: ClassVar[str] = "<please overwrite me>"
