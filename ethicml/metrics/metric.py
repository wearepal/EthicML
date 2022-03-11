"""Abstract Base Class of all metrics in the framework."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional
from typing_extensions import Protocol

from ethicml.utility import DataTuple, Prediction

__all__ = ["CfmMetric", "ClassificationMetric", "FairnessMetric", "Metric"]


class Metric(Protocol):
    """Base class for all metrics."""

    apply_per_sensitive: bool
    """Whether the metric can be applied per sensitive attribute."""
    # name: str
    # """Name of the metric."""

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


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class CfmMetric(Metric):
    """Confusion Matrix based metric."""

    pos_class: int = 1
    """The class to treat as being "positive"."""
    labels: Optional[List[int]] = None
    """List of possible target values. If `None`, then this is inferred from the data when run."""
    _name: ClassVar[str] = "<please overwrite me>"
    apply_per_sensitive = True

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._name


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class ClassificationMetric(Metric):
    """Classification metrics are not explicitly fairness related."""

    _name: ClassVar[str] = "<please overwrite me>"
    apply_per_sensitive = True

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._name


@dataclass  # type: ignore  # mypy doesn't allow abstract dataclasses because mypy is stupid
class FairnessMetric(Metric):
    """Fairness metrics explicitly measure something related to fairness."""

    _name: ClassVar[str] = "<please overwrite me>"
    apply_per_sensitive = False

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self._name
