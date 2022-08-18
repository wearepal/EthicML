"""Abstract Base Class of all metrics in the framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import ClassVar, final

from ranzen import implements

from ethicml.utility import EvalTuple, Prediction

__all__ = ["MetricStaticName", "Metric"]


class Metric(ABC):
    """Base class for all metrics."""

    apply_per_sensitive: ClassVar[bool] = True
    """Whether the metric can be applied per sensitive attribute."""

    @abstractmethod
    def score(self, prediction: Prediction, actual: EvalTuple) -> float:
        """Compute score.

        :param prediction: predicted labels
        :param actual: EvalTuple with the actual labels and the sensitive attributes
        :returns: the score as a single number
        """

    @abstractmethod
    def get_name(self) -> str:
        """Name of the metric."""

    @property
    @final
    def name(self) -> str:
        """Name of the metric."""
        return self.get_name()


class MetricStaticName(Metric, ABC):
    """Metric base class for metrics whose name does not depend on instance variables."""

    _name: ClassVar[str] = "<please overwrite me>"

    @implements(Metric)
    def get_name(self) -> str:
        return self._name
