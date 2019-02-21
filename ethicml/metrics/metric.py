"""
Abstract Base Class of all metrics in the framework
"""

from typing import Dict
from abc import ABC, abstractmethod

import numpy
import pandas


class Metric(ABC):
    """Base class for all metrics"""

    @abstractmethod
    def score(self, prediction: numpy.array, actual: Dict[str, pandas.DataFrame]) -> float:
        """
        Compute score

        Args:
            prediction: predicted labels
            actual: dictionary with the actual labels and the sensitive attributes

        Returns:
            the score as a single number
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        raise NotImplementedError()

    @property
    def apply_per_sensitive(self) -> bool:
        """
        Whether the metric can be applied per sensitive attribute
        """
        return True
