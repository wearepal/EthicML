"""
Abstract Base Class of all metrics in the framework
"""

from abc import ABC, abstractmethod


class Metric(ABC):

    @abstractmethod
    def score(self, prediction, actual) -> float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
