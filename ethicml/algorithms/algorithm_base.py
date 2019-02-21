"""
Base class for Algorithms
"""

from abc import ABC, abstractmethod


class Algorithm(ABC):
    """Base class for Algorithms"""
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""
        raise NotImplementedError()
