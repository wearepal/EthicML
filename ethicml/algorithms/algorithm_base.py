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


class ThreadedAlgorithm(ABC):
    """Base class for algorithms that run in their own thread"""
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm"""
        raise NotImplementedError()
