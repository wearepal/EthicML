"""
Base class for Algorithms
"""

from abc import ABC, abstractmethod


class Algorithm(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()
