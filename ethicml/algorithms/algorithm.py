"""
Abstract Base Class of all algorithms in the framework
"""

from abc import ABC, abstractmethod
import numpy


class Algorithm(ABC):

    @abstractmethod
    def run(self, train, test) -> numpy.array:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
