"""
Abstract Base Class of all algorithms in the framework
"""
import numpy

from abc import ABC, abstractmethod


class Algorithm(ABC):

    @abstractmethod
    def run(self, train, test) -> numpy.array:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
