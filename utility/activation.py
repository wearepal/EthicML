"""
Activations are applied to
"""

from abc import ABC, abstractmethod
import numpy


class Activation(ABC):
    @abstractmethod
    def apply(self, soft_output: numpy.array) -> numpy.array:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
