"""
Abstract Base Class of all algorithms in the framework
"""

from abc import ABC, abstractmethod
import numpy


class Algorithm(ABC):

    @abstractmethod
    def run(self, train, test) -> numpy.array:
        pass

    def run_test(self, train, test) -> numpy.array:

        train_testing = {
            'x': train['x'][:][:500],
            'y': train['y'][:][:500]
        }
        result: numpy.array = self.run(train_testing, test)
        return result

    @abstractmethod
    def get_name(self) -> str:
        pass
