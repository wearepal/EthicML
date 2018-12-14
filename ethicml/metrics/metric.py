"""
Abstract Base Class of all metrics in the framework
"""

from typing import Dict
from abc import ABC, abstractmethod

import numpy
import pandas


class Metric(ABC):

    @abstractmethod
    def score(self, prediction: numpy.array, actual: Dict[str, pandas.DataFrame]) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()
