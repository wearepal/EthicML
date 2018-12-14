"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from typing import Dict
import pandas as pd

from ..algorithm_base import Algorithm
from ..utils import get_subset


class InAlgorithm(Algorithm):

    @abstractmethod
    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError("This method needs to be implemented in all models")

    def run_test(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        train_testing = get_subset(train)
        return self.run(train_testing, test)
