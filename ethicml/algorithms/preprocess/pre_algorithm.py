"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from typing import Dict, Tuple
import pandas as pd

from ethicml.algorithms.algorithm_base import Algorithm
from ethicml.algorithms.utils import get_subset


class PreAlgorithm(Algorithm):
    @abstractmethod
    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Run needs to be implemented")

    def run_test(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_testing = get_subset(train)
        return self.run(train_testing, test)
