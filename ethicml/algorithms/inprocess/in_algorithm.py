"""
Abstract Base Class of all algorithms in the framework
"""

from abc import abstractmethod
from typing import Dict
import pandas as pd

from ..algorithm_base import Algorithm
from ..utils import get_subset


class InAlgorithm(Algorithm):
    """Abstract Base Class dor algorithms that run in the middle of the pipeline"""

    @abstractmethod
    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run Algorithm

        Args:
            train:
            test:
        """
        raise NotImplementedError("This method needs to be implemented in all models")

    def run_test(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> (
            pd.DataFrame):
        """

        Args:
            train:
            test:

        Returns:

        """
        train_testing = get_subset(train)
        return self.run(train_testing, test)
