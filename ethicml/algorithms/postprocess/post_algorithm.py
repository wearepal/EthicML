"""
Abstract Base Class of all post-processing algorithms in the framework
"""

from abc import abstractmethod
from typing import Dict
import pandas as pd

from ..algorithm_base import Algorithm


class PostAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do post-processing"""
    @abstractmethod
    def run(self, predictions: pd.DataFrame, test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Make predictions fair

        Args:
            predictions:
            test:
        """
        raise NotImplementedError("Run needs to be implemented")

    def run_test(self, predictions: pd.DataFrame, test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """

        Args:
            predictions:
            test:
        """
        raise NotImplementedError("Need to reduce data to 500 datapoints")
