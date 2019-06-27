"""
Abstract Base Class of all post-processing algorithms in the framework
"""

from abc import abstractmethod
import pandas as pd

from ethicml.utility.data_structures import DataTuple
from ..algorithm_base import Algorithm


class PostAlgorithm(Algorithm):
    """Abstract Base Class for all algorithms that do post-processing"""

    @abstractmethod
    def run(self, predictions: pd.DataFrame, test: DataTuple) -> pd.DataFrame:
        """Make predictions fair

        Args:
            predictions:
            test:
        """
        raise NotImplementedError("Run needs to be implemented")

    def run_test(self, predictions: pd.DataFrame, test: DataTuple) -> pd.DataFrame:
        """

        Args:
            predictions:
            test:
        """
        raise NotImplementedError("Need to reduce data to 500 datapoints")
