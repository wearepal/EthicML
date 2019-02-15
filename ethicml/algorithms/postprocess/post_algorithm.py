"""
Abstract Base Class of all post-processing algorithms in the framework
"""

from abc import abstractmethod
from typing import Dict
import pandas as pd

from ..algorithm_base import Algorithm


class PostAlgorithm(Algorithm):
    @abstractmethod
    def run(self, predictions: pd.DataFrame, test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError("Run needs to be implemented")

    def run_test(self, predictions: pd.DataFrame, test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError("Need to reduce data to 500 datapoints")
