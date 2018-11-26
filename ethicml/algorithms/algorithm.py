"""
Abstract Base Class of all algorithms in the framework
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy
import pandas as pd


class Algorithm(ABC):

    @abstractmethod
    def run(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        pass

    def run_test(self, train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame]) -> pd.DataFrame:

        train_testing = {
            'x': train['x'][:][:500],
            's': train['s'][:][:500],
            'y': train['y'][:][:500]
        }
        result: pd.DataFrame = self.run(train_testing, test)
        return result

    @abstractmethod
    def get_name(self) -> str:
        pass
