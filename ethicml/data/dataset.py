"""
Abstract Base Class for all datasets that come with the framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class Dataset(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def filename(self) -> str:
        pass

    @property
    @abstractmethod
    def feature_split(self) -> Dict[str, List[str]]:
        pass

    @property
    @abstractmethod
    def continuous_features(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def discrete_features(self) -> List[str]:
        pass
