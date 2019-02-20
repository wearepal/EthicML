"""
Abstract Base Class for all datasets that come with the framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class Dataset(ABC):
    features: List[str]
    y_prefix: List[str]
    y_labels: List[str]
    s_prefix: List[str]
    sens_attrs: List[str]
    _cont_features: List[str]
    _disc_features: List[str]

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
