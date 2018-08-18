"""
Abstract Base Class for all datasets that come with the framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class Dataset(ABC):

    @abstractmethod
    def get_dataset_name(self) -> str:
        pass

    @abstractmethod
    def get_filename(self) -> str:
        pass

    @abstractmethod
    def get_feature_split(self) -> Dict[str, List[str]]:
        pass
