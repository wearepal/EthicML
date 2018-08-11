"""
Abstract Base Class for all datasets that come with the framework
"""

from abc import ABC, abstractmethod


class Dataset(ABC):

    @abstractmethod
    def get_filename(self):
        pass

    @abstractmethod
    def get_feature_split(self):
        pass
