"""
Abstract Base Class for all datasets that come with the framework
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from ethicml.data.util import filter_features_by_prefixes, get_discrete_features


class Dataset(ABC):
    def __init__(self):
        self._features: List[str] = []
        self._class_label_prefix: List[str] = []
        self._class_labels: List[str] = []
        self._s_prefix: List[str] = []
        self._sens_attrs: List[str] = []
        self._cont_features: List[str] = []
        self._disc_features: List[str] = []
        self.features_to_remove: List[str] = []
        self.discrete_only: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def filename(self) -> str:
        pass

    @property
    def feature_split(self) -> Dict[str, List[str]]:

        if self.discrete_only:
            self.features_to_remove += self.continuous_features

        return {
            "x": filter_features_by_prefixes(self.features, self.features_to_remove),
            "s": self._sens_attrs,
            "y": self._class_labels
        }

    @property
    def continuous_features(self):
        return self._cont_features

    @continuous_features.setter
    def continuous_features(self, feats: List[str]):
        self._cont_features = feats

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, feats: List[str]):
        self._features = feats

    @property
    def s_prefix(self):
        return self._s_prefix

    @s_prefix.setter
    def s_prefix(self, sens_attrs: List[str]):
        self._s_prefix = sens_attrs
        self.features_to_remove += sens_attrs

    @property
    def sens_attrs(self):
        return self._sens_attrs

    @sens_attrs.setter
    def sens_attrs(self, sens_attrs: List[str]):
        self._sens_attrs = sens_attrs

    @property
    def class_labels(self):
        return self._class_labels

    @class_labels.setter
    def class_labels(self, labels: List[str]):
        self._class_labels = labels

    @property
    def class_label_prefix(self):
        return self._class_label_prefix

    @class_label_prefix.setter
    def class_label_prefix(self, label_prefixs: List[str]):
        self._class_label_prefix = label_prefixs
        self.features_to_remove += label_prefixs

    @property
    def discrete_features(self) -> List[str]:
        return get_discrete_features(self.features, self.features_to_remove, self.continuous_features)

    @discrete_features.setter
    def discrete_features(self, feats: List[str]):
        self._disc_features = feats
