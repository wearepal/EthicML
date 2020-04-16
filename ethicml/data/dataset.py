"""Data structure for all datasets that come with the framework."""
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from typing_extensions import final

from ethicml.common import ROOT_PATH
from ethicml.utility import DataTuple
from .util import filter_features_by_prefixes, get_discrete_features

__all__ = ["Dataset"]


@final  # this class should not be subclassed. it is just meant as a data structure
class Dataset:
    """Data structure that holds all the information needed to load a given dataset."""

    def __init__(
        self,
        name: str,
        filename_or_path: Union[str, Path],
        features: List[str],
        cont_features: List[str],
        sens_attrs: List[str],
        class_labels: List[str],
        num_samples: int,
        discrete_only: bool,
        s_prefix: Optional[List[str]] = None,
        class_label_prefix: Optional[List[str]] = None,
        disc_feature_groups: Optional[Dict[str, List[str]]] = None,
    ):
        """Init Dataset object."""
        self.features_to_remove: List[str] = []
        self._name: str = name
        self.num_samples: int = num_samples
        self.continuous_features = cont_features
        self.features = features
        self.s_prefix = s_prefix or []
        self.sens_attrs: List[str] = sens_attrs
        self.class_label_prefix = class_label_prefix or []
        self.class_labels: List[str] = class_labels
        self.discrete_only: bool = discrete_only
        self._filepath: Path
        if isinstance(filename_or_path, Path):
            self._filepath = filename_or_path
        else:
            self._filepath = ROOT_PATH / "data" / "csvs" / filename_or_path
        self._disc_feature_groups: Optional[Dict[str, List[str]]] = disc_feature_groups

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return self._name

    @property
    def filepath(self) -> Path:
        """Filepath from which to load the data."""
        return self._filepath

    @property
    def ordered_features(self) -> Dict[str, List[str]]:
        """Return an order features dictionary.

        This should have separate entries for the features, the labels and the
        sensitive attributes, but the x features are ordered so first are the discrete features,
        then the continuous.
        """
        return {
            "x": self.discrete_features + self.continuous_features,
            "s": self.sens_attrs,
            "y": self.class_labels,
        }

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        """Return a feature split dictionary.

        This should have separate entries for the features, the labels and the sensitive attributes.
        """
        features_to_remove = self.features_to_remove
        if self.discrete_only:
            features_to_remove += self.continuous_features

        return {
            "x": filter_features_by_prefixes(self.features, features_to_remove),
            "s": self.sens_attrs,
            "y": self.class_labels,
        }

    @property
    def continuous_features(self) -> List[str]:
        """List of features that are continuous."""
        return filter_features_by_prefixes(self._cont_features, self.features_to_remove)

    @continuous_features.setter
    def continuous_features(self, feats: List[str]) -> None:
        self._cont_features = feats

    @property
    def features(self) -> List[str]:
        """List of all features."""
        return self._features

    @features.setter
    def features(self, feats: List[str]) -> None:
        self._features = feats

    @property
    def s_prefix(self) -> List[str]:
        """List of prefixes of the sensitive attribute."""
        return self._s_prefix

    @s_prefix.setter
    def s_prefix(self, sens_attrs: List[str]) -> None:
        self._s_prefix = sens_attrs
        self.features_to_remove += sens_attrs

    @property
    def class_label_prefix(self) -> List[str]:
        """List of prefixes of class labels."""
        return self._class_label_prefix

    @class_label_prefix.setter
    def class_label_prefix(self, label_prefixs: List[str]) -> None:
        self._class_label_prefix = label_prefixs
        self.features_to_remove += label_prefixs

    @property
    def discrete_features(self) -> List[str]:
        """List of features that are discrete."""
        return get_discrete_features(
            self.features, self.features_to_remove, self.continuous_features
        )

    @property
    def disc_feature_groups(self) -> Optional[Dict[str, List[str]]]:
        """Dictionary of feature groups."""
        if self._disc_feature_groups is None:
            return None
        return {
            k: v for k, v in self._disc_feature_groups.items() if k not in self.features_to_remove
        }

    def __len__(self) -> int:
        """Number of elements in the dataset."""
        return self.num_samples

    def load(self, ordered: bool = False, generate_dummies: bool = False) -> DataTuple:
        """Load dataset from its CSV file.

        Args:
            ordered: if True, return features such that discrete come first, then continuous
            generate_dummies: if True generate complementary features for standalone binary features

        Returns:
            DataTuple with dataframes of features, labels and sensitive attributes
        """
        dataframe: pd.DataFrame = pd.read_csv(self.filepath)
        assert isinstance(dataframe, pd.DataFrame)

        feature_split = self.feature_split if not ordered else self.ordered_features

        x_data = dataframe[feature_split["x"]]
        s_data = dataframe[feature_split["s"]]
        y_data = dataframe[feature_split["y"]]

        if generate_dummies:
            # check whether we have to generate some complementary columns for binary features
            disc_feature_groups = self.disc_feature_groups
            if disc_feature_groups is not None:
                for group in disc_feature_groups.values():
                    assert len(group) > 1, "binary features should be encoded as two features"
                    if len(group) == 2:
                        # check if one of the features is a dummy feature (not in the CSV file)
                        if group[0] not in x_data.columns:
                            existing_feature, non_existing_feature = group[1], group[0]
                        elif group[1] not in x_data.columns:
                            existing_feature, non_existing_feature = group[0], group[1]
                        else:
                            continue  # all features are present
                        # the dummy feature is the inverse of the existing feature
                        inverse: pd.Series = 1 - x_data[existing_feature]
                        x_data = pd.concat(
                            [x_data, inverse.to_frame(name=non_existing_feature)], axis="columns"
                        )
                # order the features: first discrete features in the groups, then continuous
                if ordered:
                    discrete_features: List[str] = []
                    for group in disc_feature_groups.values():
                        discrete_features += group
                    x_data = x_data[discrete_features + self.continuous_features]

        return DataTuple(x=x_data, s=s_data, y=y_data, name=self.name)
