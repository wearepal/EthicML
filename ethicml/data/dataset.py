"""Data structure for all datasets that come with the framework."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Sequence, Tuple

import pandas as pd
from typing_extensions import Literal

from ethicml.common import ROOT_PATH
from ethicml.utility import DataTuple
from .util import filter_features_by_prefixes, get_discrete_features

__all__ = ["Dataset"]


@dataclass(frozen=True)
class Dataset:
    """Data structure that holds all the information needed to load a given dataset."""

    name: str
    filename_or_path: Union[str, Path]
    features: Sequence[str]
    cont_features: Sequence[str]
    sens_attrs: Sequence[str]
    class_labels: Sequence[str]
    num_samples: int
    discrete_only: bool
    s_prefix: Sequence[str] = field(default_factory=list)
    class_label_prefix: Sequence[str] = field(default_factory=list)
    discrete_feature_groups: Optional[Dict[str, List[str]]] = None
    combination_multipliers: Dict[str, int] = field(default_factory=dict)

    @property
    def filepath(self) -> Path:
        """Filepath from which to load the data."""
        if isinstance(self.filename_or_path, Path):
            return self.filename_or_path
        else:
            return ROOT_PATH / "data" / "csvs" / self.filename_or_path

    @property
    def features_to_remove(self) -> List[str]:
        """Features that have to be removed from x."""
        to_remove: List[str] = []
        to_remove += self.s_prefix
        to_remove += self.class_label_prefix
        if self.discrete_only:
            to_remove += self.continuous_features
        return to_remove

    @property
    def ordered_features(self) -> Dict[str, List[str]]:
        """Return an order features dictionary.

        This should have separate entries for the features, the labels and the
        sensitive attributes, but the x features are ordered so first are the discrete features,
        then the continuous.
        """
        ordered = self.discrete_features + self.continuous_features
        return {
            "x": filter_features_by_prefixes(ordered, self.features_to_remove),
            "s": list(self.sens_attrs),
            "y": list(self.class_labels),
        }

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        """Return a feature split dictionary.

        This should have separate entries for the features, the labels and the sensitive attributes.
        """
        features_to_remove = self.features_to_remove

        return {
            "x": filter_features_by_prefixes(self.features, features_to_remove),
            "s": list(self.sens_attrs),
            "y": list(self.class_labels),
        }

    @property
    def continuous_features(self) -> List[str]:
        """List of features that are continuous."""
        return filter_features_by_prefixes(self.cont_features, self.features_to_remove)

    @property
    def discrete_features(self) -> List[str]:
        """List of features that are discrete."""
        return get_discrete_features(
            list(self.features), self.features_to_remove, self.continuous_features
        )

    @property
    def disc_feature_groups(self) -> Optional[Dict[str, List[str]]]:
        """Dictionary of feature groups."""
        if self.discrete_feature_groups is None:
            return None
        dfgs = self.discrete_feature_groups
        return {k: v for k, v in dfgs.items() if k not in self.features_to_remove}

    def __len__(self) -> int:
        """Number of elements in the dataset."""
        return self.num_samples

    def load(
        self,
        ordered: bool = False,
        discard_non_one_hot: bool = False,
        map_to_binary: bool = False,
        sens_combination: bool = False,
        class_combination: bool = False,
    ) -> DataTuple:
        """Load dataset from its CSV file.

        Args:
            ordered: if True, return features such that discrete come first, then continuous
            generate_dummies: if True generate complementary features for standalone binary features
            discard_non_one_hot: if some entries in s or y are not correctly one-hot encoded,
                                 discard those
            map_to_binary: if True, convert labels from {-1, 1} to {0, 1}
            sens_combination: if True, take as s the cartesian product of all s columns
            class_combination: if True, take as y the cartesian product of all y columns

        Returns:
            DataTuple with dataframes of features, labels and sensitive attributes
        """
        dataframe: pd.DataFrame = pd.read_csv(self.filepath)
        assert isinstance(dataframe, pd.DataFrame)

        feature_split = self.feature_split if not ordered else self.ordered_features
        feature_split_x = feature_split["x"]

        # =========================================================================================
        # Check whether we have to generate some complementary columns for binary features.
        # This happens when we have for example several races: race-asian-pac-islander etc, but we
        # want to have a an attribute called "race_other" that summarizes them all. Now the problem
        # is that this cannot be done in the before this point, because only here have we actually
        # loaded the data. So, we have to do it here, with all the information we can piece
        # together.

        disc_feature_groups = self.discrete_feature_groups
        if disc_feature_groups is not None:
            for group in disc_feature_groups.values():
                if len(group) == 1:
                    continue
                for feature in group:
                    if feature in dataframe.columns:
                        continue  # nothing to do
                    missing_feature = feature

                    existing_features = [other for other in group if other in dataframe.columns]
                    assert len(existing_features) == len(group) - 1, "at most 1 feature missing"
                    # the dummy feature is the inverse of the existing feature
                    or_combination = dataframe[existing_features[0]] == 1
                    for other in existing_features[1:]:
                        or_combination |= dataframe[other] == 1
                    inverse: pd.Series = 1 - or_combination
                    dataframe = pd.concat(
                        [dataframe, inverse.to_frame(name=missing_feature)], axis="columns"
                    )

        # =========================================================================================
        x_data = dataframe[feature_split_x]
        s_data = dataframe[feature_split["s"]]
        y_data = dataframe[feature_split["y"]]

        if map_to_binary:
            s_data = (s_data + 1) // 2  # map from {-1, 1} to {0, 1}
            y_data = (y_data + 1) // 2  # map from {-1, 1} to {0, 1}

        if sens_combination:
            s_mask = None
            s_data = self.combine_attributes(s_data)
        else:
            s_name = self.s_prefix[0] if self.s_prefix else ",".join(s_data.columns)
            s_data, s_mask = maybe_undo_one_hot(s_data, discard_non_one_hot, "s", s_name)
            if s_mask is not None:
                y_data = y_data[s_mask]
                x_data = x_data[s_mask]
        if class_combination:
            y_mask = None
            y_data = self.combine_attributes(y_data)
        else:
            y_name = (
                self.class_label_prefix[0] if self.class_label_prefix else ",".join(y_data.columns)
            )
            y_data, y_mask = maybe_undo_one_hot(y_data, discard_non_one_hot, "y", y_name)
            if y_mask is not None:
                s_data = s_data[y_mask]
                x_data = x_data[y_mask]

        return DataTuple(x=x_data, s=s_data, y=y_data, name=self.name)

    def combine_attributes(self, attributes: pd.DataFrame) -> pd.DataFrame:
        """Take the cartesian product of all attributes."""
        attr_names = list(attributes.columns)
        combination = attributes[attr_names[0]]
        multiplier = 2
        self.combination_multipliers[attr_names[0]] = 1
        for attr_name in attr_names[1:]:
            combination += multiplier * attributes[attr_name]
            self.combination_multipliers[attr_name] = multiplier
            multiplier *= 2
        return combination.to_frame(name=",".join(attr_names))


def maybe_undo_one_hot(
    attributes: pd.DataFrame,
    discard_non_one_hot: bool,
    label_type: Literal["s", "y"],
    new_name: str,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Undo one-hot encoding if it's there."""
    mask = None
    if attributes.shape[1] > 1:
        if discard_non_one_hot:
            # only use those samples where exactly one of the specified attributes is true
            mask = attributes.sum(axis="columns") == 1
            attributes = attributes.loc[mask]
        else:
            assert (attributes.sum(axis="columns") == 1).all(), f"{label_type} is not one-hot"

        # we have to overwrite the column names because `idxmax` uses the column names
        attributes.columns = list(range(attributes.shape[1]))
        attributes = attributes.idxmax(axis="columns").to_frame(name=new_name)

    return attributes, mask
