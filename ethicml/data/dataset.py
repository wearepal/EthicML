"""Data structure for all datasets that come with the framework."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from typing_extensions import Literal

from ethicml.common import ROOT_PATH
from ethicml.utility import DataTuple, undo_one_hot

from .util import (
    LabelSpec,
    filter_features_by_prefixes,
    get_discrete_features,
    label_spec_to_feature_list,
)

__all__ = ["Dataset"]


@dataclass(frozen=True)
class Dataset:
    """Data structure that holds all the information needed to load a given dataset."""

    name: str
    filename_or_path: Union[str, Path]
    features: Sequence[str]
    cont_features: Sequence[str]
    sens_attr_spec: Union[str, LabelSpec]
    class_label_spec: Union[str, LabelSpec]
    num_samples: int
    discrete_only: bool
    s_prefix: Sequence[str] = field(default_factory=list)
    class_label_prefix: Sequence[str] = field(default_factory=list)
    discrete_feature_groups: Optional[Dict[str, List[str]]] = None
    discard_non_one_hot: bool = False
    """If some entries in s or y are not correctly one-hot encoded, discard those."""
    map_to_binary: bool = False
    """If True, convert labels from {-1, 1} to {0, 1}."""

    @property
    def sens_attrs(self) -> List[str]:
        """Get the list of sensitive attributes."""
        if isinstance(self.sens_attr_spec, str):
            return [self.sens_attr_spec]
        else:
            assert isinstance(self.sens_attr_spec, dict)
            return label_spec_to_feature_list(self.sens_attr_spec)

    @property
    def class_labels(self) -> List[str]:
        """Get the list of class labels."""
        if isinstance(self.class_label_spec, str):
            return [self.class_label_spec]
        else:
            assert isinstance(self.class_label_spec, dict)
            return label_spec_to_feature_list(self.class_label_spec)

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
            "s": self.sens_attrs,
            "y": self.class_labels,
        }

    @property
    def feature_split(self) -> Dict[str, List[str]]:
        """Return a feature split dictionary.

        This should have separate entries for the features, the labels and the sensitive attributes.
        """
        features_to_remove = self.features_to_remove

        return {
            "x": filter_features_by_prefixes(self.features, features_to_remove),
            "s": self.sens_attrs,
            "y": self.class_labels,
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

    def load(self, ordered: bool = False) -> DataTuple:
        """Load dataset from its CSV file.

        Args:
            ordered: if True, return features such that discrete come first, then continuous

        Returns:
            DataTuple with dataframes of features, labels and sensitive attributes
        """
        dataframe: pd.DataFrame = pd.read_csv(self.filepath, nrows=self.num_samples)
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

        if self.map_to_binary:
            s_data = (s_data + 1) // 2  # map from {-1, 1} to {0, 1}
            y_data = (y_data + 1) // 2  # map from {-1, 1} to {0, 1}

        s_data, s_mask = self._maybe_combine_labels(s_data, label_type="s")
        if s_mask is not None:
            x_data = x_data.loc[s_mask].reset_index(drop=True)
            s_data = s_data.loc[s_mask].reset_index(drop=True)
            y_data = y_data.loc[s_mask].reset_index(drop=True)
        y_data, y_mask = self._maybe_combine_labels(y_data, label_type="y")
        if y_mask is not None:
            x_data = x_data.loc[y_mask].reset_index(drop=True)
            s_data = s_data.loc[y_mask].reset_index(drop=True)
            y_data = y_data.loc[y_mask].reset_index(drop=True)

        return DataTuple(x=x_data, s=s_data, y=y_data, name=self.name)

    def _maybe_combine_labels(
        self, attributes: pd.DataFrame, label_type: Literal["s", "y"]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Construct a new label according to the LabelSpecs."""
        mask = None  # the mask is needed when we have to discard samples

        label_mapping = self.sens_attr_spec if label_type == "s" else self.class_label_spec
        if isinstance(label_mapping, str):
            return attributes, mask

        # create a Series of zeroes with the same length as the dataframe
        combination: pd.Series[int] = pd.Series(0, index=range(len(attributes)))

        for name, spec in label_mapping.items():
            if len(spec.columns) > 1:  # data is one-hot encoded
                raw_values = attributes[spec.columns]
                if self.discard_non_one_hot:
                    # only use those samples where exactly one of the specified attributes is true
                    mask = raw_values.sum(axis="columns") == 1
                else:
                    assert (raw_values.sum(axis="columns") == 1).all(), f"{name} is not one-hot"
                values = undo_one_hot(raw_values)
            else:
                values = attributes[spec.columns[0]]
            combination += spec.multiplier * values
        return combination.to_frame(name=",".join(label_mapping)), mask

    def expand_labels(self, label: pd.DataFrame, label_type: Literal["s", "y"]) -> pd.DataFrame:
        """Expand a label in the form of an index into all the subfeatures."""
        label_mapping = self.sens_attr_spec if label_type == "s" else self.class_label_spec
        assert not isinstance(label_mapping, str)
        label_mapping: LabelSpec  # redefine the variable for mypy's sake

        # first order the multipliers; this is important for disentangling the values
        names_ordered = sorted(label_mapping, key=lambda name: label_mapping[name].multiplier)

        labels = label[label.columns[0]]

        final_df = {}
        for i, name in enumerate(names_ordered):
            spec = label_mapping[name]
            value = labels
            if i + 1 < len(names_ordered):
                next_group = label_mapping[names_ordered[i + 1]]
                value = labels % next_group.multiplier
            value = value // spec.multiplier
            value.replace(list(range(len(spec.columns))), spec.columns, inplace=True)
            restored = pd.get_dummies(value)
            final_df[name] = restored  # for the multi-level column index

        return pd.concat(final_df, axis=1)
