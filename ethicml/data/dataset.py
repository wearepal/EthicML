"""Data structure for all datasets that come with the framework."""
import typing
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import Literal, TypedDict

import pandas as pd
from ranzen import implements

from ethicml.common import ROOT_PATH
from ethicml.utility import DataTuple, undo_one_hot

from .util import (
    LabelSpec,
    filter_features_by_prefixes,
    get_discrete_features,
    label_spec_to_feature_list,
)

__all__ = ["Dataset", "FeatureSplit", "LoadableDataset"]


class FeatureSplit(TypedDict):
    """A dictionary of the list of columns that belong to the feature groups."""

    x: List[str]
    s: List[str]
    y: List[str]


class Dataset:
    """Data structure that holds all the information needed to load a given dataset.

    Args:
        discard_non_one_hot: If some entries in s or y are not correctly one-hot encoded, discard
            those.
        map_to_binary: If True, convert labels from {-1, 1} to {0, 1}.
    """

    def __init__(
        self,
        name: str,
        filename_or_path: Union[str, Path],
        features: Sequence[str],
        cont_features: Sequence[str],
        sens_attr_spec: Union[str, LabelSpec],
        class_label_spec: Union[str, LabelSpec],
        discrete_only: bool,
        num_samples: int,
        discard_non_one_hot: bool = False,
        map_to_binary: bool = False,
        s_prefix: Optional[Sequence[str]] = None,
        class_label_prefix: Optional[Sequence[str]] = None,
        discrete_feature_groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._name = name
        self._features = features
        self._sens_attr_spec = sens_attr_spec
        self._class_label_spec = class_label_spec
        self._num_samples = num_samples
        self._discrete_only = discrete_only
        self._s_prefix = [] if s_prefix is None else s_prefix
        self._class_label_prefix = [] if class_label_prefix is None else class_label_prefix
        self._discard_non_one_hot = discard_non_one_hot
        self._map_to_binary = map_to_binary
        self._discrete_feature_groups = discrete_feature_groups
        self._raw_file_name_or_path = filename_or_path
        self._cont_features_unfiltered = cont_features

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return self._name

    @property
    def sens_attrs(self) -> List[str]:
        """Get the list of sensitive attributes."""
        if isinstance(self._sens_attr_spec, str):
            return [self._sens_attr_spec]
        assert isinstance(self._sens_attr_spec, dict)
        return label_spec_to_feature_list(self._sens_attr_spec)

    @property
    def class_labels(self) -> List[str]:
        """Get the list of class labels."""
        if isinstance(self._class_label_spec, str):
            return [self._class_label_spec]
        assert isinstance(self._class_label_spec, dict)
        return label_spec_to_feature_list(self._class_label_spec)

    @property
    def filepath(self) -> Path:
        """Filepath from which to load the data."""
        if isinstance(self._raw_file_name_or_path, Path):
            return self._raw_file_name_or_path
        else:
            return ROOT_PATH / "data" / "csvs" / self._raw_file_name_or_path

    @property
    def features_to_remove(self) -> List[str]:
        """Features that have to be removed from x."""
        to_remove: List[str] = []
        to_remove += self._s_prefix
        to_remove += self._class_label_prefix
        if self._discrete_only:
            to_remove += self.continuous_features
        return to_remove

    @property
    def ordered_features(self) -> FeatureSplit:
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
    def feature_split(self) -> FeatureSplit:
        """Return a feature split dictionary.

        This should have separate entries for the features, the labels and the sensitive attributes.
        """
        features_to_remove = self.features_to_remove

        return {
            "x": filter_features_by_prefixes(self._features, features_to_remove),
            "s": self.sens_attrs,
            "y": self.class_labels,
        }

    @property
    def continuous_features(self) -> List[str]:
        """List of features that are continuous."""
        return filter_features_by_prefixes(self._cont_features_unfiltered, self.features_to_remove)

    @property
    def discrete_features(self) -> List[str]:
        """List of features that are discrete."""
        return get_discrete_features(
            list(self._features), self.features_to_remove, self.continuous_features
        )

    @property
    def disc_feature_groups(self) -> Optional[Dict[str, List[str]]]:
        """Dictionary of feature groups."""
        if self._discrete_feature_groups is None:
            return None
        dfgs = self._discrete_feature_groups
        return {k: v for k, v in dfgs.items() if k not in self.features_to_remove}

    def __len__(self) -> int:
        """Number of elements in the dataset."""
        return self._num_samples

    @abstractmethod
    def load(self, ordered: bool = False, labels_as_features: bool = False) -> DataTuple:
        """Load the dataset."""
        raise NotImplementedError("Dataset is abstract.")

    def _maybe_combine_labels(
        self, attributes: pd.DataFrame, label_type: Literal["s", "y"]
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Construct a new label according to the LabelSpecs."""
        mask = None  # the mask is needed when we have to discard samples

        label_mapping = self._sens_attr_spec if label_type == "s" else self._class_label_spec
        if isinstance(label_mapping, str):
            return attributes, mask

        # create a Series of zeroes with the same length as the dataframe
        combination: pd.Series = pd.Series(0, index=range(len(attributes)))  # type: ignore[arg-type]

        for name, spec in label_mapping.items():
            if len(spec.columns) > 1:  # data is one-hot encoded
                raw_values = attributes[spec.columns]
                if self._discard_non_one_hot:
                    # only use those samples where exactly one of the specified attributes is true
                    mask = raw_values.sum(axis="columns") == 1
                else:
                    assert (raw_values.sum(axis="columns") == 1).all(), f"{name} is not one-hot"
                values = undo_one_hot(raw_values)
            else:
                values = attributes[spec.columns[0]]
            combination += spec.multiplier * values
        return combination.to_frame(name=",".join(label_mapping)), mask

    @staticmethod
    def _from_dummies(data: pd.DataFrame, categorical_cols: Dict[str, List[str]]) -> pd.DataFrame:
        out = data.copy()

        for col_parent, filter_col in categorical_cols.items():
            if len(filter_col) > 1:
                undummified = (
                    out[filter_col].idxmax(axis=1).apply(lambda x: x.split(col_parent + "_", 1)[1])
                )

                out[col_parent] = undummified
                out.drop(filter_col, axis=1, inplace=True)

        return out

    def expand_labels(self, label: pd.DataFrame, label_type: Literal["s", "y"]) -> pd.DataFrame:
        """Expand a label in the form of an index into all the subfeatures."""
        label_mapping = self._sens_attr_spec if label_type == "s" else self._class_label_spec
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

        return pd.concat(final_df, axis=1)  # type: ignore[arg-type,return-value]


@dataclass(init=False)
class LoadableDataset(Dataset):
    """Dataset that uses the default load function."""

    # TODO: actually move the loading code into this class. This will require some wider changes.
    discrete_only: bool = False
    invert_s: bool = False

    @implements(Dataset)
    def load(self, ordered: bool = False, labels_as_features: bool = False) -> DataTuple:
        """Load dataset from its CSV file.

        Args:
            ordered: if True, return features such that discrete come first, then continuous
            labels_as_features: if True, the s and y labels are included in the x features

        Returns:
            DataTuple with dataframes of features, labels and sensitive attributes
        """
        dataframe: pd.DataFrame = pd.read_csv(self.filepath, nrows=self._num_samples)
        assert isinstance(dataframe, pd.DataFrame)

        feature_split = self.feature_split if not ordered else self.ordered_features
        if labels_as_features:
            feature_split_x = feature_split["x"] + feature_split["s"] + feature_split["y"]
        else:
            feature_split_x = feature_split["x"]

        # =========================================================================================
        # Check whether we have to generate some complementary columns for binary features.
        # This happens when we have for example several races: race-asian-pac-islander etc, but we
        # want to have a an attribute called "race_other" that summarizes them all. Now the problem
        # is that this cannot be done before this point, because only here have we actually loaded
        # the data. So, we have to do it here, with all the information we can piece together.

        disc_feature_groups = self._discrete_feature_groups
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

        if self._map_to_binary:
            s_data = (s_data + 1) // 2  # map from {-1, 1} to {0, 1}
            y_data = (y_data + 1) // 2  # map from {-1, 1} to {0, 1}

        if self.invert_s:
            assert s_data.nunique().values[0] == 2, "s must be binary"
            s_data = 1 - s_data

        # the following operations remove rows if a label group is not properly one-hot encoded
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

    @typing.no_type_check
    def load_aif(self):  # Returns aif.360 Standard Dataset
        """Load the dataset as an AIF360 dataset.

        Experimental.
        Requires the aif360 library.

        Ignores the type check as the return type is not yet defined.
        """
        from aif360.datasets import StandardDataset

        data = self.load()
        if self.disc_feature_groups is not None:
            data_collapsed = self._from_dummies(data.x, self.disc_feature_groups)
            data = data.replace(x=data_collapsed)
        df = pd.concat([data.x, data.s, data.y], axis="columns")

        return StandardDataset(
            df=df,
            label_name=data.y.columns[0],
            favorable_classes=lambda x: x > 0,
            protected_attribute_names=data.s.columns,
            privileged_classes=[lambda x: x == 1],
            categorical_features=self.disc_feature_groups.keys()
            if self.disc_feature_groups is not None
            else None,
        )
