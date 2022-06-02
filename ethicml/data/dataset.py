"""Data structure for all datasets that come with the framework."""
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from typing_extensions import Literal, TypeAlias, TypedDict, final

import pandas as pd
from ranzen import implements

from ethicml.common import ROOT_PATH
from ethicml.utility import DataTuple, undo_one_hot

from .util import (
    LabelSpec,
    label_spec_to_feature_list,
    from_dummies,
    single_col_spec,
)

__all__ = [
    "Dataset",
    "DiscFeatureGroup",
    "FeatureSplit",
    "LabelSpecsPair",
    "LegacyDataset",
    "CSVDataset",
    "CSVDatasetDC",
]

DiscFeatureGroup: TypeAlias = Dict[str, List[str]]


class LabelSpecsPair(NamedTuple):
    s: LabelSpec
    y: LabelSpec


class FeatureSplit(TypedDict):
    """A dictionary of the list of columns that belong to the feature groups."""

    x: List[str]
    s: List[str]
    y: List[str]


class Dataset(ABC):
    """Data structure that holds all the information needed to load a given dataset."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the dataset."""

    @abstractmethod
    def load(self, labels_as_features: bool = False) -> DataTuple:
        """Load dataset from its CSV file.

        :param labels_as_features: if True, the s and y labels are included in the x features
        :returns: DataTuple with dataframes of features, labels and sensitive attributes
        """

    @property
    @abstractmethod
    def continuous_features(self) -> List[str]:
        """Continuous features."""

    @property
    @abstractmethod
    def discrete_features(self) -> List[str]:
        """List of features that are discrete."""

    @property
    @abstractmethod
    def feature_split(self) -> FeatureSplit:
        """Return a features dictionary."""

    @property
    @abstractmethod
    def disc_feature_groups(self) -> Optional[Dict[str, List[str]]]:
        """Return Dictionary of feature groups."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of elements in the dataset."""


class CSVDataset(Dataset, ABC):
    """Dataset class that loads data from CSV files."""

    discard_non_one_hot: ClassVar[bool] = False
    """If some entries in s or y are not correctly one-hot encoded, discard those."""
    map_to_binary: ClassVar[bool] = False
    """If True, convert labels from {-1, 1} to {0, 1}."""

    @abstractmethod
    def get_unfiltered_continuous(self) -> List[str]:
        """Continuous features."""

    @abstractmethod
    def get_unfiltered_discrete(self) -> List[str]:
        """Discrete features."""

    @abstractmethod
    def get_name(self) -> str:
        """Name of the dataset."""

    @abstractmethod
    def get_label_specs(self) -> Tuple[LabelSpecsPair, List[str]]:
        """Label specs and discrete feature groups that have to be removed from ``x``."""

    @abstractmethod
    def get_num_samples(self) -> int:
        """Number of samples in the dataset."""

    @abstractmethod
    def get_filename_or_path(self) -> Union[str, Path]:
        """Filename of CSV files containing the data."""

    @property
    @abstractmethod
    def load_discrete_only(self) -> bool:
        """Whether to only load discrete features."""

    @property
    @abstractmethod
    def invert_sens_attr(self) -> bool:
        """Whether to invert the sensitive attribute."""

    def get_disc_feature_groups(self) -> Optional[DiscFeatureGroup]:
        """Discrete feature groups."""
        return None

    @property
    @final
    def name(self) -> str:
        """Name of the dataset."""
        return self.get_name()

    @property
    @final
    def filepath(self) -> Path:
        """Filepath from which to load the data."""
        file_name_or_path = self.get_filename_or_path()
        if isinstance(file_name_or_path, Path):
            return file_name_or_path
        else:
            return ROOT_PATH / "data" / "csvs" / file_name_or_path

    @property
    @final
    def sens_attrs(self) -> List[str]:
        """Get the list of sensitive attributes."""
        return label_spec_to_feature_list(self.get_label_specs()[0].s)

    @property
    @final
    def class_labels(self) -> List[str]:
        """Get the list of class labels."""
        return label_spec_to_feature_list(self.get_label_specs()[0].y)

    @property
    @final
    def features_to_remove(self) -> List[str]:
        """Features that have to be removed from x."""
        to_remove: List[str] = []
        disc_feature_groups = self.get_disc_feature_groups()
        if disc_feature_groups is not None:
            for group in self.get_label_specs()[1]:
                to_remove += disc_feature_groups[group]

        if self.load_discrete_only:
            to_remove += self.continuous_features
        return to_remove

    @final
    def filter_features(self, features: Sequence[str]) -> List[str]:
        return [feat for feat in features if feat not in self.features_to_remove]

    @property
    def feature_split(self) -> FeatureSplit:
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
    def continuous_features(self) -> List[str]:
        """List of features that are continuous."""
        return self.filter_features(self.get_unfiltered_continuous())

    @property
    def discrete_features(self) -> List[str]:
        """List of features that are discrete."""
        return self.filter_features(self.get_unfiltered_discrete())

    @property
    def disc_feature_groups(self) -> Optional[Dict[str, List[str]]]:
        """Return Dictionary of feature groups."""
        dfgs = self.get_disc_feature_groups()
        if dfgs is None:
            return None
        return {k: v for k, v in dfgs.items() if k not in self.features_to_remove}

    @implements(Dataset)
    def __len__(self) -> int:
        return self.get_num_samples()

    @implements(Dataset)
    def load(self, labels_as_features: bool = False) -> DataTuple:
        dataframe: pd.DataFrame = pd.read_csv(self.filepath, nrows=self.get_num_samples())
        assert isinstance(dataframe, pd.DataFrame)

        feature_split = self.feature_split
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

        disc_feature_groups = self.get_disc_feature_groups()
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
        s_df = dataframe[feature_split["s"]]
        y_df = dataframe[feature_split["y"]]

        if self.map_to_binary:
            s_df = (s_df + 1) // 2  # map from {-1, 1} to {0, 1}
            y_df = (y_df + 1) // 2  # map from {-1, 1} to {0, 1}

        if self.invert_sens_attr:
            assert s_df.nunique().values[0] == 2, "s must be binary"
            s_df = 1 - s_df

        # the following operations remove rows if a label group is not properly one-hot encoded
        s_data, s_mask = self._one_hot_encode_and_combine(s_df, label_type="s")
        if s_mask is not None:
            x_data = x_data.loc[s_mask].reset_index(drop=True)
            s_data = s_data.loc[s_mask].reset_index(drop=True)
            y_df = y_df.loc[s_mask].reset_index(drop=True)
        y_data, y_mask = self._one_hot_encode_and_combine(y_df, label_type="y")
        if y_mask is not None:
            x_data = x_data.loc[y_mask].reset_index(drop=True)
            s_data = s_data.loc[y_mask].reset_index(drop=True)
            y_data = y_data.loc[y_mask].reset_index(drop=True)

        return DataTuple.from_df(x=x_data, s=s_data, y=y_data, name=self.name)

    def _one_hot_encode_and_combine(
        self, attributes: pd.DataFrame, label_type: Literal["s", "y"]
    ) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Construct a new label according to the LabelSpecs.

        :param attributes: DataFrame containing the attributes.
        :param label_type: Type of label to construct.
        """
        mask = None  # the mask is needed when we have to discard samples

        label_specs, _ = self.get_label_specs()
        label_mapping = label_specs.s if label_type == "s" else label_specs.y

        # create a Series of zeroes with the same length as the dataframe
        combination: pd.Series = pd.Series(  # type: ignore[call-overload]
            0, index=range(len(attributes)), name=",".join(label_mapping)
        )

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
        return combination, mask

    def expand_labels(self, label: pd.Series, label_type: Literal["s", "y"]) -> pd.DataFrame:
        """Expand a label in the form of an index into all the subfeatures.

        :param label: DataFrame containing the labels.
        :param label_type: Type of label to expand.
        """
        label_specs, _ = self.get_label_specs()
        label_mapping = label_specs.s if label_type == "s" else label_specs.y

        # first order the multipliers; this is important for disentangling the values
        names_ordered = sorted(label_mapping, key=lambda name: label_mapping[name].multiplier)

        final_df = {}
        for i, name in enumerate(names_ordered):
            spec = label_mapping[name]
            value = label
            if i + 1 < len(names_ordered):
                next_group = label_mapping[names_ordered[i + 1]]
                value = label % next_group.multiplier
            value = value // spec.multiplier
            value.replace(list(range(len(spec.columns))), spec.columns, inplace=True)
            restored = pd.get_dummies(value)
            final_df[name] = restored  # for the multi-level column index

        return pd.concat(final_df, axis=1)

    @typing.no_type_check
    def load_aif(self):  # Returns aif.360 Standard Dataset
        """Load the dataset as an AIF360 dataset.

        Experimental.
        Requires the aif360 library.

        Ignores the type check as the return type is not yet defined.

        """
        from aif360.datasets import StandardDataset

        data = self.load()
        disc_feature_groups = self.get_disc_feature_groups()
        if disc_feature_groups is not None:
            data_collapsed = from_dummies(data.x, disc_feature_groups)
            data = data.replace(x=data_collapsed)
        df = data.data

        return StandardDataset(
            df=df,
            label_name=data.y_column,
            favorable_classes=lambda x: x > 0,
            protected_attribute_names=[data.s_column],
            privileged_classes=[lambda x: x == 1],
            categorical_features=disc_feature_groups.keys()
            if disc_feature_groups is not None
            else None,
        )


@dataclass  # type: ignore[misc]  # mypy doesn't allow abstract dataclasses because mypy is stupid
class CSVDatasetDC(CSVDataset):
    """Dataset that uses the default load function."""

    discrete_only: bool = False
    invert_s: bool = False

    @property
    def load_discrete_only(self) -> bool:
        """Whether to only load discrete features."""
        return self.discrete_only

    @property
    def invert_sens_attr(self) -> bool:
        """Whether to invert the sensitive attribute."""
        return self.invert_s


@dataclass(init=False)
class LegacyDataset(CSVDataset):
    """Dataset that uses the default load function."""

    discrete_only: bool = False
    invert_s: bool = False

    def __init__(
        self,
        name: str,
        filename_or_path: Union[str, Path],
        features: Sequence[str],
        cont_features: Sequence[str],
        sens_attr_spec: Union[str, LabelSpec],
        class_label_spec: Union[str, LabelSpec],
        num_samples: int,
        s_prefix: Optional[Sequence[str]] = None,
        class_label_prefix: Optional[Sequence[str]] = None,
        discrete_feature_groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._name = name
        self._features = features
        self._sens_attr_spec = sens_attr_spec
        self._class_label_spec = class_label_spec
        self._num_samples = num_samples
        self._s_prefix = [] if s_prefix is None else s_prefix
        self._class_label_prefix = [] if class_label_prefix is None else class_label_prefix
        self._discrete_feature_groups = discrete_feature_groups
        self._raw_file_name_or_path = filename_or_path
        self._cont_features_unfiltered = list(cont_features)

    @implements(CSVDataset)
    def get_disc_feature_groups(self) -> Optional[DiscFeatureGroup]:
        """Discrete feature groups."""
        return self._discrete_feature_groups

    @implements(CSVDataset)
    def get_unfiltered_continuous(self) -> List[str]:
        """Continuous features."""
        return self._cont_features_unfiltered

    @implements(CSVDataset)
    def get_unfiltered_discrete(self) -> List[str]:
        """Discrete features."""
        return [feat for feat in self._features if feat not in self.continuous_features]

    @implements(CSVDataset)
    def get_name(self) -> str:
        """Name of the dataset."""
        return self._name

    @implements(CSVDataset)
    def get_label_specs(self) -> Tuple[LabelSpecsPair, List[str]]:
        """Label specs and discrete feature groups that have to be removed from ``x``."""
        s_spec = self._sens_attr_spec
        y_spec = self._class_label_spec
        return (
            LabelSpecsPair(
                s=single_col_spec(s_spec) if isinstance(s_spec, str) else s_spec,
                y=single_col_spec(y_spec) if isinstance(y_spec, str) else y_spec,
            ),
            list(self._s_prefix) + list(self._class_label_prefix),
        )

    @implements(CSVDataset)
    def get_num_samples(self) -> int:
        """Number of samples in the dataset."""
        return self._num_samples

    @implements(CSVDataset)
    def get_filename_or_path(self) -> Union[str, Path]:
        return self._raw_file_name_or_path

    @property
    def load_discrete_only(self) -> bool:
        """Whether to only load discrete features."""
        return self.discrete_only

    @property
    def invert_sens_attr(self) -> bool:
        """Whether to invert the sensitive attribute."""
        return self.invert_s
