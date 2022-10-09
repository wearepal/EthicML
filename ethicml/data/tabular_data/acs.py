"""Class to describe features of the Adult dataset.

Uses FolkTables: https://github.com/zykls/folktables
Paper: https://arxiv.org/abs/2108.04884
"""
# pylint: skip-file
# Pylint will go crazy as we're reimplementing the Dataset Init.
from __future__ import annotations
import contextlib
import os
from pathlib import Path
from typing import Generator, Iterable, Literal

from folktables import ACSDataSource, adult_filter, folktables, state_list
import numpy as np
import pandas as pd
from ranzen import implements

from ethicml.utility.data_helpers import undo_one_hot
from ethicml.utility.data_structures import DataTuple

from ..dataset import Dataset, FeatureOrder, FeatureSplit
from ..util import (
    DiscFeatureGroups,
    LabelSpec,
    filter_features_by_prefixes,
    flatten_dict,
    get_discrete_features,
    label_spec_to_feature_list,
    spec_from_binary_cols,
)

__all__ = ["AcsIncome", "AcsEmployment"]


@contextlib.contextmanager
def download_dir(root: Path) -> Generator[None, None, None]:
    curdir = os.getcwd()
    os.chdir(root.expanduser().resolve())
    try:
        yield
    finally:
        os.chdir(curdir)


class AcsBase(Dataset):
    split: str
    target: str

    def __init__(
        self,
        name: str,
        root: str | Path,
        year: str,
        horizon: int,
        states: list[str],
        class_label_spec: str,
        class_label_prefix: list[str],
        discrete_only: bool = False,
        invert_s: bool = False,
    ):
        if isinstance(root, str):
            root = Path(root)
        self.root = root.expanduser().resolve()
        self.root.mkdir(exist_ok=True)

        self.year = year
        self.horizon = horizon
        self.survey = "person"
        self.states = states
        self._invert_s = invert_s

        assert all(state in state_list for state in states)

        state_string = "_".join(states)
        self._name = f"{name}_{year}_{horizon}_{state_string}_{self.split}"
        self._sens_attr_spec: str | LabelSpec = ""
        self._s_prefix: list[str] = []
        self._class_label_spec: str | LabelSpec = class_label_spec
        self._class_label_prefix = class_label_prefix
        self._discrete_only = discrete_only
        self._features: list[str] = []
        self._cont_features_unfiltered: list[str] = []
        self._map_to_binary = False
        self._discrete_feature_groups: DiscFeatureGroups | None = None
        self._num_samples = 0

    @property
    def name(self) -> str:
        """Name of the dataset."""
        return self._name

    def __len__(self) -> int:
        return self._num_samples

    @property
    def sens_attrs(self) -> list[str]:
        """Get the list of sensitive attributes."""
        if isinstance(self._sens_attr_spec, str):
            return [self._sens_attr_spec]
        assert isinstance(self._sens_attr_spec, dict)
        return label_spec_to_feature_list(self._sens_attr_spec)

    @property
    def class_labels(self) -> list[str]:
        """Get the list of class labels."""
        if isinstance(self._class_label_spec, str):
            return [self._class_label_spec]
        assert isinstance(self._class_label_spec, dict)
        return label_spec_to_feature_list(self._class_label_spec)

    @property
    def features_to_remove(self) -> list[str]:
        """Features that have to be removed from x."""
        to_remove: list[str] = []
        to_remove += self._s_prefix
        to_remove += self._class_label_prefix
        if self._discrete_only:
            to_remove += self.continuous_features
        return to_remove

    @implements(Dataset)
    def feature_split(self, order: FeatureOrder = FeatureOrder.disc_first) -> FeatureSplit:
        features_to_remove = self.features_to_remove

        return {
            "x": filter_features_by_prefixes(self._features, features_to_remove),
            "s": self.sens_attrs,
            "y": self.class_labels,
        }

    @property
    def continuous_features(self) -> list[str]:
        """List of features that are continuous."""
        return filter_features_by_prefixes(self._cont_features_unfiltered, self.features_to_remove)

    @property
    def discrete_features(self) -> list[str]:
        """List of features that are discrete."""
        return get_discrete_features(
            list(self._features), self.features_to_remove, self.continuous_features
        )

    @property
    def disc_feature_groups(self) -> DiscFeatureGroups:
        """Return Dictionary of feature groups."""
        dfgs = self._discrete_feature_groups
        assert dfgs is not None
        return {k: v for k, v in dfgs.items() if k not in self.features_to_remove}

    def _backend_load(self, dataframe: pd.DataFrame, *, labels_as_features: bool) -> DataTuple:
        # +++ BELOW HERE IS A COPY OF DATASET LOAD +++

        assert isinstance(dataframe, pd.DataFrame)

        feature_split = self.feature_split()
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

        if self._invert_s:
            assert s_data.nunique().values[0] == 2, "s must be binary"
            s_data = 1 - s_data

        # the following operations remove rows if a label group is not properly one-hot encoded
        s_data = self._one_hot_encode_and_combine(s_data, label_type="s")
        y_data = self._one_hot_encode_and_combine(y_data, label_type="y")

        return DataTuple.from_df(x=x_data, s=s_data, y=y_data, name=self.name)

    def _one_hot_encode_and_combine(
        self, attributes: pd.DataFrame, label_type: Literal["s", "y"]
    ) -> pd.Series:
        """Construct a new label according to the LabelSpecs.

        :param attributes: DataFrame containing the attributes.
        :param label_type: Type of label to construct.
        :returns: A Series object with the new combined labels.
        """
        label_mapping = self._sens_attr_spec if label_type == "s" else self._class_label_spec
        if isinstance(label_mapping, str):
            return attributes[label_mapping]

        # create a Series of zeroes with the same length as the dataframe
        combination: pd.Series = pd.Series(
            0, index=range(len(attributes)), name=",".join(label_mapping)
        )

        for name, spec in label_mapping.items():
            if len(spec.columns) > 1:  # data is one-hot encoded
                raw_values = attributes[spec.columns]
                assert (raw_values.sum(axis="columns") == 1).all(), f"{name} is not one-hot"
                values = undo_one_hot(raw_values)
            else:
                values = attributes[spec.columns[0]]
            combination += spec.multiplier * values
        return combination


class AcsIncome(AcsBase):
    """The ACS Income Dataset from EAAMO21/NeurIPS21 - Retiring Adult."""

    def __init__(
        self,
        root: str | Path,
        year: str,
        horizon: int,
        states: list[str],
        split: str = "Sex",
        target_threshold: int = 50_000,
        discrete_only: bool = False,
        invert_s: bool = False,
    ):

        self.split = split
        self.target = "PINCP"
        self.target_threshold = target_threshold

        self.sens_lookup = {"Sex": "SEX", "Race": "RAC1P"}

        super().__init__(
            name="ACS_Income",
            root=root,
            class_label_spec="PINCP_1",
            class_label_prefix=["PINCP"],
            year=year,
            horizon=horizon,
            states=states,
            discrete_only=discrete_only,
            invert_s=invert_s,
        )

    @staticmethod
    def cat_lookup(key: str) -> Iterable:
        """Look up categories."""
        table = {
            "COW": range(1, 9),
            'MAR': range(1, 6),
            'RELP': range(18),
            'SEX': range(1, 3),
            'RAC1P': range(1, 10),
            'PINCP': range(2),
        }

        return table[key]

    @implements(Dataset)
    def load(
        self, labels_as_features: bool = False, order: FeatureOrder = FeatureOrder.disc_first
    ) -> DataTuple:
        datasource = ACSDataSource(
            survey_year=self.year, horizon=f'{self.horizon}-Year', survey=self.survey
        )

        with download_dir(self.root):
            dataframe = datasource.get_data(states=self.states, download=True)

        disc_feats = [
            'COW',
            'MAR',
            'RELP',
            'SEX',
            'RAC1P',
            'PINCP',
        ]

        continuous_features = [
            'AGEP',
            'OCCP',
            'POBP',
            'SCHL',
            'WKHP',
        ]

        data_obj = folktables.BasicProblem(
            features=disc_feats + continuous_features,
            target=self.target,
            target_transform=lambda x: x > self.target_threshold,
            group=self.split,
            preprocess=adult_filter,
            postprocess=lambda x: np.nan_to_num(x, nan=-1),
        )

        dataframe = data_obj._preprocess(dataframe)
        dataframe[data_obj.target] = dataframe[data_obj.target].apply(data_obj._target_transform)

        for feat in disc_feats:
            dataframe[feat] = (
                dataframe[feat]
                .astype('int')
                .astype(pd.CategoricalDtype(categories=self.cat_lookup(feat)))
            )

        dataframe[continuous_features] = dataframe[continuous_features].astype('int')

        dataframe = pd.get_dummies(dataframe[disc_feats + continuous_features])

        dataframe = dataframe.apply(data_obj._postprocess)

        cow_cols = [col for col in dataframe.columns if col.startswith('COW')]
        mar_cols = [col for col in dataframe.columns if col.startswith('MAR')]
        relp_cols = [col for col in dataframe.columns if col.startswith('RELP')]
        sex_cols = [col for col in dataframe.columns if col.startswith('SEX')]
        rac1p_cols = [col for col in dataframe.columns if col.startswith('RAC1P')]
        pincp_cols = [col for col in dataframe.columns if col.startswith('PINCP')]
        disc_feature_groups = {
            "COW": cow_cols,
            'MAR': mar_cols,
            'RELP': relp_cols,
            'SEX': sex_cols,
            'RAC1P': rac1p_cols,
            'PINCP': pincp_cols,
        }
        discrete_features = flatten_dict(disc_feature_groups)

        # set all the attributes manually that we couldn't set in the __init__
        self._features = discrete_features + continuous_features
        self._cont_features_unfiltered = continuous_features

        self._num_samples = dataframe.shape[0]

        self._discrete_feature_groups = disc_feature_groups

        if self.split == "Sex":
            self._sens_attr_spec = f"{self.sens_lookup[self.split]}_1"
            self._s_prefix = [self.sens_lookup[self.split]]
        elif self.split == "Race":
            self._sens_attr_spec = spec_from_binary_cols(
                {self.sens_lookup[self.split]: disc_feature_groups[self.sens_lookup[self.split]]}
            )
            self._s_prefix = [self.sens_lookup[self.split]]
        elif self.split == "Sex-Race":
            self._sens_attr_spec = spec_from_binary_cols(
                {
                    f"{self.sens_lookup['Sex']}": [f"{self.sens_lookup['Sex']}_1"],
                    f"{self.sens_lookup['Race']}": disc_feature_groups[
                        f"{self.sens_lookup['Race']}"
                    ],
                }
            )
            self._s_prefix = [self.sens_lookup["Sex"], self.sens_lookup["Race"]]
        else:
            raise NotImplementedError

        dataframe = dataframe.reset_index(drop=True)

        return self._backend_load(dataframe, labels_as_features=labels_as_features)


class AcsEmployment(AcsBase):
    """The ACS Employmment Dataset from EAAMO21/NeurIPS21 - Retiring Adult."""

    def __init__(
        self,
        root: str | Path,
        year: str,
        horizon: int,
        states: list[str],
        split: str = "Sex",
        discrete_only: bool = False,
        invert_s: bool = False,
    ):

        self.split = split
        self.target = "ESR"

        self.sens_lookup = {"Sex": "SEX", "Race": "RAC1P"}

        super().__init__(
            name="ACS_Employment",
            root=root,
            class_label_spec="ESR_1",
            class_label_prefix=["ESR"],
            year=year,
            horizon=horizon,
            states=states,
            discrete_only=discrete_only,
            invert_s=invert_s,
        )

    @staticmethod
    def cat_lookup(key: str) -> Iterable:
        """Look up categories."""
        table = {
            'SCHL': range(1, 25),
            'MAR': range(1, 6),
            'SEX': range(1, 3),
            'DIS': range(1, 3),
            'ESP': range(1, 9),
            'MIG': range(1, 4),
            'CIT': range(1, 5),
            'MIL': range(1, 5),
            'ANC': (1, 2, 4, 8),
            'NATIVITY': range(1, 3),
            'RELP': range(18),
            'DEAR': range(1, 3),
            'DEYE': range(1, 3),
            'DREM': range(1, 3),
            'RAC1P': range(1, 10),
            'ESR': range(1, 7),
        }

        return table[key]

    @implements(Dataset)
    def load(
        self, labels_as_features: bool = False, order: FeatureOrder = FeatureOrder.disc_first
    ) -> DataTuple:
        datasource = ACSDataSource(
            survey_year=self.year, horizon=f'{self.horizon}-Year', survey=self.survey
        )

        with download_dir(self.root):
            dataframe = datasource.get_data(states=self.states, download=True)

        disc_feats = [
            'SCHL',
            'MAR',
            'RELP',
            'DIS',
            'ESP',
            'CIT',
            'MIG',
            'MIL',
            'ANC',
            'NATIVITY',
            'DEAR',
            'DEYE',
            'DREM',
            'SEX',
            'RAC1P',
            'ESR',
        ]

        continuous_features = [
            'AGEP',
        ]

        data_obj = folktables.BasicProblem(
            features=disc_feats + continuous_features,
            target=self.target,
            target_transform=lambda x: x == 1,
            group=self.split,
            preprocess=lambda x: x,
            postprocess=lambda x: np.nan_to_num(x, nan=-1),
        )

        dataframe = data_obj._preprocess(dataframe)
        dataframe[data_obj.target] = dataframe[data_obj.target].apply(data_obj._target_transform)
        dataframe = dataframe.apply(data_obj._postprocess)

        for feat in disc_feats:
            dataframe[feat] = (
                dataframe[feat]
                .astype('int')
                .astype(pd.CategoricalDtype(categories=self.cat_lookup(feat)))
            )

        dataframe[continuous_features] = dataframe[continuous_features].astype('int')

        dataframe = pd.get_dummies(dataframe[disc_feats + continuous_features])

        dataframe = dataframe.apply(data_obj._postprocess)

        schl_cols = [col for col in dataframe.columns if col.startswith('SCHL')]
        mar_cols = [col for col in dataframe.columns if col.startswith('MAR')]
        relp_cols = [col for col in dataframe.columns if col.startswith('RELP')]
        dis_cols = [col for col in dataframe.columns if col.startswith('DIS')]
        esp_cols = [col for col in dataframe.columns if col.startswith('ESP')]
        cit_cols = [col for col in dataframe.columns if col.startswith('CIT')]
        mig_cols = [col for col in dataframe.columns if col.startswith('MIG')]
        mil_cols = [col for col in dataframe.columns if col.startswith('MIL')]
        anc_cols = [col for col in dataframe.columns if col.startswith('ANC')]
        nativity_cols = [col for col in dataframe.columns if col.startswith('NATIVITY')]
        dear_cols = [col for col in dataframe.columns if col.startswith('DEAR')]
        deye_cols = [col for col in dataframe.columns if col.startswith('DEYE')]
        drem_cols = [col for col in dataframe.columns if col.startswith('DREM')]
        sex_cols = [col for col in dataframe.columns if col.startswith('SEX')]
        rac1p_cols = [col for col in dataframe.columns if col.startswith('RAC1P')]
        esr_cols = [col for col in dataframe.columns if col.startswith('ESR')]
        disc_feature_groups = {
            "SCHL": schl_cols,
            'MAR': mar_cols,
            'RELP': relp_cols,
            'DIS': dis_cols,
            'ESP': esp_cols,
            'CIT': cit_cols,
            'MIG': mig_cols,
            'MIL': mil_cols,
            'ANC': anc_cols,
            'NATIVITY': nativity_cols,
            'DEAR': dear_cols,
            'DEYE': deye_cols,
            'DREM': drem_cols,
            'SEX': sex_cols,
            'RAC1P': rac1p_cols,
            'ESR': esr_cols,
        }
        discrete_features = flatten_dict(disc_feature_groups)

        # set all the attributes manually that we couldn't set in the __init__
        self._features = discrete_features + continuous_features
        self._cont_features_unfiltered = continuous_features

        self._num_samples = dataframe.shape[0]

        self._discrete_feature_groups = disc_feature_groups

        if self.split == "Sex":
            self._sens_attr_spec = f"{self.sens_lookup[self.split]}_1"
            self._s_prefix = [self.sens_lookup[self.split]]
        elif self.split == "Race":
            self._sens_attr_spec = spec_from_binary_cols(
                {self.sens_lookup[self.split]: disc_feature_groups[self.sens_lookup[self.split]]}
            )
            self._s_prefix = [self.sens_lookup[self.split]]
        elif self.split == "Sex-Race":
            self._sens_attr_spec = spec_from_binary_cols(
                {
                    f"{self.sens_lookup['Sex']}": [f"{self.sens_lookup['Sex']}_1"],
                    f"{self.sens_lookup['Race']}": disc_feature_groups[
                        f"{self.sens_lookup['Race']}"
                    ],
                }
            )
            self._s_prefix = [self.sens_lookup["Sex"], self.sens_lookup["Race"]]
        else:
            raise NotImplementedError

        dataframe = dataframe.reset_index(drop=True)

        return self._backend_load(dataframe, labels_as_features=labels_as_features)
