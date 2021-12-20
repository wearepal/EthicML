"""Class to describe features of the Adult dataset.

Uses FolkTables: https://github.com/zykls/folktables
Paper: https://arxiv.org/abs/2108.04884
"""
# pylint: skip-file
# Pylint will go crazy as we're reimplementing the Dataset Init.
import contextlib
import os
from enum import Enum
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from folktables import ACSDataSource, ACSIncome, adult_filter, folktables, state_list
from ranzen import implements

from ethicml.utility import DataTuple

from ..dataset import Dataset
from ..util import LabelSpec, flatten_dict, simple_spec

__all__ = ["acs_income", "AcsIncome"]


class AdultSplits(Enum):
    SEX = "Sex"
    EDUCTAION = "Education"
    NATIONALITY = "Nationality"
    RACE = "Race"
    RACE_BINARY = "Race-Binary"
    RACE_SEX = "Race-Sex"
    CUSTOM = "Custom"


@contextlib.contextmanager
def download_dir(root: Path):
    curdir = os.getcwd()
    os.chdir(root.expanduser().resolve())
    try:
        yield
    finally:
        os.chdir(curdir)


def acs_income(
    root: Path,
    year: str,
    horizon: int,
    survey: str,
    states: List[str],
    split: str = "Sex",
    target_threshold: int = 50_000,
    discrete_only: bool = False,
) -> ACSIncome:
    """The ACS Income Dataset from EAAMO21/NeurIPS21 - Retiring Adult."""
    return AcsIncome(
        root=root,
        year=year,
        horizon=horizon,
        survey=survey,
        states=states,
        split=split,
        target_threshold=target_threshold,
        discrete_only=discrete_only,
    )


class AcsIncome(Dataset):
    """The ACS Income Dataset from EAAMO21/NeurIPS21 - Retiring Adult."""

    def __init__(
        self,
        root: Union[str, Path],
        year: str,
        horizon: int,
        survey: str,
        states: List[str],
        split: str = "Sex",
        target_threshold: int = 50_000,
        discrete_only: bool = False,
        invert_s: bool = False,
    ):

        if isinstance(root, str):
            root = Path(root)
        self.root = root.expanduser().resolve()
        self.root.mkdir(exist_ok=True)

        self.year = year
        self.horizon = horizon
        self.survey = survey
        self.states = states
        self.split = split
        self.target = "PINCP"
        self.target_threshold = target_threshold
        self.discrete_only = discrete_only

        assert all(state in state_list for state in states)

        self.sens_lookup = {"Sex": "SEX", "Race": "RAC1P"}

        state_string = "_".join(states)
        self.name = f"ACS_Income_{year}_{survey}_{horizon}_{state_string}_{split}"
        self.class_label_spec = "PINCP_1"
        self.class_label_prefix = ["PINCP"]
        self.discard_non_one_hot = False
        self.map_to_binary = False
        self.invert_s = invert_s

    @implements(Dataset)
    def load(self, ordered: bool = False, labels_as_features: bool = False) -> DataTuple:
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
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        dataframe = data_obj._preprocess(dataframe)
        dataframe[data_obj.target] = dataframe[data_obj.target].apply(data_obj._target_transform)

        dataframe[disc_feats] = dataframe[disc_feats].astype('int').astype('category')
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

        self.features = discrete_features + continuous_features
        self.cont_features = continuous_features

        self.num_samples = dataframe.shape[0]

        self.discrete_feature_groups = disc_feature_groups

        sens_attr_spec: Union[str, LabelSpec]
        if self.split == "Sex":
            self.sens_attr_spec = f"{self.sens_lookup[self.split]}_1"
            self.s_prefix = [self.sens_lookup[self.split]]
        elif self.split == "Race":
            self.sens_attr_spec = simple_spec(
                {self.sens_lookup[self.split]: disc_feature_groups[self.sens_lookup[self.split]]}
            )
            self.s_prefix = [self.sens_lookup[self.split]]
        elif self.split == "Sex-Race":
            self.sens_attr_spec = simple_spec(
                {
                    f"{self.sens_lookup['Sex']}": [f"{self.sens_lookup['Sex']}_1"],
                    f"{self.sens_lookup['Race']}": disc_feature_groups[
                        f"{self.sens_lookup['Race']}"
                    ],
                }
            )
            self.s_prefix = [self.sens_lookup["Sex"], self.sens_lookup["Race"]]
        else:
            raise NotImplementedError

        self._discrete_feature_groups = self.discrete_feature_groups
        self._cont_features_unfiltered = self.cont_features

        dataframe = dataframe.reset_index(drop=True)

        # +++ BELOW HERE IS A COPY OF DATASET LOAD +++

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

        if self.map_to_binary:
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
