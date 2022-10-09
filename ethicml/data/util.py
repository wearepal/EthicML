"""Useful methods that are used in some of the data objects."""
from __future__ import annotations
from itertools import groupby
from typing import Dict, List, Mapping, NamedTuple, Sequence
from typing_extensions import TypeAlias

import pandas as pd

__all__ = [
    "DiscFeatureGroups",
    "LabelGroup",
    "LabelSpec",
    "filter_features_by_prefixes",
    "flatten_dict",
    "from_dummies",
    "get_discrete_features",
    "group_disc_feat_indices",
    "label_spec_to_feature_list",
    "reduce_feature_group",
    "spec_from_binary_cols",
    "single_col_spec",
]

DiscFeatureGroups: TypeAlias = Dict[str, List[str]]
"""A grouping of discrete features such that the result is one-hot encoded."""


class LabelGroup(NamedTuple):
    """Definition of a group of columns that should be interpreted as a single label."""

    columns: list[str]  # list of columns that belong to the group
    multiplier: int = 1  # multiplier assigned to this group (needed when combining groups)


LabelSpec: TypeAlias = Mapping[str, LabelGroup]
"""A label specification consisting of :class:`LabelGroup` entries with names."""


def filter_features_by_prefixes(features: Sequence[str], prefixes: Sequence[str]) -> list[str]:
    """Filter the features by prefixes.

    :param features: list of features names
    :param prefixes: list of prefixes
    :returns: filtered feature names
    """
    res: list[str] = []
    for name in features:
        filtered = any(name.startswith(pref) for pref in prefixes)
        if not filtered:
            res.append(name)
    return res


def get_discrete_features(
    all_feats: list[str], feats_to_remove: list[str], cont_feats: list[str]
) -> list[str]:
    """Get a list of the discrete features in a dataset.

    :param all_feats: List of all features in the dataset.
    :param feats_to_remove: List of features that aren't used.
    :param cont_feats: List of continuous features in the dataset.
    :returns: List of features not marked as continuous or to be removed.
    """
    return [
        item
        for item in filter_features_by_prefixes(all_feats, feats_to_remove)
        if item not in cont_feats
    ]


def group_disc_feat_indices(disc_feat_names: list[str], prefix_sep: str = "_") -> list[slice]:
    """Group discrete features names according to the first segment of their name.

    Returns a list of their corresponding slices (assumes order is maintained).

    :param disc_feat_names: List of discrete feature names.
    :param prefix_sep: Separator between the prefix and the rest of the name. (Default: "_")
    :returns: List of slices.
    """

    def _first_segment(feature_name: str) -> str:
        return feature_name.split(prefix_sep)[0]

    group_iter = groupby(disc_feat_names, _first_segment)

    feature_slices: list[slice] = []
    start_idx = 0
    for _, group in group_iter:
        len_group = len(list(group))
        indices = slice(start_idx, start_idx + len_group)
        feature_slices.append(indices)
        start_idx += len_group

    return feature_slices


def flatten_dict(dictionary: Mapping[str, list[str]] | None) -> list[str]:
    """Flatten a dictionary of lists by joining all lists to one big list."""
    if dictionary is None:
        return []
    return [x for inner_list in dictionary.values() for x in inner_list]


def reduce_feature_group(
    disc_feature_groups: Mapping[str, list[str]],
    feature_group: str,
    to_keep: Sequence[str],
    remaining_feature_name: str,
) -> DiscFeatureGroups:
    """Drop all features in the given feature group except the ones in to_keep.

    :param disc_feature_groups: Dictionary of feature groups.
    :param feature_group: Name of the feature group that will be replaced by ``to_keep``.
    :param to_keep: List of features that will be kept in the feature group.
    :param remaining_feature_name: Name of the dummy feature that will be used to summarize the
        removed features.
    :returns: Modified dictionary of feature groups.
    """
    # first set the given feature group to just the value that we want to keep
    features_to_keep = list(to_keep)
    # then add a new dummy feature to the feature group. `load_data()` will create this for us
    features_to_keep.append(f"{feature_group}{remaining_feature_name}")
    new_dfgs = dict(disc_feature_groups)
    new_dfgs[feature_group] = features_to_keep
    return new_dfgs


def reduce_feature_group_mut(
    disc_feature_groups: dict[str, list[str]],
    feature_group: str,
    to_keep: Sequence[str],
    remaining_feature_name: str,
) -> list[str]:
    """Drop all features in the given feature group except the ones in to_keep.

    :param disc_feature_groups: Dictionary of feature groups.
    :param feature_group: Name of the feature group that will be replaced by ``to_keep``.
    :param to_keep: List of features that will be kept in the feature group.
    :param remaining_feature_name: Name of the dummy feature that will be used to summarize the
        removed features.
    :returns: Flattened version of the modified dictionary of feature groups.
    """
    # first set the given feature group to just the value that we want to keep
    disc_feature_groups[feature_group] = list(to_keep)
    # then add a new dummy feature to the feature group. `load_data()` will create this for us
    disc_feature_groups[feature_group].append(f"{feature_group}{remaining_feature_name}")
    # then, regenerate the list of discrete features; just like it's done in the constructor
    return flatten_dict(disc_feature_groups)


def label_spec_to_feature_list(spec: LabelSpec) -> list[str]:
    """Extract all the feature column names from a dictionary of label specifications.

    :param spec: Dictionary of label specifications.
    :returns: A flattend list of all the columns occuring in the label specs.
    """
    feature_list: list[str] = []
    for group in spec.values():
        feature_list += group.columns
    return feature_list


def spec_from_binary_cols(label_defs: Mapping[str, Sequence[str]]) -> LabelSpec:
    """Create label specs for the most common case where columns contain 0s and 1s.

    :param label_defs: Mapping of label names to column names.
    :returns: Label specifications.
    """
    multiplier = 1
    label_spec = {}
    for name, columns in label_defs.items():
        label_spec[name] = LabelGroup(list(columns), multiplier=multiplier)
        num_columns = len(columns)
        # we assume here that the columns only contain 0s and 1s
        multiplier *= num_columns if num_columns > 1 else 2
    return label_spec


def single_col_spec(col: str, feature_name: str | None = None) -> LabelSpec:
    """Create a label spec for the case where the label is defined by a single column."""
    return {col if feature_name is None else feature_name: LabelGroup([col])}


def from_dummies(data: pd.DataFrame, categorical_cols: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Convert one-hot encoded columns into categorical columns."""
    out = data.copy()

    for col_parent, filter_col in categorical_cols.items():
        if len(filter_col) > 1:
            undummified = (
                out[list(filter_col)]
                .idxmax(axis=1)
                .apply(lambda x: x.split(f"{col_parent}_", 1)[1])
            )

            out[col_parent] = undummified
            out.drop(filter_col, axis=1, inplace=True)

    return out
