"""Useful methods that are used in some of the data objects."""
from itertools import groupby
from typing import Dict, List, Mapping, NamedTuple, Sequence

__all__ = [
    "LabelSpec",
    "PartialLabelSpec",
    "simple_spec",
    "get_concatenated_features",
    "filter_features_by_prefixes",
    "get_discrete_features",
    "group_disc_feat_indexes",
    "label_specs_to_feature_list",
]


class PartialLabelSpec(NamedTuple):
    """Specification for part of a label (can be class label or sensitive attribute)."""

    columns: List[str]
    multiplier: int = 1


LabelSpec = Mapping[str, PartialLabelSpec]


def get_concatenated_features(
    s_prefix: List[str], y_prefix: List[str], continuous_features: List[str], discrete_only: bool
) -> List[str]:
    """Absolutely no idea why this is here.

    It's not used anywhere. Mark for deletion.

    Args:
        s_prefix:
        y_prefix:
        continuous_features:
        discrete_only:

    Returns:
        list of features prepended with the sensitive label and class label.
    """
    return s_prefix + y_prefix + continuous_features if discrete_only else s_prefix + y_prefix


def filter_features_by_prefixes(features: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """Filter the features by prefixes.

    Args:
        features: list of features names
        prefixes: list of prefixes

    Returns:
        filtered feature names
    """
    res: List[str] = []
    for name in features:
        filtered = any(name.startswith(pref) for pref in prefixes)
        if not filtered:
            res.append(name)
    return res


def get_discrete_features(
    all_feats: List[str], feats_to_remove: List[str], cont_feats: List[str]
) -> List[str]:
    """Get a list of the discrete features in a dataset.

    Args:
        all_feats: List of all features in the dataset.
        feats_to_remove: List of features that aren't used.
        cont_feats: List of continuous features in the dataset.

    Returns:
        List of features not marked as continuous or to be removed.
    """
    return [
        item
        for item in filter_features_by_prefixes(all_feats, feats_to_remove)
        if item not in cont_feats
    ]


def group_disc_feat_indexes(disc_feat_names: List[str], prefix_sep: str = "_") -> List[slice]:
    """Group discrete features names according to the first segment of their name.

    Returns a list of their corresponding slices (assumes order is maintained).
    """

    def _first_segment(feature_name: str) -> str:
        return feature_name.split(prefix_sep)[0]

    group_iter = groupby(disc_feat_names, _first_segment)

    feature_slices: List[slice] = []
    start_idx = 0
    for _, group in group_iter:
        len_group = len(list(group))
        indexes = slice(start_idx, start_idx + len_group)
        feature_slices.append(indexes)
        start_idx += len_group

    return feature_slices


def flatten_dict(dictionary: Mapping[str, List[str]]) -> List[str]:
    """Flatten a dictionary of lists by joining all lists to one big list."""
    return [x for inner_list in dictionary.values() for x in inner_list]


def reduce_feature_group(
    disc_feature_groups: Dict[str, List[str]],
    feature_group: str,
    to_keep: Sequence[str],
    remaining_feature_name: str,
) -> List[str]:
    """Drop all features in the given feature group except the ones in to_keep."""
    # first set the given feature group to just the value that we want to keep
    disc_feature_groups[feature_group] = list(to_keep)
    # then add a new dummy feature to the feature group. `load_data()` will create this for us
    disc_feature_groups[feature_group].append(f"{feature_group}{remaining_feature_name}")
    # then, regenerate the list of discrete features; just like it's done in the constructor
    return flatten_dict(disc_feature_groups)


def label_specs_to_feature_list(specs: LabelSpec) -> List[str]:
    """Extract all the feature column names from a dictionary of label specifications."""
    feature_list: List[str] = []
    for spec in specs.values():
        feature_list += spec.columns
    return feature_list


def simple_spec(label_defs: Mapping[str, Sequence[str]]) -> LabelSpec:
    """Create label specs for the most common case where columns contain 0s and 1s."""
    multiplier = 1
    label_specs = {}
    for name, columns in label_defs.items():
        label_specs[name] = PartialLabelSpec(list(columns), multiplier=multiplier)
        num_columns = len(columns)
        # we assume here that the columns only contain 0s and 1s
        multiplier *= num_columns if num_columns > 1 else 2
    return label_specs
