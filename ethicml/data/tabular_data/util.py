"""Useful methods that are used in some of the data objects."""

from itertools import groupby
from typing import List

__all__ = [
    "get_concatenated_features",
    "filter_features_by_prefixes",
    "get_discrete_features",
    "group_disc_feat_indexes",
]


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


def filter_features_by_prefixes(features: List[str], prefixes: List[str]) -> List[str]:
    """Filter the features by prefixes.

    Args:
        features: list of features names
        prefixes: list of prefixes

    Returns:
        filtered feature names
    """
    res: List[str] = []
    for name in features:
        filtered = False
        for pref in prefixes:
            if name.startswith(pref):
                filtered = True
                break
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
