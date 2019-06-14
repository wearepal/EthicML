"""
Useful methods that are used in some of the data objects
"""

from typing import List


def get_concatenated_features(
    s_prefix: List[str], y_prefix: List[str], continuous_features: List[str], discrete_only: bool
) -> List[str]:
    """

    Args:
        s_prefix:
        y_prefix:
        continuous_features:
        discrete_only:

    Returns:

    """
    return s_prefix + y_prefix + continuous_features if discrete_only else s_prefix + y_prefix


def filter_features_by_prefixes(features: List[str], prefixes: List[str]):
    """Filter the features by prefixes

    Args:
        features: list of features names
        prefixes: list of prefixes

    Returns:
        filtered feature names
    """
    res = []
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
    """

    Args:
        all_feats:
        feats_to_remove:
        cont_feats:

    Returns:

    """
    return [
        item
        for item in filter_features_by_prefixes(all_feats, feats_to_remove)
        if item not in cont_feats
    ]
