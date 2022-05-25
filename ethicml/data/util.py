"""Useful methods that are used in some of the data objects."""
import functools
import warnings
from itertools import groupby
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, TypeVar
from typing_extensions import ParamSpec

__all__ = [
    "LabelGroup",
    "LabelSpec",
    "deprecated",
    "filter_features_by_prefixes",
    "get_discrete_features",
    "group_disc_feat_indices",
    "label_spec_to_feature_list",
    "simple_spec",
]

_P = ParamSpec("_P")
_T = TypeVar("_T")


def deprecated(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Decorate functions as deprecated.

    It will result in a warning being emitted when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(
            f"The {func.__name__} class is deprecated. "
            f"Use the function `{func.__name__.lower()}` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


class LabelGroup(NamedTuple):
    """Definition of a group of columns that should be interpreted as a single label."""

    columns: List[str]  # list of columns that belong to the group
    multiplier: int = 1  # multiplier assigned to this group (needed when combining groups)


LabelSpec = Mapping[str, LabelGroup]


def filter_features_by_prefixes(features: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """Filter the features by prefixes.

    :param features: list of features names
    :param prefixes: list of prefixes
    :returns: filtered feature names
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


def group_disc_feat_indices(disc_feat_names: List[str], prefix_sep: str = "_") -> List[slice]:
    """Group discrete features names according to the first segment of their name.

    Returns a list of their corresponding slices (assumes order is maintained).

    :param disc_feat_names: List of discrete feature names.
    :param prefix_sep: Separator between the prefix and the rest of the name. (Default: "_")
    :returns: List of slices.
    """

    def _first_segment(feature_name: str) -> str:
        return feature_name.split(prefix_sep)[0]

    group_iter = groupby(disc_feat_names, _first_segment)

    feature_slices: List[slice] = []
    start_idx = 0
    for _, group in group_iter:
        len_group = len(list(group))
        indices = slice(start_idx, start_idx + len_group)
        feature_slices.append(indices)
        start_idx += len_group

    return feature_slices


def flatten_dict(dictionary: Optional[Mapping[str, List[str]]]) -> List[str]:
    """Flatten a dictionary of lists by joining all lists to one big list."""
    if dictionary is None:
        return []
    return [x for inner_list in dictionary.values() for x in inner_list]


def reduce_feature_group(
    disc_feature_groups: Dict[str, List[str]],
    feature_group: str,
    to_keep: Sequence[str],
    remaining_feature_name: str,
) -> List[str]:
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


def label_spec_to_feature_list(spec: LabelSpec) -> List[str]:
    """Extract all the feature column names from a dictionary of label specifications.

    :param spec: Dictionary of label specifications.
    """
    feature_list: List[str] = []
    for group in spec.values():
        feature_list += group.columns
    return feature_list


def simple_spec(label_defs: Mapping[str, Sequence[str]]) -> LabelSpec:
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
