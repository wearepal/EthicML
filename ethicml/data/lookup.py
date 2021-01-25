"""Lookup tables / switch statements for project level objects."""

from typing import Callable, Dict, List

from .dataset import Dataset
from .tabular_data.adult import adult
from .tabular_data.compas import compas
from .tabular_data.credit import credit
from .tabular_data.crime import crime
from .tabular_data.german import german
from .tabular_data.health import health
from .tabular_data.non_binary_toy import nonbinary_toy
from .tabular_data.sqf import sqf
from .tabular_data.toy import toy

__all__ = ["available_tabular", "get_dataset_obj_by_name"]


def _lookup_table() -> Dict[str, Callable[[], Dataset]]:
    return {
        adult.__name__: adult,
        compas.__name__: compas,
        credit.__name__: credit,
        crime.__name__: crime,
        german.__name__: german,
        nonbinary_toy.__name__: nonbinary_toy,
        health.__name__: health,
        sqf.__name__: sqf,
        toy.__name__: toy,
    }


def get_dataset_obj_by_name(name: str) -> Callable[[], Dataset]:
    """Given a dataset name, get the corresponding dataset object."""
    lookup = _lookup_table()
    lowercase_name = name.lower()
    if lowercase_name not in lookup:
        raise NotImplementedError("That dataset doesn't exist")

    return lookup[lowercase_name]


def available_tabular() -> List[str]:
    """List of tabular dataset names."""
    return list(_lookup_table())
