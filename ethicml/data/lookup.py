"""Lookup tables / switch statements for project level objects."""
from __future__ import annotations
from typing import Callable

from .dataset import Dataset
from .tabular_data.adult import Adult
from .tabular_data.compas import Compas
from .tabular_data.credit import Credit
from .tabular_data.crime import Crime
from .tabular_data.german import German
from .tabular_data.health import Health
from .tabular_data.non_binary_toy import NonBinaryToy
from .tabular_data.sqf import Sqf
from .tabular_data.toy import Toy

__all__ = ["available_tabular", "get_dataset_obj_by_name"]


def _lookup_table() -> dict[str, Callable[[], Dataset]]:
    return {
        Adult.__name__.lower(): Adult,
        Compas.__name__.lower(): Compas,
        Credit.__name__.lower(): Credit,
        Crime.__name__.lower(): Crime,
        German.__name__.lower(): German,
        NonBinaryToy.__name__.lower(): NonBinaryToy,
        Health.__name__.lower(): Health,
        Sqf.__name__.lower(): Sqf,
        Toy.__name__.lower(): Toy,
    }


def get_dataset_obj_by_name(name: str) -> Callable[[], Dataset]:
    """Given a dataset name, get the corresponding dataset object.

    :param name: Name of the dataset.
    :returns: A callable that can be used to construct the dataset object.
    :raises NotImplementedError: If the given name does not correspond to a dataset.
    """
    lookup = _lookup_table()
    lowercase_name = name.lower()
    if lowercase_name not in lookup:
        raise NotImplementedError("That dataset doesn't exist")

    return lookup[lowercase_name]


def available_tabular() -> list[str]:
    """List of tabular dataset names."""
    return list(_lookup_table())
