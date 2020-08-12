"""Lookup tables / Swutch statements for project level objects."""

from typing import Callable, Dict

from ethicml.data import (
    Dataset,
    adult,
    compas,
    credit,
    crime,
    german,
    health,
    nonbinary_toy,
    sqf,
    toy,
)

__all__ = ["get_dataset_obj_by_name"]


def get_dataset_obj_by_name(name: str) -> Callable[[], Dataset]:
    """Given a dataset name, get the corresponding dataset object."""
    lookup: Dict[str, Callable[[], Dataset]] = {
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
    lowercase_name = name.lower()
    if lowercase_name not in lookup:
        raise NotImplementedError("That dataset doesn't exist")

    return lookup[lowercase_name]
