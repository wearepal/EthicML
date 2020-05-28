"""Lookup tables / Swutch statements for project level objects."""

from typing import Callable, Dict

from ethicml.data import Dataset, adult, compas, credit, german, nonbinary_toy, sqf, toy

__all__ = ["get_dataset_obj_by_name"]


def get_dataset_obj_by_name(name: str) -> Callable[[], Dataset]:
    """Given a dataset name, get the corresponding dataset object."""
    lookup: Dict[str, Callable[[], Dataset]] = {
        "Adult": adult,
        "Compas": compas,
        "Credit": credit,
        "German": german,
        "NonBinaryToy": nonbinary_toy,
        "SQF": sqf,
        "Toy": toy,
    }

    if name not in lookup:
        raise NotImplementedError("That dataset doesn't exist")

    return lookup[name]
