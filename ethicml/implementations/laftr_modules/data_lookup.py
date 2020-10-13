"""Lookup."""
from typing import Tuple

from ethicml import Dataset, DataTuple, adult, health, toy, train_test_split


def data_lookup(dataset: str) -> Dataset:
    """Get an EthicML dataset from a string."""
    lookup = {
        "adult": adult(binarize_nationality=True),
        "health": health(),
        "toy": toy(),
    }
    return lookup[dataset]


def get_data(dataset: str) -> Tuple[DataTuple, DataTuple]:
    """Lookup EthicML dataset based on str."""
    data = data_lookup(dataset).load()
    return train_test_split(data)
