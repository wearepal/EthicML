"""Util functions for the vision datasets."""
from typing import Any, List

import teext as tx
from torch.utils.data import Dataset, Subset, random_split

__all__ = ["set_transform", "train_test_split"]


def set_transform(dataset: Dataset, transform: Any) -> None:
    """Set the transform of a dataset to the specified transform."""
    if hasattr(dataset, "dataset"):
        set_transform(dataset.dataset, transform)  # type: ignore[attr-defined]
    elif isinstance(dataset, Dataset):
        if hasattr(dataset, "transform"):
            dataset.transform = transform  # type: ignore[attr-defined]
        elif hasattr(dataset, "datasets"):
            for dtst in dataset.datasets:  # type: ignore[attr-defined]
                set_transform(dtst, transform)


def train_test_split(dataset: Dataset, train_pcnt: float) -> List[Subset]:
    """Split a dataset into train and test splits, of sizes dictated by the train percentage."""
    tx.assert_percentage(train_pcnt)
    curr_len = len(dataset)  # type: ignore[arg-type]
    train_len = round(train_pcnt * curr_len)
    test_len = curr_len - train_len

    return random_split(dataset, lengths=[train_len, test_len])
