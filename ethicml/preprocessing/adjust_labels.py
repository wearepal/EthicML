"""Simple class to make class labels binary. Useful if a network uses BCELoss for example."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ethicml.utility import DataTuple

__all__ = ["LabelBinarizer"]


def assert_binary_labels(data_tuple: DataTuple) -> None:
    """Assert that datasets only include binary labels."""
    assert data_tuple.y.nunique() == 2
    assert (np.unique(data_tuple.y.to_numpy()) == np.array([0, 1])).all()


class LabelBinarizer:
    """If a dataset has labels [-1,1], then this will make it so the labels = [0,1]."""

    def __init__(self) -> None:
        self.min_val: int
        self.max_val: int

    def adjust(self, dataset: DataTuple) -> DataTuple:
        """Take a datatuple and make the labels [0,1]."""
        assert dataset.y.nunique() == 2

        # make copy of dataset
        new_y = dataset.y.copy()

        self.min_val = new_y.to_numpy().min().item()
        self.max_val = new_y.to_numpy().max().item()

        new_y = new_y.replace(self.min_val, 0)
        new_y = new_y.replace(self.max_val, 1)
        return dataset.replace(y=new_y)

    def post_only_labels(self, labels: pd.Series) -> pd.Series:
        """Inverse of adjust but only for a DataFrame instead of a DataTuple."""
        assert labels.nunique() == 2

        # make copy of the labels
        labels_copy = labels.copy()

        labels_copy = labels_copy.replace(0, self.min_val)
        labels_copy = labels_copy.replace(1, self.max_val)
        return labels_copy

    def post(self, dataset: DataTuple) -> DataTuple:
        """Inverse of adjust."""
        transformed_y = self.post_only_labels(dataset.y)
        return dataset.replace(y=pd.Series(transformed_y, name=dataset.y.name))
