"""Simple class to make class labels binary. Useful if a network uses BCELoss for example."""

import numpy as np
import pandas as pd

from ethicml.utility import DataTuple

__all__ = ["LabelBinarizer"]


def assert_binary_labels(data_tuple: DataTuple) -> None:
    """Assert that datasets only include binary labels."""
    y_col = data_tuple.y.columns[0]
    assert data_tuple.y[y_col].nunique() == 2
    assert (np.unique(data_tuple.y[y_col].to_numpy()) == np.array([0, 1])).all()


class LabelBinarizer:
    """If a dataset has labels [-1,1], then this will make it so the labels = [0,1]."""

    def __init__(self) -> None:
        self.min_val: int
        self.max_val: int

    def adjust(self, dataset: DataTuple) -> DataTuple:
        """Take a datatuple and make the labels [0,1]."""
        y_col = dataset.y.columns[0]
        assert dataset.y[y_col].nunique() == 2

        # make copy of dataset
        dataset = dataset.replace(y=dataset.y.copy())

        self.min_val = dataset.y.to_numpy().min()
        self.max_val = dataset.y.to_numpy().max()

        y_col = dataset.y.columns[0]

        dataset.y[y_col] = dataset.y[y_col].replace(self.min_val, 0)
        dataset.y[y_col] = dataset.y[y_col].replace(self.max_val, 1)

        return DataTuple(x=dataset.x, s=dataset.s, y=dataset.y, name=dataset.name)

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
        y_col = dataset.y.columns[0]
        transformed_y = self.post_only_labels(dataset.y[y_col])
        return dataset.replace(y=pd.DataFrame(transformed_y, columns=[y_col]))
