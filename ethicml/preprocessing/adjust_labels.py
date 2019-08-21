"""
Simplae class to make class labels binary. Useful if a network uses BCELoss for example
"""

import numpy as np
import pandas as pd

from ethicml.utility.data_structures import DataTuple


def assert_binary_labels(train: DataTuple):
    """
    Assert that datasets only include binary labels
    Args:
        train:

    Returns:

    """

    y_col = train.y.columns[0]
    assert train.y[y_col].nunique() == 2
    assert (np.unique(train.y[y_col].values) == np.array([0, 1])).all()


class LabelBinarizer:
    """
    If a dataset has labels [-1,1], then this will make it so the labels = [0,1]
    """

    def __init__(self):
        self.min_val: int
        self.max_val: int

    def adjust(self, dataset: DataTuple) -> DataTuple:
        """
        Take a datatuple and make the labels [0,1]
        Args:
            dataset:

        Returns:

        """
        y_col = dataset.y.columns[0]
        assert dataset.y[y_col].nunique() == 2

        # make copy of dataset
        dataset = dataset.make_copy_with(y=dataset.y.copy())

        self.min_val = dataset.y.values.min()
        self.max_val = dataset.y.values.max()

        y_col = dataset.y.columns[0]

        dataset.y[y_col].replace(self.min_val, 0, inplace=True)
        dataset.y[y_col].replace(self.max_val, 1, inplace=True)

        return DataTuple(x=dataset.x, s=dataset.s, y=dataset.y, name=dataset.name)

    def post_only_labels(self, labels: pd.DataFrame) -> pd.DataFrame:
        """Inverse of adjust but only for a DataFrame instead of a DataTuple"""
        y_col = labels.columns[0]
        assert labels[y_col].nunique() == 2

        # make copy of the labels
        labels_copy = labels.copy()

        y_col = labels_copy.columns[0]
        labels_copy[y_col].replace(0, self.min_val, inplace=True)
        labels_copy[y_col].replace(1, self.max_val, inplace=True)
        return labels_copy

    def post(self, dataset: DataTuple) -> DataTuple:
        """Inverse of adjust"""

        transformed_y = self.post_only_labels(dataset.y)
        return dataset.make_copy_with(y=transformed_y)
