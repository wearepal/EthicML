import numpy as np
import pandas as pd

from ethicml.algorithms.utils import DataTuple


def assert_binary_labels(train: DataTuple, test: DataTuple):

    y_col = train.y.columns[0]
    assert train.y[y_col].nunique() == 2
    assert test.y[y_col].nunique() == 2

    assert np.unique(train.y[y_col].values) == [0, 1]
    assert np.unique(test.y[y_col].values) == [0, 1]


class Processor():
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def pre(self, dataset: DataTuple) -> DataTuple:

        # make copy of dataset
        dataset = DataTuple(x=dataset.x, s=dataset.s, y=dataset.y.copy())

        self.min_val = dataset.y.values.min()
        self.max_val = dataset.y.values.max()

        y_col = dataset.y.columns[0]

        dataset.y[y_col].replace(self.min_val, 0, inplace=True)
        dataset.y[y_col].replace(self.max_val, 1, inplace=True)

        return DataTuple(
            x=dataset.x,
            s=dataset.s,
            y=dataset.y
        )
