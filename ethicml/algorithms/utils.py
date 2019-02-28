"""
Returns a subset of the data. Used primarily in testing so that kernel methods finish in a
reasonable time
"""
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch


class DataTuple(NamedTuple):
    """A tuple of dataframes for the features, the sensitive attribute and the class labels"""
    x: pd.DataFrame  # features
    s: pd.DataFrame  # senstitive attributes
    y: pd.DataFrame  # class labels


class PathTuple(NamedTuple):
    """For algorithms that run in their own process, we pass around paths to the data"""
    x: Path  # path to file with features
    s: Path  # path to file with sensitive attributes
    y: Path  # path to file with class labels


def get_subset(train: DataTuple) -> DataTuple:
    """Get the first elements of the given dataset

    Args:
        train: training data

    Returns:
        subset of training data
    """
    return DataTuple(
        x=train.x[:][:500],
        s=train.s[:][:500],
        y=train.y[:][:500]
    )


def quadratic_time_mmd(data_first, data_second, sigma):
    """

    Args:
        data_first:
        data_second:
        sigma:

    Returns:

    """
    xx_gm = data_first@data_first.t()
    xy_gm = data_first@data_second.t()
    yy_gm = data_second@data_second.t()
    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    pad_first = lambda x: torch.unsqueeze(x, 0)
    pad_second = lambda x: torch.unsqueeze(x, 1)

    gamma = 1 / (2 * sigma ** 2)
    # use the second binomial formula
    kernel_xx = torch.exp(-gamma * (-2 * xx_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms)))
    kernel_xy = torch.exp(-gamma * (-2 * xy_gm + pad_second(x_sqnorms) + pad_first(y_sqnorms)))
    kernel_yy = torch.exp(-gamma * (-2 * yy_gm + pad_second(y_sqnorms) + pad_first(y_sqnorms)))

    xx_num = float(kernel_xx.shape[0])
    yy_num = float(kernel_yy.shape[0])

    mmd2 = (kernel_xx.sum() / (xx_num * xx_num)
            + kernel_yy.sum() / (yy_num * yy_num)
            - 2 * kernel_xy.sum() / (xx_num * yy_num))
    return mmd2
