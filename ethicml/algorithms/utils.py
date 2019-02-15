"""
Returns a subset of the data. Used primarily in testing so that kernel methods finish in a reasonable time
"""

from typing import Dict
import pandas as pd
import torch


def get_subset(train: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    return {
        'x': train['x'][:][:500],
        's': train['s'][:][:500],
        'y': train['y'][:][:500]
    }


def make_dict(x_val: pd.DataFrame, s_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        'x': x_val,
        's': s_val,
        'y': y_val
    }


def quadratic_time_mmd(data_first, data_second, sigma):
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
