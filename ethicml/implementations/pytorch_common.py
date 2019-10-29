"""Functions that are common to PyTorch models"""
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ethicml.utility.data_structures import DataTuple, TestTuple


def _get_info(
    data: TestTuple,
) -> Tuple["np.ndarray[np.float32]", "np.ndarray[np.float32]", int, int, int, List[str], List[str]]:
    features = data.x.to_numpy(dtype=np.float32)
    sens_labels = data.s.to_numpy(dtype=np.float32)
    num = data.s.shape[0]
    xdim = data.x.shape[1]
    sdim = data.s.shape[1]
    x_names = data.x.columns
    s_names = data.s.columns
    return features, sens_labels, num, xdim, sdim, x_names, s_names


class TestDataset(Dataset):
    """Shared Dataset for pytorch models without labels"""

    def __init__(self, data: TestTuple):
        super().__init__()
        self.x, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.x[index, ...], self.s[index, ...]

    def __len__(self) -> int:
        return self.num

    @property
    def names(self) -> Tuple[List[str], List[str]]:
        return self.x_names, self.s_names


class CustomDataset(Dataset):
    """Shared Dataset for pytorch models"""

    def __init__(self, data: DataTuple):
        super().__init__()
        test = data.remove_y()
        self.x, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(test)
        self.y = data.y.to_numpy(dtype=np.float32)
        self.ydim = data.y.shape[1]
        self.y_names = data.y.columns

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x[index, ...], self.s[index, ...], self.y[index, ...]

    def __len__(self) -> int:
        return self.num

    @property
    def names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.x_names, self.s_names, self.y_names


def quadratic_time_mmd(data_first: Tensor, data_second: Tensor, sigma: float) -> Tensor:
    """

    Args:
        data_first:
        data_second:
        sigma:

    Returns:

    """
    xx_gm = data_first @ data_first.t()
    xy_gm = data_first @ data_second.t()
    yy_gm = data_second @ data_second.t()
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

    mmd2 = (
        kernel_xx.sum() / (xx_num * xx_num)
        + kernel_yy.sum() / (yy_num * yy_num)
        - 2 * kernel_xy.sum() / (xx_num * yy_num)
    )
    return mmd2
