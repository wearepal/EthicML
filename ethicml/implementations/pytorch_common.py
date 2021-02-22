"""Functions that are common to PyTorch models."""
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from ethicml.utility import DataTuple, TestTuple


def _get_info(data: TestTuple) -> Tuple[np.ndarray, np.ndarray, int, int, int, pd.Index, pd.Index]:
    features = data.x.to_numpy(dtype=np.float32)
    sens_labels = data.s.to_numpy(dtype=np.float32)
    num = data.s.shape[0]
    xdim = data.x.shape[1]
    sdim = data.s.shape[1]
    x_names = data.x.columns
    s_names = data.s.columns
    return features, sens_labels, num, xdim, sdim, x_names, s_names


class TestDataset(Dataset):
    """Shared Dataset for pytorch models without labels."""

    def __init__(self, data: TestTuple):
        super().__init__()
        self.x, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Implement __getitem__ magic method."""
        return self.x[index, ...], self.s[index, ...]

    def __len__(self) -> int:
        """Implement __len__ magic method."""
        return self.num

    @property
    def names(self) -> Tuple[pd.Index, pd.Index]:
        """Get tuple of x names and s names."""
        return self.x_names, self.s_names


class CustomDataset(Dataset):
    """Shared Dataset for pytorch models."""

    def __init__(self, data: DataTuple):
        super().__init__()
        test = data.remove_y()
        self.x, self.s, self.num, self.xdim, self.sdim, self.x_names, self.s_names = _get_info(test)
        self.y = data.y.to_numpy(dtype=np.float32)
        self.ydim = data.y.shape[1]
        self.y_names = data.y.columns

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement __getitem__ magic method."""
        return self.x[index, ...], self.s[index, ...], self.y[index, ...]

    def __len__(self) -> int:
        """Implement __len__ magic method."""
        return self.num

    @property
    def names(self) -> Tuple[pd.Index, pd.Index, pd.Index]:
        """Get tuple of x names, s names and y names."""
        return self.x_names, self.s_names, self.y_names


def quadratic_time_mmd(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    """Calculate MMD betweer 2 tensors of equal size.

    Args:
        x: Sample 1.
        y: Sample 2.
        sigma: Scale of the RBF kernel.

    Returns:
        Tensor of MMD in each dim.
    """
    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()
    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    def pad_first(x):
        return torch.unsqueeze(x, 0)

    def pad_second(x):
        return torch.unsqueeze(x, 1)

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


def compute_projection_gradients(
    model: nn.Module, loss_p: Tensor, loss_a: Tensor, alpha: float
) -> None:
    """Computes the adversarial gradient projection term.

    Args:
        model (nn.Module): Model whose parameters the gradients are to be computed w.r.t.
        loss_p (Tensor): Prediction loss.
        loss_a (Tensor): Adversarial loss.
        alpha (float): Pre-factor for adversarial loss.
    """
    grad_p = torch.autograd.grad(loss_p, model.parameters(), retain_graph=True)
    grad_a = torch.autograd.grad(loss_a, model.parameters(), retain_graph=True)

    def _proj(a: Tensor, b: Tensor) -> Tensor:
        return b * torch.sum(a * b) / torch.sum(b * b)

    grad_p = [p - _proj(p, a) - alpha * a for p, a in zip(grad_p, grad_a)]

    for param, grad in zip(model.parameters(), grad_p):
        param.grad = grad
