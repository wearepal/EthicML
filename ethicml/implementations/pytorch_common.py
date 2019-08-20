"""Functions that are common to PyTorch models"""
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import torch
from torch.optim.optimizer import Optimizer

from ethicml.utility.data_structures import DataTuple, TestTuple
from itertools import groupby


class TestDataset(Dataset):
    """Shared Dataset for pytorch models without labels"""

    def __init__(self, data: TestTuple):
        self.features = data.x.to_numpy(dtype=np.float32)
        self.sens_labels = data.s.to_numpy(dtype=np.float32)
        self.num = data.s.shape[0]
        self.s_size = data.s.shape[1]
        self.size = data.x.shape[1]
        self.x_names = data.x.columns
        self.s_names = data.s.columns
        self.column_lookup = {col: i for i, col in enumerate(self.x_names)}
        self.groups = [
            list(group) for key, group in groupby(self.x_names, lambda x: x.split('_')[0])
        ]
        self.group_columns = [[self.column_lookup[g] for g in group] for group in self.groups]

    def __getitem__(self, index):
        return self.features[index], self.sens_labels[index]

    def __len__(self):
        return self.num

    def names(self):
        return self.x_names, self.s_names


class CustomDataset(TestDataset):
    """Shared Dataset for pytorch models"""

    def __init__(self, data: DataTuple):
        super().__init__(data.remove_y())
        self.class_labels = data.y.to_numpy(dtype=np.float32)
        self.y_size = data.y.shape[1]
        self.y_names = data.y.columns

    def __getitem__(self, index):

        _x = self.features[index]
        _s = self.sens_labels[index]
        _y = self.class_labels[index]
        _x_groups = [_x[group] for group in self.group_columns]

        return (
            torch.from_numpy(_x),
            torch.from_numpy(_s),
            torch.from_numpy(_y),
            [torch.from_numpy(_g) for _g in _x_groups],
        )

    def names(self):
        return self.x_names, self.s_names, self.y_names


def quadratic_time_mmd(data_first, data_second, sigma):
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


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
