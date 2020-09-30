"""Common items for Facct paper."""

from typing import Optional, Tuple

import torch
from torch import Tensor, autograd, nn


class GradReverse(autograd.Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:
        """Do GRL."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Do GRL."""
        return grad_output.neg().mul(ctx.lambda_), None


def grad_reverse(features: Tensor, lambda_: float = 1.0) -> Tensor:
    """Gradient Reversal layer."""
    return GradReverse.apply(features, lambda_)


def init_weights(m):
    """Make Linear layer weights initialised with Xavier Norm."""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


def block(in_dim: int, out_dim: int):
    """Make a simple block."""
    linear = nn.Linear(in_dim, out_dim)
    bn = nn.BatchNorm1d(out_dim)
    relu = nn.LeakyReLU()
    seq = nn.Sequential(linear, bn, relu)
    seq.apply(init_weights)
    return seq


def mid_blocks(latent_dim, latent_multipler, blocks):
    """Build middle blocks for hidden layers."""
    return (
        [
            block(latent_dim * latent_multipler, latent_dim * latent_multipler)
            for _ in range(blocks - 1)
        ]
        if blocks > 1
        else []
    )


def get_device(device: int) -> torch.device:
    """Assign a torch device."""
    if torch.cuda.is_available() and device >= 0:
        return torch.device(f"cuda:{device}")
    return torch.device("cpu")
