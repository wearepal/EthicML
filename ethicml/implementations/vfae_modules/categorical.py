"""Implementation for Louizos et al Variational Fair Autoencoder."""
# pylint: disable=arguments-differ
from __future__ import annotations

import torch
from torch import nn


class _OneHotEncoder(nn.Module):
    """One Hot Encode the output based on feature groups."""

    def __init__(self, n_dims: int, index_dim: int = 1):
        super().__init__()
        self.n_dims = n_dims
        self.index_dim = index_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = x.argmax(dim=self.index_dim)
        indices = indices.type(torch.int64).view(-1, 1)
        n_dims = self.n_dims  # if self.n_dims is not None else int(torch.max(indices)) + 1
        one_hots = torch.zeros(indices.size()[0], n_dims).scatter_(1, indices, 1)
        one_hots = one_hots.view(*indices.shape, -1)
        return one_hots


class Categorical(nn.Module):
    """Group a category together."""

    def __init__(self, in_feat: int, dims: int):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(in_feat, dims), nn.Softmax(dim=-1))
        self.ohe = _OneHotEncoder(n_dims=dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.layer(x)
        if not self.training:
            out = self.ohe(out)
        return out
