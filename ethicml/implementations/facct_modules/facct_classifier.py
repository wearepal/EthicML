"""The Classification AE module."""

import torch
import torch.distributions as td
from torch import Tensor, cat, nn

from ethicml import implements
from ethicml.implementations.facct_modules.common import block, grad_reverse, mid_blocks

LATENT_MULTIPLIER = 10


class Encoder(nn.Module):
    """Encoder Module."""

    @implements(nn.Module)
    def __init__(self, in_size: int, latent_dim: int, blocks: int):
        super().__init__()
        _blocks = [block(in_size, latent_dim * LATENT_MULTIPLIER)] + mid_blocks(
            latent_dim, LATENT_MULTIPLIER, blocks
        )
        self.hid = nn.Sequential(*_blocks)

        self.out = nn.Linear(latent_dim * LATENT_MULTIPLIER, latent_dim)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, x: Tensor):
        z = self.hid(x)
        return self.out(z)


class Adversary(nn.Module):
    """Adversary module."""

    @implements(nn.Module)
    def __init__(self, latent_dim: int, blocks: int):
        super().__init__()
        _blocks = [block(latent_dim, latent_dim * LATENT_MULTIPLIER)] + mid_blocks(
            latent_dim, LATENT_MULTIPLIER, blocks
        )
        self.hid = nn.Sequential(*_blocks)

        self.out = nn.Linear(latent_dim * LATENT_MULTIPLIER, 1)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, z: Tensor):
        sp = self.hid(grad_reverse(z))
        return self.out(sp)


class Predictor(nn.Module):
    """Predictor module."""

    @implements(nn.Module)
    def __init__(self, latent_dim: int, blocks: int):
        super().__init__()
        _blocks = [block(latent_dim + 2, latent_dim * LATENT_MULTIPLIER)] + mid_blocks(
            latent_dim, LATENT_MULTIPLIER, blocks
        )
        self.hid = nn.Sequential(*_blocks)

        self.out = nn.Linear(latent_dim * LATENT_MULTIPLIER, 1)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, z: td.Distribution, s: Tensor):
        z = self.hid(cat([z, s], dim=1))
        return self.out(z)


class FacctClassifier(nn.Module):
    """Classification AE."""

    @implements(nn.Module)
    def __init__(self, in_size: int, latent_dim: int):
        super().__init__()
        self.enc = Encoder(in_size, latent_dim, blocks=2)
        self.adv = Adversary(latent_dim, blocks=2)
        self.pred = Predictor(latent_dim, blocks=2)

    @implements(nn.Module)
    def forward(self, x, s):
        # OHE Sens
        s_prime = torch.ones_like(s) - s
        s = torch.cat([s, s_prime], dim=1)

        z = self.enc(x)
        s_pred = self.adv(z)
        y = self.pred(z, s)
        return z, s_pred, y
