"""Autoencoder for X."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from typing_extensions import Literal

from ethicml import implements
from ethicml.implementations.laftr_modules.base_model import BaseModel
from ethicml.implementations.laftr_modules.common import block, grad_reverse, mid_blocks


class Encoder(nn.Module):
    """Encoder model."""

    @implements(nn.Module)
    def __init__(self, in_size: int, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(in_size, latent_dim)
        else:
            _blocks = [block(in_size, latent_dim * hid_multiplier)] + mid_blocks(
                latent_dim, hid_multiplier, blocks
            )
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(latent_dim * hid_multiplier, latent_dim)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        z = self.hid(x)
        return self.out(z)


class Adversary(nn.Module):
    """Adversary for AE."""

    @implements(nn.Module)
    def __init__(self, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(latent_dim, 1)
        else:
            _blocks = [block(latent_dim, latent_dim * hid_multiplier)] + mid_blocks(
                latent_dim, hid_multiplier, blocks
            )
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(latent_dim * hid_multiplier, 1)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, z: Tensor) -> Tensor:
        s = self.hid(grad_reverse(z))
        return self.out(s).squeeze(-1)


class Decoder(nn.Module):
    """Decoder model."""

    @implements(nn.Module)
    def __init__(self, latent_dim: int, in_size: int, blocks: int, hid_multiplier: int) -> None:
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(latent_dim + 2, in_size)
        else:
            _blocks = [block(latent_dim + 2, latent_dim * hid_multiplier)] + mid_blocks(
                latent_dim, hid_multiplier, blocks
            )
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(latent_dim * hid_multiplier, in_size)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, z: td.Distribution, s: torch.Tensor) -> Tensor:
        y = self.hid(torch.cat([z, s], dim=1))
        return self.out(y)


class Predictor(nn.Module):
    """Predictor module."""

    @implements(nn.Module)
    def __init__(self, latent_dim: int, blocks: int, hid_multiplier: int):
        super().__init__()
        if blocks == 0:
            self.hid = nn.Identity()
            self.out = nn.Linear(latent_dim, 1)
        else:
            _blocks = [block(latent_dim, latent_dim * hid_multiplier)] + mid_blocks(
                latent_dim, hid_multiplier, blocks
            )
            self.hid = nn.Sequential(*_blocks)
            self.out = nn.Linear(latent_dim * hid_multiplier, 1)
        nn.init.xavier_normal_(self.out.weight)

    @implements(nn.Module)
    def forward(self, z: td.Distribution) -> Tensor:
        z = self.hid(z)
        return self.out(z).squeeze(-1)


class LaftrAE(BaseModel):
    """Autoencoder model."""

    @implements(nn.Module)
    def __init__(
        self,
        in_size: int,
        latent_dim: int,
        blocks: int,
        hidden_multiplier: int,
        disc_feature_group_slices: List[slice],
    ):
        super().__init__()
        self.enc = Encoder(in_size, latent_dim, blocks=blocks, hid_multiplier=hidden_multiplier)
        self.adv = Adversary(latent_dim, blocks=blocks, hid_multiplier=hidden_multiplier)
        self.dec = Decoder(latent_dim, in_size, blocks=blocks, hid_multiplier=hidden_multiplier)
        self.pred = Predictor(latent_dim, blocks=blocks, hid_multiplier=hidden_multiplier)
        self._optim: Optional[torch.optim.Optimizer] = None
        self.disc_feature_group_slices = disc_feature_group_slices

    @implements(nn.Module)
    def forward(self, x: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # OHE Sens
        s_prime = torch.ones_like(s) - s
        s = torch.cat([torch.unsqueeze(s, -1), torch.unsqueeze(s_prime, -1)], dim=-1)

        z = self.enc(x)
        recon = self.dec(z, s)
        s_pred = self.adv(z)
        y_pred = self.pred(z)
        return z, s_pred, recon, y_pred

    def predict(self, x: Tensor, s: Tensor) -> np.ndarray:
        """Return class predictions."""
        self.eval()
        _, _, _, yh = self(x, s)
        return yh.sigmoid().round().detach().cpu().numpy().squeeze()

    def fit(
        self,
        *,
        data_loader: DataLoader,
        epochs: int,
        device: torch.device,
        warmup_steps: int,
        reg_weight: float,
        adv_weight: float,
        scheduler: Optional[torch.optim.lr_scheduler.ExponentialLR],
        additional_adv_steps: int,
        pred_weight: float,
    ) -> LaftrAE:
        """Train the model."""
        self.train()
        for epoch in range(epochs):
            results_dict = self._train_iteration(
                data_loader=data_loader,
                device=device,
                warmup=epoch < warmup_steps,
                reg_weight=reg_weight,
                adv_weight=adv_weight,
                additional_adv_steps=additional_adv_steps,
                pred_weight=pred_weight,
            )
            results_dict.update({"epoch": epoch, "epochs": epochs})
            self.log_results(results_dict)
            if scheduler:
                scheduler.step()
        self.eval()
        return self

    def _train_iteration(
        self,
        *,
        data_loader: DataLoader,
        device: torch.device,
        warmup: bool,
        reg_weight: float,
        adv_weight: float,
        additional_adv_steps: int,
        pred_weight: float,
    ) -> Dict[str, float]:
        enc_loss = 0.0
        enc_recon_loss = 0.0
        enc_adv_loss = 0.0
        enc_latent_norm = 0.0
        enc_pred_loss = 0.0
        _adv_weight = 0.0 if warmup else adv_weight
        for _, (data_x, data_s, data_y) in enumerate(data_loader):
            data_x = data_x.to(device)
            data_s = data_s.to(device)

            self.optim.zero_grad()
            z, s_pred, x_pred, y_pred = self(data_x, data_s)

            feat_sens_loss = F.binary_cross_entropy_with_logits(s_pred, data_s, reduction="mean")
            pred_loss = F.binary_cross_entropy_with_logits(y_pred, data_y, reduction="mean")

            if self.disc_feature_group_slices:
                recon_loss = F.mse_loss(
                    x_pred[:, slice(self.disc_feature_group_slices[-1].stop, data_x.shape[1])],
                    data_x[:, slice(self.disc_feature_group_slices[-1].stop, data_x.shape[1])],
                    reduction="mean",
                )
                for group_slice in self.disc_feature_group_slices:
                    recon_loss += F.cross_entropy(
                        x_pred[:, group_slice],
                        torch.argmax(data_x[:, group_slice], dim=-1),
                        reduction="mean",
                    )
                # recon_loss /= len(self.disc_feature_group_slices) + 1

            else:
                recon_loss = F.mse_loss(x_pred, data_x, reduction="mean")

            loss = (
                recon_loss
                + _adv_weight * feat_sens_loss
                + reg_weight * z.norm(dim=1).mean()
                + pred_weight * pred_loss
            )

            loss.backward()
            self.optim.step()

            # calculate the loss again for monitoring
            self.eval()
            z, s_pred, x_pred, y_pred = self(data_x, data_s)

            enc_adv_loss += _adv_weight * feat_sens_loss.item()
            enc_latent_norm += reg_weight * z.norm(dim=1).mean().item()
            enc_loss += loss.item()
            enc_recon_loss += recon_loss.item()
            enc_pred_loss += pred_weight * pred_loss.item()
            self.train()

            if not warmup:
                for _ in range(additional_adv_steps):
                    self.optim.zero_grad()
                    _, _sp, _ = self(data_x, data_s)
                    loss = F.binary_cross_entropy_with_logits(_sp, data_s, reduction="mean")
                    loss.backward()
                    self.optim.step()

        return {
            "enc_adv_loss": _adv_weight * enc_adv_loss / len(data_loader),
            "enc_latent_norm": reg_weight * enc_latent_norm / len(data_loader),
            "enc_loss": enc_loss / len(data_loader),
            "enc_recon_loss": enc_recon_loss / len(data_loader),
            "pred_loss": enc_pred_loss / len(data_loader),
        }

    def bottleneck(self, x: Tensor, s: Tensor) -> np.ndarray:
        """Return the latent embedding."""
        self.eval()
        z, _, _, _ = self(x, s)
        return z.detach().cpu().numpy()

    @staticmethod
    def to_discrete(inputs):
        """Discretize the data."""
        if inputs.dim() <= 1 or inputs.size(1) <= 1:
            return inputs.round()
        else:
            argmax = inputs.argmax(dim=1)
            return F.one_hot(argmax, num_classes=inputs.size(1))

    def invert(self, z, discretize: bool = True) -> Tensor:
        """Go from soft to discrete features."""
        if discretize and self.disc_feature_group_slices:
            for group_slice in self.disc_feature_group_slices:
                one_hot = self.to_discrete(z[:, group_slice])
                z[:, group_slice] = one_hot

        return z

    @overload
    def recon(self, x: Tensor, s: Tensor, as_numpy: Literal[True] = ...) -> np.ndarray:
        ...

    @overload
    def recon(self, x: Tensor, s: Tensor, as_numpy: Literal[False]) -> Tensor:
        ...

    def recon(self, x: Tensor, s: Tensor, as_numpy: bool = True) -> Union[np.ndarray, Tensor]:
        """Return Reconstruction."""
        self.eval()
        _, _, recon, _ = self(x, s)
        recon = self.invert(recon)
        return recon.detach().cpu().numpy() if as_numpy else recon

    def get_recon(self, data_loader: DataLoader):
        """Run dataloader through model."""
        recon = None
        for j in data_loader:
            x, s = self.unpack(j)
            z = self.recon(x, s)
            recon = z if recon is None else np.append(recon, z, axis=0)
        return recon

    def get_latent(self, data_loader: DataLoader):
        """Run dataloader through model."""
        latent = None
        for j in data_loader:
            x, s = self.unpack(j)
            z = self.bottleneck(x, s)
            latent = z if latent is None else np.append(latent, z, axis=0)
        return latent
